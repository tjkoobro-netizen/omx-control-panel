#!/usr/bin/env python3
"""Camera fault watch of an evaluation (3.31.0, TK 11/26).

11/26 04:36-05:00 the top camera front sent a fully black picture for about 25 minutes at a normal 29.8 fps with no
read error: nothing warned, the one-arm loading's placement judge (front) never saw the box placed and the policy ran
123 s, pushing the Pinky. This watch looks at the pictures themselves, not only at the frame rate.

States of one camera (from its latest frame, every 8th pixel, about 5 times a second):
  stall   no new frame for STALL_S (the read thread stopped, the USB dropped)
  black   mean < BLACK_MEAN and std < FLAT_STD         (11/26 front: mean 0.0)
  white   mean > WHITE_MEAN and std < FLAT_STD
  flat    std < SOLID_STD (one solid colour: a "no signal" screen)
  frozen  new frames keep coming but the picture is bit for bit the same (mean abs diff < FROZEN_DIFF) for FROZEN_S
  dim / glare  mean < DIM_MEAN / > GLARE_MEAN: shown and written, never acted on
A fault state counts once it has lasted CONFIRM_S (frozen: FROZEN_S). Normal pictures on 11/26: mean 48 (wrist_right)
to 162 (front_2), std 43-69; the black front: mean 0-1, std 0.

Roles (the panel's OMX_CAM_WATCH file): "input" = the model's cameras, "judge" = cameras a judge that ends or reports
the mission reads (placement judge front, tray judge side camera + front_2). A fault of an input or judge camera:
  before the policy starts -> the run refuses to start (err_code_14), the arms stay home
  during an episode (mode "on") -> the policy stops and the arms hold in place, torque on, like 비상 정지
    ("[omx] cam_hold {json}"; the control tower gets a failed report with a HOLD)
Other cameras: written only. mode "record": everything written, nothing acted on.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

VERSION = "1126a"
SUB = 8                 # every 8th pixel (60 x 80 of a 480 x 640 frame)
BLACK_MEAN = 5.0        # the panel's camera view has shown "검은 화면" below 5 since 3.x
WHITE_MEAN = 250.0
FLAT_STD = 3.0          # black / white need a flat picture too (a real dark scene still has noise and edges)
SOLID_STD = 0.5         # one solid colour at any brightness
FROZEN_DIFF = 0.05      # mean abs difference of two frames: sensor noise alone gives about 1-3
DIM_MEAN, GLARE_MEAN = 25.0, 235.0   # warnings only (normal 48-162 on 11/26)
STALL_S = 1.0
CONFIRM_S = 1.0         # black / white / flat / stall must last this long
FROZEN_S = 3.0
PERIOD_S = 0.2          # 5 checks a second
LINE_S = 5.0            # the "[omx] camwatch:" line during an episode
START_S = 1.5           # the check before the policy starts
KEEP_FRAMES = 40        # fault photos kept in the logs folder
FAULTS = ("stall", "black", "white", "flat", "frozen")
WARNINGS = ("dim", "glare")
KO = {"ok": "정상", "stall": "영상 끊김", "black": "검은 화면", "white": "하얀 화면", "flat": "단색 화면",
      "frozen": "멈춘 화면", "dim": "어두움", "glare": "너무 밝음", "none": "영상 없음"}
ROLE_KO = {"input": "모델 입력", "judge": "판정", "other": "기타"}


def measure(frame):
    """(mean, std, small gray float32) of a camera frame (RGB / BGR / gray), or None."""
    import numpy as np

    if frame is None:
        return None
    try:
        small = np.asarray(frame)[::SUB, ::SUB]
        small = small.mean(axis=2, dtype=np.float32) if small.ndim == 3 else small.astype(np.float32)
        return float(small.mean()), float(small.std()), small
    except Exception:  # noqa: BLE001
        return None


def picture_state(mean: float, std: float) -> str:
    """The state one picture alone shows (no time: stall and frozen need the watch)."""
    if std < FLAT_STD and mean < BLACK_MEAN:
        return "black"
    if std < FLAT_STD and mean > WHITE_MEAN:
        return "white"
    if std < SOLID_STD:
        return "flat"
    if mean < DIM_MEAN:
        return "dim"
    if mean > GLARE_MEAN:
        return "glare"
    return "ok"


def frame_unusable(frame) -> str | None:
    """For the judges (3.31.0): a frame that is black, white or one solid colour is not judged ("unread"), so a broken
    camera can never read as "the tray changed" (one-arm unloading: tray_changed >= 0.25 counted as on the tray)."""
    m = measure(frame)
    if m is None:
        return None
    st = picture_state(m[0], m[1])
    return st if st in FAULTS else None


class _Cam:
    def __init__(self) -> None:
        self.raw = "none"           # what the latest look shows
        self.raw_since = None
        self.state = "ok"           # confirmed (a fault after CONFIRM_S / FROZEN_S; cleared after CONFIRM_S of good)
        self.mean = self.std = None
        self.diff = None            # mean abs difference of the last two new frames (sensor noise ~1-3; frozen < 0.05)
        self.prev = None            # the small picture of the last new frame
        self.prev_ts = None         # its capture stamp (LeRobot latest_timestamp)
        self.prev_seen = None       # when this watch first saw it
        self.same_since = None      # new frames identical to the one before since (frozen)
        self.last_new = None        # when a new frame was last seen
        self.hist: deque = deque(maxlen=40)   # (t, raw) of the last 8 s


class Watch:
    """cams: name -> LeRobot camera (latest_frame, latest_timestamp). conf: the panel's settings (roles, mode)."""

    def __init__(self, cams: dict, conf: dict, say, clock=time.perf_counter):
        self.cams = dict(cams)
        self.mode = conf.get("mode", "on")
        self.input = [c for c in conf.get("input") or [] if c in self.cams]
        self.judge = {c: list(v) for c, v in (conf.get("judge") or {}).items() if c in self.cams}
        self.frame_dir = conf.get("frame_dir")
        self.say, self.clock = say, clock
        self.st = {n: _Cam() for n in self.cams}
        self.lock = threading.Lock()
        self.episode = False
        self.events: list = []          # every fault / warning reported (start, hold, other cameras), last 20
        self.told: dict = {}            # camera -> state last reported (one line per change)
        self.held = False               # this run already held the arms once
        self.t_line = 0.0
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------ roles
    def role(self, name: str) -> str:
        return "input" if name in self.input else "judge" if name in self.judge else "other"

    def required(self, name: str) -> bool:
        return self.role(name) != "other"

    # ------------------------------------------------------------ one look at every camera
    def sample(self) -> dict:
        now = self.clock()
        with self.lock:
            for name, cam in self.cams.items():
                self._look(self.st[name], cam, now)
            return self._snapshot()

    def _look(self, s: _Cam, cam, now: float) -> None:
        frame = getattr(cam, "latest_frame", None)
        ts = getattr(cam, "latest_timestamp", None)
        new = ts is not None and ts != s.prev_ts
        if new or s.last_new is None:
            s.last_new = now  # nothing yet: the stall counts from the first look
        m = measure(frame) if frame is not None else None
        raw = "none"
        if m is not None:
            s.mean, s.std = round(m[0], 1), round(m[1], 1)
            raw = picture_state(m[0], m[1])
            if new:
                if s.prev is not None and s.prev.shape == m[2].shape:
                    s.diff = round(float(abs(m[2] - s.prev).mean()), 2)
                same = s.diff is not None and s.prev is not None and s.diff < FROZEN_DIFF
                s.same_since = (s.same_since if s.same_since is not None else s.prev_seen) if same else None
                s.prev, s.prev_ts, s.prev_seen = m[2], ts, now
        if now - s.last_new >= STALL_S:
            raw = "stall"
        elif raw not in FAULTS and raw != "none" and s.same_since is not None and now - s.same_since >= FROZEN_S:
            raw = "frozen"
        if raw != s.raw:
            s.raw, s.raw_since = raw, now
        s.hist.append((now, raw))
        if raw == "none":
            return
        if raw in FAULTS:
            need = 0.0 if raw == "frozen" else CONFIRM_S  # frozen already waited FROZEN_S of identical frames
        else:
            need = CONFIRM_S if s.state in FAULTS else 0.0  # a fault clears only after as long a good picture
        if now - s.raw_since >= need - 1e-9:
            s.state = raw

    def _snapshot(self) -> dict:
        return {n: {"state": s.state, "state_ko": KO.get(s.state, s.state), "raw": s.raw, "mean": s.mean, "std": s.std,
                    "diff": s.diff, "role": self.role(n)} for n, s in self.st.items()}

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def faulty(self, name: str) -> str | None:
        """The confirmed fault of a camera now (the judges skip its frames), or None."""
        s = self.st.get(name)
        return s.state if s is not None and s.state in FAULTS else None

    # ------------------------------------------------------------ what a fault looks like in the reports
    def event(self, name: str, when: str, action: str) -> dict:
        s = self.st[name]
        now = self.clock()
        since = s.same_since if s.state == "frozen" and s.same_since is not None else s.raw_since
        ev = {"camera": name, "state": s.state, "state_ko": KO.get(s.state, s.state), "role": self.role(name),
              "role_ko": ROLE_KO[self.role(name)], "judges": self.judge.get(name, []), "brightness": s.mean,
              "std": s.std, "for_s": round(now - (since or now), 1), "when": when, "action": action,
              "mode": self.mode, "version": VERSION, "at": _iso(time.time())}
        ev["frame_file"] = self._save_frame(name)
        return ev

    @staticmethod
    def describe(ev: dict) -> str:
        b = f", 밝기 {ev['brightness']}" if ev.get("brightness") is not None else ""
        j = f" {'·'.join(ev['judges'])}" if ev.get("judges") else ""
        return f"{ev['camera']} {ev['state_ko']} ({ev['role_ko']}{j}{b}, {ev['for_s']}초째)"

    def _save_frame(self, name: str) -> str | None:
        if not self.frame_dir:
            return None
        try:
            import cv2
            import numpy as np

            frame = getattr(self.cams[name], "latest_frame", None)
            if frame is None:
                return None
            d = Path(self.frame_dir)
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"camfault_{time.strftime('%Y%m%d_%H%M%S')}_{name}.jpg"
            img = np.asarray(frame)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
            if cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                old = sorted(d.glob("camfault_*.jpg"))
                for o in old[: max(len(old) - KEEP_FRAMES, 0)]:
                    o.unlink(missing_ok=True)
                return p.name
        except Exception:  # noqa: BLE001  a photo never stops the watch
            pass
        return None

    def _tell(self, ev: dict) -> None:
        self.events.append(ev)
        del self.events[:-20]
        self.say("[omx] cam_fault " + json.dumps(ev, ensure_ascii=False))

    # ------------------------------------------------------------ before the policy starts
    def start_check(self, seconds: float = START_S) -> list:
        """Look for `seconds` (the cameras already warmed up at connect). Returns the fault events of input / judge
        cameras in mode "on" (empty = start); the other faults and every warning are written as lines only."""
        if self.mode == "off":
            return []
        end = self.clock() + seconds
        while True:
            self.sample()
            if self.clock() >= end:
                break
            time.sleep(PERIOD_S)
        bad = []
        with self.lock:
            for name, s in self.st.items():
                whole = [r for _, r in s.hist]
                st = None
                if whole and all(r in ("none", "stall") for r in whole):
                    st = "stall"  # not one new frame in the whole look
                elif whole and all(r == whole[-1] for r in whole) and whole[-1] in FAULTS + WARNINGS:
                    st = whole[-1]  # the same fault (or warning) for the whole look
                elif s.same_since is not None and len(whole) >= 5 and self.clock() - s.same_since >= seconds - 2 * PERIOD_S:
                    st = "frozen"   # new frames but the same picture for the whole look
                if st is None:
                    continue
                s.state = st
                act = "refuse" if st in FAULTS and self.required(name) and self.mode == "on" else "warn"
                ev = self.event(name, "start", act)
                self.told[name] = st
                self._tell(ev)
                if act == "refuse":
                    bad.append(ev)
        return bad

    # ------------------------------------------------------------ during the evaluation
    def start(self) -> None:
        if self.mode == "off" or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name="camwatch")
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.wait(PERIOD_S):
            try:
                snap = self.sample()
                self._changes(snap)
                if self.episode and self.clock() - self.t_line >= LINE_S:
                    self.t_line = self.clock()
                    self.say(self.line(snap))
            except Exception as exc:  # noqa: BLE001  the watch never stops the run
                self.say(f"[omx] camwatch: {type(exc).__name__}: {exc}")

    def _changes(self, snap: dict) -> None:
        """One line per change of a camera's state. A fault the control loop acts on (mode on, an input / judge camera,
        during an episode) is reported by the loop itself ("[omx] cam_hold")."""
        for name, v in snap.items():
            st = v["state"]
            if st == self.told.get(name, "ok"):
                continue
            self.told[name] = st
            if st in FAULTS or st in WARNINGS:
                if st in FAULTS and self.mode == "on" and self.required(name) and self.episode and not self.held:
                    continue
                act = "record" if st in FAULTS and self.mode == "record" and self.required(name) else "warn"
                with self.lock:
                    ev = self.event(name, "eval" if self.episode else "between", act)
                self._tell(ev)
            else:
                self.say(f"[omx] camwatch: {name} 다시 정상 (밝기 {v['mean']})")

    def begin_episode(self) -> None:
        self.episode, self.t_line = True, self.clock()

    def end_episode(self) -> None:
        self.episode = False

    def take_hold(self) -> dict | None:
        """The control loop, every step of a policy episode: a confirmed fault of an input / judge camera (mode "on"),
        once per run - also one that was already there when the episode began. Cheap when all is well."""
        if self.held or self.mode != "on":
            return None
        with self.lock:
            for name, s in self.st.items():
                if s.state in FAULTS and self.required(name):
                    self.held = True
                    ev = self.event(name, "eval", "hold")
                    self.events.append(ev)
                    del self.events[:-20]
                    return ev
        return None

    def line(self, snap: dict | None = None) -> str:
        snap = snap or self.snapshot()
        fmt = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
        return "[omx] camwatch: " + " | ".join(f"{n} {v['state']} bri {fmt(v['mean'])} sd {fmt(v['std'])} "
                                               f"df {'-' if v['diff'] is None else v['diff']}" for n, v in snap.items())

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        return {"version": VERSION, "mode": self.mode, "input": self.input, "judge": self.judge,
                "events": list(self.events), "now": self.snapshot()}


def _iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def load(path: str, cams: dict, say) -> Watch | None:
    conf = json.loads(Path(path).read_text())
    return Watch(cams, conf, say)
