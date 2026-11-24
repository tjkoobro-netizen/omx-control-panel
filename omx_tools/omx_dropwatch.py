#!/usr/bin/env python3
"""Box dropped during an evaluation (3.23.0, record mode): only records what it saw; never ends the run, never moves.

Stage 1, the gripper (Guide's offline analysis 11/24, detect2.py: 181 evaluations, 3 signals, all 3 real drops):
  an arm that held something (command - measured < -HOLD_G for HOLD_S) closes within WIN_S by at least
  max(MIN_DROP, FRAC * |the held gap|) while its gripper command moved less than CMD_TOL (not opened on purpose).
  One arm is enough (11/09 19:07 the right hand still held a corner while the box fell).
  The fingertips (handover_judge FK) inside the shelf zone (y > 0.09, z > 0.28) or the tray zone (y < 0.06, z < 0.28)
  are a normal let-go: recorded, not checked further. These zones are for the 2B layout.
Stage 2, the top camera (untested rule, values from Guide 11/24): from the signal on for CONFIRM_S, the commanded
  box's marker (whole raw front frame, centre of its four corners). Not seen at all -> confirmed "unseen". Last seen
  LOST_S or more before the window end (seen while falling, then gone) -> confirmed "lost". Seen
  standing still (STILL_PX for STILL_S) outside the tray and 2B regions -> confirmed "rest". Seen inside them ->
  rejected. Otherwise undecided. Known limits (Guide): a box fallen on its side in the tray hides its marker
  (unseen); a box hanging from one hand that stops a moment can read as rest. Every frame of the window is kept
  (t, marker id, centre, side in px, other boxes seen) so the rule can be tuned offline.
  The wrist cameras are not used (they see the marker in 5.6% of the held frames).

The control loop only hands over numbers (and a frame copy at FRAME_HZ while a check runs); the work is done in
this module's own thread, so the 30 Hz loop is not slowed. Config (OMX_DROP_WATCH=<json>): box_ids, camera,
tray_roi [v0, v1, u0, u1] or null, shelf_roi (default handover_fk.json shelf.top_roi), frame_dir, fps.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from pathlib import Path

VERSION = "1124b"  # 1124b: + "lost" (Guide 11/24 replay of real top-camera video: real drops 2/4 -> 4/4, no false one)
HOLD_G, HOLD_S, WIN_S, CMD_TOL, FRAC, MIN_DROP, COOL_S = 4.0, 0.3, 0.5, 1.0, 0.6, 3.0, 2.0
SHELF_ZONE = (0.09, 0.28)   # tip y >, z >
TRAY_ZONE = (0.06, 0.28)    # tip y <, z <
CONFIRM_S, STILL_S, STILL_PX, FRAME_HZ = 1.5, 0.5, 3.0, 10.0  # Guide 11/24: the box rests about 1 s after the signal
TRAY_ROI = [95, 215, 240, 420]  # front pixels [v0, v1, u0, u1] (Guide 11/24: the tray of 11/17 and 11/24 00:31 with margin)
ALL_BOXES = (16, 17, 20)
LOST_S = 0.8  # seen once or twice while falling, then gone this long by the window end: confirmed "lost"
MAX_FRAMES_KEPT = 20  # per signal: (t, markers) of the frames in its window
MAX_SIGNALS = 20


def zone(tip) -> str:
    _, y, z = (float(v) for v in tip)
    if y > SHELF_ZONE[0] and z > SHELF_ZONE[1]:
        return "shelf"
    if y < TRAY_ZONE[0] and z < TRAY_ZONE[1]:
        return "tray"
    return "mid"


class GripSignal:
    """Stage 1 for one arm, one sample per control step (fps fixed by the loop)."""

    def __init__(self, fps: float):
        self.hold_n, self.win, self.cool_n = round(HOLD_S * fps), round(WIN_S * fps), round(COOL_S * fps)
        self.n, self.run, self.cool = 0, 0, -1
        self.anchors: collections.deque = collections.deque()  # (index, command, measured, gap)

    def step(self, a: float, s: float) -> dict | None:
        i, g = self.n, a - s
        self.n += 1
        hit = None
        alive = collections.deque()
        for k, a0, s0, g0 in self.anchors:
            if i - k > self.win or abs(a - a0) >= CMD_TOL:
                continue  # too late, or the command moved (opened on purpose)
            if hit is None and s0 - s >= max(MIN_DROP, FRAC * abs(g0)):
                hit = {"lat_s": round((i - k) * WIN_S / self.win, 2) if self.win else 0.0, "g0": round(g0, 1),
                       "drop": round(s0 - s, 1), "grip_state": round(s, 1), "grip_cmd": round(a, 1)}
            alive.append((k, a0, s0, g0))
        self.anchors = alive
        if hit is not None:
            self.anchors.clear()
            self.run, self.cool = 0, i + self.cool_n
            return hit
        self.run = self.run + 1 if g < -HOLD_G else 0
        if self.run >= self.hold_n and i > self.cool:
            self.anchors.append((i, a, s, g))
        return None


def _iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def _inside(uv, roi) -> bool:
    if not roi:
        return False
    v0, v1, u0, u1 = roi
    return u0 <= uv[0] <= u1 and v0 <= uv[1] <= v1


class DropWatch:
    def __init__(self, path: str):
        import cv2

        import handover_judge

        cfg = json.loads(Path(path).read_text())
        self.cfg = cfg
        self.camera = cfg.get("camera", "front")
        self.box_ids = {int(b) for b in cfg.get("box_ids") or handover_judge.BOX_IDS}
        tab = json.loads(Path(cfg.get("fk_json") or handover_judge.HERE / "handover_fk.json").read_text())
        self.fk, self.names = handover_judge.FK(tab), tab["names"]
        self.shelf_roi = cfg.get("shelf_roi") or tab["shelf"]["top_roi"]
        self.tray_roi = cfg["tray_roi"] if "tray_roi" in cfg else TRAY_ROI
        self.frame_dir = Path(cfg["frame_dir"]) if cfg.get("frame_dir") else None
        self.fps = float(cfg.get("fps", 30))
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                           cv2.aruco.DetectorParameters())
        self.lock, self.wake = threading.Lock(), threading.Event()
        self.samples: collections.deque = collections.deque(maxlen=600)  # (t, joints, {side: (cmd, state)})
        self.frames: collections.deque = collections.deque(maxlen=40)    # (t, frame) while a check runs
        self.frame_until, self.frame_t = 0.0, 0.0
        self.reset(time.time())
        threading.Thread(target=self._run, daemon=True, name="dropwatch").start()

    def reset(self, t0: float) -> None:
        with self.lock:
            self.t0 = t0
            self.sig = {}
            self.signals: list = []
            self.pending: list = []
            self.updates = self.errors = self.frames_checked = 0
            self.latest_t = t0
            self.samples.clear()
            self.frames.clear()
            self.frame_until = 0.0
            self.episode = getattr(self, "episode", -1) + 1

    # ------------------------------------------------------------ the control loop's side (cheap)
    def feed(self, obs: dict, action: dict, t: float) -> None:
        """Every control step of an evaluation episode: the measured joints, the gripper commands just computed, and
        (only while a check runs, at FRAME_HZ) a copy of the raw top-camera frame."""
        grips = {}
        for side in ("left", "right"):
            k = f"{side}_gripper.pos"
            if k in obs and k in action:
                grips[side] = (float(action[k]), float(obs[k]))
        if not grips:
            return
        joints = {k[:-4]: float(v) for k, v in obs.items() if k.endswith(".pos")}
        frame = None
        if t <= self.frame_until and t - self.frame_t >= 1.0 / FRAME_HZ and obs.get(self.camera) is not None:
            import numpy as np

            frame = np.array(obs[self.camera], copy=True)
            self.frame_t = t
        with self.lock:
            self.samples.append((t, joints, grips))
            if frame is not None:
                self.frames.append((t, frame))
        self.wake.set()

    # ------------------------------------------------------------ the watcher's thread
    def _run(self) -> None:
        while True:
            self.wake.wait(0.2)
            self.wake.clear()
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001  never take the run down; counted
                with self.lock:
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"[:200]

    def _step(self) -> None:
        with self.lock:
            samples = list(self.samples)
            self.samples.clear()
            frames = list(self.frames)
            self.frames.clear()
        for t, joints, grips in samples:
            self.updates += 1
            for side, (a, s) in grips.items():
                det = self.sig.setdefault(side, GripSignal(self.fps))
                hit = det.step(a, s)
                if hit is not None:
                    self._signal(t, side, joints, hit)
        for t, frame in frames:
            self._frame(t, frame)
        if samples:
            self.latest_t = max(self.latest_t, samples[-1][0])
        now = self.latest_t  # the loop's own time (a window closes once the loop has gone past it)
        for c in [c for c in self.pending if now > c["t"] + CONFIRM_S + 0.15]:
            gone = c["t"] + CONFIRM_S - c.get("last_seen_t", c["t"])
            if not c["seen_n"]:
                self._close(c, "confirmed", f"{CONFIRM_S}초 동안 상자 마커가 안 보임", c.get("last_frame"), "unseen")
            elif gone >= LOST_S:
                self._close(c, "confirmed", f"상자 마커가 {c['last_uv']}에서 보인 뒤 {gone:.2f}초 동안 안 보임", c.get("last_frame"), "lost")
            else:
                self._close(c, "undecided", "움직이는 채로 보이기만 함", c.get("last_frame"))

    def _signal(self, t: float, side: str, joints: dict, hit: dict) -> None:
        jd = {k: 0.0 for k in self.names}
        jd.update({k: v for k, v in joints.items() if k in jd})
        tip = [round(float(x), 3) for x in self.fk.tip(jd, side)]
        z = zone(tip)
        rec = {"t": round(t, 3), "at": _iso(t), "since_start_s": round(t - self.t0, 2), "side": side, **hit,
               "tip": tip, "zone": z, "ignored": z != "mid"}
        if z == "mid":
            rec["stage2"] = {"result": "checking"}
            c = {"rec": rec, "t": t, "seen_n": 0, "still_since": None, "still_uv": None, "last_frame": None, "frames": []}
            self.pending.append(c)
            self.frame_until = max(self.frame_until, t + CONFIRM_S + 0.2)
        with self.lock:
            if len(self.signals) < MAX_SIGNALS:
                self.signals.append(rec)

    def _markers(self, gray) -> list:
        """Every box marker on the frame: id, centre of its corners, mean side in px."""
        import numpy as np

        cs, ids, _ = self.det.detectMarkers(gray)
        out = []
        for k, q in ([] if ids is None else zip(ids.ravel(), cs)):
            if int(k) in self.box_ids or int(k) in ALL_BOXES:
                q = q.reshape(4, 2)
                c = q.mean(0)
                side = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))
                out.append({"id": int(k), "uv": [round(float(c[0]), 1), round(float(c[1]), 1)], "side_px": round(side, 1)})
        return out

    def _frame(self, t: float, frame) -> None:
        import cv2

        live = [c for c in self.pending if c["t"] <= t <= c["t"] + CONFIRM_S]
        if not live:
            return
        self.frames_checked += 1
        seen = self._markers(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame)
        m = next((x for x in seen if x["id"] in self.box_ids), None)
        for c in live:
            c["last_frame"] = frame
            if len(c["frames"]) < MAX_FRAMES_KEPT:
                c["frames"].append({"dt": round(t - c["t"], 2), "markers": seen})
            if m is None:
                c["still_since"] = None
                continue
            c["seen_n"] += 1
            c["last_uv"] = m["uv"]
            c["last_seen_t"] = t
            if _inside(m["uv"], self.shelf_roi) or _inside(m["uv"], self.tray_roi):
                where = "2B" if _inside(m["uv"], self.shelf_roi) else "쟁반"
                self._close(c, "rejected", f"상자 마커가 {where} 영역 안에서 보임 {m['uv']}", frame)
                continue
            if c["still_uv"] is None or max(abs(m["uv"][0] - c["still_uv"][0]), abs(m["uv"][1] - c["still_uv"][1])) > STILL_PX:
                c["still_uv"], c["still_since"] = m["uv"], t
            elif t - c["still_since"] >= STILL_S:
                if self.tray_roi is None:
                    self._close(c, "undecided", f"영역 밖에서 멈춤 {m['uv']}이지만 쟁반 영역 설정이 없음", frame)
                else:
                    self._close(c, "confirmed", f"상자 마커가 쟁반·2B 밖에서 {STILL_S}초 멈춤 {m['uv']}", frame, "rest")

    def _close(self, c: dict, result: str, why: str, frame, kind: str | None = None) -> None:
        if c not in self.pending:
            return
        self.pending.remove(c)
        out = {"result": result, "kind": kind, "why": why, "seen_frames": c["seen_n"], "last_uv": c.get("last_uv"),
               "decided_at": _iso(time.time()), "frames": c["frames"]}
        if result == "confirmed" and frame is not None and self.frame_dir is not None:
            try:
                import cv2

                self.frame_dir.mkdir(parents=True, exist_ok=True)
                f = self.frame_dir / f"drop_{time.strftime('%Y%m%d_%H%M%S')}_{c['rec']['side']}.jpg"
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                out["frame_file"] = str(f)
            except Exception as exc:  # noqa: BLE001
                out["frame_error"] = str(exc)[:120]
        with self.lock:
            c["rec"]["stage2"] = out

    def finish(self) -> dict:
        """The episode ended: let the thread take what is left, close the checks still running, then the summary."""
        for _ in range(20):
            with self.lock:
                left = bool(self.samples or self.frames)
            if not left:
                break
            self.wake.set()
            time.sleep(0.02)
        with self.lock:
            pending = list(self.pending)
        for c in pending:
            self._close(c, "undecided", "확인 창이 끝나기 전에 시도가 끝남", c.get("last_frame"))
        return self.summary()

    def summary(self) -> dict:
        with self.lock:
            sig = [dict(s) for s in self.signals]
            return {"version": VERSION, "on": True, "mode": "record", "box_ids": sorted(self.box_ids),
                    "tray_roi": self.tray_roi, "updates": self.updates, "errors": self.errors,
                    "last_error": getattr(self, "last_error", None), "frames_checked": self.frames_checked,
                    "signals": sig, "confirmed": sum((s.get("stage2") or {}).get("result") == "confirmed" for s in sig)}
