#!/usr/bin/env python3
"""Every 5 s of an evaluation episode, three log lines on the cameras, the UDP inference link and the control loop
(3.26.0, format agreed with Guide 11/24; read by Guide's omx_live_monitor). Values are for those 5 s only, and once
more at the end of the episode for the whole episode ("... total:").

  [omx] cams: front 30.0fps age 35/60ms dup 2 err 0 | front_2 ...
      fps = frames the camera's read thread delivered (omx_camstats), age p50/p95 = from the capture of the frame the
      loop used (LeRobot latest_timestamp) to that step, dup = the step got the same frame as the step before,
      err = exceptions of the read thread's hardware read.
  [omx] udp: req 52 bytes 8.7MB frag 7300 resp 50 timeout 2 stale 1 rtt 114/180ms
      req = attempts (a timeout is re-sent), bytes = observation bytes sent, frag = bytes / 1200 per attempt
      (DATAGRAM_BYTES), resp = replies, timeout = attempts without a reply in time, stale = infer_link's
      stats["stale_reply"] (read only), rtt p50/p95 = client.last["round_ms"]. Only with UDP inference.
  [omx] loop5s: p5 28.9Hz min 24.1Hz over 3
      Hz from the step-to-step times, over = steps whose own work took longer than 1/fps.

The control loop only adds numbers to lists (no lock, no string); the lines are built once every 5 s.
"""

from __future__ import annotations

import math
import time

PERIOD_S = 5.0
DATAGRAM_BYTES = 1200


def _pct(vals: list, q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


class Telemetry:
    def __init__(self, robot, backend, fps: int, say, period: float = PERIOD_S):
        import omx_camstats

        omx_camstats.install()  # idempotent; the read threads look the method up on each frame
        self.camstats = omx_camstats
        self.cams = dict(getattr(robot, "cameras", None) or {})
        self.udp = backend if hasattr(backend, "tele") else None
        self.interval, self.say, self.period = 1.0 / fps, say, period
        self.prev_frame: dict = {}
        self.ep = self._window()
        self.win = self._window()
        self.t_start = self.t_win = time.perf_counter()
        self.cam0 = self.cam_win = self._cam_counts()
        self.udp0 = self.udp_win = self._udp_counts()

    def _window(self) -> dict:
        return {"t0": [], "over": 0, "age": {n: [] for n in self.cams}, "dup": dict.fromkeys(self.cams, 0)}

    def _cam_counts(self) -> dict:
        return {n: (self.camstats.frames(c), self.camstats.errors(c)) for n, c in self.cams.items()}

    def _udp_counts(self) -> dict | None:
        if self.udp is None:
            return None
        t = dict(self.udp.tele)
        t["stale"] = int(self.udp.client.stats.get("stale_reply", 0))
        return t

    # ------------------------------------------------------------ the control loop's side (cheap)
    def step(self, t0: float, obs: dict, work_s: float) -> None:
        now = time.perf_counter()
        for w in (self.win, self.ep):
            w["t0"].append(t0)
            if work_s > self.interval:
                w["over"] += 1
        for name, cam in self.cams.items():
            f = obs.get(name)
            if f is None:
                continue
            if f is self.prev_frame.get(name):
                self.win["dup"][name] += 1
                self.ep["dup"][name] += 1
            else:
                self.prev_frame[name] = f
            ts = getattr(cam, "latest_timestamp", None)
            if ts is not None and getattr(cam, "latest_frame", None) is f:  # no lock: a newer frame = skip this one
                age = (now - ts) * 1000
                self.win["age"][name].append(age)
                self.ep["age"][name].append(age)
        if now - self.t_win >= self.period:
            self.emit()

    # ------------------------------------------------------------ the lines
    def _lines(self, w: dict, cam0: dict, udp0: dict | None, secs: float, total: bool) -> list:
        tag = " total" if total else ""
        out = []
        cam1 = self._cam_counts()
        parts = []
        for n in self.cams:
            fr, er = cam1[n][0] - cam0[n][0], cam1[n][1] - cam0[n][1]
            ages = w["age"][n]
            a50, a95 = (round(_pct(ages, 0.5)), round(_pct(ages, 0.95))) if ages else (0, 0)
            parts.append(f"{n} {fr / max(secs, 1e-6):.1f}fps age {a50}/{a95}ms dup {w['dup'][n]} err {er}")
        if parts:
            out.append(f"[omx] cams{tag}: " + " | ".join(parts))
        if udp0 is not None:
            u = self._udp_counts()
            d = {k: u[k] - udp0[k] for k in ("req", "bytes", "frag", "resp", "timeout", "stale", "rtt_n")}
            rtt = list(self.udp.rtts)[-d["rtt_n"]:] if d["rtt_n"] else []
            r50, r95 = (round(_pct(rtt, 0.5)), round(_pct(rtt, 0.95))) if rtt else (0, 0)
            out.append(f"[omx] udp{tag}: req {d['req']} bytes {d['bytes'] / 1e6:.1f}MB frag {d['frag']} resp {d['resp']} "
                       f"timeout {d['timeout']} stale {d['stale']} rtt {r50}/{r95}ms")
        ts = w["t0"]
        if len(ts) >= 2:
            hz = [1.0 / max(b - a, 1e-6) for a, b in zip(ts, ts[1:])]
            out.append(f"[omx] loop5s{tag}: p5 {_pct(hz, 0.05):.1f}Hz min {min(hz):.1f}Hz over {w['over']}")
        return out

    def emit(self) -> None:
        now = time.perf_counter()
        try:
            for line in self._lines(self.win, self.cam_win, self.udp_win, now - self.t_win, False):
                self.say(line)
        except Exception as exc:  # noqa: BLE001  a report line must never stop the run
            self.say(f"[omx] telemetry: {type(exc).__name__}: {exc}")
        self.win, self.t_win = self._window(), now
        self.cam_win, self.udp_win = self._cam_counts(), self._udp_counts()

    def finish(self) -> None:
        """The episode ended: the last partial window, then the whole episode."""
        now = time.perf_counter()
        if self.win["t0"]:
            self.emit()
        try:
            for line in self._lines(self.ep, self.cam0, self.udp0, now - self.t_start, True):
                self.say(line)
        except Exception as exc:  # noqa: BLE001
            self.say(f"[omx] telemetry: {type(exc).__name__}: {exc}")


def frag(n_bytes: int) -> int:
    return math.ceil(n_bytes / DATAGRAM_BYTES)
