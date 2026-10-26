#!/usr/bin/env python3
"""Measured camera frame rate (fps) and image brightness, for the panel's camera views.

install() wraps LeRobot's OpenCVCamera._read_from_hardware, which runs once per frame the camera
delivers (in the camera's own reader thread), so fps() is the camera's real rate - not how often the
panel or the robot loop looks at it. Brightness is the mean pixel value (0-255) of the latest frame:
there is no light sensor, but it drops right away when the light fades or the lens is covered.
"""

from __future__ import annotations

import threading
import time
from collections import deque

_TIMES: dict = {}  # camera object -> deque of frame times
_LOCK = threading.Lock()


def install() -> None:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    if getattr(OpenCVCamera, "_omx_camstats", False):
        return
    OpenCVCamera._omx_camstats = True
    original = OpenCVCamera._read_from_hardware

    def _read_from_hardware(self, *args, **kwargs):
        frame = original(self, *args, **kwargs)
        now = time.perf_counter()
        with _LOCK:
            q = _TIMES.get(id(self))
            if q is None:
                q = _TIMES[id(self)] = deque(maxlen=120)
            q.append(now)
        return frame

    OpenCVCamera._read_from_hardware = _read_from_hardware


def fps(cam, window: float = 1.5) -> float | None:
    """Frames per second over the last `window` seconds (None before any frame)."""
    with _LOCK:
        q = list(_TIMES.get(id(cam)) or ())
    if not q:
        return None
    now = time.perf_counter()
    if now - q[-1] > 1.0:
        return 0.0  # nothing for a second: stalled
    recent = [t for t in q if now - t <= window]
    if len(recent) < 2:
        return 0.0
    return round((len(recent) - 1) / (recent[-1] - recent[0]), 1)


def brightness(frame) -> float | None:
    try:
        return round(float(frame[::8, ::8].mean()), 1)  # every 8th pixel: plenty for a mean, cheap
    except Exception:
        return None


def stats(cameras: dict, latest: dict | None = None) -> dict:
    """{name: {"fps", "brightness"}} for a dict of OpenCVCamera objects (latest: name -> last frame)."""
    out = {}
    for name, cam in cameras.items():
        frame = (latest or {}).get(name)
        if frame is None:
            frame = getattr(cam, "latest_frame", None)
        out[name] = {"fps": fps(cam), "brightness": brightness(frame) if frame is not None else None}
    return out
