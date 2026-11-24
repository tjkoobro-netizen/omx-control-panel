#!/usr/bin/env python3
"""Which wrist camera is on which arm, seen from a small gripper move (3.22.0).

The two wrist cameras of the cooperative robot are the same model with the same serial, so robot.json keeps them by
USB port, and the panel's camera search tells them apart only by the arm board on the same USB hub. When the hub
wiring changes they can be exchanged without anything noticing, and the policy then runs on the other arm's view.

Before an evaluation starts (both arms at home, the cameras open, the output still capped by the safe start), each
gripper in turn opens a little and comes back while both wrist cameras are read. The camera whose picture changed is
that arm's. Opening only (closing only when already near fully open), so nothing held is squeezed.

  - both arms point at different cameras, as saved          -> start as before
  - both arms point at different cameras, the other way     -> exchange wrist_left / wrist_right in this run, and the
                                                               panel corrects robot.json (no person needed: the
                                                               evidence is the move itself; TK 11/23)
  - anything else (too small a change, both changed, ...)   -> no guess: the policy does not start
One arm of the cooperative robot run alone (3.21.0) has only its own wrist camera open: that camera must change.

omx_run.py prints one machine line "[omx] wrist_check {json}" that the panel keeps (last_result.wrist_check).
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

AMP = 10.0          # gripper units (0..100) of the small move
SPEED = 25.0        # units per second (the output stays capped by the safe start)
SETTLE_S = 0.25     # after the move, before the pictures are read
PIX = 25            # grey levels: a pixel that changed
MIN_CHANGED = 0.02  # part of the picture that must change on the moving arm's own camera
RATIO = 3.0         # ... and at least this many times the other camera's change
FLOOR = 0.005       # the other camera's change counts as at least this much (picture noise)
GRIP_TOP = 100.0


def small(frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA), (5, 5), 0)


def changed(a: np.ndarray, b: np.ndarray) -> float:
    """Part of the picture (0..1) that changed by more than PIX grey levels."""
    return float((cv2.absdiff(a, b) > PIX).mean())


def grab(cam) -> np.ndarray:
    return small(cam.async_read(timeout_ms=1000))  # a fresh frame (the camera's own reader thread)


def gripper_target(g0: float) -> float:
    return g0 + AMP if g0 + AMP <= GRIP_TOP else g0 - AMP


def move_gripper(bus, start: float, goal: float, stop=lambda: False) -> bool:
    steps = max(1, int(np.ceil(abs(goal - start) / SPEED * 30)))
    for i in range(1, steps + 1):
        if stop():
            return False
        bus.sync_write("Goal_Position", {"gripper": start + (goal - start) * i / steps})
        time.sleep(1 / 30)
    return True


def wiggle(arm, cams: dict, stop=lambda: False) -> dict:
    """{camera name: change} while this arm's gripper opened a little (noise of a still picture taken off)."""
    bus = arm.bus
    g0 = float(bus.sync_read("Present_Position")["gripper"])
    g1 = gripper_target(g0)
    base = {n: grab(c) for n, c in cams.items()}
    still = {n: grab(c) for n, c in cams.items()}
    noise = {n: changed(base[n], still[n]) for n in cams}
    try:
        if not move_gripper(bus, g0, g1, stop):
            raise InterruptedError("stopped")
        time.sleep(SETTLE_S)
        moved = [{n: grab(c) for n, c in cams.items()} for _ in range(2)]
    finally:
        move_gripper(bus, g1, g0)  # always back, also after a stop or a camera error
    return {n: round(max(0.0, max(changed(still[n], m[n]) for m in moved) - noise[n]), 4) for n in cams}


def winner(change: dict) -> tuple[str | None, str]:
    ranked = sorted(change, key=change.get, reverse=True)
    top = ranked[0]
    if change[top] < MIN_CHANGED:
        return None, f"{top} 변화가 작음({change[top]:.3f} < {MIN_CHANGED})"
    rival = max([change[n] for n in ranked[1:]], default=0.0)
    if change[top] < RATIO * max(rival, FLOOR):
        return None, f"두 카메라가 함께 바뀜({change[top]:.3f} 대 {rival:.3f})"
    return top, "ok"


def decide_pair(left_move: dict, right_move: dict) -> dict:
    """left_move / right_move: {camera name: change} while the left / right gripper moved."""
    wl, why_l = winner(left_move)
    wr, why_r = winner(right_move)
    if wl is None or wr is None:
        return {"ok": False, "swapped": False, "why": f"왼팔: {why_l}" if wl is None else f"오른팔: {why_r}"}
    if wl == wr:
        return {"ok": False, "swapped": False, "why": f"두 팔 모두 {wl}을 바꿈"}
    if (wl, wr) == ("wrist_left", "wrist_right"):
        return {"ok": True, "swapped": False, "why": "저장된 배정과 같음"}
    if (wl, wr) == ("wrist_right", "wrist_left"):
        return {"ok": True, "swapped": True, "why": "저장된 배정과 반대: 이번 실행에서 바꿈"}
    return {"ok": False, "swapped": False, "why": f"알 수 없는 카메라 {wl}, {wr}"}


def decide_one(change: float) -> dict:
    if change >= MIN_CHANGED:
        return {"ok": True, "swapped": False, "why": "이 팔의 손목 카메라가 바뀜"}
    return {"ok": False, "swapped": False,
            "why": f"변화가 작음({change:.3f} < {MIN_CHANGED}): 다른 팔의 손목 카메라일 수 있음"}


def cam_path(cam) -> str:
    return str(getattr(getattr(cam, "config", None), "index_or_path", "") or "")


def check(robot, stop=lambda: False) -> dict | None:
    """Run the check on a connected robot. None when there is nothing to check (no wrist camera of an arm)."""
    cams = getattr(robot, "cameras", None) or {}
    t0 = time.monotonic()
    if getattr(robot, "arms", None) and {"wrist_left", "wrist_right"} <= set(cams):
        pair = {n: cams[n] for n in ("wrist_left", "wrist_right")}
        was = {"left": cam_path(pair["wrist_left"]), "right": cam_path(pair["wrist_right"])}
        moves = {"left": wiggle(robot.left_arm, pair, stop), "right": wiggle(robot.right_arm, pair, stop)}
        out = decide_pair(moves["left"], moves["right"])
        if out["swapped"]:
            cams["wrist_left"], cams["wrist_right"] = pair["wrist_right"], pair["wrist_left"]
        out |= {"mode": "pair", "was": was, "left": cam_path(cams["wrist_left"]), "right": cam_path(cams["wrist_right"]),
                "change": moves}
    else:
        side = getattr(robot, "_omx_side", None)
        name = f"wrist_{side}" if side in ("left", "right") else None
        if not name or name not in cams or getattr(robot, "arms", None):
            return None
        change = wiggle(robot, {name: cams[name]}, stop)
        out = decide_one(change[name]) | {"mode": "one", "side": side, side: cam_path(cams[name]), "change": {side: change}}
    out |= {"ms": int((time.monotonic() - t0) * 1000),
            "limits": {"amp": AMP, "min_changed": MIN_CHANGED, "ratio": RATIO}}
    return out


def line(result: dict) -> str:
    return "[omx] wrist_check " + json.dumps(result, ensure_ascii=False)
