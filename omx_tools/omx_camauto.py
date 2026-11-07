"""Which camera is which after a USB replug.

Identical cameras share serials, so robot.json keeps each one by USB port and a replug loses it. A wrist camera is
told by its model and the arm boards on its USB hub, a global camera by its view against the one saved when the user
last confirmed the setup. Anything else stays unsure for the user to look at: never a guess.
"""
import re
from pathlib import Path

import cv2
import numpy as np

BUILTIN = re.compile(r"True Vision|Integrated|Webcam|FaceTime", re.I)  # the laptop's own camera
WRIST_MODEL = re.compile(r"Innomaker", re.I)
# same values as the kit's sim_realcam.py: no wrong answer on 18 recordings and 3564 occluded / brightened variants
MATCH_MIN, MATCH_GAP = 0.3, 0.25
DARK = 5  # mean brightness under this says nothing about the view


def _usb_device(sys_path: Path) -> Path | None:
    for p in [sys_path, *sys_path.parents]:
        if (p / "idVendor").exists():
            return p
    return None


def _hub(sys_link: str) -> str | None:
    """USB hub above a device, e.g. "3-4.1". None when it hangs on the laptop itself: that says nothing about an arm."""
    dev = _usb_device(Path(sys_link).resolve())
    return dev.parent.name if dev and "-" in dev.parent.name else None


def camera_hub(device: str) -> str | None:
    return _hub(f"/sys/class/video4linux/{Path(device).name}/device")


def board_hub(port: str) -> str | None:
    p = Path(port)
    return _hub(f"/sys/class/tty/{p.resolve().name}/device") if p.exists() else None


def grab_frame(path: str, frames: int = 8):
    """One BGR frame, or None. The first frames are often dark (auto exposure)."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(3, 640)
        cap.set(4, 480)
        ok, frame = False, None
        for _ in range(frames):
            ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def grab_gray(path: str):
    frame = grab_frame(path, 15)
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray if gray.mean() >= DARK else None


def view_vector(gray) -> np.ndarray:
    v = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def _pick(path: str | None, sure: bool, why: str, score: float | None = None) -> dict:
    return {"path": path, "sure": sure, "why": why, "score": score}


def _place_wrists(roles: list, saved: dict, cams: list, set_hubs: dict, role_sets: dict) -> dict:
    out, free, plugged = {}, {c["path"]: c for c in cams}, {c["path"] for c in cams}
    for role in roles:
        name = role_sets.get(role)
        others = set().union(*[h for n, h in set_hubs.items() if n != name])
        hits = [c for c in cams if c["hub"] and c["hub"] in set_hubs.get(name, set()) - others]
        if len(hits) == 1:
            path = hits[0]["path"]
            if saved.get(role) in plugged and saved[role] != path:
                out[role] = _pick(path, False, f"허브에는 {name} 보드가 있지만 저장된 포트와 다릅니다")
            else:
                out[role] = _pick(path, True, f"같은 USB 허브에 {name} 보드")
            free.pop(path)
    left = [r for r in roles if r not in out]
    for role in list(left):
        if saved.get(role) in free:
            out[role] = _pick(saved[role], True, "포트 그대로")
            free.pop(saved[role])
            left.remove(role)
    if len(left) == 1 and len(free) == 1:
        path = next(iter(free))
        alone = len(roles) == 1 and len(cams) == 1
        out[left[0]] = _pick(path, alone, "손목 카메라 1대뿐" if alone else "남은 카메라라 이쪽으로 추정")
        left = []
    for role in left:
        out[role] = _pick(None, False, "판별하지 못했습니다")
    return out


def _place_globals(roles: list, saved: dict, cams: list, frames: dict, refs: dict) -> dict:
    if len(roles) == 1 and len(cams) == 1:
        return {roles[0]: _pick(cams[0]["path"], True, "글로벌 카메라 1대뿐")}
    out, free = {}, [c["path"] for c in cams]
    rated = [r for r in roles if r in refs]
    vec = {c["path"]: view_vector(frames[c["path"]]) for c in cams if frames.get(c["path"]) is not None}
    scores = {p: {r: float(v @ refs[r]) / refs[r].size for r in rated} for p, v in vec.items()}
    for role in rated:
        ranked = sorted(scores, key=lambda p: scores[p][role], reverse=True)
        if not ranked:
            continue
        top, s = ranked[0], scores[ranked[0]][role]
        rivals = [scores[p][role] for p in ranked[1:]] + [v for r, v in scores[top].items() if r != role]
        if s >= MATCH_MIN and s - max(rivals, default=-1.0) >= MATCH_GAP:
            out[role] = _pick(top, True, "기준 화면과 일치", round(s, 2))
            free.remove(top)
    left = [r for r in roles if r not in out]
    for role in list(left):
        if saved.get(role) in free:
            out[role] = (_pick(saved[role], False, "포트는 그대로지만 화면이 기준과 다릅니다") if role in refs
                         else _pick(saved[role], True, "포트 그대로"))
            free.remove(saved[role])
            left.remove(role)
    if len(left) == 1 and len(free) == 1:
        out[left[0]] = _pick(free[0], False, "남은 카메라라 이쪽으로 추정")
        left = []
    for role in left:
        out[role] = _pick(None, False, "판별하지 못했습니다")
    return out


def propose(roles: list, saved: dict, cams: list, set_hubs: dict, role_sets: dict, frames: dict, refs: dict) -> dict:
    """{role: {path, sure, why, score}}. saved: role -> path; cams: [{path, model, hub}]; set_hubs: robot set -> hubs of
    its boards; role_sets: wrist role -> robot set; frames: path -> gray image or None; refs: role -> view vector."""
    usable = [c for c in cams if not BUILTIN.search(c["model"])]
    wrists = [c for c in usable if WRIST_MODEL.search(c["model"])]
    others = [c for c in usable if not WRIST_MODEL.search(c["model"])]
    return {**_place_wrists([r for r in roles if r.startswith("wrist")], saved, wrists, set_hubs, role_sets),
            **_place_globals([r for r in roles if not r.startswith("wrist")], saved, others, frames, refs)}


def swapped(cameras: dict, refs: dict, frames: dict) -> list:
    """Global roles whose view clearly belongs to another of them (all ports present, two cables exchanged)."""
    roles = [r for r in cameras if r in refs]
    if len(roles) < 2:
        return []
    cams = [{"path": cameras[r], "model": "", "hub": None} for r in roles]
    picks = _place_globals(roles, {r: cameras[r] for r in roles}, cams, frames, refs)
    return [r for r in roles if picks[r]["sure"] and picks[r]["score"] is not None and picks[r]["path"] != cameras[r]]


def ref_path(folder: Path, mode: str, role: str) -> Path:
    return folder / f"{mode}_{role}.jpg"


def save_refs(folder: Path, mode: str, cameras: dict) -> None:
    """The views the user just confirmed. A role without a usable picture loses its old one: it no longer applies."""
    folder.mkdir(parents=True, exist_ok=True)
    for role, path in cameras.items():
        gray = grab_gray(path)
        if gray is None:
            ref_path(folder, mode, role).unlink(missing_ok=True)
        else:
            cv2.imwrite(str(ref_path(folder, mode, role)), cv2.resize(gray, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 85])


def load_refs(folder: Path, mode: str, roles: list) -> dict:
    out = {}
    for role in roles:
        gray = cv2.imread(str(ref_path(folder, mode, role)), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            out[role] = view_vector(gray)
    return out
