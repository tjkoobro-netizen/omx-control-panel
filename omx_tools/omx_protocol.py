#!/usr/bin/env python3
"""Evaluation protocol v2: fixed position presets, per-attempt records, failure stages.

Presets (team shared, one file per task): <shared_root>/omx_presets/<id>.json
  {"id", "name", "task", "camera": "front", "size": [640, 480], "radius": 18,
   "detector": "red_hsv" | "none",                         (omx_vision.DETECTORS)
   "positions": [{"id": 1, "x": 300, "y": 225, "label": "", "role": "eval"}, ...],   role: eval | collect
   "fixtures": [{"name": "bowl", "x": 140, "y": 185, "radius": 45}],                  drawn as references
   "created_by", "updated"}
Attempts go to the eval's notes (.omx_notes/eval_*.json) next to the old "results":
  "protocol": {"version": 2, "preset_id", "preset_name", "detector", "positions": [...], "fixtures": [...]},
  "attempts": {"<episode>": {"position_id", "target_px", "start_px", "start_offset_px",
                             "result": success | fail | void | null, "fail_stage", "fail_stage_auto",
                             "fail_stage_source": auto | user, "grasp": {...}, "loop": {...},
                             "note": "", "train_candidate": null}}
Positions are used in preset order; an attempt marked void gives its position back, so the next attempt
retries it. Success rates leave void attempts out.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path

import omx_config

FAIL_STAGES = ("approach", "grasp", "carry", "release", "start_misbehavior", "other")
STAGE_KO = {"approach": "다가가기", "grasp": "잡기", "carry": "운반", "release": "놓기",
            "start_misbehavior": "시작 오동작", "other": "기타"}
RESULTS = ("success", "fail", "void")
TOLERANCE_PX = [8, 20]  # object vs target: green up to 8 px, yellow up to 20 px, red beyond (per preset: "tolerance_px")
LOCK = threading.Lock()

DEFAULT_PRESETS = [{
    "id": "strawberry_10", "name": "지정된 위치 10곳", "task": "Pick up the strawberry and put it in the bowl",
    "camera": "front", "size": [640, 480], "radius": 18, "detector": "red_hsv", "tolerance_px": TOLERANCE_PX,
    "detect_roi": [0, 40, 610, 425],  # the mat without the robot base (red motor parts at y < 20)
    "positions": [{"id": i + 1, "x": x, "y": y, "label": "", "role": "eval"} for i, (x, y) in enumerate(
        [(300, 225), (480, 250), (380, 330), (220, 300), (420, 130), (250, 110), (540, 110), (90, 300), (560, 320), (340, 170)])],
    "fixtures": [{"name": "bowl", "x": 140, "y": 185, "radius": 45}],
    "created_by": "omx", "updated": "2026-10-23 20:30",
}]


# ---------------------------------------------------------------- presets
def preset_dir() -> Path:
    root = Path(omx_config.CFG.get("shared_root") or "")
    if root.is_dir() and os.access(root, os.W_OK):
        d = root / "omx_presets"
    else:
        d = Path.home() / "il_ws" / "omx_presets"
    d.mkdir(parents=True, exist_ok=True)
    if not any(d.glob("*.json")):
        for p in DEFAULT_PRESETS:
            _write(d / f"{p['id']}.json", p)
    return d


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)
    try:
        os.chmod(path, 0o664)
    except OSError:
        pass


def list_presets() -> list[dict]:
    out = []
    for f in sorted(preset_dir().glob("*.json")):
        try:
            p = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(p, dict) and "positions" in p:  # other team files live here too (data_codes.json)
            out.append(p)
    return out


def load_preset(pid: str) -> dict:
    path = preset_dir() / f"{pid}.json"
    if not re.fullmatch(r"[a-z0-9_]+", pid or "") or not path.exists():
        raise RuntimeError("평가 방식(지정된 위치)을 찾을 수 없습니다.")
    return json.loads(path.read_text())


def save_preset(p: dict, user: str) -> dict:
    from omx_vision import DETECTORS

    pid = str(p.get("id") or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{1,40}", pid):
        raise RuntimeError("세트 ID는 영문 소문자·숫자·_ 40자 이내로 정하세요 (예: strawberry_10).")
    w, h = (p.get("size") or [640, 480])[:2]
    positions = []
    for i, q in enumerate(p.get("positions") or []):
        x, y = int(q["x"]), int(q["y"])
        if not (0 <= x < w and 0 <= y < h):
            raise RuntimeError(f"위치 {i + 1}이 화면({w}x{h}) 밖입니다.")
        positions.append({"id": i + 1, "x": x, "y": y, "label": str(q.get("label", ""))[:40],
                          "role": q.get("role", "eval") if q.get("role", "eval") in ("eval", "collect") else "eval"})
    if not positions:
        raise RuntimeError("위치를 하나 이상 정하세요.")
    fixtures = [{"name": str(f.get("name", ""))[:20] or "fixture", "x": int(f["x"]), "y": int(f["y"]),
                 "radius": int(f.get("radius", 40))} for f in (p.get("fixtures") or [])]
    detector = p.get("detector", "red_hsv")
    if detector not in DETECTORS:
        raise RuntimeError(f"감지기는 {', '.join(DETECTORS)} 중 하나입니다.")
    tol = p.get("tolerance_px") or TOLERANCE_PX
    try:
        tol = [float(tol[0]), float(tol[1])]
    except (TypeError, ValueError, IndexError):
        raise RuntimeError("허용 거리는 [초록 기준, 노랑 기준] 두 숫자입니다 (예: [8, 20]).")
    if not 0 < tol[0] <= tol[1] <= 200:
        raise RuntimeError("허용 거리는 0 < 초록 기준 ≤ 노랑 기준 ≤ 200 이어야 합니다.")
    roi = p.get("detect_roi")
    if roi:
        try:
            roi = [int(v) for v in roi][:4]
        except (TypeError, ValueError):
            raise RuntimeError("감지 범위는 [x0, y0, x1, y1] 네 숫자입니다.")
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= w and 0 <= roi[1] < roi[3] <= h):
            raise RuntimeError(f"감지 범위가 화면({w}x{h}) 안의 사각형이 아닙니다.")
    data = {"id": pid, "name": str(p.get("name") or pid)[:40], "task": str(p.get("task") or "")[:200],
            "camera": p.get("camera", "front"), "size": [int(w), int(h)], "radius": int(p.get("radius", 18)),
            "detector": detector, "tolerance_px": tol, "detect_roi": roi or None, "positions": positions, "fixtures": fixtures,
            "created_by": user, "updated": time.strftime("%Y-%m-%d %H:%M")}
    _write(preset_dir() / f"{pid}.json", data)
    return data


def delete_preset(pid: str) -> None:
    load_preset(pid)
    (preset_dir() / f"{pid}.json").unlink()


def eval_positions(proto: dict) -> list[dict]:
    return [p for p in proto.get("positions", []) if p.get("role", "eval") == "eval"]


# ---------------------------------------------------------------- attempts
def protocol_snapshot(preset: dict) -> dict:
    return {"version": 2, "preset_id": preset["id"], "preset_name": preset["name"], "detector": preset["detector"],
            "camera": preset.get("camera", "front"), "radius": preset.get("radius", 18),
            "tolerance_px": preset.get("tolerance_px") or TOLERANCE_PX, "detect_roi": preset.get("detect_roi"),
            "positions": preset["positions"], "fixtures": preset.get("fixtures", [])}


def free_positions(note: dict) -> list[dict]:
    """Eval positions in preset order that no attempt holds (void attempts give theirs back)."""
    used = {a.get("position_id") for a in note.get("attempts", {}).values() if a.get("result") != "void"}
    return [p for p in eval_positions(note.get("protocol", {})) if p["id"] not in used]


def begin_attempt(note: dict, episode: int, start_px: tuple | None) -> dict | None:
    """An episode starts: keep its position on a re-record, else take the first free position."""
    attempts = note.setdefault("attempts", {})
    a = attempts.get(str(episode))
    if a is None:
        free = free_positions(note)
        if not free:
            return None
        pos = free[0]
        a = {"position_id": pos["id"], "target_px": [pos["x"], pos["y"]], "result": None, "fail_stage": None,
             "fail_stage_auto": None, "fail_stage_source": None, "note": "", "train_candidate": None}
        attempts[str(episode)] = a
    a["start_px"] = list(start_px) if start_px else None
    a["start_offset_px"] = round(math.dist(a["start_px"], a["target_px"]), 1) if start_px else None
    a["started"] = time.strftime("%H:%M:%S")
    return a


KEEP = object()


def set_start(note: dict, episode: int, px, source: str) -> None:
    """Where the object was when the attempt started (source: live view, or the saved first frame)."""
    a = note.get("attempts", {}).get(str(episode))
    if a is None:
        return
    a["start_px"] = list(px) if px else None
    a["start_offset_px"] = round(math.dist(a["start_px"], a["target_px"]), 1) if px else None
    a["start_px_source"] = source


def mark(note: dict, episode: int, result=KEEP, fail_stage=KEEP, text=KEEP, train_candidate=KEEP) -> dict:
    """User marks one attempt. result: success | fail | void | None (clear); fail_stage: a stage,
    or "" to go back to the estimate; text: a note. Only the given fields change."""
    ep = str(int(episode))
    a = note.get("attempts", {}).get(ep)
    if result is not KEEP:
        if result not in RESULTS + (None,):
            raise RuntimeError("결과는 성공·실패·무효 중 하나입니다.")
        results = note.setdefault("results", {})
        if result in ("success", "fail"):
            results[ep] = result  # the team table (INDEX "평가" column) reads these; void is left out
        else:
            results.pop(ep, None)
        (note.get("aborted") or {}).pop(ep, None)  # a free-position attempt that was aborted: now marked
        if a is not None:
            a["result"] = result
            if a.get("fail_stage_source") != "user":
                est = a.get("fail_stage_auto") if result == "fail" else None
                a["fail_stage"], a["fail_stage_source"] = est, ("auto" if est else None)
    if a is not None and fail_stage is not KEEP:
        if fail_stage in FAIL_STAGES:
            a["fail_stage"], a["fail_stage_source"] = fail_stage, "user"
        else:  # back to the estimate
            est = a.get("fail_stage_auto") if a.get("result") == "fail" else None
            a["fail_stage"], a["fail_stage_source"] = est, ("auto" if est else None)
    if a is not None and text is not KEEP:
        a["note"] = str(text or "")[:200]
    if a is not None and train_candidate is not KEEP:
        if train_candidate and a.get("result") != "success":
            raise RuntimeError("학습 후보는 성공으로 표시한 시도만 고를 수 있습니다.")
        a["train_candidate"], a["train_candidate_source"] = bool(train_candidate), "user"
    if a is not None and a.get("result") != "success" and a.get("train_candidate"):
        a["train_candidate"], a["train_candidate_source"] = False, "auto"  # no longer a success
    return note


ABORT_KO = {"abort": "즉시 중단", "freeze": "비상 정지"}


def mark_aborted(note: dict, episode: int, mode: str, secs: float) -> dict:
    """An attempt cut by 즉시 중단 / 비상 정지 and saved as far as it got: result "aborted". Left out of
    success rates (counted apart), never a training candidate, kept for the failure analysis and the
    report; the user can still change it to fail or void."""
    ep = str(int(episode))
    note.setdefault("results", {})[ep] = "aborted"
    info = {"result": "aborted", "abort_mode": mode, "abort_mode_ko": ABORT_KO.get(mode, mode), "abort_at_s": round(secs, 1),
            "train_candidate": False}
    a = note.get("attempts", {}).get(ep)
    if a is not None:
        a.update(info)
    else:
        note.setdefault("aborted", {})[ep] = info
    return note


def select_candidates(note: dict, mode: str) -> dict:
    """Bulk selection: rec_all (priority 1 and 2), rec1 (priority 1 only), clear."""
    for a in note.get("attempts", {}).values():
        if mode == "clear":
            a["train_candidate"], a["train_candidate_source"] = False, "auto"
        elif a.get("result") == "success":
            want = a.get("train_rec_priority") in ((1, 2) if mode == "rec_all" else (1,))
            a["train_candidate"], a["train_candidate_source"] = want, "auto"
    return note


def candidate_episodes(note: dict) -> dict:
    """Checked training candidates of one evaluation and how many of each kind."""
    eps, kinds = [], {"one_shot": 0, "recovery": 0, "wandering_by_user": 0}
    for ep, a in sorted(note.get("attempts", {}).items(), key=lambda kv: int(kv[0])):
        if a.get("train_candidate") and a.get("result") == "success":
            eps.append(int(ep))
            rec = a.get("train_rec")
            kinds["wandering_by_user" if rec == "wandering" or rec is None else rec] += 1
    return {"episodes": eps, **kinds}


# ---------------------------------------------------------------- automatic failure stage
_DEMO_STATS: dict = {}


def attempt_metrics(data_home: Path, repo: str, episode: int, model_path: str | None) -> dict | None:
    """Grasp metrics and the estimated failure stage of one saved episode (None if not readable yet)."""
    import numpy as np

    import omx_eval_report as R

    try:
        data = R.joint_data(data_home / repo)
        rows = data[data.episode_index == episode]
        if not len(rows):
            return None
        track = np.stack(rows["observation.state"].to_numpy())
        fps = int(json.loads((data_home / repo / "meta/info.json").read_text()).get("fps", 30))
    except Exception:
        return None
    g = R.grasp_info(track, fps)
    stats = None
    if model_path:
        try:
            demo = json.loads((Path(model_path) / "train_config.json").read_text())["dataset"]["repo_id"]
            if demo not in _DEMO_STATS:
                ddata = R.joint_data(data_home / demo)
                tracks = [np.stack(ddata[ddata.episode_index == e]["observation.state"].to_numpy())
                          for e in sorted(ddata.episode_index.unique())]
                _DEMO_STATS[demo] = R.demo_stats([R.grasp_info(t, fps) for t in tracks])
            stats = _DEMO_STATS[demo]
        except Exception:
            stats = None
    keep = ("gripper_close_s", "gripper_release_s", "closed_value", "holding_guess", "grasp_count", "length_s",
            "max_pause_s", "idle_s")
    out = {"grasp": {k: g.get(k) for k in keep},
           "fail_stage_auto": R.fail_stage(g, (stats or {}).get("release_pose"))}
    out.update(R.train_recommendation(g, stats, first_episode=episode == 0))
    if stats:  # the thresholds used, so the decision can be traced after they are tuned
        out["train_rec_thresholds"] = {k: stats.get(k) for k in ("long_pause_s", "pause_mean_s", "pause_std_s",
                                                                  "long_length_s", "length_mean_s")}
        out["train_rec_thresholds"].update(pause_sigma=R.PAUSE_SIGMA, pause_min_s=R.PAUSE_MIN_S, length_factor=R.LENGTH_FACTOR)
    return out


def first_frame_start(data_home: Path, repo: str, episode: int, proto_: dict) -> tuple[bool, list | None]:
    """Object position on the saved episode's first front frame - what the policy saw at the start.
    Returns (saved, position): saved is False while the episode is not written yet."""
    import omx_eval_report as R
    import omx_vision

    root = data_home / repo
    meta = R.episodes_meta(root)
    rows = [r for r in (meta.to_dict("records") if len(meta) else []) if int(r["episode_index"]) == episode]
    if not rows:
        return False, None
    frame = R.episode_frames(root, rows[0], proto_.get("camera", "front"), [0.0])
    px = omx_vision.detect(proto_["detector"], frame[0], proto_.get("detect_roi")) if frame else None
    return True, (list(px) if px else None)


def fill_auto(note: dict, data_home: Path, repo: str) -> bool:
    """Compute the estimate for attempts that lack it (episodes are readable once saved), and take
    the start position from the saved first frame."""
    changed = False
    p = note.get("protocol") or {}
    for ep, a in note.get("attempts", {}).items():
        if p and p.get("detector", "none") != "none" and a.get("start_px_source") != "frame0":
            try:
                saved, px = first_frame_start(data_home, repo, int(ep), p)
            except Exception:
                saved, px = False, None
            if saved:
                set_start(note, int(ep), px, "frame0")
                changed = True
        if a.get("grasp") is not None:
            continue
        m = attempt_metrics(data_home, repo, int(ep), note.get("model"))
        if m is None:
            continue
        a.update(m)
        a.setdefault("train_candidate", False)
        if a.get("result") == "fail" and a.get("fail_stage_source") != "user":
            a["fail_stage"], a["fail_stage_source"] = m["fail_stage_auto"], "auto"
        changed = True
    return changed
