#!/usr/bin/env python3
"""Standard home pose: where the follower starts (and ends) every run, taken from the training data.

A model's home pose is the per-joint median of the first frame (observation.state, gripper included)
of the episodes it was trained on. The median ignores the few demos that started elsewhere. The
gripper's start state follows the data the same way (open if most demos started open); a mixed start
(the smaller side above 20 %) is reported as a warning. It is cached in the run folder as
home_pose.json (written once, never overwritten) and kept with each evaluation's notes.
Teleop and recording have no model: they use the task's home pose from
<shared_root>/omx_presets/home_poses.json ({"tasks": {"straw": {...}}, "default": "straw"}), computed the
same way from every recording of that task ([다시 계산] in the panel).

  python omx_home.py model <model_path>      print (and cache) a model's home pose
  python omx_home.py task <code> [--apply]   compute a task's home pose (dry run unless --apply)

Units are the panel's normalized joint units (-100..100, ~1.8 deg each; gripper 0..100).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

DATA_HOME = Path(os.environ.get("HF_LEROBOT_HOME", str(omx_config.path("data_home"))))
CKPT_HOME = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints"))))
GRIP_OPEN, GRIP_CLOSED = 57.0, 52.0   # follower gripper: open >= 57, closed < 52 (in between: half)
MIXED = 0.2                           # smaller gripper group above this share -> warning
FLAG_SIGMA = 3.0                      # a joint further than this many std from home is flagged
MIN_STD = 0.5                         # units: floor for the std, so a very consistent joint is not flagged for noise


def enabled() -> bool:
    return bool(omx_config.CFG.get("home_pose"))


def first_frames(root: Path, episodes: list[int] | None = None):
    """(names, array n x joints) of the first frame of every (or the given) episode."""
    import numpy as np
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    names = [n.removesuffix(".pos") for n in info["features"]["observation.state"]["names"]]
    rows = []
    for f in sorted((root / "data").glob("*/*.parquet")):
        d = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        d = d[d.frame_index == 0]
        if episodes is not None:
            d = d[d.episode_index.isin(episodes)]
        rows += list(d["observation.state"])
    return names, np.stack(rows) if rows else np.zeros((0, len(names)))


def compute(parts: list[tuple[Path, list[int] | None]], source: str) -> dict:
    import numpy as np

    names, arrays = None, []
    for root, eps in parts:
        n, a = first_frames(root, eps)
        if names is not None and n != names:
            raise RuntimeError(f"{root}: 관절 구성이 다릅니다 ({n} / {names})")
        names = n
        arrays.append(a)
    data = np.concatenate(arrays) if arrays else np.zeros((0, 0))
    if not len(data):
        raise RuntimeError(f"{source}: 첫 프레임을 읽지 못했습니다")
    med, std = np.median(data, axis=0), np.std(data, axis=0)
    out = {"joints": {k: round(float(v), 2) for k, v in zip(names, med)},
           "std": {k: round(float(v), 2) for k, v in zip(names, std)},
           "episodes": int(len(data)), "source": source, "method": "median of first frames",
           "computed": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    if "gripper" in names:
        g = data[:, names.index("gripper")]
        counts = {"open": int((g >= GRIP_OPEN).sum()), "half": int(((g >= GRIP_CLOSED) & (g < GRIP_OPEN)).sum()),
                  "closed": int((g < GRIP_CLOSED).sum())}
        out["gripper_counts"] = counts
        out["gripper_start"] = "open" if out["joints"]["gripper"] >= GRIP_OPEN else "closed" if out["joints"]["gripper"] < GRIP_CLOSED else "half"
        if min(counts["open"], counts["closed"]) > MIXED * len(g):
            out["warning"] = f"학습 데이터 시작 그리퍼: 열림 {counts['open']} / 닫힘 {counts['closed']} (섞여 있음)"
    return out


# ---------------------------------------------------------------- per model
def run_dir(model_path: str | Path) -> Path:
    """<run>/pretrained_model or <run>/checkpoints/012500/pretrained_model -> <run>."""
    p = Path(model_path)
    if p.name == "pretrained_model":
        p = p.parent
    if p.parent.name == "checkpoints":
        p = p.parent.parent
    return p


def model_home(model_path: str | Path, cache: bool = True) -> dict:
    """Home pose of a model from its training data (cached in the run folder, written once)."""
    rdir = run_dir(model_path)
    cached = rdir / "home_pose.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text())
        except (OSError, ValueError):
            pass
    cfg = json.loads((Path(model_path) / "train_config.json").read_text())
    d = cfg["dataset"]
    root = Path(d["root"]) if d.get("root") else DATA_HOME / d["repo_id"]
    home = compute([(root, d.get("episodes"))], d["repo_id"])
    if cache and os.access(rdir, os.W_OK) and not cached.exists():  # never overwrite; a teammate's folder is left alone
        try:
            cached.write_text(json.dumps(home, ensure_ascii=False, indent=2))
            os.chmod(cached, 0o664)
        except OSError:
            pass
    return home


# ---------------------------------------------------------------- per task (teleop, recording)
def task_file() -> Path:
    import omx_protocol as proto

    return proto.preset_dir() / "home_poses.json"


def load_tasks() -> dict:
    try:
        return json.loads(task_file().read_text())
    except (OSError, ValueError):
        return {"tasks": {}, "default": None}


def task_home(code: str | None = None) -> dict | None:
    t = load_tasks()
    return (t.get("tasks") or {}).get(code or t.get("default") or "")


def task_recordings(code: str) -> list[Path]:
    """Original recordings (not training/eval datasets) whose task sentence has this task code."""
    import pandas as pd

    import omx_names

    codes = omx_names.load_codes()
    out = []
    for info in sorted(DATA_HOME.glob("*/*/meta/info.json")):
        root = info.parent.parent
        if root.name.startswith(("eval_", "train_", "_tmp")) or (root / "meta/omx_sources.json").exists():
            continue
        try:
            tasks = list(pd.read_parquet(root / "meta/tasks.parquet").index)
        except Exception:
            continue
        if omx_names.task_code(tasks, codes) == code:
            out.append(root)
    return out


def compute_task(code: str, apply: bool = False) -> dict:
    roots = task_recordings(code)
    if not roots:
        raise RuntimeError(f"작업 '{code}'의 녹화를 찾지 못했습니다")
    home = compute([(r, None) for r in roots], f"{code}: " + ", ".join(f"{r.parent.name}/{r.name}" for r in roots))
    if apply:
        t = load_tasks()
        t.setdefault("tasks", {})
        if code in t["tasks"]:
            home["previous"] = {k: v for k, v in t["tasks"][code].items() if k != "previous"}
        t["tasks"][code] = home
        t["default"] = t.get("default") or code
        task_file().write_text(json.dumps(t, ensure_ascii=False, indent=2))
        try:
            os.chmod(task_file(), 0o664)
        except PermissionError:
            pass
    return home


# ---------------------------------------------------------------- comparing with the arm
def diff(home: dict, current: dict | None) -> list[dict]:
    """Per joint: home, current, difference, and whether it is further than FLAG_SIGMA std."""
    rows = []
    for k, v in (home.get("joints") or {}).items():
        c = (current or {}).get(k)
        d = None if c is None else round(c - v, 1)
        sd = max((home.get("std") or {}).get(k, 0.0), MIN_STD)
        rows.append({"joint": k, "home": v, "current": None if c is None else round(c, 1), "diff": d,
                     "limit": round(FLAG_SIGMA * sd, 1), "flag": d is not None and abs(d) > FLAG_SIGMA * sd})
    return rows


def read_follower(port: str | None = None) -> dict:
    """Read the follower's present position (bus only, torque untouched). Only when nothing else uses it."""
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

    robot = OmxFollower(OmxFollowerConfig(port=port or omx_config.CFG["follower_port"], id="omx_follower_arm"))
    robot.bus.connect()
    try:
        pos = robot.bus.sync_read("Present_Position")
    finally:
        robot.bus.disconnect(disable_torque=False)
    return {m: round(float(v), 2) for m, v in pos.items()}


def write_run_file(home: dict, path: Path) -> Path:
    """The home pose for one run, read by omx_safe_start (OMX_HOME_POSE)."""
    path.write_text(json.dumps(home, ensure_ascii=False, indent=2))
    return path


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "model":
        print(json.dumps(model_home(sys.argv[2], cache="--no-cache" not in sys.argv), ensure_ascii=False, indent=2))
    elif len(sys.argv) >= 3 and sys.argv[1] == "task":
        apply = "--apply" in sys.argv
        print(json.dumps(compute_task(sys.argv[2], apply), ensure_ascii=False, indent=2))
        print("적용함" if apply else "미리보기만 했습니다 (--apply로 저장)")
    else:
        raise SystemExit(__doc__)
