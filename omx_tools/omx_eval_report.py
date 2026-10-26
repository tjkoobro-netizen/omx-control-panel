#!/usr/bin/env python3
"""Evaluation report: compare policy runs (eval_ datasets) with the demonstrations a model learned from.

  python omx_eval_report.py tkoo/eval_x [tkoo/eval_y ...] --demos tkoo/strawberry_train_1013 [--episodes 1-20 | 1,2,5-9]

Writes to <data_home>/<owner of first eval>/.omx_reports/<name>/
  sheet_<eval>_ep<N>.jpg   front (top row) and wrist (bottom row) every ~2 s of each attempt
  start_map.jpg            where the strawberry started: demos (green) vs attempts (red, numbered)
  joints.jpg               joint angles over time: demo range (grey band), demo mean, attempts
  report.json              per-attempt numbers (grasp time/pose, idle time, distance to demos)
Works with video only when an eval run lost its joint data (e.g. force-quit while saving).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import sys

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_vision  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

DATA_HOME = Path(os.environ.get("HF_LEROBOT_HOME", str(omx_config.path("data_home"))))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def episodes_meta(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index") if files else pd.DataFrame()


def parse_episodes(spec: str) -> list[int]:
    """'1-20' or '1,2,5-9' -> list of episode indices."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out += list(range(a, b + 1))
        elif part:
            out.append(int(part))
    return out


def unfinished_rows(root: Path) -> list[dict]:
    """An eval run that was force-quit has no episode table: treat each whole video file as one attempt."""
    info = json.loads((root / "meta/info.json").read_text())
    keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    rows = []
    for vid in sorted((root / "videos" / keys[0]).glob("chunk-*/file-*.mp4")):
        chunk, file = int(vid.parent.name.split("-")[1]), int(vid.stem.split("-")[1])
        with av.open(str(vid)) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
        row = {"episode_index": len(rows), "length": int(dur * info["fps"]), "unfinished": True}
        for k in keys:
            row.update({f"videos/{k}/chunk_index": chunk, f"videos/{k}/file_index": file,
                        f"videos/{k}/from_timestamp": 0.0, f"videos/{k}/to_timestamp": dur})
        rows.append(row)
    return rows


def joint_data(root: Path) -> pd.DataFrame | None:
    try:
        return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet")))])
    except Exception:
        # e.g. the parquet footer is missing after a force-quit: use the per-episode backups
        return backup_joint_data(root)


def backup_joint_data(root: Path) -> pd.DataFrame | None:
    """Joint data from omx_record.py's backups (<owner>/.omx_backup/<dataset>/episode_*.npz)."""
    files = sorted((root.parent / ".omx_backup" / root.name).glob("episode_*.npz"))
    frames = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            n = len(z["observation__state"])
            frames.append(pd.DataFrame({
                "episode_index": [meta["episode_index"]] * n,
                "frame_index": z["frame_index"] if "frame_index" in z else np.arange(n),
                "observation.state": list(z["observation__state"]), "action": list(z["action"])}))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


# Smoothness of an attempt (Validation Leader's measures). Active span = first to last arm motion.
MOVE_SPEED = 3.0    # units/s summed over the 5 arm joints: "moving"
STOP_SPEED = 1.0    # units/s: a frame inside the active span slower than this counts as a stop


def smoothness(state: np.ndarray, action: np.ndarray | None, fps: float) -> dict | None:
    """stop_frac: share of stopped frames inside the active span; accel: mean |second difference of
    the state| x fps^2 summed over the arm joints (units/s^2); cmd_jump: share of command steps larger
    than 4 x their median + 1 (target jumps, e.g. at chunk changes)."""
    s = np.asarray(state, dtype=float)[:, :5]
    v = np.abs(np.diff(s, axis=0)).sum(1) * fps
    idx = np.where(v > MOVE_SPEED)[0]
    if len(idx) < 10:
        return None
    span = slice(idx[0], idx[-1])
    out = {"stop_frac": round(float((v[span] < STOP_SPEED).mean()), 3),
           "accel": round(float(np.abs(np.diff(s[span], 2, axis=0)).sum(1).mean() * fps * fps), 1)}
    if action is not None:
        a = np.asarray(action, dtype=float)[:, :5]
        step = np.abs(np.diff(a[span], axis=0)).sum(1)
        out["cmd_jump"] = round(float((step > 4 * np.median(step) + 1).mean()), 3) if len(step) else None
    return out


def mean_smoothness(items: list[dict]) -> dict | None:
    items = [i for i in items if i]
    if not items:
        return None
    return {k: round(float(np.mean([i[k] for i in items if i.get(k) is not None])), 3 if k != "accel" else 1)
            for k in ("stop_frac", "accel", "cmd_jump") if any(i.get(k) is not None for i in items)}


def eval_smoothness(root: Path) -> dict | None:
    """Mean smoothness over the saved attempts of one evaluation (with the per-attempt values)."""
    data = joint_data(root)
    if data is None or not len(data):
        return None
    try:
        fps = json.loads((root / "meta/info.json").read_text()).get("fps", 30)
    except (OSError, ValueError):
        fps = 30
    per = {}
    for ep, d in data.groupby("episode_index"):
        d = d.sort_values("frame_index")
        per[int(ep)] = smoothness(np.stack(d["observation.state"].to_numpy()),
                                  np.stack(d["action"].to_numpy()) if "action" in d else None, fps)
    return {"mean": mean_smoothness(list(per.values())), "attempts": per}


def episode_frames(root: Path, row: dict, cam: str, times: list[float]) -> list[np.ndarray]:
    """Decode the frames of one episode nearest to the given episode-relative times (seconds)."""
    key = f"observation.images.{cam}"
    info = json.loads((root / "meta/info.json").read_text())
    path = root / info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                            file_index=int(row[f"videos/{key}/file_index"]))
    t0 = float(row[f"videos/{key}/from_timestamp"])
    want = [t0 + t for t in times]
    out, i = [], 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1, 0) / s.time_base), stream=s, any_frame=False, backward=True)
        for fr in c.decode(s):
            if fr.time is None:
                continue
            while i < len(want) and fr.time >= want[i] - 1e-3:
                out.append(fr.to_ndarray(format="bgr24"))
                i += 1
            if i >= len(want):
                break
    return out


def strawberry_xy(img: np.ndarray) -> tuple[int, int] | None:
    """Centre of the largest saturated red blob (see omx_vision.red_hsv)."""
    return omx_vision.red_hsv(img)


def label(img, text, org=(6, 20), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1, cv2.LINE_AA)


def plot_joints(demo_tracks: list[np.ndarray], attempts: list[tuple[str, np.ndarray]], fps: int) -> np.ndarray:
    """3x2 grid of joint-vs-time panels drawn with OpenCV (no matplotlib in the venv)."""
    W, H, pad = 420, 220, 34
    colors = [(40, 40, 230), (230, 120, 30), (40, 170, 40), (200, 40, 200), (30, 160, 220), (120, 80, 20)]
    tmax = max([len(t) for t in demo_tracks] + [len(a) for _, a in attempts]) / fps
    panels = []
    for j, name in enumerate(JOINTS):
        img = np.full((H, W, 3), 255, np.uint8)
        vals = np.concatenate([t[:, j] for t in demo_tracks] + [a[:, j] for _, a in attempts])
        lo, hi = float(vals.min()), float(vals.max())
        lo, hi = lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1)
        X = lambda t: int(pad + t / tmax * (W - pad - 8))
        Y = lambda v: int(8 + (1 - (v - lo) / (hi - lo)) * (H - pad - 8))
        n = min(len(t) for t in demo_tracks)
        stack = np.stack([t[:n, j] for t in demo_tracks])
        band_lo, band_hi, mean = stack.min(0), stack.max(0), stack.mean(0)
        poly = [(X(i / fps), Y(v)) for i, v in enumerate(band_hi)] + [(X(i / fps), Y(v)) for i, v in reversed(list(enumerate(band_lo)))]
        cv2.fillPoly(img, [np.array(poly, np.int32)], (225, 225, 225))
        cv2.polylines(img, [np.array([(X(i / fps), Y(v)) for i, v in enumerate(mean)], np.int32)], False, (110, 110, 110), 2)
        for k, (_, a) in enumerate(attempts):
            cv2.polylines(img, [np.array([(X(i / fps), Y(v)) for i, v in enumerate(a[:, j])], np.int32)], False, colors[k % len(colors)], 2)
        cv2.rectangle(img, (pad, 8), (W - 8, H - pad), (160, 160, 160), 1)
        for tick in range(0, int(tmax) + 1, 5):
            cv2.putText(img, f"{tick}s", (X(tick) - 8, H - pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        cv2.putText(img, f"{hi:.0f}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        cv2.putText(img, f"{lo:.0f}", (2, H - pad), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
        cv2.putText(img, name, (pad + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        panels.append(img)
    grid = np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, 6, 2)])
    legend = np.full((26 + 20 * len(attempts), grid.shape[1], 3), 255, np.uint8)
    cv2.putText(legend, "grey band = demo min~max, grey line = demo mean", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    for k, (name, _) in enumerate(attempts):
        y = 38 + 20 * k
        cv2.line(legend, (8, y - 4), (38, y - 4), colors[k % len(colors)], 3)
        cv2.putText(legend, name, (46, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([legend, grid])


# Follower gripper values (0..100) seen on this robot: open 57-61 (the leader trigger stops at
# gripper_open_pos=60), closed on a strawberry 52-54, closed on nothing 49-51. A state machine with a
# gap between the two thresholds ignores the small wobble while holding.
GRIP_OPEN, GRIP_CLOSED, GRIP_EMPTY = 57.0, 55.0, 51.5


def grasp_info(track: np.ndarray, fps: int) -> dict:
    """Grasp events from the follower gripper value.

    close   = first open -> closed transition (gripper_close_s, pose_at_close)
    release = first closed -> open transition after it (gripper_release_s, e.g. put in the bowl)
    closed_value = median gripper value while closed the first time (on a strawberry ~52-54,
                   on nothing ~49-51 -> holding_guess false)
    grasp_count  = number of open -> closed transitions (more than 1 = it tried again)
    idle = time with almost no arm motion.
    """
    g = track[:, 5].astype(float)
    k = 5  # smoothing window; pad with the edge values so the ends do not sag towards 0
    smooth = np.convolve(np.pad(g, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    state = "open" if smooth[0] >= GRIP_OPEN else "closed" if smooth[0] <= GRIP_CLOSED else None
    closes, releases = [], []
    for i, v in enumerate(smooth):
        if v >= GRIP_OPEN:
            if state == "closed" and closes:
                releases.append(i)
            state = "open"
        elif v <= GRIP_CLOSED:
            if state == "open":
                closes.append(i)
            state = "closed"
    close_i = closes[0] if closes else None
    release_i = next((r for r in releases if close_i is not None and r > close_i), None)
    closed_value = None
    if close_i is not None:
        seg = g[close_i:release_i] if release_i else g[close_i:]
        closed_value = round(float(np.median(seg)), 1) if len(seg) else None
    speed = np.abs(np.diff(track[:, :5], axis=0)).sum(1) * fps
    # longest stop between the first arm motion and the end of the task (the release, else the last
    # motion): waiting before the start and after the strawberry is in the bowl are left out
    moving = np.flatnonzero(speed >= 1.0)
    if release_i is not None:
        moving = moving[moving <= release_i]
    max_pause = 0
    if len(moving) > 1:
        gaps = np.diff(moving) - 1
        max_pause = int(gaps.max()) if len(gaps) else 0
    return {"max_pause_s": round(max_pause / fps, 1),
            "gripper_close_s": round(close_i / fps, 1) if close_i is not None else None,
            "gripper_release_s": round(release_i / fps, 1) if release_i is not None else None,
            "pose_at_close": [round(float(v), 1) for v in track[close_i, :5]] if close_i is not None else None,
            "pose_at_release": [round(float(v), 1) for v in track[release_i, :5]] if release_i is not None else None,
            "pose_at_start": [round(float(v), 1) for v in track[0, :5]],
            "closed_value": closed_value,
            "holding_guess": closed_value >= GRIP_EMPTY if closed_value is not None else None,
            "grasp_count": len(closes),
            "gripper_min": round(float(g.min()), 1), "gripper_max": round(float(g.max()), 1),
            "idle_s": round(float((speed < 1.0).sum() / fps), 1), "length_s": round(len(track) / fps, 1)}


# Failure stages of an attempt (evaluation protocol v2). Estimated from the gripper metrics; the user
# can correct them in the panel. Thresholds in joint units (-100..100, 1 unit ~ 1.8 deg).
FAIL_STAGES = ("approach", "grasp", "carry", "release", "start_misbehavior", "other")
START_MOVE = 5.0     # mean |joint - start pose| below this = the arm had not really left its start pose
CARRY_FAR = 25.0     # mean |release pose - demos' mean release pose| above this = dropped on the way
                     # (demos themselves spread up to ~35 at most, p90 ~14)
REGRASP = 3          # closed this many times = kept missing the object


def fail_stage(g: dict, demo_release_pose: list | None = None) -> str:
    """Rules, first match wins:
    start_misbehavior  closed within 1 s while still at the start pose (e.g. Diffusion 10/23 #1)
    approach           never closed the gripper
    grasp              closed on nothing (closed_value below the "holding" level), or closed 3+ times
    release            held it and never let go
    carry              let go far from where the demos let go (dropped on the way)
    release            otherwise (let go near the bowl but missed)
    """
    pose = lambda key: np.array(g[key], dtype=float) if g.get(key) is not None else None  # noqa: E731
    close, start = pose("pose_at_close"), pose("pose_at_start")
    if (g.get("gripper_close_s") is not None and g["gripper_close_s"] <= 1.0 and close is not None
            and start is not None and float(np.mean(np.abs(close[:4] - start[:4]))) < START_MOVE):
        return "start_misbehavior"
    if g.get("gripper_close_s") is None:
        return "approach"
    if g.get("holding_guess") is False or (g.get("grasp_count") or 0) >= REGRASP:
        return "grasp"
    if g.get("gripper_release_s") is None:
        return "release"
    release = pose("pose_at_release")
    if demo_release_pose is not None and release is not None:
        if float(np.mean(np.abs(release[:4] - np.array(demo_release_pose[:4], dtype=float)))) > CARRY_FAR:
            return "carry"
    return "release"


# Training-candidate recommendation for successful attempts (evaluation protocol v2). Recovery
# (a second grasp) is worth learning; wandering (many grasps, long stops, very long attempts) is not.
PAUSE_SIGMA, PAUSE_MIN_S, LENGTH_FACTOR = 2.0, 1.0, 1.5


def demo_stats(demo_grasps: list[dict]) -> dict:
    """What "normal" looks like in the demos: release pose, longest stop, attempt length."""
    pauses = [g["max_pause_s"] for g in demo_grasps if g.get("max_pause_s") is not None]
    # task time = until the strawberry is let go (an eval episode keeps running to its time limit)
    lengths = [g["gripper_release_s"] for g in demo_grasps if g.get("gripper_release_s") is not None]
    p_mean, p_std = (float(np.mean(pauses)), float(np.std(pauses))) if pauses else (0.0, 0.0)
    return {"release_pose": demo_release_pose(demo_grasps),
            "pause_mean_s": round(p_mean, 2), "pause_std_s": round(p_std, 2),
            "long_pause_s": round(max(PAUSE_MIN_S, p_mean + PAUSE_SIGMA * p_std), 1),
            "length_mean_s": round(float(np.mean(lengths)), 1) if lengths else None,
            "long_length_s": round(float(np.mean(lengths)) * LENGTH_FACTOR, 1) if lengths else None}


def train_recommendation(g: dict, stats: dict | None, first_episode: bool = False) -> dict:
    """one_shot (priority 1): one grasp, no long stop, normal task time
    recovery (priority 2): two grasps, no long stop, normal task time
    wandering (not recommended): 3+ grasps, a long stop, or a task time (until the release) longer
    than LENGTH_FACTOR x the demos' mean"""
    stats = stats or {}
    n = g.get("grasp_count") or 0
    pause = g.get("max_pause_s") or 0.0
    length = g["gripper_release_s"] if g.get("gripper_release_s") is not None else (g.get("length_s") or 0.0)
    long_pause = pause > (stats.get("long_pause_s") or 1e9)
    long_len = stats.get("long_length_s") is not None and length > stats["long_length_s"]
    if n >= 3 or long_pause or long_len or n == 0:
        rec, prio = "wandering", None
    elif n == 1:
        rec, prio = "one_shot", 1
    else:
        rec, prio = "recovery", 2
    reason = f"잡기 {n}회 · 멈춤 {pause:.1f}s · 놓기까지 {length:.1f}s"
    if first_episode:
        # the first attempt of a run starts with the safe start (slow, 30% output approach), which no
        # human demo contains: at most priority 2
        reason += " · 안전 시작 포함"
        if prio == 1:
            prio = 2
    return {"train_rec": rec, "train_rec_priority": prio, "train_rec_reason": reason}


def demo_release_pose(demo_grasps: list[dict]) -> list | None:
    poses = [g["pose_at_release"] for g in demo_grasps if g.get("pose_at_release") is not None]
    return [round(float(v), 1) for v in np.mean(poses, axis=0)] if poses else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--demos", required=True)
    ap.add_argument("--episodes", help="demo episodes used for training, e.g. 1-20")
    ap.add_argument("--out")
    args = ap.parse_args()

    out = Path(args.out) if args.out else DATA_HOME / args.evals[0].split("/")[0] / ".omx_reports" / Path(args.evals[0]).name
    out.mkdir(parents=True, exist_ok=True)
    droot = DATA_HOME / args.demos
    dmeta = episodes_meta(droot)
    if args.episodes:
        dmeta = dmeta[dmeta.episode_index.isin(parse_episodes(args.episodes))]
    fps = json.loads((droot / "meta/info.json").read_text())["fps"]
    ddata = joint_data(droot)
    demo_tracks = [np.stack(ddata[ddata.episode_index == e]["observation.state"].to_numpy()) for e in dmeta.episode_index]

    # Strawberry start positions in the demos (first front frame of each episode)
    demo_starts = []
    for row in dmeta.to_dict("records"):
        f = episode_frames(droot, row, "front", [0.0])
        if f and (xy := strawberry_xy(f[0])):
            demo_starts.append(xy)

    report = {"demos": args.demos, "demo_episodes": len(dmeta), "attempts": []}
    try:  # which model, by its folder (run id) and the team display name
        import omx_names

        first = args.evals[0]
        note = json.loads((DATA_HOME / first.split("/")[0] / ".omx_notes" / f"{first.split('/')[1]}.json").read_text())
        report["model"] = note.get("run_id")
        report["model_display"] = omx_names.display_name(note.get("run_id"))
    except Exception:
        pass
    attempts_tracks, base_img, k = [], None, 0
    for repo in args.evals:
        root = DATA_HOME / repo
        meta = episodes_meta(root)
        data = joint_data(root)
        rows = meta.to_dict("records") if len(meta) else unfinished_rows(root)
        for row in rows:
            ep = int(row["episode_index"])
            length = int(row["length"]) / fps
            times = list(np.linspace(0, max(length - 0.2, 0), 8))
            front, wrist = episode_frames(root, row, "front", times), episode_frames(root, row, "wrist", times)
            tiles = []
            for row_imgs in (front, wrist):
                small = [cv2.resize(im, (240, 180)) for im in row_imgs]
                for im, t in zip(small, times):
                    label(im, f"{t:.1f}s")
                tiles.append(np.hstack(small))
            sheet = np.vstack(tiles)
            k += 1
            label(sheet, f"#{k} {Path(repo).name} ep{ep}", (6, 172), 0.6)
            cv2.imwrite(str(out / f"sheet_{k:02d}_{Path(repo).name}_ep{ep}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
            start_xy = strawberry_xy(front[0]) if front else None
            if base_img is None and front:
                base_img = front[0].copy()
            item = {"n": k, "eval": repo, "episode": ep, "length_s": round(length, 1), "strawberry_start_px": start_xy}
            if row.get("unfinished"):
                item["note"] = "force-quit run: the whole video (several attempts) is shown as one"
            if demo_starts and start_xy:
                d = np.linalg.norm(np.array(demo_starts) - np.array(start_xy), axis=1)
                item["start_offset_from_nearest_demo_px"] = round(float(d.min()), 1)
            if data is not None and not row.get("unfinished"):
                rows_ep = data[data.episode_index == ep]
                track = np.stack(rows_ep["observation.state"].to_numpy())
                item.update(grasp_info(track, fps))
                item["smoothness"] = smoothness(track, np.stack(rows_ep["action"].to_numpy()) if "action" in rows_ep else None, fps)
                attempts_tracks.append((f"#{k} {Path(repo).name} ep{ep}", track))
            else:
                item["joint_data"] = "missing (dataset was not finalized)"
            report["attempts"].append(item)

    # Start-position map on the first attempt's first frame
    if base_img is not None:
        m = base_img.copy()
        for xy in demo_starts:
            cv2.circle(m, xy, 7, (40, 200, 40), 2)
        for a in report["attempts"]:
            if a["strawberry_start_px"]:
                x, y = a["strawberry_start_px"]
                cv2.drawMarker(m, (x, y), (40, 40, 240), cv2.MARKER_TILTED_CROSS, 18, 3)
                label(m, f"#{a['n']}", (x + 10, y - 8))
        label(m, "green = demo starts, red = attempt starts", (8, 22))
        cv2.imwrite(str(out / "start_map.jpg"), m, [cv2.IMWRITE_JPEG_QUALITY, 85])

    demo_grasps = [grasp_info(t, fps) for t in demo_tracks]
    closed = [g for g in demo_grasps if g["gripper_close_s"] is not None]

    def mean(key, items):
        vals = [g[key] for g in items if g[key] is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    report["smoothness_mean"] = mean_smoothness([a.get("smoothness") for a in report["attempts"]])
    report["smoothness_note"] = ("stop_frac: 활성 구간(첫~마지막 움직임)에서 속도 <1u/s 프레임 비율, accel: 평균 |가속도| "
                                 "(상태 2차 차분 x fps², 팔 5축 합, u/s²), cmd_jump: 명령 1차 차분이 중앙값x4+1보다 큰 비율")
    report["demo_summary"] = {
        "length_s_mean": mean("length_s", demo_grasps),
        "gripper_close_s_mean": mean("gripper_close_s", closed),
        "gripper_release_s_mean": mean("gripper_release_s", closed),
        "closed_value_mean": mean("closed_value", closed),
        "grasp_count_mean": mean("grasp_count", demo_grasps),
        "demos_without_close": len(demo_grasps) - len(closed),
        "pose_at_close_mean": [round(float(v), 1) for v in np.mean([g["pose_at_close"] for g in closed], axis=0)] if closed else None,
        "pose_at_close_std": [round(float(v), 1) for v in np.std([g["pose_at_close"] for g in closed], axis=0)] if closed else None,
        "idle_s_mean": mean("idle_s", demo_grasps),
        "grasp_thresholds": {"open": GRIP_OPEN, "closed": GRIP_CLOSED, "empty_below": GRIP_EMPTY},
        "strawberry_starts_px": demo_starts,
    }
    if attempts_tracks:
        cv2.imwrite(str(out / "joints.jpg"), plot_joints(demo_tracks, attempts_tracks, fps), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    os.system(f"chmod -R g+rwX '{out}' 2>/dev/null")
    print(out)


if __name__ == "__main__":
    main()
