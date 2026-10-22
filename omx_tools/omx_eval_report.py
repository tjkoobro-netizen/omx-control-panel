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

import av
import cv2
import numpy as np
import pandas as pd

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
        return None  # e.g. the parquet footer is missing after a force-quit


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
    """Centre of the largest saturated red blob (strawberry, possibly merged with the red tape under it)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 60), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(cents[k][0]), int(cents[k][1])) if stats[k, cv2.CC_STAT_AREA] > 150 else None


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


def grasp_info(track: np.ndarray, fps: int) -> dict:
    """Gripper closing = largest drop of the gripper value; idle = time with almost no joint motion."""
    g = track[:, 5]
    smooth = np.convolve(g, np.ones(5) / 5, mode="same")
    close_i = int(np.argmin(np.diff(smooth))) if len(smooth) > 2 else 0
    speed = np.abs(np.diff(track[:, :5], axis=0)).sum(1) * fps
    return {"gripper_close_s": round(close_i / fps, 1), "pose_at_close": [round(float(v), 1) for v in track[close_i, :5]],
            "gripper_min": round(float(g.min()), 1), "gripper_max": round(float(g.max()), 1),
            "idle_s": round(float((speed < 1.0).sum() / fps), 1), "length_s": round(len(track) / fps, 1)}


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
                track = np.stack(data[data.episode_index == ep]["observation.state"].to_numpy())
                item.update(grasp_info(track, fps))
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
    report["demo_summary"] = {
        "length_s_mean": round(float(np.mean([g["length_s"] for g in demo_grasps])), 1),
        "gripper_close_s_mean": round(float(np.mean([g["gripper_close_s"] for g in demo_grasps])), 1),
        "pose_at_close_mean": [round(float(v), 1) for v in np.mean([g["pose_at_close"] for g in demo_grasps], axis=0)],
        "pose_at_close_std": [round(float(v), 1) for v in np.std([g["pose_at_close"] for g in demo_grasps], axis=0)],
        "idle_s_mean": round(float(np.mean([g["idle_s"] for g in demo_grasps])), 1),
        "strawberry_starts_px": demo_starts,
    }
    if attempts_tracks:
        cv2.imwrite(str(out / "joints.jpg"), plot_joints(demo_tracks, attempts_tracks, fps), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    os.system(f"chmod -R g+rwX '{out}' 2>/dev/null")
    print(out)


if __name__ == "__main__":
    main()
