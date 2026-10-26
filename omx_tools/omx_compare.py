#!/usr/bin/env python3
"""Model comparison for evaluation protocol v2, and migration of earlier (manual) evaluations.

  python omx_compare.py compare <preset_id> [--save]
      position x evaluation table (result, failure stage, start offset), per evaluation: settings,
      control rate, success rate without void attempts, failures per stage; per position: success
      over all evaluations. --save writes .omx_reports/compare_<preset>_<MMDD_HHMM>/compare.{json,csv}

  python omx_compare.py migrate <eval> --preset <preset_id> [--order 1,2,...] [--episodes 0,1,...] [--apply]
      give an evaluation that was run by hand in preset order the v2 record: episode k -> the k-th
      position of --order (default: preset order), start position detected on the first front frame,
      results from the existing success/fail marks, failure stages estimated. Without --apply it only
      prints what it would write (dry run); with --apply the old notes file is kept as
      <name>.json.bak.<MMDD_HHMM> before it is rewritten.

Stages estimated automatically are marked "estimated" (shown as "(추정)") until the user confirms them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402
import omx_names as names  # noqa: E402
import omx_protocol as proto  # noqa: E402

DATA_HOME = Path(os.environ.get("HF_LEROBOT_HOME", str(omx_config.path("data_home"))))


def _notes(data_home: Path):
    for f in sorted(data_home.glob("*/.omx_notes/eval_*.json")):
        try:
            yield f, json.loads(f.read_text())
        except (OSError, ValueError):
            continue


_SMOOTH_CACHE: dict = {}


def _smooth(root: Path) -> dict | None:
    """Mean smoothness of an evaluation (stop_frac, accel, cmd_jump), cached until its data changes."""
    import omx_eval_report as R

    try:
        key = (str(root), max((f.stat().st_mtime for f in (root / "data").rglob("*.parquet")), default=0))
    except OSError:
        return None
    if key not in _SMOOTH_CACHE:
        try:
            r = R.eval_smoothness(root)
            _SMOOTH_CACHE[key] = (r or {}).get("mean")
        except Exception:
            _SMOOTH_CACHE[key] = None
    return _SMOOTH_CACHE[key]


def compare(preset_id: str, data_home: Path = DATA_HOME) -> dict:
    preset = proto.load_preset(preset_id)
    positions = proto.eval_positions(preset)
    columns, cells, excluded = [], {}, []
    aliases = names.load_aliases()
    steps = names.run_steps()
    for f, n in _notes(data_home):
        p = n.get("protocol") or {}
        if p.get("preset_id") != preset_id:
            continue
        repo = f"{f.parent.parent.name}/{f.stem}"
        if n.get("compare_exclude"):  # kept on record, left out of the comparison
            excluded.append({"eval": repo, "started": n.get("started"), "model": n.get("run_id"),
                             "model_display": names.display_name(n.get("run_id"), aliases) if n.get("run_id") else "",
                             "attempts": len(n.get("attempts", {}))})
            continue
        attempts = n.get("attempts", {})
        loops = [a["loop"] for a in attempts.values() if a.get("loop")]
        col = {
            "eval": repo, "started": n.get("started"), "model": n.get("run_id"), "model_path": n.get("model"),
            "model_display": names.model_label(names.display_name(n.get("run_id"), aliases), names.train_amount(n, steps),
                                               n.get("final_steps") or steps.get(n.get("run_id"))) if n.get("run_id") else "",
            "train_step": names.train_amount(n, steps),
            "policy_type": n.get("policy_type"), "async": n.get("async_mode"), "rtc": n.get("rtc"),
            "smooth": n.get("smooth"), "diffusion_steps": n.get("diffusion_steps"), "threads": n.get("threads"),
            "actions_per_chunk": n.get("actions_per_chunk"), "migrated": bool(p.get("migrated")),
            "chunk_size_threshold": n.get("chunk_size_threshold"), "output_filter": n.get("output_filter"),
            "loop_median_hz": round(statistics.median(x["median_hz"] for x in loops), 1) if loops else None,
            "loop_min_hz": round(min(x["min_hz"] for x in loops), 1) if loops else None,
        }
        stages = {s: 0 for s in proto.FAIL_STAGES}
        ok = fail = void = aborted = 0
        offsets = []
        for pos in positions:
            # the last attempt at this position that was not void decides the cell
            tries = sorted(((int(e), a) for e, a in attempts.items() if a.get("position_id") == pos["id"]), key=lambda x: x[0])
            voids = sum(1 for _, a in tries if a.get("result") == "void")
            # an aborted attempt (즉시 중단 / 비상 정지) is counted apart, like void it does not decide the cell
            cut = sum(1 for _, a in tries if a.get("result") == "aborted")
            real = [a for _, a in tries if a.get("result") not in ("void", "aborted")]
            a = real[-1] if real else None
            cell = {"result": a.get("result") if a else None, "voids": voids, "aborted": cut}
            if a:
                cell["episode"] = next(e for e, x in tries if x is a)
                cell["start_offset_px"] = a.get("start_offset_px")
                if a.get("start_offset_px") is not None:
                    offsets.append(a["start_offset_px"])
                if a.get("result") == "fail":
                    cell["stage"] = a.get("fail_stage") or a.get("fail_stage_auto")
                    cell["estimated"] = a.get("fail_stage_source") != "user"
                    if cell["stage"] in stages:
                        stages[cell["stage"]] += 1
            ok += cell["result"] == "success"
            fail += cell["result"] == "fail"
            void += voids
            aborted += cut
            cells.setdefault(pos["id"], {})[repo] = cell
        col.update(success=ok, fail=fail, void=void, aborted=aborted, unmarked=len(positions) - ok - fail,
                   success_rate=round(ok / (ok + fail), 2) if ok + fail else None,
                   fail_stages=stages, mean_start_offset_px=round(statistics.mean(offsets), 1) if offsets else None)
        col.update(_smooth(data_home / repo) or {})
        columns.append(col)
    columns.sort(key=lambda c: c.get("started") or "")
    rows = []
    for pos in positions:
        row_cells = cells.get(pos["id"], {})
        marked = [c for c in row_cells.values() if c["result"] in ("success", "fail")]
        rows.append({"position": pos, "cells": row_cells,
                     "success": sum(c["result"] == "success" for c in marked), "marked": len(marked)})
    return {"preset": preset, "columns": columns, "rows": rows, "stages": proto.STAGE_KO, "excluded": excluded}


def save(result: dict, data_home: Path = DATA_HOME) -> Path:
    owner = os.environ.get("USER") or Path.home().name
    out = data_home / owner / ".omx_reports" / f"compare_{result['preset']['id']}_{dt.datetime.now():%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with open(out / "compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = result["columns"]
        w.writerow(["position", "x", "y"] + [c["eval"] for c in cols])
        w.writerow(["model", "", ""] + [c.get("model_display") or c.get("model") for c in cols])
        for r in result["rows"]:
            vals = []
            for c in cols:
                cell = r["cells"].get(c["eval"], {})
                v = {"success": "O", "fail": "X"}.get(cell.get("result"), "")
                if cell.get("stage"):
                    v += f" {result['stages'].get(cell['stage'], cell['stage'])}{'(추정)' if cell.get('estimated') else ''}"
                vals.append(v)
            w.writerow([r["position"]["id"], r["position"]["x"], r["position"]["y"]] + vals)
        w.writerow([])
        for key in ("model", "model_display", "train_step", "policy_type", "async", "rtc", "smooth", "diffusion_steps", "threads", "loop_median_hz",
                    "loop_min_hz", "success", "fail", "void", "aborted", "success_rate", "stop_frac", "accel", "cmd_jump",
                    "output_filter", "model_path"):
            w.writerow([key, "", ""] + [c.get(key) for c in cols])
    os.system(f"chmod -R g+rwX '{out}' 2>/dev/null")
    return out


def migrate(name: str, preset_id: str, order: list[int] | None, episodes: list[int] | None,
            data_home: Path = DATA_HOME, apply: bool = False) -> dict:
    import omx_eval_report as R
    import omx_vision

    repo = name if "/" in name else f"{os.environ.get('USER') or Path.home().name}/{name}"
    path = data_home / repo.split("/")[0] / ".omx_notes" / f"{repo.split('/')[1]}.json"
    note = json.loads(path.read_text())
    preset = proto.load_preset(preset_id)
    pos_by_id = {p["id"]: p for p in proto.eval_positions(preset)}
    order = order or list(pos_by_id)
    root = data_home / repo
    meta = R.episodes_meta(root)
    rows = meta.to_dict("records") if len(meta) else R.unfinished_rows(root)
    eps = episodes if episodes is not None else [int(r["episode_index"]) for r in rows]
    if len(eps) > len(order):
        raise SystemExit(f"에피소드 {len(eps)}개인데 위치 순서는 {len(order)}개입니다 (--order 또는 --episodes 확인)")
    note["protocol"] = proto.protocol_snapshot(preset) | {"migrated": True}
    note["attempts"] = {}
    row_by_ep = {int(r["episode_index"]): r for r in rows}
    for ep, pid in zip(eps, order):
        pos = pos_by_id[pid]
        start = None
        if preset["detector"] != "none" and ep in row_by_ep:
            try:
                frame = R.episode_frames(root, row_by_ep[ep], preset.get("camera", "front"), [0.0])
                start = omx_vision.detect(preset["detector"], frame[0] if frame else None, preset.get("detect_roi"))
            except Exception:
                start = None
        result = note.get("results", {}).get(str(ep))
        note["attempts"][str(ep)] = {
            "position_id": pid, "target_px": [pos["x"], pos["y"]],
            "start_px": list(start) if start else None,
            "start_offset_px": round(math.dist(start, (pos["x"], pos["y"])), 1) if start else None,
            "result": result, "fail_stage": None, "fail_stage_auto": None, "fail_stage_source": None,
            "note": "", "train_candidate": None}
    proto.fill_auto(note, data_home, repo)
    if apply:
        shutil.copy(path, path.parent / f"{path.name}.bak.{dt.datetime.now():%m%d_%H%M}")
        with proto.LOCK:
            path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
    return note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("preset")
    c.add_argument("--save", action="store_true")
    m = sub.add_parser("migrate")
    m.add_argument("eval")
    m.add_argument("--preset", required=True)
    m.add_argument("--order", help="position ids in the order they were used, e.g. 1,2,3")
    m.add_argument("--episodes", help="episode indices to map, e.g. 0,1,2 (default: all)")
    m.add_argument("--apply", action="store_true", help="write the notes (default: dry run, print only)")
    a = ap.parse_args()
    if a.cmd == "compare":
        r = compare(a.preset)
        if a.save:
            print(save(r))
        for col in r["columns"]:
            print(f"{col['eval']}: {col['success']}/{col['success'] + col['fail']} 성공 (무효 {col['void']}), "
                  f"{col['policy_type']} async={col['async']} smooth={col['smooth']} ddim={col['diffusion_steps']} "
                  f"루프 {col['loop_median_hz']}/{col['loop_min_hz']}Hz, 실패 단계 {col['fail_stages']}")
    else:
        parse = lambda s: [int(x) for x in s.split(",")] if s else None  # noqa: E731
        note = migrate(a.eval, a.preset, parse(a.order), parse(a.episodes), apply=a.apply)
        print("적용함 (원본은 .bak.<시각>으로 보관)" if a.apply else "미리보기만 했습니다 (--apply로 실제 기록)")
        for ep, att in note["attempts"].items():
            print(f"  에피소드 {ep}: 위치 {att['position_id']} 목표 {att['target_px']} 시작 {att['start_px']} "
                  f"(거리 {att['start_offset_px']}) 결과 {att['result']} 단계 {att.get('fail_stage')} 추정 {att.get('fail_stage_auto')} "
                  f"추천 {att.get('train_rec')}/{att.get('train_rec_priority')} ({att.get('train_rec_reason')})")


if __name__ == "__main__":
    main()
