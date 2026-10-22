#!/usr/bin/env python3
"""Team index of trained models <-> the recordings they were trained on.

Scans the checkpoints folder (omx_config "checkpoints", e.g. /srv/il/checkpoints) and writes
  <checkpoints>/INDEX.md     one row per training run (numbered, cumulative)
  <checkpoints>/index.json   the same data, plus the run numbers (kept stable)
  <run>/TRAINED_ON.md              per-run sheet, only for runs this account owns

Links are resolved through two files:
  <training dataset>/meta/omx_sources.json   episode -> original recording + episode
  <run>/summary.json                         trainer, start/end time, final loss, note
Runs without them still appear (from train_config.json); missing fields show "-".

  python omx_index.py                                   rebuild the index
  python omx_index.py --sources tkoo/train_x A:0-9 B:1  write omx_sources.json for a merged dataset
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
import pwd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

DATA_HOME = Path(os.environ.get("HF_LEROBOT_HOME", str(omx_config.path("data_home"))))
CKPT_HOME = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints"))))


def sources_path(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "omx_sources.json"


def episode_table(dataset_root: Path) -> list[dict]:
    import pandas as pd

    files = sorted(glob.glob(str(dataset_root / "meta/episodes/*/*.parquet")))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
    return [{"index": int(r["episode_index"]), "length": int(r["length"])} for r in df.to_dict("records")]


def write_sources(dataset_repo: str, parts: list[str]) -> Path:
    """parts like 'tkoo/1013_2203:0-9' in merge order; maps merged episodes to their origin."""
    root = DATA_HOME / dataset_repo
    mapping, merged = [], 0
    for part in parts:
        repo, _, rng = part.partition(":")
        if "-" in rng:
            a, b = (int(x) for x in rng.split("-"))
            eps = list(range(a, b + 1))
        else:
            eps = [int(x) for x in rng.split(",") if x != ""]
        for ep in eps:
            mapping.append({"episode": merged, "source": repo, "source_episode": ep})
            merged += 1
    total = len(episode_table(root))
    if total and total != merged:
        sys.exit(f"[omx] {dataset_repo} has {total} episodes but the sources list {merged}")
    path = sources_path(root)
    path.write_text(json.dumps({"dataset": dataset_repo, "episodes": mapping}, indent=2))
    return path


def fingerprint(dataset_root: Path) -> str:
    """Short hash of the dataset's metadata files, to notice if the data changed after training."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(str(dataset_root / "meta/**/*"), recursive=True)):
        p = Path(f)
        if p.is_file() and p.name != "omx_sources.json":
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def ranges(nums: list[int]) -> str:
    nums = sorted(nums)
    out, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i == len(nums) - 1 or nums[i + 1] != n + 1:
            out.append(f"{start}" if start == n else f"{start}~{n}")
            start = None
    return ", ".join(out)


def find_runs() -> list[Path]:
    """Directories holding a final policy (config.json + model.safetensors + train_config.json)."""
    runs = []
    for cfg in glob.glob(str(CKPT_HOME / "**/train_config.json"), recursive=True):
        model_dir = Path(cfg).parent
        if not (model_dir / "model.safetensors").exists():
            continue
        if "checkpoints" in model_dir.relative_to(CKPT_HOME).parts:  # <run>/checkpoints/NNNNNN = intermediate saves
            continue
        runs.append(model_dir)
    return runs


def run_dir(model_dir: Path) -> Path:
    return model_dir.parent if model_dir.name == "pretrained_model" else model_dir


def load_evals() -> dict:
    """Policy runs marked in the panel's evaluation tab, grouped by model folder."""
    by_model: dict = {}
    for note in glob.glob(str(DATA_HOME / "*/.omx_notes/eval_*.json")):
        try:
            n = json.loads(Path(note).read_text())
        except (OSError, ValueError):
            continue
        if not n.get("model"):
            continue
        res = n.get("results", {})
        marked = [v for v in res.values() if v in ("success", "fail")]
        if not marked:
            continue
        repo = f"{Path(note).parent.parent.name}/{Path(note).stem}"
        by_model.setdefault(str(Path(n["model"]).resolve()), []).append({
            "repo": repo, "date": n.get("started", ""), "success": marked.count("success"), "total": len(marked)})
    return by_model


EVALS: dict = {}


def describe(model_dir: Path) -> dict:
    import pandas as pd

    rdir = run_dir(model_dir)
    cfg = json.loads((model_dir / "train_config.json").read_text())
    summary = {}
    if (rdir / "summary.json").exists():
        summary = json.loads((rdir / "summary.json").read_text())
    d, p = cfg["dataset"], cfg["policy"]
    ds_repo = d["repo_id"]
    ds_root = Path(d["root"]) if d.get("root") else DATA_HOME / ds_repo
    all_eps = episode_table(ds_root) if ds_root.exists() else []
    used = d.get("episodes") or [e["index"] for e in all_eps]
    lengths = {e["index"]: e["length"] for e in all_eps}
    fps = None
    tasks = []
    if (ds_root / "meta/info.json").exists():
        fps = json.loads((ds_root / "meta/info.json").read_text()).get("fps")
        try:
            tasks = list(pd.read_parquet(ds_root / "meta/tasks.parquet").index)
        except Exception:
            pass

    origin, excluded = {}, {}
    src_file = sources_path(ds_root)
    if src_file.exists():
        src = json.loads(src_file.read_text())["episodes"]
        for e in src:
            bucket = origin if e["episode"] in used else excluded
            bucket.setdefault(e["source"], []).append(e["source_episode"])
    elif all_eps:
        # No merge record: the training dataset is itself an original recording
        origin = {ds_repo: list(used)}
        rest = [e["index"] for e in all_eps if e["index"] not in set(used)]
        if rest:
            excluded = {ds_repo: rest}
    stat = (model_dir / "model.safetensors").stat()
    return {
        "id": str(rdir.relative_to(CKPT_HOME)),
        "model_path": str(model_dir),
        "trainer": summary.get("trainer") or pwd.getpwuid(stat.st_uid).pw_name,
        "finished": summary.get("finished") or dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "started": summary.get("started"),
        "duration": summary.get("duration"),
        "final_loss": summary.get("final_loss"),
        "note": summary.get("note", ""),
        "policy": p.get("type"),
        "steps": cfg.get("steps"),
        "batch_size": cfg.get("batch_size"),
        "num_workers": cfg.get("num_workers"),
        "augmentation": bool(d.get("image_transforms", {}).get("enable")),
        "dataset": ds_repo,
        "dataset_episodes": len(used),
        "dataset_minutes": round(sum(lengths.get(i, 0) for i in used) / fps / 60, 1) if fps and lengths else None,
        "dataset_fingerprint": fingerprint(ds_root) if ds_root.exists() else None,
        "fingerprint_at_training": summary.get("dataset_fingerprint"),
        "origin": {k: sorted(v) for k, v in origin.items()},
        "excluded": {k: sorted(v) for k, v in excluded.items()},
        "tasks": tasks,
        "evals": sorted(EVALS.get(str(model_dir.resolve()), []), key=lambda e: e["date"]),
    }


def eval_text(evals: list, sep: str = "<br>") -> str:
    if not evals:
        return "-"
    return sep.join(f"{e['success']}/{e['total']} ({e['date'][5:10]})" for e in evals)


def origin_text(origin: dict, sep: str = "<br>") -> str:
    if not origin:
        return "- (출처 기록 없음)"
    return sep.join(f"{repo} ep{ranges(eps)} ({len(eps)}개)" for repo, eps in origin.items())


def trained_on_md(r: dict, number: int) -> str:
    lines = [
        f"# #{number} {r['id']}",
        "",
        f"- 모델 폴더 (`--policy.path`): `{r['model_path']}`",
        f"- 학습 계정: {r['trainer']} · 시작 {r['started'] or '-'} · 끝 {r['finished']} · 소요 {r['duration'] or '-'}",
        f"- 정책·설정: {r['policy']}, {r['steps']:,}스텝, 배치 {r['batch_size']}, 데이터 로딩 {r.get('num_workers') if r.get('num_workers') is not None else '-'}, 이미지 증강 {'켬' if r['augmentation'] else '끔'}",
        f"- 최종 손실: {r['final_loss'] if r['final_loss'] is not None else '-'}",
        f"- 학습 데이터: `{r['dataset']}` (에피소드 {r['dataset_episodes']}개, {r['dataset_minutes'] or '-'}분)",
        "- 원본 녹화:",
        *[f"  - `{repo}` ep{ranges(eps)} ({len(eps)}개)" for repo, eps in r["origin"].items()],
    ]
    if r["excluded"]:
        lines.append("- 학습 데이터에 있었지만 제외한 에피소드:")
        lines += [f"  - `{repo}` ep{ranges(eps)}" for repo, eps in r["excluded"].items()]
    if r["tasks"]:
        lines.append(f"- 작업: {' / '.join(r['tasks'])}")
    if r["dataset_fingerprint"]:
        changed = r["fingerprint_at_training"] and r["fingerprint_at_training"] != r["dataset_fingerprint"]
        lines.append(f"- 데이터 지문: {r['dataset_fingerprint']}" + (" (학습 이후 데이터가 바뀜)" if changed else ""))
    if r["evals"]:
        lines.append("- 평가 (로봇 실행, 성공/시도):")
        lines += [f"  - {e['date']} `{e['repo']}`: {e['success']}/{e['total']}" for e in r["evals"]]
    if r["note"]:
        lines.append(f"- 비고: {r['note']}")
    return "\n".join(lines) + "\n"


def build_index() -> list[dict]:
    CKPT_HOME.mkdir(parents=True, exist_ok=True)
    lock = open(CKPT_HOME / ".index.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    index_json = CKPT_HOME / "index.json"
    numbers = {}
    if index_json.exists():
        numbers = {r["id"]: r["number"] for r in json.loads(index_json.read_text()).get("runs", [])}
    EVALS.clear()
    EVALS.update(load_evals())
    runs = []
    for model_dir in find_runs():
        try:
            runs.append(describe(model_dir))
        except Exception as exc:
            print(f"[omx] skipped {model_dir}: {exc}", flush=True)
    runs.sort(key=lambda r: (numbers.get(r["id"], 10**6), r["finished"]))
    next_no = max(numbers.values(), default=0) + 1
    for r in runs:
        if r["id"] not in numbers:
            numbers[r["id"]] = next_no
            next_no += 1
        r["number"] = numbers[r["id"]]
    runs.sort(key=lambda r: r["number"])

    me = pwd.getpwuid(os.getuid()).pw_name
    for r in runs:
        rdir = Path(r["model_path"])
        rdir = rdir.parent if rdir.name == "pretrained_model" else rdir
        if pwd.getpwuid(rdir.stat().st_uid).pw_name == me:  # never write into a teammate's folder
            (rdir / "TRAINED_ON.md").write_text(trained_on_md(r, r["number"]))

    md = [
        "# 학습 기록 (자동 생성)",
        "",
        "학습이 끝날 때마다 omx_index.py가 갱신합니다. 직접 고치지 말고, 비고는 각 모델 폴더의 summary.json `note`에 적으세요.",
        "",
        "| # | 끝난 시각 | 계정 | 모델 폴더 (`--policy.path`) | 정책·설정 | 학습 데이터 | 원본 녹화 | 최종 손실 | 소요 | 평가 (성공/시도) | 비고 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        steps = f"{r['steps'] // 1000}K" if r["steps"] >= 1000 else str(r["steps"])
        setting = f"{r['policy']} · {steps}스텝 · 배치{r['batch_size']} · 로딩{r.get('num_workers') if r.get('num_workers') is not None else '-'} · 증강{'O' if r['augmentation'] else 'X'}"
        data = f"`{r['dataset']}`<br>{r['dataset_episodes']}개" + (f", {r['dataset_minutes']}분" if r["dataset_minutes"] else "")
        md.append(
            f"| {r['number']} | {r['finished']} | {r['trainer']} | `{r['model_path']}` | {setting} | {data} | "
            f"{origin_text(r['origin'])} | {r['final_loss'] if r['final_loss'] is not None else '-'} | "
            f"{r['duration'] or '-'} | {eval_text(r['evals'])} | {r['note'] or ''} |"
        )
    (CKPT_HOME / "INDEX.md").write_text("\n".join(md) + "\n")
    index_json.write_text(json.dumps({"updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "runs": runs},
                                     ensure_ascii=False, indent=2))
    for f in ("INDEX.md", "index.json", ".index.lock"):
        try:
            os.chmod(CKPT_HOME / f, 0o664)
        except PermissionError:
            pass
    lock.close()
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", metavar=("DATASET", "SOURCE:EPISODES"),
                    help="write omx_sources.json: DATASET then sources in merge order, e.g. tkoo/1013_2203:0-9")
    args = ap.parse_args()
    if args.sources:
        print("[omx] wrote", write_sources(args.sources[0], args.sources[1:]))
        return
    for r in build_index():
        print(f"#{r['number']} {r['id']} <- {origin_text(r['origin'], sep=' + ')}")


if __name__ == "__main__":
    main()
