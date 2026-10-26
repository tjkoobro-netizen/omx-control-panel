#!/usr/bin/env python3
"""Run one training job end to end, with provenance (used by the panel's training tab).

  python omx_train.py job.json

job.json:
  {"policy": "act" | "diffusion" | "smolvla", "steps": 40000, "batch_size": 8, "num_workers": 8, "augmentation": true, "note": "...",
   "datasets": [{"repo": "tkoo/1013_2203", "exclude": [0]}, {"repo": "tkoo/1013_2211", "exclude": []}]}

Steps
  1. training data: one dataset with nothing excluded is used as is; otherwise excluded episodes are
     removed and the rest merged into <user>/train_<keyword>_<MMDD_HHMM> with meta/omx_sources.json
  2. lerobot-train into <checkpoints>/<user>/<policy>_<keyword>_<MMDD_HHMM>
  3. copy the final policy to <run>/pretrained_model, write summary.json and train.log,
     give the team group access, and refresh <checkpoints>/INDEX.md (folders from omx_config)
Progress goes to ~/il_ws/omx_logs/train_state.json, which the panel reads. The job keeps running
when the panel server restarts. SIGINT/SIGTERM stop the training (intermediate saves are kept).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

os.environ.setdefault("HF_LEROBOT_HOME", str(omx_config.path("data_home")))
os.environ["HF_HUB_OFFLINE"] = "1"

HOME = Path.home()
USER = os.environ.get("USER") or Path.home().name
DATA_HOME = Path(os.environ["HF_LEROBOT_HOME"])
CKPT_HOME = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints"))))
VENV_BIN = omx_config.venv_bin()
LEROBOT_DIR = next((p for p in (HOME / "il_ws/src/lerobot", HOME / "dev/lerobot") if p.is_dir()), HOME)
LOG_DIR = HOME / "il_ws/omx_logs"
STATE = LOG_DIR / "train_state.json"
TRAIN_LOCK = omx_config.lock_dir() / "train.lock"
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import omx_index  # noqa: E402

TQDM_RE = re.compile(r"(\d+)/(\d+) \[([0-9:]+)<([0-9:?]+),\s*([0-9.]+)?(?:step/s|s/step)?")
LOSS_RE = re.compile(r"step:(\S+) .*?loss:([0-9.]+)")

state: dict = {}
child: subprocess.Popen | None = None
stop_requested = False


def save_state(**updates) -> None:
    state.update(updates)
    state["updated"] = time.time()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(STATE)


def log(line: str) -> None:
    state.setdefault("log", []).append(line)
    state["log"] = state["log"][-200:]
    with open(state["log_path"], "a") as f:
        f.write(line + "\n")


def on_signal(signum, frame) -> None:
    global stop_requested
    stop_requested = True
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)


def share_with_team(path: Path) -> None:
    omx_config.share_with_team(path)


def run_logged(cmd: list[str], progress: bool = False) -> int:
    """Run a command, stream its output to the log, and track lerobot-train progress."""
    global child
    log("$ " + " ".join(cmd))
    child = subprocess.Popen(cmd, cwd=LEROBOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True,
                             env=dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1"))
    buf = b""
    last_save = 0.0
    while True:
        chunk = child.stdout.read1(4096) if hasattr(child.stdout, "read1") else child.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            # tqdm redraws its bar with \r and the loss line may be glued to it, so check both
            bar = TQDM_RE.search(line) if progress else None
            loss = LOSS_RE.search(line) if progress else None
            if bar:
                state["step"], state["total"] = int(bar.group(1)), int(bar.group(2))
                state["eta"], state["rate"] = bar.group(4), bar.group(5)
            if loss:
                step = loss.group(1)
                num = float(step.rstrip("KM")) * (1000 if step.endswith("K") else 1_000_000 if step.endswith("M") else 1)
                state.setdefault("losses", []).append([int(num), float(loss.group(2))])
                line = line[line.find("INFO"):] if "INFO" in line else line
            if bar and not loss:
                continue  # progress bars are not logged line by line
            log(line)
        if time.time() - last_save > 1:
            save_state()
            last_save = time.time()
    return child.wait()


SMOLVLA_BASE = "lerobot/smolvla_base"


def smolvla_rename_map(dataset_root: Path) -> dict:
    """Our camera names -> smolvla_base's inputs (camera1..camera3): front first, then wrist.

    The pretrained model was trained on community low-cost arm data with up to three cameras named
    camera1..3; a camera it does not get is simply left out (no padding image needed).
    """
    feats = json.loads((dataset_root / "meta/info.json").read_text())["features"]
    cams = [k for k, f in feats.items() if k.startswith("observation.images.") and f.get("dtype") in ("video", "image")]
    order = {"front": 0, "wrist": 1}
    cams.sort(key=lambda k: (order.get(k.rsplit(".", 1)[-1], 9), k))
    return {k: f"observation.images.camera{i + 1}" for i, k in enumerate(cams[:3])}


def find_reusable(datasets: list[dict]) -> str | None:
    """An earlier merged training dataset of this account with exactly the same source episodes."""
    want = []
    for d in datasets:
        n = len(omx_index.episode_table(DATA_HOME / d["repo"]))
        excluded = set(d.get("exclude", []))
        want += [[d["repo"], e] for e in range(n) if e not in excluded]
    if not want:
        return None
    for src in sorted((DATA_HOME / USER).glob("train_*/meta/omx_sources.json"), reverse=True):
        try:
            mapping = json.loads(src.read_text()).get("episodes", [])
        except (OSError, ValueError):
            continue
        root = src.parent.parent
        if ([[m["source"], m["source_episode"]] for m in mapping] == want
                and (root / "meta/info.json").exists() and len(omx_index.episode_table(root)) == len(want)):
            return f"{USER}/{root.name}"
    return None


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = job.get("stamp") or dt.datetime.now().strftime("%m%d_%H%M")
    started = dt.datetime.now()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    state.update({"pid": os.getpid(), "phase": "준비 중", "started": started.strftime("%Y-%m-%d %H:%M"),
                  "job": job, "log": [], "losses": [], "step": 0, "total": job["steps"],
                  "log_path": str(LOG_DIR / f"train_{stamp}.log")})
    save_state()

    lock = open(TRAIN_LOCK, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        save_state(phase="실패", error=f"다른 학습이 실행 중입니다: {lock.read().strip() or '알 수 없음'}")
        return 1
    lock.seek(0)
    lock.truncate()
    lock.write(f"{USER} {started:%F %T}\n")
    lock.flush()

    try:
        datasets = job["datasets"]
        # Keyword for the names: from the first dataset's task sentence and recording note
        from omx_upload import derive_keywords  # noqa: E402
        import pandas as pd

        first_root = DATA_HOME / datasets[0]["repo"]
        tasks = list(pd.read_parquet(first_root / "meta/tasks.parquet").index)
        note_file = DATA_HOME / datasets[0]["repo"].split("/")[0] / ".omx_notes" / f"{datasets[0]['repo'].split('/')[1]}.json"
        rec_note = json.loads(note_file.read_text()).get("description", "") if note_file.exists() else ""
        keyword = (derive_keywords(tasks, rec_note) or ["task"])[0]
        # the name from the training tab (team rule, see omx_names.py), else the older automatic name
        run_name = job.get("name") or f"{job['policy']}_{keyword}_{stamp}"
        run_dir = CKPT_HOME / USER / run_name
        if run_dir.exists():
            raise RuntimeError(f"{run_dir} 가 이미 있습니다. " + ("이름을 바꿔 다시 넣으세요." if job.get("name") else "1분 뒤 다시 시작하세요."))
        save_state(run_name=run_name, run_dir=str(run_dir))

        # 1) training data. Excluded episodes are removed into a new dataset rather than skipped with
        #    --dataset.episodes: in LeRobot v0.4.4 that option breaks multi-frame policies (diffusion
        #    indexes past the end of the subset).
        edit = str(VENV_BIN / "lerobot-edit-dataset")
        has_eval = any(d.get("eval") for d in datasets)
        reuse = None if len(datasets) == 1 and not datasets[0].get("exclude") else find_reusable(datasets)
        if has_eval:
            reuse = None  # the candidate counts are recorded per training dataset
        if len(datasets) == 1 and not datasets[0].get("exclude") and not has_eval:
            train_repo = datasets[0]["repo"]  # an original recording, used as is
        elif reuse:
            train_repo = reuse  # same recordings and episodes were merged before: use that data again
            log(f"[omx] 같은 조합의 학습 데이터를 다시 씁니다: {train_repo}")
        else:
            train_repo = f"{USER}/train_{keyword}_{stamp}"
            save_state(phase="학습 데이터 만드는 중")
            parts, pieces, temps = [], [], []
            for i, d in enumerate(datasets):
                n = len(omx_index.episode_table(DATA_HOME / d["repo"]))
                excluded = set(d.get("exclude", []))
                keep = [e for e in range(n) if e not in excluded]
                if not keep:
                    continue
                parts.append(f"{d['repo']}:" + ",".join(map(str, keep)))
                if excluded:
                    target = train_repo if len(datasets) == 1 else f"{USER}/_tmp_{stamp}_{i}"
                    rc = run_logged([edit, f"--repo_id={d['repo']}", f"--new_repo_id={target}",
                                     "--operation.type=delete_episodes",
                                     "--operation.episode_indices=" + json.dumps(sorted(excluded))])
                    if rc != 0 or stop_requested:
                        raise RuntimeError(f"{d['repo']}에서 에피소드를 빼지 못했습니다. 로그를 확인하세요.")
                    if target != train_repo:
                        temps.append(target)
                    pieces.append(target)
                else:
                    pieces.append(d["repo"])
            if not parts:
                raise RuntimeError("학습에 쓸 에피소드가 없습니다.")
            if len(pieces) > 1 or pieces[0] != train_repo:
                rc = run_logged([edit, f"--repo_id={train_repo}", "--operation.type=merge",
                                 "--operation.repo_ids=" + json.dumps(pieces)])
                if rc != 0 or stop_requested:
                    raise RuntimeError("데이터셋 합치기에 실패했습니다. 로그를 확인하세요 (카메라·fps 구성이 다른 데이터는 합칠 수 없습니다).")
            for t in temps:
                shutil.rmtree(DATA_HOME / t, ignore_errors=True)
            omx_index.write_sources(train_repo, parts)
            if has_eval:
                # which evaluation attempts went in, by kind, to compare models trained with / without them
                src = omx_index.sources_path(DATA_HOME / train_repo)
                meta = json.loads(src.read_text())
                evs = [dict(d["eval"], repo=d["repo"]) for d in datasets if d.get("eval")]
                meta["eval_sources"] = evs
                meta["eval_totals"] = {k: sum(e[k] for e in evs) for k in ("one_shot", "recovery", "wandering_by_user")}
                src.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            share_with_team(DATA_HOME / train_repo)
        train_root = DATA_HOME / train_repo
        used = [e["index"] for e in omx_index.episode_table(train_root)]
        if not used:
            raise RuntimeError("학습에 쓸 에피소드가 없습니다.")
        fingerprint = omx_index.fingerprint(train_root)
        save_state(train_dataset=train_repo, episodes=len(used))

        # 2) lerobot-train
        (CKPT_HOME / USER).mkdir(parents=True, exist_ok=True)
        # SmolVLA is fine-tuned from the community-pretrained lerobot/smolvla_base (downloaded once,
        # training runs offline); ACT and Diffusion start from scratch
        policy_arg = f"--policy.path={SMOLVLA_BASE}" if job["policy"] == "smolvla" else f"--policy.type={job['policy']}"
        cmd = [str(VENV_BIN / "lerobot-train"), f"--dataset.repo_id={train_repo}",
               policy_arg, "--policy.device=cuda", "--policy.push_to_hub=false",
               f"--output_dir={run_dir}", f"--job_name={run_name}",
               f"--steps={int(job['steps'])}", f"--batch_size={int(job['batch_size'])}", f"--num_workers={int(job.get('num_workers', 8))}",
               "--log_freq=100", f"--save_freq={max(int(job['steps']) // 4, 1000)}", "--eval_freq=0",
               "--wandb.enable=false"]
        if job.get("augmentation"):
            cmd.append("--dataset.image_transforms.enable=true")
        if job["policy"] == "diffusion":
            cmd += ["--policy.resize_shape=[240, 320]", "--policy.crop_ratio=0.95"]
        if job["policy"] == "smolvla":
            cmd.append("--rename_map=" + json.dumps(smolvla_rename_map(train_root)))
        save_state(phase="학습 중", train_started=time.time())
        rc = run_logged(cmd, progress=True)

        finished = dt.datetime.now()
        minutes = int((finished - started).total_seconds() // 60)
        duration = f"{minutes // 60}시간 {minutes % 60}분" if minutes >= 60 else f"{minutes}분"
        final_loss = state["losses"][-1][1] if state.get("losses") else None
        status = "완료" if rc == 0 and not stop_requested else ("중단" if stop_requested else "실패")
        summary = {"trainer": USER, "started": state["started"], "finished": finished.strftime("%Y-%m-%d %H:%M"),
                   "duration": duration, "final_loss": final_loss, "dataset_fingerprint": fingerprint,
                   "train_dataset": train_repo, "episodes": used, "status": status,
                   "note": (job.get("note", "") + (f" · 평가 시도 {sum(len(d['eval']['episodes']) for d in datasets if d.get('eval'))}개 포함"
                                                   if any(d.get("eval") for d in datasets) else "")).strip(" ·"),
                   "policy": job["policy"], "steps": int(job["steps"]), "batch_size": int(job["batch_size"]),
                   "num_workers": int(job.get("num_workers", 8)), "augmentation": bool(job.get("augmentation"))}

        # 3) results next to the model, team access, index
        if run_dir.exists():
            last = run_dir / "checkpoints/last/pretrained_model"
            if status == "완료" and last.exists():
                shutil.copytree(last, run_dir / "pretrained_model", symlinks=False)
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            shutil.copy(state["log_path"], run_dir / "train.log")
            share_with_team(run_dir)
        omx_index.build_index()
        save_state(phase=status, finished=summary["finished"], duration=duration, final_loss=final_loss)
        return 0 if status == "완료" else 1
    except Exception as exc:
        log(f"[omx] error: {exc}")
        save_state(phase="중단" if stop_requested else "실패", error=str(exc))
        return 1
    finally:
        lock.seek(0)
        lock.truncate()
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
