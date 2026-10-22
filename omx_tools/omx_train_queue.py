#!/usr/bin/env python3
"""Training plan (sequential queue) for the panel's training tab.

  python omx_train_queue.py            # run the plan: start waiting items one after another

The plan lives in ~/il_ws/omx_logs/train_queue.json, so it survives panel restarts:
  {"paused": false, "runner": {"pid": 123, "status": "..."},
   "items": [{"id": "q...", "job": {...omx_train job...}, "status": "대기", ...}]}

Item status: 대기 -> 실행 중 -> 완료 / 실패 / 중단, or 취소 (removed from the plan before it ran).
The runner is an independent process (own session), like omx_train.py, so it keeps going when the
panel server, the browser or the SSH connection goes away. For each waiting item it
  1. waits while any account trains on the GPU (lerobot-train process or the team train.lock),
  2. runs omx_train.py with the item's job and waits for it,
  3. copies the result (model path, final loss, duration, error) from train_state.json into the item.
A failed item (e.g. out of GPU memory) is marked and the next one starts. A stopped item (학습 중지)
pauses the plan unless the stop asked to continue. The runner exits when nothing is waiting or the
plan is paused.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import pwd
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

HOME = Path.home()
LOG_DIR = HOME / "il_ws/omx_logs"
QUEUE = LOG_DIR / "train_queue.json"
QUEUE_LOCK = LOG_DIR / "train_queue.lock"
RUNNER_LOCK = LOG_DIR / "train_queue.runner.lock"
TRAIN_STATE = LOG_DIR / "train_state.json"
TRAIN_LOCK = omx_config.lock_dir() / "train.lock"
TOOLS = Path(__file__).resolve().parent
VENV_BIN = omx_config.venv_bin()
DATA_HOME = os.environ.get("HF_LEROBOT_HOME") or str(omx_config.path("data_home"))
POLL_S = 20

WAITING, RUNNING, DONE, FAILED, STOPPED, CANCELLED = "대기", "실행 중", "완료", "실패", "중단", "취소"


# ---------------------------------------------------------------- plan file (shared with omx_web.py)
@contextlib.contextmanager
def edit_queue():
    """Read-modify-write the plan under a file lock; yields the dict, saved on exit."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_LOCK, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = read_queue()
        yield q
        tmp = QUEUE.with_suffix(".tmp")
        tmp.write_text(json.dumps(q, ensure_ascii=False, indent=1))
        tmp.replace(QUEUE)


def read_queue() -> dict:
    try:
        q = json.loads(QUEUE.read_text())
    except (OSError, ValueError):
        q = {}
    q.setdefault("paused", True)
    q.setdefault("items", [])
    q.setdefault("runner", {})
    return q


def new_item(job: dict) -> dict:
    return {"id": "q" + uuid.uuid4().hex[:8], "job": job, "status": WAITING,
            "added": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def runner_alive() -> bool:
    """True while a runner holds its lock (the lock dies with the process)."""
    try:
        with open(RUNNER_LOCK, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    except BlockingIOError:
        return True
    except OSError:
        return False


def spawn_runner() -> None:
    """Start the runner as an independent process if none is running."""
    if runner_alive():
        return
    subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS / "omx_train_queue.py")],
                     cwd=HOME, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, env=dict(os.environ, HF_LEROBOT_HOME=DATA_HOME))


# ---------------------------------------------------------------- GPU availability
def gpu_busy() -> str:
    """Who is training right now (any account), or '' when the GPU is free for a new run."""
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        # the program itself (lerobot-train) or the script a python runs (omx_train.py), not any
        # command line that merely mentions the name
        names = {Path(a.decode(errors="replace")).name for a in argv[:2]}
        if {"lerobot-train", "omx_train.py"} & names and b"multiprocessing" not in b" ".join(argv):
            try:
                user = pwd.getpwuid(proc.stat().st_uid).pw_name
            except (KeyError, OSError):
                user = "?"
            return user
    try:
        with open(TRAIN_LOCK, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
    except BlockingIOError:
        return "다른"  # someone holds the team train.lock
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------- runner
child: subprocess.Popen | None = None
stop_flag = False


def set_runner(status: str) -> None:
    with edit_queue() as q:
        q["runner"] = {"pid": os.getpid(), "status": status, "updated": time.time()}


def on_signal(signum, frame) -> None:
    # stopping the runner itself: stop the current training too, keep the plan (paused by main).
    # No file locking here: the signal may arrive while main holds the plan lock.
    global stop_flag
    stop_flag = True
    if child and child.poll() is None:
        os.kill(child.pid, signal.SIGINT)


def memory_check(job: dict) -> str:
    """Refuse to start when the job is expected to need more RAM than is free right now."""
    try:
        import omx_ram
        roots = [Path(DATA_HOME) / d["repo"] for d in job.get("datasets", [])]
        est = omx_ram.estimate(job["policy"], int(job["batch_size"]), int(job.get("num_workers", 8)),
                               int(job["steps"]), roots, Path(omx_config.path("checkpoints")))
    except Exception:
        return ""
    if est["level"] != "bad":
        return ""
    tip = est.get("suggest")
    tip = f" 권장: 배치 {tip['batch_size']}, 데이터 로딩 {tip['num_workers']} (약 {tip['ram_gb']}GB)." if tip else ""
    return (f"메모리 부족 예상: 예상 RAM 약 {est['ram_gb']}GB / 지금 사용 가능 {est['avail_gb']}GB. "
            f"배치나 데이터 로딩 수를 줄여 다시 넣으세요.{tip}")


def failure_reason(log_path: str | None) -> str | None:
    """A short reason from the end of the training log."""
    try:
        tail = Path(log_path).read_text(errors="replace")[-20000:] if log_path else ""
    except OSError:
        return None
    if "killed by signal: Killed" in tail or "Killed" in tail.splitlines()[-1:]:
        return "시스템 RAM 부족으로 데이터 로딩 프로세스가 강제 종료됨. 데이터 로딩 수나 배치를 줄이세요"
    if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
        return "GPU 메모리 부족. 배치를 줄이세요"
    return "학습 프로그램이 오류로 끝남. 로그를 확인하세요"


def run_item(item: dict) -> dict:
    """Run one item with omx_train.py and return the fields to store on it."""
    global child
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_file = LOG_DIR / f"train_job_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    job = dict(item["job"], queue_item=item["id"])  # omx_train keeps it in train_state.json["job"]
    job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    TRAIN_STATE.write_text(json.dumps({"phase": "준비 중", "job": job, "log": [], "losses": []}, ensure_ascii=False))
    started = time.time()
    child = subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS / "omx_train.py"), str(job_file)],
                             cwd=HOME, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             env=dict(os.environ, HF_LEROBOT_HOME=DATA_HOME))
    with edit_queue() as q:
        for it in q["items"]:
            if it["id"] == item["id"]:
                it["pid"] = child.pid
    rc = child.wait()
    child = None
    try:
        st = json.loads(TRAIN_STATE.read_text())
    except (OSError, ValueError):
        st = {}
    phase = st.get("phase", "")
    status = DONE if rc == 0 and phase == "완료" else STOPPED if phase == "중단" else FAILED
    out = {"status": status, "finished": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "run_name": st.get("run_name"), "run_dir": st.get("run_dir"),
           "train_dataset": st.get("train_dataset"), "final_loss": st.get("final_loss"),
           "duration": st.get("duration") or f"{int((time.time() - started) // 60)}분",
           "log_path": st.get("log_path"), "error": st.get("error")}
    if status == FAILED and not out["error"]:
        out["error"] = failure_reason(st.get("log_path"))
    if out["run_dir"] and status == DONE:
        out["model_path"] = str(Path(out["run_dir"]) / "pretrained_model")
    # omx_train refused because another account took the GPU in between: wait and retry
    if status == FAILED and (st.get("error") or "").startswith("다른 학습이 실행 중"):
        out = {"status": WAITING, "error": None}
    return out


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(RUNNER_LOCK, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0  # another runner is already working on the plan
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        while True:
            if stop_flag:
                with edit_queue() as q:
                    q["paused"] = True
                set_runner("일시정지")
                return 0
            q = read_queue()
            nxt = next((it for it in q["items"] if it["status"] == WAITING), None)
            if q["paused"] or nxt is None:
                set_runner("일시정지" if q["paused"] and nxt else "끝")
                return 0
            busy = gpu_busy()
            if busy:
                set_runner(f"GPU 대기 중: {busy} 계정의 학습이 끝나면 시작")
                time.sleep(POLL_S)
                continue
            short = memory_check(nxt["job"])
            if short:
                with edit_queue() as q:
                    for x in q["items"]:
                        if x["id"] == nxt["id"] and x["status"] == WAITING:
                            x.update(status=FAILED, error=short, finished=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
                continue
            with edit_queue() as q:
                it = next((x for x in q["items"] if x["id"] == nxt["id"]), None)
                if it is None or it["status"] != WAITING or q["paused"]:
                    continue
                it.update(status=RUNNING, started=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), error=None)
                q["runner"] = {"pid": os.getpid(), "status": f"실행 중: {it['job']['policy']}", "updated": time.time()}
                item = dict(it)
            result = run_item(item)
            with edit_queue() as q:
                for x in q["items"]:
                    if x["id"] == item["id"]:
                        x.update(result)
                        x.pop("pid", None)
                        # a stop pauses the plan unless the stop asked to continue (set by the panel)
                        if result["status"] == STOPPED and not x.pop("continue_after_stop", False):
                            q["paused"] = True
            if result["status"] == WAITING:
                time.sleep(POLL_S)
    finally:
        with contextlib.suppress(Exception):
            with edit_queue() as q:
                for x in q["items"]:
                    if x["status"] == RUNNING and x.get("pid") and not Path(f"/proc/{x['pid']}").exists():
                        x["status"] = FAILED
                        x["error"] = "계획 실행기가 멈춰 결과를 알 수 없습니다"
                q["runner"]["pid"] = None
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
