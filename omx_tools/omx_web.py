#!/usr/bin/env python3
"""OMX-AI control panel: check, camera preview, teleop and recording from a browser.

Runs on the GPU laptop and listens on 127.0.0.1 only. Laptops reach it through
an SSH tunnel (see omx-control.sh), so the panel is never exposed on the Wi-Fi.
Uses only the LeRobot v0.4.4 environment and the Python stdlib. Machine-specific settings
(ports, cameras, shared folders) come from omx_config.py / ~/.config/omx/config.json.
"""

from __future__ import annotations

import collections
import datetime as dt
import fcntl
import getpass
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

import omx_config  # noqa: E402  (per-machine settings written by install_gpu.sh)
import omx_train_queue as tq  # noqa: E402  (training plan file and runner)
import omx_ram  # noqa: E402  (RAM and time estimates)

CFG = omx_config.CFG
VERSION = omx_config.VERSION
# One panel server per account, each on its own port (port_base + uid - uid_base).
# omx-control.sh and omx-panel-server.sh ask omx_config for the same number.
HOST, PORT = "127.0.0.1", omx_config.port()
HOME = Path.home()
USER = getpass.getuser()
VENV_BIN = omx_config.venv_bin()
# Working directory for LeRobot commands: the lecture layout, or the team guide's ~/dev/lerobot
LEROBOT_DIR = next((p for p in (HOME / "il_ws/src/lerobot", HOME / "dev/lerobot") if p.is_dir()), HOME)
TOOLS_DIR = Path(__file__).resolve().parent
LOG_DIR = HOME / "il_ws/omx_logs"
# Team shared LeRobot home: calibration and datasets live here (child processes inherit it)
os.environ["HF_LEROBOT_HOME"] = str(omx_config.path("data_home"))
DATA_HOME = Path(os.environ["HF_LEROBOT_HOME"])
LOCK_PATH = omx_config.lock_dir() / "robot.lock"
OWNER_PATH = omx_config.lock_dir() / "robot.owner"

FOLLOWER = CFG["follower_port"]
LEADER = CFG["leader_port"]
FOLLOWER_IDS = list(CFG["follower_ids"])
LEADER_IDS = list(CFG["leader_ids"])
BAUDRATE = int(CFG["baudrate"])
CAMERAS = dict(CFG["cameras"])
CAMS_ARG = omx_config.cameras_arg(warmup_s=3)
ROBOT_ARGS = ["--robot.type=omx_follower", f"--robot.port={FOLLOWER}", "--robot.id=omx_follower_arm"]
TELEOP_ARGS = ["--teleop.type=omx_leader", f"--teleop.port={LEADER}", "--teleop.id=omx_leader_arm"]
VIEWER_PORT = 9876

SLOW_RE = re.compile(r"Record loop is running slower \(([0-9.]+) Hz\)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TABLE_RE = re.compile(r"^(\S+\.pos\s*\||-{5,}$|NAME\s*\|)")


def fix_wrist_fps() -> None:
    """Keep the configured camera (usually the wrist one) at full fps in dim light (resets on replug)."""
    name = CFG.get("fixed_fps_camera") or ""
    if name in CAMERAS:
        subprocess.run(["v4l2-ctl", "-d", CAMERAS[name], "-c", "exposure_dynamic_framerate=0"], capture_output=True)


def other_robot_processes() -> list[str]:
    out = subprocess.run(["ps", "-eo", "user=,pid=,args="], capture_output=True, text=True).stdout
    mine = {str(JOB.pid)} if JOB.pid else set()
    found = []
    for line in out.splitlines():
        user, pid, args = (line.split(None, 2) + ["", ""])[:3]
        if pid in mine or "omx_web.py" in args:
            continue
        if re.search(r"lerobot-(teleoperate|record)|omx_(record|teleop)\.py", args):
            found.append(f"{user} (PID {pid})")
    return found


def ping_motors() -> dict:
    from dynamixel_sdk import PacketHandler, PortHandler

    result = {}
    for name, port, ids in (("follower", FOLLOWER, FOLLOWER_IDS), ("leader", LEADER, LEADER_IDS)):
        if not Path(port).exists():
            result[name] = {"ok": False, "msg": f"{port} 없음 (USB 연결 확인)"}
            continue
        handler, packet = PortHandler(port), PacketHandler(2.0)
        if not handler.openPort():
            result[name] = {"ok": False, "msg": "포트를 열 수 없음 (다른 프로그램이 사용 중일 수 있음)"}
            continue
        handler.setBaudRate(BAUDRATE)
        missing, hw_errors = [], []
        for motor_id in ids:
            _, comm, err = packet.ping(handler, motor_id)
            if comm != 0:
                missing.append(motor_id)
            elif err & 0x80:
                hw_errors.append(motor_id)
        handler.closePort()
        if missing or hw_errors:
            parts = []
            if missing:
                parts.append(f"응답 없음 ID {missing}")
            if hw_errors:
                parts.append(f"하드웨어 에러 ID {hw_errors} (전원 껐다 켜기)")
            result[name] = {"ok": False, "msg": ", ".join(parts)}
        else:
            result[name] = {"ok": True, "msg": f"모터 {len(ids)}개 정상"}
    return result


def reset_motors() -> dict:
    """Reboot every motor on both arms: clears hardware errors (e.g. overload) and turns torque off."""
    from dynamixel_sdk import PacketHandler, PortHandler

    result = {}
    for name, port, ids in (("follower", FOLLOWER, FOLLOWER_IDS), ("leader", LEADER, LEADER_IDS)):
        if not Path(port).exists():
            result[name] = {"ok": False, "msg": f"{port} 없음 (USB 연결 확인)"}
            continue
        handler, packet = PortHandler(port), PacketHandler(2.0)
        if not handler.openPort():
            result[name] = {"ok": False, "msg": "포트를 열 수 없음 (강제 종료를 먼저 누르세요)"}
            continue
        handler.setBaudRate(BAUDRATE)
        for motor_id in ids:
            packet.reboot(handler, motor_id)
        time.sleep(1.0)
        missing, hw_errors = [], []
        for motor_id in ids:
            value, comm, _ = packet.read1ByteTxRx(handler, motor_id, 70)  # Hardware_Error_Status
            if comm != 0:
                missing.append(motor_id)
            elif value:
                hw_errors.append(motor_id)
        handler.closePort()
        if missing or hw_errors:
            parts = []
            if missing:
                parts.append(f"응답 없음 ID {missing}")
            if hw_errors:
                parts.append(f"에러 남음 ID {hw_errors} (팔로워 전원을 껐다 켜세요)")
            result[name] = {"ok": False, "msg": ", ".join(parts)}
        else:
            result[name] = {"ok": True, "msg": f"모터 {len(ids)}개 초기화됨 · 토크 꺼짐"}
    return result


def kill_own_robot_processes() -> list[str]:
    """SIGKILL this user's leftover teleop/record processes (others' cannot be touched)."""
    out = subprocess.run(["ps", "-u", USER, "-o", "pid=,pgid=,args="], capture_output=True, text=True).stdout
    killed = []
    for line in out.splitlines():
        pid, pgid, args = (line.split(None, 2) + ["", ""])[:3]
        if "omx_web.py" in args or not re.search(r"lerobot-(teleoperate|record)|omx_(record|teleop)\.py", args):
            continue
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            os.kill(int(pid), signal.SIGKILL)
        killed.append(pid)
    return killed


def gpu_status() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        name, used, total, util = [s.strip() for s in out.stdout.strip().split(",")]
        return f"{name} · 메모리 {used}/{total} MiB · 사용률 {util}%"
    except Exception:
        return "GPU 정보를 읽을 수 없음 (nvidia-smi 확인)"


def list_datasets() -> list[dict]:
    """kind: recording (original), train (built for training, has omx_sources.json), eval (policy runs)."""
    items = []
    if not DATA_HOME.exists():
        return items
    for info in sorted(DATA_HOME.glob("*/*/meta/info.json")):
        repo = info.parent.parent
        try:
            meta = json.loads(info.read_text())
        except Exception:
            continue
        kind = "eval" if repo.name.startswith("eval_") else "train" if (repo / "meta/omx_sources.json").exists() else "recording"
        items.append({
            "repo_id": f"{repo.parent.name}/{repo.name}",
            "episodes": meta.get("total_episodes", 0),
            "frames": meta.get("total_frames", 0),
            "minutes": round(meta.get("total_frames", 0) / (meta.get("fps") or 30) / 60, 1),
            "kind": kind,
        })
    return items


# ---- Training (runs omx_train.py as an independent process; state in ~/il_ws/omx_logs) ----
TRAIN_STATE = LOG_DIR / "train_state.json"
CKPT_HOME = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints"))))


def _pid_runs(pid: int, needle: str) -> bool:
    try:
        return needle in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False


def gpu_users() -> dict:
    """GPU load and who is training right now (any account)."""
    out = {"util": None, "mem_used": None, "mem_total": None, "trainings": []}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        out["util"], out["mem_used"], out["mem_total"] = [int(x) for x in q.strip().split(",")]
    except Exception:
        pass
    ps = subprocess.run(["ps", "-eo", "user=,pid=,etime=,args="], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        user, pid, etime, args = (line.split(None, 3) + ["", "", ""])[:4]
        if "lerobot-train" in args and "omx_train.py" not in args and "multiprocessing" not in args:
            m = re.search(r"--output_dir=(\S+)", args)
            out["trainings"].append({"user": user, "pid": int(pid), "elapsed": etime,
                                     "output": m.group(1) if m else ""})
    return out


def train_status() -> dict:
    state = {}
    try:
        state = json.loads(TRAIN_STATE.read_text())
    except (OSError, ValueError):
        pass
    running = bool(state.get("pid")) and _pid_runs(state["pid"], "omx_train.py")
    if state and not running and state.get("phase") in ("준비 중", "학습 데이터 만드는 중", "학습 중"):
        state["phase"] = "중단됨 (프로세스 없음)"
    return {"state": state, "running": running, "gpu": gpu_users()}


DEFAULT_WORKERS = {"act": 8, "diffusion": 4}


def train_estimate(body: dict) -> dict:
    """RAM / time estimate for the settings on the training tab (also used before starting)."""
    policy = body.get("policy") if body.get("policy") in ("act", "diffusion") else "act"
    roots, frames = [], 0
    for d in body.get("datasets", []):
        try:
            root = dataset_dir(d["repo"])
        except Exception:
            continue
        roots.append(root)
        try:
            info = json.loads((root / "meta/info.json").read_text())
            n_ep = max(info.get("total_episodes", 1), 1)
            frames += info.get("total_frames", 0) * (n_ep - len(d.get("exclude", []))) / n_ep
        except (OSError, ValueError):
            pass
    return omx_ram.estimate(policy, int(body.get("batch_size") or 8), int(body.get("num_workers", DEFAULT_WORKERS[policy])),
                            int(body.get("steps") or 0), roots, CKPT_HOME, int(frames))


def make_job(body: dict) -> dict:
    """Validate the training settings from the panel into an omx_train.py job."""
    policy = body.get("policy")
    if policy not in ("act", "diffusion"):
        raise RuntimeError("정책은 ACT 또는 Diffusion만 고를 수 있습니다.")
    steps, batch = int(body.get("steps", 0)), int(body.get("batch_size", 0))
    workers = int(body.get("num_workers", DEFAULT_WORKERS[policy]))
    if not 100 <= steps <= 500_000 or not 1 <= batch <= 64:
        raise RuntimeError("스텝은 100~500,000, 배치는 1~64 사이로 정하세요.")
    if not 0 <= workers <= 16:
        raise RuntimeError("데이터 로딩 수는 0~16 사이로 정하세요.")
    datasets = []
    for d in body.get("datasets", []):
        dataset_dir(d["repo"])
        datasets.append({"repo": d["repo"], "exclude": sorted({int(e) for e in d.get("exclude", [])})})
    if not datasets:
        raise RuntimeError("학습에 쓸 데이터셋을 하나 이상 고르세요.")
    return {"policy": policy, "steps": steps, "batch_size": batch, "num_workers": workers,
            "augmentation": bool(body.get("augmentation")),
            "note": str(body.get("note", "")).strip(), "datasets": datasets}


def train_start(body: dict) -> dict:
    status = train_status()
    if status["running"]:
        raise RuntimeError("이미 학습이 실행 중입니다.")
    if status["gpu"]["trainings"]:
        t = status["gpu"]["trainings"][0]
        raise RuntimeError(f"GPU에서 다른 학습이 실행 중입니다: {t['user']} ({t['elapsed']}째). 끝난 뒤 시작하세요.")
    job = make_job(body)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_file = LOG_DIR / f"train_job_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    TRAIN_STATE.write_text(json.dumps({"phase": "준비 중", "job": job, "log": [], "losses": []}, ensure_ascii=False))
    subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_train.py"), str(job_file)],
                     cwd=HOME, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, env=dict(os.environ, HF_LEROBOT_HOME=str(DATA_HOME)))
    return {"ok": True}


def train_stop(body: dict) -> None:
    state = train_status()
    if not state["running"]:
        raise RuntimeError("실행 중인 학습이 없습니다.")
    item_id = (state["state"].get("job") or {}).get("queue_item")
    if item_id:
        # a plan item: by default the plan pauses after the stop; "continue" starts the next item
        with tq.edit_queue() as q:
            for it in q["items"]:
                if it["id"] == item_id:
                    it["continue_after_stop"] = bool(body.get("continue"))
    os.kill(state["state"]["pid"], signal.SIGINT)


# ---- Training plan (sequential queue, run by omx_train_queue.py) ----
def queue_status() -> dict:
    q = tq.read_queue()
    alive = tq.runner_alive()
    if not alive and q["runner"].get("status", "").startswith(("GPU 대기", "실행 중")):
        q["runner"]["status"] = "멈춤"
    return {"paused": q["paused"], "items": q["items"], "runner": q["runner"], "runner_alive": alive}


def queue_action(path: str, body: dict) -> dict:
    action = path.rsplit("/", 1)[-1]
    with tq.edit_queue() as q:
        items = q["items"]
        find = lambda: next((it for it in items if it["id"] == body.get("id")), None)  # noqa: E731
        if action == "add":
            items.append(tq.new_item(make_job(body)))
        elif action == "cancel":
            it = find()
            if not it or it["status"] != tq.WAITING:
                raise RuntimeError("대기 중인 항목만 취소할 수 있습니다. 실행 중이면 학습 중지를 누르세요.")
            it["status"] = tq.CANCELLED
        elif action == "retry":
            it = find()
            if not it or it["status"] not in (tq.FAILED, tq.STOPPED, tq.CANCELLED):
                raise RuntimeError("실패·중단·취소된 항목만 다시 넣을 수 있습니다.")
            items.append(tq.new_item(it["job"]))
        elif action == "move":
            it = find()
            if not it or it["status"] != tq.WAITING:
                raise RuntimeError("대기 중인 항목만 순서를 바꿀 수 있습니다.")
            waiting = [i for i, x in enumerate(items) if x["status"] == tq.WAITING]
            pos = waiting.index(items.index(it)) + (-1 if body.get("dir", -1) < 0 else 1)
            if 0 <= pos < len(waiting):
                a, b = items.index(it), waiting[pos]
                items[a], items[b] = items[b], items[a]
        elif action == "clear":
            q["items"] = [x for x in items if x["status"] in (tq.WAITING, tq.RUNNING)]
        elif action == "run":
            if not any(x["status"] == tq.WAITING for x in items):
                raise RuntimeError("계획에 대기 중인 항목이 없습니다.")
            q["paused"] = False
        elif action == "pause":
            q["paused"] = True
        else:
            raise RuntimeError("알 수 없는 요청입니다.")
    if action in ("run", "add") and not tq.read_queue()["paused"]:
        tq.spawn_runner()  # also picks up items added while the plan runs
    return queue_status()


def train_runs(refresh: bool) -> dict:
    if refresh:
        subprocess.run([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_index.py")], capture_output=True,
                       env=dict(os.environ, HF_LEROBOT_HOME=str(DATA_HOME)), timeout=120)
    try:
        index = json.loads((CKPT_HOME / "index.json").read_text())
    except (OSError, ValueError):
        index = {"runs": []}
    for r in index.get("runs", []):
        model = Path(r["model_path"])
        rdir = model.parent if model.name == "pretrained_model" else model
        r["own"] = rdir.exists() and rdir.owner() == USER
        r["intermediate"] = (rdir / "checkpoints").is_dir()
    return index


def train_clean(run_id: str) -> None:
    """Delete a run's intermediate checkpoints (keeps pretrained_model, summary and log)."""
    rdir = (CKPT_HOME / run_id).resolve()
    if CKPT_HOME.resolve() not in rdir.parents or not (rdir / "pretrained_model/model.safetensors").exists():
        raise RuntimeError("최종 모델(pretrained_model)이 있는 학습만 정리할 수 있습니다.")
    if rdir.owner() != USER:
        raise RuntimeError("다른 계정의 학습 결과는 정리할 수 없습니다.")
    shutil.rmtree(rdir / "checkpoints", ignore_errors=True)


def dataset_dir(repo_id: str) -> Path:
    """Resolve a dataset folder inside DATA_HOME, refusing anything outside it."""
    path = (DATA_HOME / repo_id).resolve()
    if DATA_HOME.resolve() not in path.parents or not (path / "meta/info.json").exists():
        raise RuntimeError("데이터셋을 찾을 수 없습니다.")
    return path


def share_with_team(path: Path) -> None:
    """Make a dataset readable by the team group, so other accounts can train on it.

    Videos are written to a temp file (mode 600) and moved into place, so they skip the shared
    folder's default ACL and end up with `group::---`. With an ACL present, `chmod g+rX` only
    raises the mask, so the group entries are set with setfacl as well.
    """
    omx_config.share_with_team(path)


def notes_path(repo_id: str) -> Path:
    """Keywords/description for the Hub upload, kept beside (not inside) the dataset folder."""
    owner, name = repo_id.split("/", 1)
    return DATA_HOME / owner / ".omx_notes" / f"{name}.json"


def read_notes(repo_id: str) -> dict:
    try:
        return json.loads(notes_path(repo_id).read_text())
    except (OSError, ValueError):
        return {}


def write_notes(repo_id: str, keywords: str, description: str) -> None:
    path = notes_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"keywords": keywords, "description": description}, ensure_ascii=False, indent=2))


def dataset_detail(repo_id: str) -> dict:
    """Episodes of one dataset with the video file and time range of each camera."""
    import pandas as pd

    root = dataset_dir(repo_id)
    info = json.loads((root / "meta/info.json").read_text())
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    notes = read_notes(repo_id)
    if not files:
        return {"repo_id": repo_id, "fps": info.get("fps"), "episodes": [], "notes": notes}
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
    episodes = []
    for row in df.to_dict("records"):
        videos = {}
        for key in video_keys:
            chunk = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            rel = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file_idx)
            videos[key.rsplit(".", 1)[-1]] = {
                "url": f"/media/{repo_id}/{rel}",
                "from": float(row[f"videos/{key}/from_timestamp"]),
                "to": float(row[f"videos/{key}/to_timestamp"]),
            }
        tasks = row.get("tasks")
        episodes.append({
            "index": int(row["episode_index"]),
            "length": int(row["length"]),
            "seconds": round(int(row["length"]) / info["fps"], 1),
            "task": str(tasks[0]) if tasks is not None and len(tasks) else "",
            "videos": videos,
        })
    return {"repo_id": repo_id, "fps": info.get("fps"), "episodes": episodes, "notes": notes}


class Preview:
    """Owns the two cameras while the browser preview is on."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cams: dict[str, OpenCVCamera] = {}
        self.frames: dict[str, bytes] = {}
        self.brightness: dict[str, float] = {}
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            fix_wrist_fps()
            cams = {}
            for name, path in CAMERAS.items():
                cam = OpenCVCamera(OpenCVCameraConfig(
                    index_or_path=path, width=640, height=480, fps=30, fourcc="MJPG"))
                try:
                    cam.connect()
                except Exception as exc:
                    for opened in cams.values():
                        opened.disconnect()
                    raise RuntimeError(f"{name} 카메라를 열 수 없습니다. 다른 프로그램(텔레옵·녹화)이 카메라를 쓰고 있거나 USB가 빠졌는지 확인하세요.")
                cams[name] = cam
            self.cams, self.running = cams, True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self) -> None:
        while self.running:
            for name, cam in list(self.cams.items()):
                try:
                    frame = cam.async_read(timeout_ms=500)
                except Exception:
                    continue
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.frames[name] = jpg.tobytes()
                    self.brightness[name] = round(float(frame.mean()), 1)
            time.sleep(1 / 15)

    def stop(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.running = False
            if self.thread:
                self.thread.join(timeout=3)
            for cam in self.cams.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            self.cams, self.frames, self.brightness = {}, {}, {}


class Job:
    """One robot process at a time (teleop or record), guarded by the shared robot lock."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.kind = ""
        self.started = 0.0
        self.lines: collections.deque[str] = collections.deque(maxlen=400)
        self.info: dict = {}
        self.lock_file = None
        self.log_file = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _acquire_lock(self, label: str) -> None:
        try:
            self.lock_file = open(LOCK_PATH, "w")
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = OWNER_PATH.read_text().strip() if OWNER_PATH.exists() else "알 수 없음"
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError(f"로봇 사용 중: {owner}")
        except OSError:
            self.lock_file = None  # shared dir missing: fall back to the process check only
            return
        try:
            OWNER_PATH.write_text(f"{USER} | {dt.datetime.now():%F %T} | {label}\n")
        except OSError:
            pass

    def _release_lock(self) -> None:
        if self.lock_file:
            try:
                OWNER_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            self.lock_file.close()
            self.lock_file = None

    def start(self, kind: str, cmd: list[str], info: dict, need_leader: bool = True,
              env_extra: dict | None = None) -> None:
        if self.running():
            raise RuntimeError("이미 실행 중인 작업이 있습니다. 먼저 중지하세요.")
        others = other_robot_processes()
        if others:
            raise RuntimeError(f"다른 사람이 로봇을 쓰고 있습니다: {', '.join(others)}")
        for port in ((FOLLOWER, LEADER) if need_leader else (FOLLOWER,)):
            if not Path(port).exists():
                raise RuntimeError(f"{port} 가 없습니다. USB 연결을 확인하세요.")
        self._acquire_lock(kind)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = open(LOG_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{kind}.log", "w")
        self.log_file.write("$ " + " ".join(cmd) + "\n")
        self.lines.clear()
        self.kind, self.info, self.started = kind, info, time.time()
        self.slow = collections.deque(maxlen=2000)  # (time, Hz) of "Record loop is running slower" warnings
        self.slow_total = 0
        env = dict(os.environ, PYTHONUNBUFFERED="1", **(env_extra or {}))
        self.proc = subprocess.Popen(
            cmd, cwd=LEROBOT_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        proc = self.proc
        for raw in proc.stdout:
            line = ANSI_RE.sub("", raw).rstrip()
            if not line:
                continue
            if line.startswith("Teleop loop time"):
                self.info["loop"] = line.split(":", 1)[1].strip()
                continue
            if TABLE_RE.match(line):
                continue
            if m := SLOW_RE.search(line):
                # lerobot-record warns on every slow loop iteration: count them instead of flooding the log view
                self.slow.append((time.time(), float(m.group(1))))
                self.slow_total += 1
                if self.log_file:
                    self.log_file.write(line + "\n")
                continue
            self._parse(line)
            self.lines.append(line)
            if self.log_file:
                self.log_file.write(line + "\n")
                self.log_file.flush()
        proc.wait()
        self.info["phase"] = "끝남"
        self.info["returncode"] = proc.returncode
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self._release_lock()
        if self.kind == "eval" and self.info.get("repo_id"):
            # keep the control-rate summary with the eval record
            try:
                path = notes_path(self.info["repo_id"])
                note = json.loads(path.read_text())
                hz = sorted(h for _, h in self.slow)
                note["loop"] = {"slow_warnings": self.slow_total,
                                "slow_median_hz": hz[len(hz) // 2] if hz else None,
                                "slow_min_hz": hz[0] if hz else None}
                path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
            except (OSError, ValueError):
                pass
        if self.kind in ("record", "eval") and self.info.get("repo_id"):
            # Also covers force-quit runs, where omx_record.py had no chance to do it itself
            share_with_team(DATA_HOME / self.info["repo_id"])
            if CFG.get("team_group"):
                self.lines.append(f"[omx] dataset shared with group {CFG['team_group']} (read/write)")
        # Auto-upload only after a clean recording; a failed or force-quit run is left local
        if self.kind == "record" and self.info.get("upload") and proc.returncode == 0:
            try:
                UPLOADER.start(self.info["repo_id"])
            except Exception as exc:
                self.lines.append(f"[omx] upload not started: {exc}")

    def _parse(self, line: str) -> None:
        if m := re.search(r"Recording episode (\d+)", line):
            self.info["phase"] = "실행 중" if self.kind == "eval" else "녹화 중"
            self.info["episode"] = int(m.group(1))
            self.info["phase_started"] = time.time()
        elif "Reset the environment" in line or "[omx] reset period" in line:
            self.info["phase"] = "원위치 시간"
            self.info["phase_started"] = time.time()
        elif "Re-record episode" in line:
            self.info["phase"] = "다시 찍기"
        elif "Stop recording" in line:
            self.info["phase"] = "저장 중"
        elif "[omx] safe stop:" in line:
            self.info["phase"] = "천천히 내려놓는 중"
            self.info["safe_start"] = line.split("safe stop:", 1)[1].strip()
        elif "[omx] safe start:" in line:
            msg = line.split("safe start:", 1)[1].strip()
            done = msg.startswith("완료") or msg.startswith("시간 초과")
            self.info["safe_start"] = msg
            if self.kind == "teleop":
                self.info["phase"] = "조작 중" if done else "안전 시작 중"
            elif not done:
                self.info["phase"] = "안전 시작 중"
        elif "connected." in line and self.kind == "teleop" and self.info.get("phase") != "안전 시작 중":
            self.info["phase"] = "조작 중"

    def loop_warn(self) -> dict | None:
        """Control rate below target for a while (last 10 s), while the robot is being driven."""
        if not self.running() or self.info.get("phase") not in ("실행 중", "녹화 중", "원위치 시간"):
            return None
        if time.time() - self.info.get("phase_started", 0) < 3:
            return None  # the first frames of a phase are often slow (episode start)
        now = time.time()
        recent = sorted(h for t, h in self.slow if now - t < 10)
        if len(recent) < 15:  # about 5% of 300 iterations
            return None
        return {"count": len(recent), "hz": round(recent[len(recent) // 2], 1), "min_hz": round(recent[0], 1),
                "target": 30}

    def send(self, cmd: str) -> None:
        if not self.running() or self.kind not in ("record", "eval"):
            raise RuntimeError("녹화 중이 아닙니다.")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def stop(self) -> None:
        if not self.running():
            return
        sig = signal.SIGTERM if self.info.get("stopping") else signal.SIGINT
        self.info["stopping"] = True
        os.killpg(self.proc.pid, sig)

    def kill(self) -> list[str]:
        """Force-quit: SIGKILL the job's whole process group plus any leftover robot processes."""
        killed = []
        if self.running():
            self.info["phase"] = "강제 종료"
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                killed.append(str(self.proc.pid))
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        killed += kill_own_robot_processes()
        time.sleep(0.5)
        return sorted(set(killed))

    def status(self) -> dict:
        if not self.proc:
            return {"running": False}
        info = {k: v for k, v in self.info.items() if k != "phase_started"}
        if "phase_started" in self.info:
            info["phase_elapsed"] = round(time.time() - self.info["phase_started"])
        info["loop_warn"] = self.loop_warn()
        return {
            "running": self.running(),
            "kind": self.kind,
            "elapsed": round(time.time() - self.started),
            "info": info,
        }


class Uploader:
    """Uploads one dataset at a time to the Hugging Face Hub (does not use the robot)."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.state: dict = {}
        self.hf_user: str | None = None
        self.hf_checked = 0.0

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def hf_login(self) -> str | None:
        """Logged-in Hugging Face user name, re-checked at most once a minute."""
        if time.time() - self.hf_checked > 60:
            self.hf_checked = time.time()
            try:
                from huggingface_hub import HfApi
                self.hf_user = HfApi().whoami()["name"]
            except Exception:
                self.hf_user = None
        return self.hf_user

    def start(self, repo_id: str, keywords: str | None = None, description: str | None = None) -> None:
        if self.running():
            raise RuntimeError(f"이미 업로드 중입니다: {self.state.get('repo_id')}")
        dataset_dir(repo_id)
        if keywords is not None or description is not None:
            saved = read_notes(repo_id)
            write_notes(repo_id, keywords if keywords is not None else saved.get("keywords", ""),
                        description if description is not None else saved.get("description", ""))
        self.hf_checked = 0.0
        if not self.hf_login():
            raise RuntimeError(f"Hugging Face에 로그인되어 있지 않습니다. 개인 PC 터미널에서 한 번만: ssh -t omx-gpu '{VENV_BIN}/hf auth login' (Write 토큰)")
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        env["PYTHONUNBUFFERED"] = "1"
        self.state = {"repo_id": repo_id, "phase": "업로드 중", "started": time.time()}
        self.proc = subprocess.Popen(
            [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_upload.py"), repo_id],
            cwd=HOME, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        proc = self.proc
        log = open(LOG_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_upload.log", "w")
        for raw in proc.stdout:
            line = ANSI_RE.sub("", raw).rstrip()
            if not line:
                continue
            log.write(line + "\n")
            log.flush()
            if line.startswith("[omx]") or "Error" in line:
                JOB.lines.append(line)
            if m := re.search(r"\[omx\] uploaded (\S+)", line):
                self.state["url"] = m.group(1)
            elif "[omx] upload failed" in line or "[omx] upload skipped" in line:
                self.state["error"] = line.split("] ", 1)[-1]
        proc.wait()
        log.close()
        self.state["phase"] = "업로드 완료" if proc.returncode == 0 else "업로드 실패"
        self.state["returncode"] = proc.returncode

    def status(self) -> dict:
        state = {k: v for k, v in self.state.items() if k != "started"}
        if "started" in self.state:
            state["elapsed"] = round(time.time() - self.state["started"])
        state["running"] = self.running()
        return state


PREVIEW = Preview()
JOB = Job()
UPLOADER = Uploader()


def display_args(viewer_ip: str) -> list[str]:
    if not viewer_ip:
        return []
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", viewer_ip):
        raise RuntimeError("Rerun 주소는 100.x.x.x 형식의 IP여야 합니다.")
    return ["--display_data=true", f"--display_ip={viewer_ip}", f"--display_port={VIEWER_PORT}"]


def start_teleop(body: dict) -> dict:
    # omx_teleop.py = lerobot-teleoperate + safe start (lift, close gripper, slow approach at low power)
    cmd = [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_teleop.py"), *ROBOT_ARGS, *TELEOP_ARGS]
    viewer_ip = body.get("viewer_ip", "").strip()
    if viewer_ip:
        PREVIEW.stop()  # cameras go to the teleop process for Rerun
        fix_wrist_fps()
        cmd.insert(5, f"--robot.cameras={CAMS_ARG}")
        cmd += display_args(viewer_ip)
    JOB.start("teleop", cmd, {"phase": "연결 중", "with_cameras": bool(viewer_ip)})
    return {"ok": True}


def start_record(body: dict) -> dict:
    repo_id = body.get("repo_id", "").strip()
    if not repo_id:
        # The panel pre-fills the name as MMDD_HHMM; the user may edit it
        name = re.sub(r"[^a-z0-9_]+", "_", body.get("name", "").lower()).strip("_")
        if not name:
            name = f"{dt.datetime.now():%m%d_%H%M}"
        repo_id = f"{USER}/{name}"
        if (DATA_HOME / repo_id).exists() and not (DATA_HOME / repo_id / "meta/info.json").exists():
            raise RuntimeError(f"{repo_id} 폴더가 이미 있지만 데이터셋이 아닙니다. 다른 이름을 쓰세요.")
    task = body.get("task", "").strip()
    if not task:
        raise RuntimeError("작업 설명을 입력하세요. 같은 데이터셋에서는 같은 문장을 씁니다.")
    episodes = int(body.get("episodes", 10))
    episode_s = float(body.get("episode_s", 30))
    reset_s = float(body.get("reset_s", 15))
    resume = (DATA_HOME / repo_id / "meta/info.json").exists()
    keywords = body.get("keywords", "").strip()
    description = body.get("description", "").strip()
    if keywords or description or not resume:
        write_notes(repo_id, keywords, description)  # used later for the Hub name, tags and page text
    PREVIEW.stop()  # record needs the cameras itself
    fix_wrist_fps()
    cmd = [
        str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_record.py"),
        *ROBOT_ARGS, f"--robot.cameras={CAMS_ARG}", *TELEOP_ARGS,
        f"--dataset.repo_id={repo_id}", f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}", f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}", "--dataset.push_to_hub=false",
        *display_args(body.get("viewer_ip", "").strip()),
    ]
    if resume:
        cmd.append("--resume=true")
    JOB.start("record", cmd, {
        "phase": "연결 중", "repo_id": repo_id, "target": episodes, "resume": resume,
        "episode_s": episode_s, "reset_s": reset_s, "upload": bool(body.get("upload")),
    })
    return {"ok": True, "repo_id": repo_id, "resume": resume}


def find_model(model_path: str) -> dict:
    """A trained policy from the team index (or any policy folder under CKPT_HOME)."""
    path = Path(model_path).resolve()
    if CKPT_HOME.resolve() not in path.parents or not (path / "model.safetensors").exists() or not (path / "config.json").exists():
        raise RuntimeError("학습 기록에 있는 모델을 고르세요.")
    for r in train_runs(False).get("runs", []):
        if Path(r["model_path"]).resolve() == path:
            return r
    return {"id": path.name, "model_path": str(path), "tasks": []}


# CPU threads for policy runs. PyTorch defaults to one thread per core (24 here); its thread pool then
# splits every small CPU op of the pre/post-processing 24 ways and spin-waits, using ~2200% CPU and
# dropping the 30 Hz control loop to ~21 Hz. 1-8 threads all held 30 Hz in the benchmark (2026-10-22).
EVAL_THREADS = 4


def thread_env(threads: int) -> dict:
    n = str(threads)
    return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "OMX_TORCH_THREADS": n}


def start_eval(body: dict) -> dict:
    run = find_model(body.get("model_path", ""))
    task = str(body.get("task") or (run.get("tasks") or [""])[0]).strip()
    if not task:
        raise RuntimeError("작업 문장을 입력하세요. 학습 데이터와 같은 문장을 씁니다.")
    episodes = int(body.get("episodes", 5))
    episode_s = float(body.get("episode_s", 30))
    reset_s = float(body.get("reset_s", 15))
    if not 1 <= episodes <= 100 or not 3 <= episode_s <= 600 or not 0 <= reset_s <= 300:
        raise RuntimeError("에피소드 1~100개, 시도 3~600초, 원위치 0~300초 사이로 정하세요.")
    threads = int(body.get("threads") or EVAL_THREADS)
    if not 1 <= threads <= (os.cpu_count() or 24):
        raise RuntimeError(f"CPU 스레드 수는 1~{os.cpu_count() or 24} 사이로 정하세요.")
    slug = re.sub(r"[^a-z0-9_]+", "_", run["id"].replace("/", "_").lower()).strip("_")[:60]
    repo_id = f"{USER}/eval_{slug}_{dt.datetime.now():%m%d_%H%M}"
    path = notes_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "eval", "model": run["model_path"], "run_id": run["id"],
                                "run_number": run.get("number"), "task": task, "results": {},
                                "smooth": bool(body.get("smooth")), "threads": threads,
                                "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2))
    PREVIEW.stop()  # the policy run needs the cameras itself
    fix_wrist_fps()
    cmd = [
        str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_record.py"),
        *ROBOT_ARGS, f"--robot.cameras={CAMS_ARG}",
        f"--policy.path={run['model_path']}",
        f"--dataset.repo_id={repo_id}", f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}", f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}", "--dataset.push_to_hub=false",
        *display_args(body.get("viewer_ip", "").strip()),
    ]
    if body.get("smooth"):
        # ACT temporal ensembling: predict every step and average overlapping chunks (no jumps every 100 steps)
        cmd += ["--policy.n_action_steps=1", "--policy.temporal_ensemble_coeff=0.01"]
    JOB.start("eval", cmd, {"phase": "모델 불러오는 중", "repo_id": repo_id, "target": episodes,
                            "episode_s": episode_s, "reset_s": reset_s, "model": run["id"],
                            "smooth": bool(body.get("smooth")), "threads": threads}, need_leader=False,
              env_extra=thread_env(threads))
    return {"ok": True, "repo_id": repo_id}


def eval_list() -> list[dict]:
    """Policy runs (eval_ datasets) of every account, newest first, with marked results."""
    items = []
    for note in DATA_HOME.glob("*/.omx_notes/eval_*.json"):
        try:
            n = json.loads(note.read_text())
        except (OSError, ValueError):
            continue
        repo = f"{note.parent.parent.name}/{note.stem}"
        info = DATA_HOME / repo / "meta/info.json"
        total = json.loads(info.read_text()).get("total_episodes", 0) if info.exists() else 0
        res = n.get("results", {})
        items.append({"repo_id": repo, "model": n.get("model"), "run_id": n.get("run_id"), "run_number": n.get("run_number"),
                      "threads": n.get("threads"), "smooth": n.get("smooth"), "loop": n.get("loop"),
                      "task": n.get("task"), "started": n.get("started"), "episodes": total, "results": res,
                      "success": sum(1 for v in res.values() if v == "success"),
                      "marked": sum(1 for v in res.values() if v in ("success", "fail"))})
    return sorted(items, key=lambda x: x.get("started") or "", reverse=True)


def eval_mark(body: dict) -> None:
    repo = body.get("repo_id", "")
    path = notes_path(repo)
    if not path.exists():
        raise RuntimeError("평가 기록을 찾을 수 없습니다.")
    n = json.loads(path.read_text())
    ep, result = str(int(body.get("episode"))), body.get("result")
    if result in ("success", "fail"):
        n.setdefault("results", {})[ep] = result
    else:
        n.setdefault("results", {}).pop(ep, None)
    path.write_text(json.dumps(n, ensure_ascii=False, indent=2))
    # refresh the team table in the background so the "평가" column follows the marks
    subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_index.py")], cwd=HOME,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                     env=dict(os.environ, HF_LEROBOT_HOME=str(DATA_HOME)))


REPORTS: dict = {}  # repo -> Popen of omx_eval_report.py


def report_dir(repo: str) -> Path:
    owner, name = repo.split("/", 1)
    return DATA_HOME / owner / ".omx_reports" / name


def start_report(repo: str) -> None:
    """Build the evaluation report (frames, joint plots vs the model's demos, start map) in the background."""
    dataset_dir(repo)
    if repo in REPORTS and REPORTS[repo].poll() is None:
        raise RuntimeError("이미 분석 자료를 만드는 중입니다.")
    note = read_notes(repo)
    model = Path(note.get("model", ""))
    cfg_file = model / "train_config.json"
    if not cfg_file.exists():
        raise RuntimeError("이 평가의 모델 정보를 찾을 수 없습니다 (학습 설정 파일 없음).")
    d = json.loads(cfg_file.read_text())["dataset"]
    cmd = [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_eval_report.py"), repo, "--demos", d["repo_id"]]
    if d.get("episodes"):
        cmd += ["--episodes", ",".join(str(e) for e in d["episodes"])]
    out = report_dir(repo)
    shutil.rmtree(out, ignore_errors=True)
    REPORTS[repo] = subprocess.Popen(cmd, cwd=HOME, stdout=subprocess.DEVNULL, stderr=open(LOG_DIR / "report.log", "w"),
                                     env=dict(os.environ, HF_LEROBOT_HOME=str(DATA_HOME), HF_HUB_OFFLINE="1"))


def report_status(repo: str) -> dict:
    out = report_dir(repo)
    proc = REPORTS.get(repo)
    running = proc is not None and proc.poll() is None
    files = sorted(p.name for p in out.glob("*.jpg")) if out.exists() else []
    failed = proc is not None and not running and proc.returncode not in (0, None)
    return {"running": running, "ready": (out / "report.json").exists() and not running, "failed": failed,
            "dir": str(out), "files": files}


def reference_frame(repo: str, cam: str = "front") -> bytes:
    """First frame of the first episode of a dataset, as JPEG (for the camera alignment overlay)."""
    import av

    root = dataset_dir(repo)
    info = json.loads((root / "meta/info.json").read_text())
    import pandas as pd
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    row = pd.read_parquet(files[0]).sort_values("episode_index").iloc[0]
    key = f"observation.images.{cam}"
    path = root / info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                            file_index=int(row[f"videos/{key}/file_index"]))
    t0 = float(row[f"videos/{key}/from_timestamp"])
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            if fr.time is not None and fr.time >= t0 - 1e-3:
                ok, jpg = cv2.imencode(".jpg", fr.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jpg.tobytes()
    raise RuntimeError("기준 장면을 읽지 못했습니다.")


def alignment(repo: str) -> dict:
    """How far the live front view is from the reference: shift (px), scale, rotation, match quality."""
    import numpy as np

    live = PREVIEW.frames.get("front")
    if not live:
        raise RuntimeError("카메라를 먼저 켜세요.")
    a = cv2.imdecode(np.frombuffer(reference_frame(repo), np.uint8), cv2.IMREAD_GRAYSCALE)
    b = cv2.imdecode(np.frombuffer(live, np.uint8), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return {"ok": False, "msg": "특징점을 찾지 못했습니다. 조명을 확인하세요."}
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda m: m.distance)[:300]
    if len(matches) < 12:
        return {"ok": False, "msg": "기준 장면과 겹치는 부분이 거의 없습니다. 카메라를 기준 사진처럼 크게 옮겨보세요."}
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=4.0)
    if M is None:
        return {"ok": False, "msg": "위치를 계산하지 못했습니다."}
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    h, w = a.shape
    cx, cy = M @ np.array([w / 2, h / 2, 1.0])
    dx, dy = float(cx - w / 2), float(cy - h / 2)  # where the reference centre appears in the live view
    ratio = float(inliers.sum()) / len(matches)
    good = abs(dx) < 15 and abs(dy) < 15 and abs(scale - 1) < 0.04 and abs(angle) < 2 and ratio > 0.3
    return {"ok": True, "dx": round(dx), "dy": round(dy), "scale": round(scale, 3), "angle": round(angle, 1),
            "match": round(ratio, 2), "aligned": good}


def start_pose() -> dict:
    import omx_safe_start
    return {"pose": omx_safe_start.load_pose("start"), "rest": omx_safe_start.load_pose("rest"),
            "settings": omx_safe_start.settings()}


def save_start_pose(kind: str = "start") -> dict:
    """Read the leader arm (read-only) and store it as the follower start or rest pose for everyone."""
    if kind not in ("start", "rest"):
        raise RuntimeError("알 수 없는 자세 종류입니다.")
    if JOB.running() or other_robot_processes():
        raise RuntimeError("텔레옵·녹화·평가가 실행 중입니다. 끝난 뒤 저장하세요 (리더 포트를 읽어야 합니다).")
    r = subprocess.run([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_safe_start.py"), "save", kind],
                       capture_output=True, text=True, timeout=60, cwd=HOME)
    if r.returncode != 0:
        raise RuntimeError("리더 자세를 읽지 못했습니다. 리더 USB 연결을 확인하세요. " + (r.stderr.strip().splitlines() or [""])[-1][:160])
    return start_pose()


def run_check() -> dict:
    result = {
        "ports": {p: Path(p).exists() for p in (FOLLOWER, LEADER)},
        "cameras": {n: Path(p).exists() for n, p in CAMERAS.items()},
        "gpu": gpu_status(),
        "others": other_robot_processes(),
    }
    if JOB.running() or result["others"]:
        result["motors"] = {"note": "로봇 작업 실행 중이라 모터 점검을 건너뜀"}
    else:
        result["motors"] = ping_motors()
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            body = (TOOLS_DIR / "omx_panel.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json({
                "user": USER,
                "version": VERSION,
                "gpu_name": CFG.get("gpu_name"),
                "ckpt_home": str(CKPT_HOME),
                "preview": PREVIEW.running,
                "live": JOB.running() and JOB.kind in ("record", "eval"),
                "brightness": PREVIEW.brightness,
                "job": JOB.status(),
                "log": list(JOB.lines)[-120:],
                "datasets": list_datasets(),
                "data_home": str(DATA_HOME),
                "hf_user": UPLOADER.hf_login(),
                "upload": UPLOADER.status(),
            })
        elif self.path.startswith("/stream/"):
            self._stream(self.path.split("?", 1)[0].rsplit("/", 1)[-1])
        elif self.path.startswith("/api/reference?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body = reference_frame(q.get("repo", [""])[0], q.get("cam", ["front"])[0])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=300")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/align?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                self._json(alignment(q.get("repo", [""])[0]))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/eval/report?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            self._json(report_status(q.get("repo", [""])[0]))
        elif self.path.startswith("/report/"):
            rel = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path[len("/report/"):])
            path = (DATA_HOME / rel).resolve()
            if DATA_HOME.resolve() not in path.parents or ".omx_reports" not in path.parts or path.suffix != ".jpg" or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/start_pose":
            self._json(start_pose())
        elif self.path == "/api/eval/list":
            self._json({"evals": eval_list()})
        elif self.path == "/api/train/status":
            self._json(train_status())
        elif self.path == "/api/train/queue":
            self._json(queue_status())
        elif self.path.startswith("/api/train/runs"):
            self._json(train_runs("refresh=1" in self.path))
        elif self.path.startswith("/api/dataset?"):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                self._json(dataset_detail(query.get("repo", [""])[0]))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/media/"):
            self._media(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path[len("/media/"):]))
        else:
            self._json({"error": "not found"}, 404)

    def _media(self, rel: str) -> None:
        """Serve a dataset mp4 with HTTP Range support so the browser can seek."""
        path = (DATA_HOME / rel).resolve()
        if DATA_HOME.resolve() not in path.parents or path.suffix != ".mp4" or not path.is_file():
            self._json({"error": "not found"}, 404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self, name: str) -> None:
        if name not in CAMERAS:
            self._json({"error": "unknown camera"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        live = Path(f"/dev/shm/omx_live_{os.getuid()}_{name}.jpg")
        # keep the connection through short gaps (preview stops just before a record/eval job starts)
        idle_since = None
        try:
            while True:
                if PREVIEW.running or JOB.running():
                    idle_since = None
                elif idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > 5:
                    break
                frame = PREVIEW.frames.get(name) if PREVIEW.running else None
                if frame is None and JOB.running() and live.exists() and time.time() - live.stat().st_mtime < 2:
                    frame = live.read_bytes()
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path in ("/api/teleop", "/api/record", "/api/eval/start") and not body.get("confirm"):
                raise RuntimeError("로봇 옆에 사람이 있는지 확인하고 체크해주세요.")
            if self.path == "/api/check":
                self._json(run_check())
            elif self.path == "/api/preview":
                if body.get("on"):
                    if JOB.running() and JOB.kind in ("record", "eval"):
                        raise RuntimeError("녹화·평가 중에는 그 프로그램이 카메라를 쓰고 있습니다. 화면은 그대로 보입니다.")
                    if JOB.running() and JOB.info.get("with_cameras"):
                        raise RuntimeError("카메라를 쓰는 텔레옵이 실행 중입니다.")
                    PREVIEW.start()
                else:
                    PREVIEW.stop()
                self._json({"ok": True})
            elif self.path == "/api/teleop":
                self._json(start_teleop(body))
            elif self.path == "/api/record":
                self._json(start_record(body))
            elif self.path == "/api/control":
                JOB.send(body.get("cmd", ""))
                self._json({"ok": True})
            elif self.path == "/api/stop":
                JOB.stop()
                self._json({"ok": True})
            elif self.path == "/api/start_pose/save":
                self._json(save_start_pose(body.get("kind", "start")))
            elif self.path == "/api/eval/start":
                self._json(start_eval(body))
            elif self.path == "/api/eval/report":
                start_report(body.get("repo_id", ""))
                self._json({"ok": True})
            elif self.path == "/api/eval/mark":
                eval_mark(body)
                self._json({"ok": True})
            elif self.path == "/api/train/start":
                self._json(train_start(body))
            elif self.path == "/api/train/estimate":
                self._json(train_estimate(body))
            elif self.path.startswith("/api/train/queue/"):
                self._json(queue_action(self.path, body))
            elif self.path == "/api/train/stop":
                train_stop(body)
                self._json({"ok": True})
            elif self.path == "/api/train/clean":
                train_clean(body.get("id", ""))
                self._json({"ok": True})
            elif self.path == "/api/upload":
                UPLOADER.start(body.get("repo_id", ""), body.get("keywords"), body.get("description"))
                self._json({"ok": True})
            elif self.path == "/api/kill":
                killed = JOB.kill()
                self._json({"ok": True, "killed": killed, "others": other_robot_processes()})
            elif self.path == "/api/reset_motors":
                if JOB.running() or other_robot_processes():
                    raise RuntimeError("로봇 프로그램이 아직 실행 중입니다. 강제 종료를 먼저 누르세요.")
                self._json({"ok": True, "motors": reset_motors()})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    # Started in the background, so SIGINT may be ignored; make INT and TERM both shut down cleanly.
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"OMX panel {VERSION} on http://{HOST}:{PORT} (user {USER}, settings: {CFG['_source']})", flush=True)
    try:
        server.serve_forever()
    finally:
        JOB.stop()
        PREVIEW.stop()


if __name__ == "__main__":
    main()
