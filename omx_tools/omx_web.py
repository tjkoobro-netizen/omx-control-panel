#!/usr/bin/env python3
"""OMX-AI control panel: check, camera preview, teleop and recording from a browser.

Runs on the GPU laptop and listens on 127.0.0.1 only. Laptops reach it through
an SSH tunnel (see omx-control.sh), so the panel is never exposed on the Wi-Fi.
Uses only the LeRobot environment (v3: 0.6.1) and the Python stdlib. Machine-specific settings
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
import select
import shutil
import signal
import statistics
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

import omx_config  # noqa: E402  (per-machine settings written by install_gpu.sh)
import omx_train_queue as tq  # noqa: E402  (training plan file and runner)
import omx_ram  # noqa: E402  (RAM and time estimates)
import omx_protocol as proto  # noqa: E402  (evaluation protocol v2: position presets, attempts)
import omx_names as names  # noqa: E402  (team naming rule, display names of models)
import omx_data_notes as data_notes  # noqa: E402  (group and description of each dataset in the training tab)
import omx_errors  # noqa: E402  (err_code_N of an abnormal end)
import omx_home  # noqa: E402  (standard home pose from the training data; config "home_pose")
import omx_camstats  # noqa: E402  (measured camera fps and brightness)
import omx_camauto as camauto  # noqa: E402  (which camera is which after a USB replug)

omx_camstats.install()

CONTROL = omx_config.is_control()  # v2: this PC holds the robot; training and inference on the GPU laptop
if CONTROL:
    import omx_remote as remote  # noqa: E402

CFG = omx_config.CFG
VERSION = omx_config.VERSION
# One panel server per account, each on its own port (port_base + uid - uid_base).
# omx-control.sh and omx-panel-server.sh ask omx_config for the same number.
HOST, PORT = "127.0.0.1", omx_config.port()
HOME = Path.home()
USER = (omx_config.CFG.get("gpu_user") if omx_config.is_control() else "") or getpass.getuser()
VENV_BIN = omx_config.venv_bin()
# Working directory for LeRobot commands: the lecture layout, or the team guide's ~/dev/lerobot
LEROBOT_DIR = omx_config.lerobot_dir()
TOOLS_DIR = Path(__file__).resolve().parent
LOG_DIR = omx_config.log_dir()
LIVE = omx_config.live_prefix()  # the running robot program's camera frames and motor output (/dev/shm)
LOG_DIR.mkdir(parents=True, exist_ok=True)  # a new control PC has none yet
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

# ---- Robot setup (v2.1): one arm, or two arms as one cooperative robot, chosen in the panel ----
# ~/.config/omx/v3/robot.json: {"mode": "single" | "bimanual", "single": {"set": "OMX 2"},
#   "bimanual": {"left": "OMX 2", "right": "OMX 3"},
#   "cameras": {"single": [{"name": "front", "path": "/dev/v4l/by-path/...", "label": "..."}, ...], "bimanual": [...]}}
# Without the file (e.g. the GPU laptop) the configured follower/leader/cameras are used as before.
SETUP_FILE = Path(os.environ.get("OMX_ROBOT_SETUP") or omx_config.CONF_DIR / "robot.json").expanduser()
ROBOT_SETS = CFG.get("robot_sets") or {f"OMX {n}": {"follower": f"/dev/omx_follower_{n}", "leader": f"/dev/omx_leader_{n}"}
                                       for n in (1, 2, 3)}
CAMERA_ROLES = {"single": ["front", "front_2", "wrist"], "bimanual": ["front", "front_2", "wrist_left", "wrist_right"]}
ROBOT_MODE = "single"
FOLLOWERS: list = []   # [(label, port)]
LEADERS: list = []
CAMERA_LABELS: dict = {}


def camera_label(mode: str, name: str, setup: dict) -> str:
    """Which robot's which camera, e.g. "OMX 2 · 왼팔 wrist" or "OMX 2+3 · 글로벌 front"."""
    if mode == "bimanual":
        left, right = setup["bimanual"]["left"], setup["bimanual"]["right"]
        pair = f"{left.replace('OMX ', 'OMX ')}+{right.removeprefix('OMX ')}"
        return {"front": f"{pair} · 글로벌 front", "front_2": f"{pair} · 글로벌 front 2",
                "wrist_left": f"{left} · 왼팔 wrist", "wrist_right": f"{right} · 오른팔 wrist"}.get(name, name)
    robot = setup["single"]["set"]
    return {"front": f"{robot} · 글로벌 front", "front_2": f"{robot} · 글로벌 front 2", "wrist": f"{robot} · 팔로워 wrist"}.get(name, name)


def load_setup() -> dict | None:
    try:
        return json.loads(SETUP_FILE.read_text())
    except (OSError, ValueError):
        return None


def apply_setup(setup: dict | None) -> None:
    """Ports, cameras and the lerobot arguments of the chosen robot setup."""
    global FOLLOWER, LEADER, CAMERAS, CAMS_ARG, ROBOT_ARGS, TELEOP_ARGS, ROBOT_MODE, FOLLOWERS, LEADERS, CAMERA_LABELS, ARM_NAMES
    if not setup:
        ROBOT_MODE = "single"
        ARM_NAMES = {"": CFG.get("robot_set") or "OMX"}
        FOLLOWERS, LEADERS = [("팔로워", CFG["follower_port"])], [("리더", CFG["leader_port"])]
        CAMERAS = dict(CFG["cameras"])
        CAMERA_LABELS = {n: {"front": "글로벌 카메라", "wrist": "팔로워 장착"}.get(n, n) for n in CAMERAS}
    else:
        ROBOT_MODE = setup.get("mode", "single")
        cams = (setup.get("cameras") or {}).get(ROBOT_MODE) or []
        CAMERAS = {c["name"]: c["path"] for c in cams}
        CAMERA_LABELS = {c["name"]: c.get("label") or camera_label(ROBOT_MODE, c["name"], setup) for c in cams}
        if ROBOT_MODE == "bimanual":
            ARM_NAMES = {s: setup["bimanual"][s] for s in ("left", "right")}  # joint prefix -> set name
            left, right = (ROBOT_SETS[setup["bimanual"][s]] for s in ("left", "right"))
            FOLLOWERS = [(f"왼팔 팔로워 ({setup['bimanual']['left']})", left["follower"]),
                         (f"오른팔 팔로워 ({setup['bimanual']['right']})", right["follower"])]
            LEADERS = [(f"왼팔 리더 ({setup['bimanual']['left']})", left["leader"]),
                       (f"오른팔 리더 ({setup['bimanual']['right']})", right["leader"])]
        else:
            ARM_NAMES = {"": setup["single"]["set"]}
            one = ROBOT_SETS[setup["single"]["set"]]
            FOLLOWERS = [(f"팔로워 ({setup['single']['set']})", one["follower"])]
            LEADERS = [(f"리더 ({setup['single']['set']})", one["leader"])]
    FOLLOWER, LEADER = FOLLOWERS[0][1], LEADERS[0][1]
    CAMS_ARG = omx_config.cameras_arg(warmup_s=3, cameras=CAMERAS)
    if ROBOT_MODE == "bimanual":
        ROBOT_ARGS = ["--robot.type=bi_omx_follower", f"--robot.left_port={FOLLOWERS[0][1]}",
                      f"--robot.right_port={FOLLOWERS[1][1]}", "--robot.id=bi_omx_follower"]
        TELEOP_ARGS = ["--teleop.type=bi_omx_leader", f"--teleop.left_port={LEADERS[0][1]}",
                       f"--teleop.right_port={LEADERS[1][1]}", "--teleop.id=bi_omx_leader"]
    else:
        ROBOT_ARGS = ["--robot.type=omx_follower", f"--robot.port={FOLLOWER}", "--robot.id=omx_follower_arm"]
        TELEOP_ARGS = ["--teleop.type=omx_leader", f"--teleop.port={LEADER}", "--teleop.id=omx_leader_arm"]


apply_setup(load_setup())


def list_cameras() -> list[dict]:
    """Every camera plugged into this PC, by USB port (identical cameras share serials, so the port is the id)."""
    out = []
    for p in sorted(Path("/dev/v4l/by-path").glob("*-video-index0")):
        if "usbv2" in p.name:  # the same device under an older path alias
            continue
        dev = p.resolve().name
        try:
            model = Path(f"/sys/class/video4linux/{dev}/name").read_text().split(":")[0].strip()
        except OSError:
            model = ""
        port = re.search(r"usb-\d+:([\d.]+):", p.name)
        used = next((n for n, path in CAMERAS.items() if path == str(p)), None)
        out.append({"path": str(p), "device": f"/dev/{dev}", "model": model, "usb_port": port.group(1) if port else "",
                    "role": used, "label": CAMERA_LABELS.get(used) if used else None})
    return out


def camera_probe(path: str) -> bytes:
    """One JPEG from a camera (for choosing roles); only when nothing else holds the cameras."""
    if PREVIEW.running or JOB.running():
        raise RuntimeError("카메라 보기나 작업이 켜져 있어 따로 읽을 수 없습니다. 끈 뒤 다시 누르세요.")
    if not path.startswith("/dev/v4l/by-path/") or not Path(path).exists():
        raise RuntimeError("카메라를 찾을 수 없습니다.")
    frame = camauto.grab_frame(path)
    if frame is None:
        raise RuntimeError("영상을 읽지 못했습니다.")
    return cv2.imencode(".jpg", cv2.resize(frame, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


CAM_REFS = SETUP_FILE.parent / "cam_refs"  # the views of the global cameras when the setup was last saved
CAM_DETECT = threading.Lock()


def camera_watch() -> dict:
    """For the status poll (opens no camera): saved cameras that are gone, and whether the user still has to look.
    Not without a board of this setup plugged in: the cameras that account could see are someone else's."""
    setup = load_setup()
    if not setup or not any(Path(p).exists() for _, p in FOLLOWERS + LEADERS):
        return {"missing": [], "unconfirmed": False, "sig": ""}
    return {"missing": [n for n, p in CAMERAS.items() if not Path(p).exists()],
            "unconfirmed": ROBOT_MODE in setup.get("cameras_unconfirmed", []),
            "sig": "|".join(sorted(p.name for p in Path("/dev/v4l/by-path").glob("*-video-index0")))}


def required_camera_roles(mode: str) -> list:
    return ["front", "wrist"] if mode == "single" else ["front", "wrist_left", "wrist_right"]


def detect_cameras(apply: bool) -> dict:
    """Propose the camera of each role. With apply, a result that is sure for every role is saved at once; a record
    or evaluation then still waits for the user to look at the pictures (camera_gate)."""
    setup = load_setup()
    if not setup:
        raise RuntimeError("저장된 로봇 구성이 없습니다. 카메라를 직접 지정하세요.")
    if JOB.running() or PREVIEW.running or other_robot_processes():
        raise RuntimeError("작업이나 카메라 보기를 끈 뒤 다시 찾으세요.")
    if not CAM_DETECT.acquire(blocking=False):
        raise RuntimeError("카메라를 찾는 중입니다.")
    try:
        mode = ROBOT_MODE
        entries = {c["name"]: c for c in (setup.get("cameras") or {}).get(mode) or []}
        roles = [r for r in CAMERA_ROLES[mode] if r in entries] or required_camera_roles(mode)
        saved = {r: c["path"] for r, c in entries.items()}
        role_sets = ({"wrist_left": setup["bimanual"]["left"], "wrist_right": setup["bimanual"]["right"]}
                     if mode == "bimanual" else {"wrist": setup["single"]["set"]} if setup.get("single") else {})
        set_hubs = {n: {h for h in map(camauto.board_hub, s.values()) if h} for n, s in ROBOT_SETS.items()}
        cams = [{"path": c["path"], "model": c["model"], "hub": camauto.camera_hub(c["device"])} for c in list_cameras()]
        refs = camauto.load_refs(CAM_REFS, mode, [r for r in roles if not r.startswith("wrist")])
        frames = {c["path"]: camauto.grab_gray(c["path"]) for c in cams
                  if refs and not camauto.BUILTIN.search(c["model"]) and not camauto.WRIST_MODEL.search(c["model"])}
        picks = camauto.propose(roles, saved, cams, set_hubs, role_sets, frames, refs)
        sure = all(p["path"] and p["sure"] for p in picks.values())
        changed = any(p["path"] != saved.get(r) for r, p in picks.items())
        applied = False
        if apply and sure and changed:
            setup.setdefault("cameras", {})[mode] = [
                {"name": r, "path": picks[r]["path"], "label": entries.get(r, {}).get("label", "")} for r in roles]
            setup["cameras_unconfirmed"] = sorted({*setup.get("cameras_unconfirmed", []), mode})
            SETUP_FILE.write_text(json.dumps(setup, ensure_ascii=False, indent=2))
            apply_setup(setup)
            applied = True
        return {"picks": picks, "sure": sure, "changed": changed, "applied": applied, "has_refs": bool(refs),
                "labels": {r: c.get("label", "") for r, c in entries.items()}}
    finally:
        CAM_DETECT.release()


def camera_gate() -> None:
    """Before a record or evaluation: names on the wrong pictures would silently spoil the data or feed the policy
    the wrong views. Saving the setup answers both, it also takes the views now seen as the new reference."""
    setup = load_setup()
    if not setup:
        return
    if ROBOT_MODE in setup.get("cameras_unconfirmed", []):
        raise RuntimeError("카메라 포트가 바뀌어 자동으로 다시 찾았습니다. 로봇 구성에서 화면을 확인하고 저장한 뒤 시작하세요.")
    if JOB.running():
        return
    refs = camauto.load_refs(CAM_REFS, ROBOT_MODE, [n for n in CAMERAS if not n.startswith("wrist")])
    cameras = {n: CAMERAS[n] for n in refs}
    if len(refs) < 2:
        return
    PREVIEW.stop()  # the check needs the cameras itself
    wrong = camauto.swapped(cameras, refs, {p: camauto.grab_gray(p) for p in cameras.values() if Path(p).exists()})
    if wrong:
        raise RuntimeError(f"{', '.join(wrong)} 화면이 다른 카메라의 기준 화면과 일치합니다. 케이블이 바뀐 것 같습니다. "
                           "로봇 구성에서 화면을 확인하고 저장한 뒤 시작하세요.")


WRIST_OFFSETS = (0, 90, -90)  # degrees; 180 would leave no range once clamped to one turn


def wrist_offset_env() -> dict:
    """Teleop / record: the follower's wrist_roll target (and the recorded action) is the leader's plus the
    offset of that arm's set (robot.json "wrist_offset_deg"), for a set whose leader gripper is mounted turned
    against the follower's. Not for an evaluation: the policy's output already is the follower target."""
    setup = load_setup() or {}
    offsets = setup.get("wrist_offset_deg") or {}
    if ROBOT_MODE == "bimanual":
        keys = {f"{side}_wrist_roll.pos": setup["bimanual"][side] for side in ("left", "right")}
    elif setup.get("single"):
        keys = {"wrist_roll.pos": setup["single"]["set"]}
    else:
        return {}
    units = {k: offsets[n] / 1.8 for k, n in keys.items() if offsets.get(n)}  # 200 units = 360 degrees
    return {"OMX_WRIST_OFFSET": json.dumps(units)} if units else {}


def setup_status() -> dict:
    setup = load_setup()
    return {"setup": setup, "mode": ROBOT_MODE, "sets": {n: {k: {"port": v, "present": Path(v).exists()} for k, v in s.items()}
                                                          for n, s in ROBOT_SETS.items()},
            "roles": CAMERA_ROLES, "cameras": list_cameras(),
            "followers": [{"label": l, "port": p, "present": Path(p).exists()} for l, p in FOLLOWERS],
            "leaders": [{"label": l, "port": p, "present": Path(p).exists()} for l, p in LEADERS],
            "active_cameras": [{"name": n, "path": p, "label": CAMERA_LABELS.get(n, n), "present": Path(p).exists()}
                               for n, p in CAMERAS.items()]}


def save_setup(body: dict) -> dict:
    """Panel: choose single / cooperative, the robot set(s) and the camera roles (labels follow, or typed)."""
    if JOB.running() or PREVIEW.running:
        raise RuntimeError("작업이나 카메라 보기를 끈 뒤 바꾸세요.")
    mode = body.get("mode")
    if mode not in ("single", "bimanual"):
        raise RuntimeError("단일 로봇 또는 협동 로봇을 고르세요.")
    setup = load_setup() or {}
    setup["mode"] = mode
    if "wrist_offset_deg" in body:
        offsets = {n: int(v) for n, v in (body["wrist_offset_deg"] or {}).items() if n in ROBOT_SETS}
        if any(v not in WRIST_OFFSETS for v in offsets.values()):
            raise RuntimeError("손목 회전 보정은 0, +90, -90도 중에서 고르세요.")
        setup["wrist_offset_deg"] = {n: v for n, v in offsets.items() if v}
    if mode == "single":
        if body.get("set") not in ROBOT_SETS:
            raise RuntimeError("로봇 세트를 고르세요.")
        setup["single"] = {"set": body["set"]}
    else:
        left, right = body.get("left"), body.get("right")
        if left not in ROBOT_SETS or right not in ROBOT_SETS or left == right:
            raise RuntimeError("왼팔과 오른팔에 서로 다른 로봇 세트를 고르세요.")
        setup["bimanual"] = {"left": left, "right": right}
    cams, seen = [], set()
    for c in body.get("cameras", []):
        role = c.get("role")
        if not role:
            continue
        if role not in CAMERA_ROLES[mode] or role in seen:
            raise RuntimeError(f"카메라 역할 {role}이 잘못됐거나 두 번 쓰였습니다.")
        if not str(c.get("path", "")).startswith("/dev/v4l/by-path/"):
            raise RuntimeError("카메라 경로가 잘못됐습니다.")
        seen.add(role)
        cams.append({"name": role, "path": c["path"], "label": (c.get("label") or "").strip()[:60]})
    need = {"front", "wrist"} if mode == "single" else {"front", "wrist_left", "wrist_right"}
    if not need <= seen:
        raise RuntimeError("필요한 카메라를 모두 지정하세요: " + ", ".join(sorted(need - seen)))
    order = CAMERA_ROLES[mode]
    setup.setdefault("cameras", {})[mode] = sorted(cams, key=lambda c: order.index(c["name"]))
    unconfirmed = [m for m in setup.get("cameras_unconfirmed", []) if m != mode]
    if unconfirmed:
        setup["cameras_unconfirmed"] = unconfirmed
    else:
        setup.pop("cameras_unconfirmed", None)
    SETUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETUP_FILE.write_text(json.dumps(setup, ensure_ascii=False, indent=2))
    apply_setup(setup)
    camauto.save_refs(CAM_REFS, mode, {c["name"]: c["path"] for c in cams if not c["name"].startswith("wrist")})
    return setup_status()

SLOW_RE = re.compile(r"Record loop is running slower \(([0-9.]+) Hz\)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TABLE_RE = re.compile(r"^(\S+\.pos\s*\||-{5,}$|NAME\s*\|)")


def fix_wrist_fps() -> None:
    """Keep the configured camera (usually the wrist one) at full fps in dim light (resets on replug).
    "wrist" also covers a cooperative robot's wrist_left / wrist_right."""
    name = CFG.get("fixed_fps_camera") or ""
    if not name:
        return
    for cam, dev in CAMERAS.items():
        if cam == name or cam.startswith(f"{name}_"):
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_dynamic_framerate=0"], capture_output=True)


def other_robot_processes() -> list[str]:
    out = subprocess.run(["ps", "-eo", "user=,pid=,args="], capture_output=True, text=True).stdout
    mine = {str(JOB.pid)} if JOB.pid else set()
    found = []
    for line in out.splitlines():
        user, pid, args = (line.split(None, 2) + ["", ""])[:3]
        if pid in mine or "omx_web.py" in args:
            continue
        if re.search(r"lerobot-(teleoperate|record)|omx_(run|record|teleop|return_home)\.py", args):
            found.append(f"{user} (PID {pid})")
    return found


CPU_NOW = {"t": 0.0, "stat": {}, "throttle": None, "out": None,
           "temps": collections.deque(maxlen=5), "throttle_ms": collections.deque(maxlen=5)}
CPU_BUSY_PCT = 25  # a hardware thread at or above this use counts as busy in the panel's top bar


def cpu_status() -> dict:
    """This PC's CPU for the panel's top bar: use of each hardware thread between two status polls, busy
    threads, physical cores, temperature, thermal throttling. Only reads /proc and /sys, so it never waits."""
    now = time.time()
    if now - CPU_NOW["t"] < 0.9 and CPU_NOW["out"]:
        return CPU_NOW["out"]
    stat = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if re.match(r"cpu\d+ ", line):
            name, *vals = line.split()
            vals = [int(v) for v in vals[:8]]
            stat[int(name[3:])] = (sum(vals), vals[3] + vals[4])  # total, idle + iowait
    threads = []
    for i in sorted(stat):
        total, idle = stat[i]
        was = CPU_NOW["stat"].get(i, (0, 0))
        threads.append(round(100 * (1 - (idle - was[1]) / max(total - was[0], 1))))
    cores = set()
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_id"):
        try:
            cores.add((f.parent.joinpath("physical_package_id").read_text().strip(), f.read_text().strip()))
        except OSError:
            pass
    temps = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if zone.joinpath("type").read_text().strip() in ("x86_pkg_temp", "TCPU"):
                temps.append(int(zone.joinpath("temp").read_text()) // 1000)
        except (OSError, ValueError):
            pass
    try:
        throttle = int(Path("/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_total_time_ms").read_text())
    except (OSError, ValueError):
        throttle = None
    if throttle is not None and CPU_NOW["throttle"] is not None:
        CPU_NOW["throttle_ms"].append(throttle - CPU_NOW["throttle"])
    if temps:
        CPU_NOW["temps"].append(max(temps))
    # The package sensor jumps ~30 C for a second on any short burst and a few ms of throttling is normal:
    # show a 5 s average, and "throttled" only when it lasts (at least 1 s of the last 5)
    throttled = sum(CPU_NOW["throttle_ms"]) >= 1000
    first = not CPU_NOW["stat"]
    CPU_NOW.update(t=now, stat=stat, throttle=throttle)
    if first:  # no earlier reading yet: use since boot would mislead
        threads = [0 for _ in threads]
    CPU_NOW["out"] = {"pct": round(sum(threads) / max(len(threads), 1)), "threads": threads,
                      "busy": sum(t >= CPU_BUSY_PCT for t in threads), "busy_pct": CPU_BUSY_PCT, "cores": len(cores) or None,
                      "temp": round(sum(CPU_NOW["temps"]) / len(CPU_NOW["temps"])) if CPU_NOW["temps"] else None, "throttled": throttled}
    return CPU_NOW["out"]


def arms_to_check() -> list:
    """(name, port, motor ids) of every arm of the current setup; the first follower/leader keep the old keys."""
    out = []
    for i, (label, port) in enumerate(FOLLOWERS):
        out.append(("follower" if i == 0 else f"follower_{i + 1}", port, FOLLOWER_IDS))
    for i, (label, port) in enumerate(LEADERS):
        out.append(("leader" if i == 0 else f"leader_{i + 1}", port, LEADER_IDS))
    return out


def close_port(handler) -> None:
    """Close a check's port even when its board is gone, so the panel does not keep it open."""
    try:
        handler.closePort()
    except Exception:
        pass


def ping_motors() -> dict:
    from dynamixel_sdk import PacketHandler, PortHandler

    result = {}
    for name, port, ids in arms_to_check():
        if not Path(port).exists():
            result[name] = {"ok": False, "msg": f"{port} 없음 (USB 연결 확인)"}
            continue
        handler, packet = PortHandler(port), PacketHandler(2.0)
        if not handler.openPort():
            result[name] = {"ok": False, "msg": "포트를 열 수 없음 (다른 프로그램이 사용 중일 수 있음)"}
            continue
        missing, hw_errors = [], []
        try:
            handler.setBaudRate(BAUDRATE)
            for motor_id in ids:
                _, comm, err = packet.ping(handler, motor_id)
                if comm != 0:
                    missing.append(motor_id)
                elif err & 0x80:
                    hw_errors.append(motor_id)
        except Exception as exc:  # the board went away during the check
            result[name] = {"ok": False, "msg": f"점검 중 연결 끊김 ({exc}). USB 확인 후 다시 점검"}
            continue
        finally:
            close_port(handler)
        if missing or hw_errors:
            parts = []
            if missing:
                parts.append(f"응답 없음 ID {missing}")
            if hw_errors:
                parts.append(f"하드웨어 에러 ID {hw_errors} (모터 초기화, 안 되면 전원 껐다 켜기)")
            result[name] = {"ok": False, "msg": ", ".join(parts)}
        else:
            result[name] = {"ok": True, "msg": f"모터 {len(ids)}개 정상"}
    return result


def reset_motors() -> dict:
    """Reboot every motor on both arms: clears hardware errors (e.g. overload) and turns torque off."""
    from dynamixel_sdk import PacketHandler, PortHandler

    result = {}
    for name, port, ids in arms_to_check():
        if not Path(port).exists():
            result[name] = {"ok": False, "msg": f"{port} 없음 (USB 연결 확인)"}
            continue
        handler, packet = PortHandler(port), PacketHandler(2.0)
        if not handler.openPort():
            result[name] = {"ok": False, "msg": "포트를 열 수 없음 (강제 종료를 먼저 누르세요)"}
            continue
        missing, hw_errors = [], []
        try:
            handler.setBaudRate(BAUDRATE)
            for motor_id in ids:
                packet.reboot(handler, motor_id)
            time.sleep(1.0)
            for motor_id in ids:
                value, comm, _ = packet.read1ByteTxRx(handler, motor_id, 70)  # Hardware_Error_Status
                if comm != 0:
                    missing.append(motor_id)
                elif value:
                    hw_errors.append(motor_id)
        except Exception as exc:  # the board went away during the reset
            result[name] = {"ok": False, "msg": f"초기화 중 연결 끊김 ({exc}). USB 확인 후 다시"}
            continue
        finally:
            close_port(handler)
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
        if "omx_web.py" in args or not re.search(r"lerobot-(teleoperate|record)|omx_(run|record|teleop|return_home)\.py", args):
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
    if CONTROL:  # this PC has no GPU of its own: show the GPU laptop's (training and inference run there)
        try:
            g = remote.gpu_json("/api/train/status", timeout=10).get("gpu") or {}
            return (f"GPU 노트북({remote.HOST}) · 메모리 {g.get('mem_used')}/{g.get('mem_total')} MiB · 사용률 {g.get('util')}%"
                    if g.get("mem_total") else f"GPU 노트북({remote.HOST}) GPU 정보를 읽지 못했습니다")
        except Exception:
            return f"GPU 노트북({remote.HOST})에 연결되지 않았습니다"
    return gpu_status_local()


def gpu_status_local() -> str:
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


def camera_stats() -> dict:
    """{camera: {"fps", "brightness"}}: from the camera view, or from the running record/eval program."""
    if PREVIEW.running:
        return {n: {**v, "brightness": PREVIEW.brightness.get(n, v["brightness"])}
                for n, v in omx_camstats.stats(PREVIEW.cams).items()}
    f = Path(f"{LIVE}stats.json")
    try:
        if JOB.running() and time.time() - f.stat().st_mtime < 3:
            return json.loads(f.read_text())
    except (OSError, ValueError):
        pass
    return {}


def motor_output() -> dict:
    """The 모터 출력 view: each follower joint's share of its max torque from the running teleop / record /
    eval program. Asking marks the view open, so the program reads the motors only while someone watches."""
    want = Path(f"{LIVE}torque.want")
    try:
        want.touch()
    except OSError:
        pass
    f = Path(f"{LIVE}torque.json")
    out = {"running": JOB.running(), "kind": JOB.kind, "arms": None}
    try:
        if out["running"] and f.stat().st_mtime > JOB.started and time.time() - f.stat().st_mtime < 2:
            data = json.loads(f.read_text())
            out["arms"], out["age"] = data["arms"], round(time.time() - data["t"], 2)
    except (OSError, ValueError, KeyError):
        pass
    return out


# ---- Control PC: copies to the GPU laptop that did not go through (network down at the end of a run) ----
SYNC = {"busy": set(), "lock": threading.Lock(), "cache": (0.0, []), "last_auto": ""}


def sync_one(root: Path, note: Path | None, lines: list | None = None) -> bool:
    """One dataset at a time (the end of a run, the retry loop and the button never copy it twice at once)."""
    key = str(root)
    with SYNC["lock"]:
        if key in SYNC["busy"]:
            return False
        SYNC["busy"].add(key)
    try:
        return remote.sync_with_state(root, note, lines)
    finally:
        SYNC["busy"].discard(key)
        SYNC["cache"] = (0.0, [])


def local_runs() -> list[Path]:
    """Recordings and evaluations made on this PC (they all belong on the GPU laptop)."""
    base = DATA_HOME / USER
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")) if base.exists() else []


def running_repo_root() -> Path | None:
    return DATA_HOME / JOB.info["repo_id"] if JOB.running() and JOB.info.get("repo_id") else None


def sync_status() -> dict:
    """What is not on the GPU laptop yet (cached 10 s for the status poll)."""
    t, items = SYNC["cache"]
    if time.time() - t > 10:
        items, now_root = [], running_repo_root()
        for root in local_runs():
            if root == now_root or not remote.needs_sync(root):
                continue
            st = remote.sync_state(root)
            items.append({"repo_id": f"{USER}/{root.name}", "status": st.get("status") or "new",
                          "error": st.get("error", ""), "at": st.get("at", "")})
        SYNC["cache"] = (time.time(), items)
    busy = [f"{USER}/{Path(k).name}" for k in SYNC["busy"]]
    return {"pending": items, "busy": busy, "link": remote.TUNNEL.status()["ok"], "last_auto": SYNC["last_auto"]}


def sync_retry(repo: str = "") -> dict:
    """[다시 보내기]: one dataset, or every one that is not there yet (blocked ones too, on request)."""
    if not remote.TUNNEL.status()["ok"]:
        raise RuntimeError("GPU 노트북에 연결되지 않아 보낼 수 없습니다. Tailscale 연결을 확인하세요.")
    roots = [DATA_HOME / repo] if repo else [r for r in local_runs() if remote.needs_sync(r)]
    now_root = running_repo_root()
    roots = [r for r in roots if r.exists() and r != now_root and r.is_relative_to(DATA_HOME / USER)]
    if not roots:
        raise RuntimeError("보낼 데이터가 없습니다.")
    for root in roots:
        remote.sync_later(sync_one, root, notes_path(f"{USER}/{root.name}"))
    SYNC["cache"] = (0.0, [])
    return {"ok": True, "started": [f"{USER}/{r.name}" for r in roots]}


def sync_loop() -> None:
    """Every minute, while the GPU laptop is reachable: send what is new or failed (not 'blocked': a dataset
    of the same name from somewhere else is there, that needs a person)."""
    while True:
        time.sleep(60)
        try:
            if not remote.TUNNEL.status()["ok"]:
                continue
            now_root = running_repo_root()
            for root in local_runs():
                if root == now_root or not remote.needs_sync(root) or remote.sync_state(root).get("status") == "blocked":
                    continue
                if sync_one(root, notes_path(f"{USER}/{root.name}")):
                    SYNC["last_auto"] = f"{dt.datetime.now():%H:%M} {USER}/{root.name} 자동 전송"
                    print(f"[omx] sync: {USER}/{root.name} 자동 전송 완료", flush=True)
        except Exception as exc:
            print(f"[omx] sync: 자동 전송 중 오류 ({exc})", flush=True)


def merged_datasets() -> list[dict]:
    """Control PC: the team's datasets on the GPU laptop (training uses those), plus the ones recorded here
    that are not there yet (marked local)."""
    shared = remote.gpu_datasets()
    names_there = {d["repo_id"] for d in shared}
    local = [dict(d, local_only=True) for d in list_datasets() if d["repo_id"] not in names_there]
    return shared + local


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
            **data_notes.describe(repo.parent.name, repo.name, meta.get("total_episodes", 0), USER),
            **({"candidates": proto.candidate_episodes(read_notes(f"{repo.parent.name}/{repo.name}"))} if kind == "eval" else {}),
        })
    return items


# ---- Training (runs omx_train.py as an independent process; state in <log_dir>) ----
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


DEFAULT_WORKERS = {"act": 8, "diffusion": 4, "smolvla": 4}
POLICIES = ("act", "diffusion", "smolvla")


def train_estimate(body: dict) -> dict:
    """RAM / time estimate for the settings on the training tab (also used before starting)."""
    policy = body.get("policy") if body.get("policy") in POLICIES else "act"
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
    if policy not in POLICIES:
        raise RuntimeError("정책은 ACT, Diffusion, SmolVLA 중에서 고르세요.")
    steps, batch = int(body.get("steps", 0)), int(body.get("batch_size", 0))
    workers = int(body.get("num_workers", DEFAULT_WORKERS[policy]))
    if not 100 <= steps <= 500_000 or not 1 <= batch <= 64:
        raise RuntimeError("스텝은 100~500,000, 배치는 1~64 사이로 정하세요.")
    if not 0 <= workers <= 16:
        raise RuntimeError("데이터 로딩 수는 0~16 사이로 정하세요.")
    datasets = []
    for d in body.get("datasets", []):
        root = dataset_dir(d["repo"])
        item = {"repo": d["repo"], "exclude": sorted({int(e) for e in d.get("exclude", [])})}
        if root.name.startswith("eval_"):
            # evaluation attempts: only the ones checked as training candidates go in
            note = read_notes(d["repo"])
            c = proto.candidate_episodes(note)
            total = json.loads((root / "meta/info.json").read_text()).get("total_episodes", 0)
            keep = set(c["episodes"]) - set(item["exclude"])
            if not keep:
                raise RuntimeError(f"{d['repo']}에 학습 후보로 체크한 시도가 없습니다.")
            item["exclude"] = sorted(set(range(total)) - keep)
            item["eval"] = {"one_shot": c["one_shot"], "recovery": c["recovery"],
                            "wandering_by_user": c["wandering_by_user"], "episodes": sorted(keep),
                            "preset_id": (note.get("protocol") or {}).get("preset_id"), "model": note.get("run_id")}
        datasets.append(item)
    if not datasets:
        raise RuntimeError("학습에 쓸 데이터셋을 하나 이상 고르세요.")
    job = {"policy": policy, "steps": steps, "batch_size": batch, "num_workers": workers,
           "augmentation": bool(body.get("augmentation")),
           "note": str(body.get("note", "")).strip(), "datasets": datasets}
    name = names.clean_name(str(body.get("name", "")))
    if name:
        if (CKPT_HOME / USER / name).exists():
            raise RuntimeError(f"모델 이름 {name}은 이미 있습니다. 끝에 -2 같은 특이사항을 붙여 바꾸세요.")
        job["name"] = name
    return job


def train_name(body: dict) -> dict:
    """Rule name for the training-tab settings: {task}_{policy}_{data}_{steps}_{MMDD}[-extras]."""
    import pandas as pd

    import omx_index

    origin: dict = {}
    tasks: list = []
    eval_attempts = 0
    for d in body.get("datasets", []):
        try:
            root = dataset_dir(d["repo"])
        except Exception:
            continue
        excl = {int(e) for e in d.get("exclude", [])}
        if not tasks:
            try:
                tasks = list(pd.read_parquet(root / "meta/tasks.parquet").index)
            except Exception:
                pass
        if root.name.startswith("eval_"):
            eval_attempts += len(set(proto.candidate_episodes(read_notes(d["repo"]))["episodes"]) - excl)
            continue
        src = omx_index.sources_path(root)
        if src.exists():  # a training dataset built earlier: count its original recordings
            for e in json.loads(src.read_text())["episodes"]:
                if e["episode"] not in excl:
                    origin.setdefault(e["source"], []).append(e["source_episode"])
            continue
        try:
            total = json.loads((root / "meta/info.json").read_text()).get("total_episodes", 0)
        except (OSError, ValueError):
            total = 0
        origin.setdefault(d["repo"], []).extend(i for i in range(total) if i not in excl)
    policy = body.get("policy") if body.get("policy") in POLICIES else "act"
    out = names.model_name(policy, int(body.get("steps") or 0), dt.datetime.now().strftime("%m%d"), origin, tasks,
                           bool(body.get("augmentation")), int(body.get("batch_size") or 0) or None, eval_attempts)
    out["exists"] = (CKPT_HOME / USER / out["name"]).exists()
    return out


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
            job = make_job(body)
            if job.get("name") and any(x["status"] in (tq.WAITING, tq.RUNNING) and x["job"].get("name") == job["name"] for x in items):
                raise RuntimeError(f"계획에 같은 이름({job['name']})의 학습이 이미 있습니다. 이름을 바꿔 주세요.")
            items.append(tq.new_item(job))
        elif action == "cancel":
            it = find()
            if not it or it["status"] != tq.WAITING:
                raise RuntimeError("대기 중인 항목만 취소할 수 있습니다. 실행 중이면 학습 중지를 누르세요.")
            it["status"] = tq.CANCELLED
        elif action == "retry":
            it = find()
            if not it or it["status"] not in (tq.FAILED, tq.STOPPED, tq.CANCELLED):
                raise RuntimeError("실패·중단·취소된 항목만 다시 넣을 수 있습니다.")
            job = dict(it["job"])
            if job.get("name") and (CKPT_HOME / USER / job["name"]).exists():  # a failed run may have left its folder
                k = 2
                while (CKPT_HOME / USER / f"{job['name']}-r{k}").exists():
                    k += 1
                job["name"] = f"{job['name']}-r{k}"
            items.append(tq.new_item(job))
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
    aliases = names.load_aliases()
    for r in index.get("runs", []):
        r["folder"] = r["id"].rsplit("/", 1)[-1]
        r["display"] = names.display_name(r["id"], aliases)
        model = Path(r["model_path"])
        rdir = model.parent if model.name == "pretrained_model" else model
        r["own"] = rdir.exists() and rdir.owner() == USER
        r["intermediate"] = (rdir / "checkpoints").is_dir()
        # intermediate saves that can be evaluated (the final one is the model itself)
        r["checkpoints"] = sorted(
            ({"step": int(d.name), "model_path": str(d / "pretrained_model")} for d in (rdir / "checkpoints").glob("[0-9]*")
             if (d / "pretrained_model/model.safetensors").exists() and int(d.name) != r.get("steps")),
            key=lambda c: c["step"]) if r["intermediate"] else []
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
    if CONTROL:
        push_note_later(path)


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
        self.raw: dict[str, bytes] = {}  # the frame before the marker check drew on it (camera alignment)
        self.brightness: dict[str, float] = {}
        self.running = False
        self.thread: threading.Thread | None = None
        # marker check on the front camera (display only, nothing is recorded here)
        self.marker_show = False                 # outline every ArUco marker with its number
        self.marker_target: int | None = None    # circle this one, exactly as the record/eval run will
        self._marker_mark = None
        self.markers: dict = {}                  # {"seen": [ids], "target": id, "found": bool}

    def set_marker_view(self, show: bool, target: int | None) -> dict:
        self.marker_show = bool(show)
        if target != self.marker_target:
            self.marker_target, self._marker_mark = target, None
        if not self.marker_show and target is None:
            self.markers = {}
        return {"ok": True, "show": self.marker_show, "target": self.marker_target}

    def _mark_front(self, frame):
        """Front frame with the marker check drawn (RGB in, RGB out)."""
        from target_mark import TargetMarker
        out = frame
        if self.marker_target is not None:
            if self._marker_mark is None:
                self._marker_mark = TargetMarker(self.marker_target)
            out = self._marker_mark(out)
        if getattr(self, "_marker_det", None) is None:
            self._marker_det = TargetMarker(0).det
        det = self._marker_det
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        seen = sorted(int(i) for i in ids.flatten()) if ids is not None else []
        if self.marker_show and ids is not None:
            out = out.copy() if out is frame else out
            cv2.aruco.drawDetectedMarkers(out, corners, ids, (0, 200, 0))
        found = self.marker_target in seen if self.marker_target is not None else None
        self.markers = {"seen": seen, "target": self.marker_target, "found": found}
        return out

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
                raw = None
                if name == "front" and (self.marker_show or self.marker_target is not None):
                    try:
                        raw, frame = frame, self._mark_front(frame)
                    except Exception:
                        pass  # the check is optional: never stop the preview over it
                if raw is not None:
                    self.raw[name] = cv2.imencode(".jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                else:
                    self.raw.pop(name, None)
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
            self.cams, self.frames, self.raw, self.brightness = {}, {}, {}, {}


def output_lines(proc: subprocess.Popen, grace_s: float = 2.0):
    """Lines a run prints, until its output closes - or grace_s after the run has exited, because a
    helper it leaves behind (multiprocessing's resource tracker) can hold the pipe open a while longer."""
    fd, buf, exited = proc.stdout.fileno(), b"", None
    while True:
        if select.select([fd], [], [], 0.2)[0]:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            *done, buf = re.split(rb"\r\n|\r|\n", buf + chunk)  # \r too: progress bars redraw a line
            yield from (d.decode(errors="replace") for d in done)
        elif proc.poll() is not None:
            exited = exited or time.time()
            if time.time() - exited > grace_s:
                break
    if buf:
        yield buf.decode(errors="replace")


class Job:
    """One robot process at a time (teleop or record), guarded by the shared robot lock."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.abort_lock = threading.Lock()
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
        # until the output reader has finished too: it records the exit code and releases the robot lock
        return self.proc is not None and (self.proc.poll() is None or bool(getattr(self, "reader", None) and self.reader.is_alive()))

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
        for port in [p for _, p in FOLLOWERS] + ([p for _, p in LEADERS] if need_leader else []):
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
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMX_PORT=str(PORT), **(env_extra or {}))  # same LIVE files
        self.proc = subprocess.Popen(
            cmd, cwd=LEROBOT_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        proc = self.proc
        for raw in output_lines(proc):
            line = ANSI_RE.sub("", raw).rstrip()
            if not line:
                continue
            if line.startswith("Teleop loop time"):
                self.info["loop"] = line.split(":", 1)[1].strip()
                continue
            if TABLE_RE.match(line):
                continue
            if m := SLOW_RE.search(line):
                # omx_run.py warns on every slow loop iteration: count them instead of flooding the log view
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
        self.info["err"] = omx_errors.classify(self.lines, proc.returncode)
        if self.info["err"]:
            line = f"[omx] {self.info['err']['name']}: {self.info['err']['title']} ({self.info['err']['detail']})"
            self.lines.append(line)
            if self.log_file:
                self.log_file.write(line + "\n")
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self._release_lock()
        if self.info["err"]:
            set_armed(False, f"이상 정지 {self.info['err']['name']}")
        if self.kind == "eval":
            LAST_RESULT.clear()
            LAST_RESULT.update(eval_result(self))
        if self.kind == "eval" and self.info.get("protocol"):
            fill_auto_later(self.info["repo_id"], delay=0)
        if self.kind == "eval" and self.info.get("repo_id"):
            # keep the control-rate summary with the eval record
            try:
                path = notes_path(self.info["repo_id"])
                note = json.loads(path.read_text())
                hz = sorted(h for _, h in self.slow)
                note["loop"] = {"slow_warnings": self.slow_total,
                                "slow_median_hz": hz[len(hz) // 2] if hz else None,
                                "slow_min_hz": hz[0] if hz else None}
                if self.info["err"]:
                    note["stop_error"] = {k: self.info["err"][k] for k in ("name", "title", "detail")}
                path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
            except (OSError, ValueError):
                pass
        if self.kind == "eval" and self.info.get("repo_id") and proc.returncode != 0:
            remove_unstarted_eval(self.info["repo_id"], self.lines)
        if self.kind in ("record", "eval") and self.info.get("repo_id"):
            # Also covers force-quit runs, where omx_run.py had no chance to do it itself
            share_with_team(DATA_HOME / self.info["repo_id"])
            if CONTROL and (DATA_HOME / self.info["repo_id"]).exists():
                # control PC: the recording / evaluation goes to the team's shared folder on the GPU laptop
                root, note, lines = DATA_HOME / self.info["repo_id"], notes_path(self.info["repo_id"]), self.lines

                def push():
                    lines.append("[omx] sync: GPU 노트북 공유 폴더로 보내는 중...")
                    sync_one(root, note, lines)
                remote.sync_later(push)
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
            if self.kind == "eval" and self.info.get("protocol"):
                protocol_attempt_start(self, int(m.group(1)))
        elif "Reset the environment" in line or "[omx] reset period" in line:
            self.info["phase"] = "원위치 시간"
            self.info.pop("home_msg", None)
            self.info["phase_started"] = time.time()
            if self.kind == "eval" and self.info.get("protocol"):
                protocol_attempt_end(self)
        elif m := re.search(r"\[omx\] loop: episode (\d+) median ([0-9.]+) p5 ([0-9.]+) min ([0-9.]+)", line):
            if self.kind == "eval" and self.info.get("protocol"):
                ep, med, p5, mn = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
                update_note(self.info["repo_id"], lambda n: n.get("attempts", {}).get(str(ep), {}).update(
                    loop={"median_hz": med, "p5_hz": p5, "min_hz": mn}))
        elif "Re-record episode" in line:
            self.info["phase"] = "다시 찍기"
        elif "Stop recording" in line:
            self.info["phase"] = "저장 중"
        elif "[omx] safe reset:" in line:
            self.info["home_msg"] = line.split("safe reset:", 1)[1].strip()
        elif m := re.search(r"\[omx\] abort: saving the episode in progress as aborted: episode (\d+) at ([0-9.]+)s mode (\w+)", line):
            self.info["aborting"] = True
            if self.kind == "eval" and self.info.get("repo_id"):
                ep, secs, mode = int(m.group(1)), float(m.group(2)), m.group(3)
                update_note(self.info["repo_id"], lambda n: proto.mark_aborted(n, ep, mode, secs))
        elif "[omx] abort:" in line:
            self.info["aborting"] = True
        elif "[omx] freeze:" in line:
            self.info["freeze_ack"] = True
            self.info["phase"] = "비상 정지 · 제자리에서 멈춤 (토크 유지)"
        elif "[omx] safe freeze:" in line:
            self.info["safe_start"] = line.split("safe freeze:", 1)[1].strip()
            if "기다립니다" in line:
                self.info["phase"] = "비상 정지 · 제자리에서 기다림 ([홈 위치로 이동])"
        elif "[omx] safe stop:" in line:
            self.info["landing"] = True
            self.info["phase"] = "천천히 내려놓는 중"
            self.info["safe_start"] = line.split("safe stop:", 1)[1].strip()
        elif "[omx] safe start:" in line:
            msg = line.split("safe start:", 1)[1].strip()
            if msg.startswith("시작하지 않습니다"):  # omx_safe_start.preflight: nothing moved; the page shows why
                self.info["refused"] = " / ".join(filter(None, [self.info.get("refused"), msg]))
            done = msg.startswith("완료") or msg.startswith("시간 초과")
            if done and self.kind == "eval" and self.info.get("protocol") and self.info.get("episode") == 0 \
                    and self.info.get("phase") == "실행 중":
                # frame where the safe start ended inside the first attempt (it is recorded in the episode)
                frame = int(round((time.time() - self.info.get("phase_started", time.time())) * 30))
                update_note(self.info["repo_id"], lambda n: n.get("attempts", {}).get("0", {}).update(safe_start_end_frame=frame))
            self.info["safe_start"] = msg
            if self.kind == "teleop":
                self.info["phase"] = "조작 중" if done else "안전 시작 중"
            elif not done:
                self.info["phase"] = "안전 시작 중"
            elif self.info.get("phase") == "안전 시작 중":  # the safe start ended inside the first episode
                self.info["phase"] = "실행 중" if self.kind == "eval" else "녹화 중"
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

    def abort(self) -> str:
        """즉시 중단 (eval): stop the policy now and return the follower to the start pose.

        SIGUSR1 makes omx_run.py leave the loop and land at once. If the safe stop has not
        started within ABORT_WAIT_S (process stuck), or on a second press, the run is killed and
        omx_return_home.py brings the arm back from the saved start pose instead.
        """
        if not self.running() or self.kind != "eval":
            raise RuntimeError("실행 중인 평가가 없습니다.")
        if self.info.get("frozen") and not self.info.get("released"):
            # after 비상 정지 the run holds the arm and waits: 즉시 중단 now means "go back"
            self.send("home")
            self.info["phase"] = "홈 위치로 이동 중"
            self.info["released"] = True
            return "release"
        if self.info.get("abort_sent"):
            if self.info.get("landing"):
                # already returning to the start pose and saving: killing now would cut the save
                # (the joint data of the whole run would be lost) - the return finishes on its own
                return "landing"
            threading.Thread(target=self._kill_and_return, args=("두 번째 즉시 중단",), daemon=True).start()
            return "kill"
        self.info["abort_sent"] = time.time()
        self.info["phase"] = "즉시 중단 · 처음 자세로 돌아가는 중"
        os.kill(self.proc.pid, signal.SIGUSR1)
        proc = self.proc

        def watchdog() -> None:
            end = time.time() + ABORT_WAIT_S
            while time.time() < end and proc.poll() is None and not self.info.get("landing"):
                time.sleep(0.2)
            if self.proc is proc and not self.info.get("landing"):
                self._kill_and_return("응답 없음" if proc.poll() is None else "복귀 없이 종료됨")

        threading.Thread(target=watchdog, daemon=True).start()
        return "abort"

    def _kill_and_return(self, why: str) -> None:
        with self.abort_lock:
            if self.kind != "eval" or self.info.get("return_started"):
                return
            self.info["return_started"] = True
            repo = self.info.get("repo_id", "")
            gripper = self.info.get("open_gripper", True)
            self.lines.append(f"[omx] abort: {why} -> 평가를 강제로 끝내고 복귀 프로그램으로 처음 자세로 돌아갑니다")
            self.kill()
            if getattr(self, "reader", None):
                self.reader.join(timeout=10)  # let the killed run release the robot lock first
            if not EVAL_HOME_FILE.exists():
                self.lines.append("[omx] abort: 처음 자세 파일이 없어(팔로워가 아직 움직이기 전) 복귀하지 않습니다")
                return
            try:
                self.start("home", [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_return_home.py"),
                                    str(EVAL_HOME_FILE), repo],
                           {"phase": "즉시 중단 · 처음 자세로 돌아가는 중", "repo_id": repo},
                           need_leader=False, env_extra={"OMX_HOME_GRIPPER": "open" if gripper else "keep"})
            except Exception as exc:
                self.lines.append(f"[omx] abort: 복귀 프로그램을 시작하지 못했습니다 ({exc})")

    def freeze(self) -> str:
        """비상 정지: an eval holds the arm where it is (torque on), keeps its records and waits for
        [홈 위치로 이동]. Anything else, or an eval that does not answer within FREEZE_WAIT_S, is
        force-quit as before (the follower also stays where it is with the torque on)."""
        if not self.running():
            return "none"
        if self.kind != "eval":
            self.kill()
            return "kill"
        if self.info.get("frozen"):
            return "frozen"
        self.info["frozen"] = time.time()
        self.info["phase"] = "비상 정지 · 멈추는 중"
        os.kill(self.proc.pid, signal.SIGUSR2)
        proc = self.proc

        def watchdog() -> None:
            end = time.time() + FREEZE_WAIT_S
            while time.time() < end and proc.poll() is None and not self.info.get("freeze_ack"):
                time.sleep(0.1)
            if self.proc is proc and proc.poll() is None and not self.info.get("freeze_ack"):
                self.lines.append("[omx] freeze: 응답이 없어 강제로 끝냅니다 (팔로워는 토크를 켠 채 제자리)")
                self.kill()
                self.info["frozen"] = time.time()  # [홈 위치로 이동] then runs omx_return_home.py

        threading.Thread(target=watchdog, daemon=True).start()
        return "freeze"

    def kill(self) -> list[str]:
        """Force-quit: SIGKILL the job's whole process group plus any leftover robot processes."""
        killed = []
        if self.running():
            self.info["phase"] = "강제 종료"
            self.info["killed"] = time.time()
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


_DISPLAY: dict = {}  # camera -> (frame key, small jpeg), shared by all viewers
_DISPLAY_LOCK = threading.Lock()


def display_jpeg(name: str, key, frame: bytes) -> bytes:
    """The picture for the browser: 480x360, quality 60 (~4x smaller). Detection uses the full frame."""
    with _DISPLAY_LOCK:
        hit = _DISPLAY.get(name)
        if hit and hit[0] == key:
            return hit[1]
    try:
        img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        small = jpg.tobytes() if ok else frame
    except Exception:
        small = frame
    with _DISPLAY_LOCK:
        _DISPLAY[name] = (key, small)
    return small


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
        """Logged-in Hugging Face user name, re-checked at most once a minute in the background (a network
        call; the status poll must not wait for it)."""
        if time.time() - self.hf_checked > 60:
            self.hf_checked = time.time()
            threading.Thread(target=self._check_hf, daemon=True).start()
        return self.hf_user

    def _check_hf(self) -> None:
        try:
            from huggingface_hub import HfApi
            self.hf_user = HfApi().whoami()["name"]
        except Exception:
            self.hf_user = None

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
        self.state["phase"] = {0: "업로드 완료", 4: "업로드 안 함"}.get(proc.returncode, "업로드 실패")
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
    # omx_run.py without a dataset = teleop with the safe start (lift, close gripper, slow approach at low power)
    cmd = [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_run.py"), *ROBOT_ARGS, *TELEOP_ARGS]
    viewer_ip = body.get("viewer_ip", "").strip()
    if viewer_ip:
        PREVIEW.stop()  # cameras go to the teleop process for Rerun
        fix_wrist_fps()
        cmd.insert(5, f"--robot.cameras={CAMS_ARG}")
        cmd += display_args(viewer_ip)
    info = {"phase": "연결 중", "with_cameras": bool(viewer_ip)}
    env = task_home_env(None, info)
    if env:  # the kill-and-return path also needs the home file
        env |= {"OMX_HOME_FILE": str(EVAL_HOME_FILE), "OMX_REPO_ID": ""}
    env = (env or stop_home_env()) | wrist_offset_env()
    JOB.start("teleop", cmd, info, env_extra=env)
    return {"ok": True}


TARGET_FILE = SETUP_FILE.parent / "target_marker"  # the marker number omx_run.py re-reads (next to robot.json)


def target_env(body: dict, info: dict) -> dict:
    """Optional target marker (record / eval): omx_run.py circles it on the front camera.
    Empty = off, the run is exactly as before."""
    marker = str(body.get("target_marker", "")).strip()
    if not marker:
        return {}
    marker = str(int(marker))
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(marker)
    info["target_marker"] = int(marker)
    return {"OMX_TARGET_MARKER": marker, "OMX_TARGET_CAMERA": "front", "OMX_TARGET_FILE": str(TARGET_FILE)}


def set_target(body: dict) -> dict:
    """Change the target marker of the running record / eval (re-read by omx_run.py within 0.5 s)."""
    marker = int(str(body.get("target_marker", "")).strip())
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(str(marker))
    if JOB.running() and JOB.kind in ("record", "eval"):
        JOB.info["target_marker"] = marker
    if PREVIEW.marker_target is not None:  # the camera check follows the new number too
        PREVIEW.set_marker_view(PREVIEW.marker_show, marker)
    return {"ok": True, "target_marker": marker}


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
    camera_gate()
    resume = (DATA_HOME / repo_id / "meta/info.json").exists()
    keywords = body.get("keywords", "").strip()
    description = body.get("description", "").strip()
    if keywords or description or not resume:
        write_notes(repo_id, keywords, description)  # used later for the Hub name, tags and page text
    PREVIEW.stop()  # record needs the cameras itself
    fix_wrist_fps()
    cmd = [
        str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_run.py"),
        *ROBOT_ARGS, f"--robot.cameras={CAMS_ARG}", *TELEOP_ARGS,
        f"--dataset.repo_id={repo_id}", f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}", f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}", "--dataset.push_to_hub=false",
        *display_args(body.get("viewer_ip", "").strip()),
    ]
    if resume:
        cmd.append("--resume=true")
    info = {"phase": "연결 중", "repo_id": repo_id, "target": episodes, "resume": resume,
            "episode_s": episode_s, "reset_s": reset_s, "upload": bool(body.get("upload"))}
    env = task_home_env(task, info)
    if env:
        env |= {"OMX_HOME_FILE": str(EVAL_HOME_FILE), "OMX_REPO_ID": repo_id}
    env = (env or stop_home_env()) | wrist_offset_env() | target_env(body, info)
    JOB.start("record", cmd, info, env_extra=env)
    return {"ok": True, "repo_id": repo_id, "resume": resume}


def find_model_remote(model_path: str) -> dict:
    """Control PC: a policy of the GPU laptop's training index (final model or an intermediate save)."""
    for r in remote.gpu_json("/api/train/runs").get("runs", []):
        if r["model_path"] == model_path:
            return r
        for c in r.get("checkpoints") or []:
            if c["model_path"] == model_path:
                return dict(r, model_path=model_path, step=c["step"])
    raise RuntimeError("GPU 노트북의 학습 기록에 있는 모델을 고르세요.")


def remote_preset(pid: str) -> dict:
    for p in remote.gpu_json("/api/presets").get("presets", []):
        if p.get("id") == pid:
            return p
    raise RuntimeError(f"평가 방식 {pid}을 GPU 노트북에서 찾지 못했습니다.")


def remote_eval_settings(local_model: Path, gpu_model: str, extra_args: list, async_env: dict, opts: dict, body: dict) -> dict:
    """Control PC: the inference settings for the GPU laptop's server, and the async engine (always on:
    "sync" = take the next chunk only when the last one is used up)."""
    if opts.get("smooth"):
        raise RuntimeError("원격 추론에서는 ACT '부드럽게 실행'(매 스텝 추론)을 쓸 수 없습니다. 출력 필터를 쓰세요.")
    overrides: dict = {}
    for a in extra_args:
        if a.startswith("--dataset.rename_map="):
            overrides["rename_map"] = json.loads(a.split("=", 1)[1])
        elif a.startswith("--policy.n_action_steps="):
            overrides["n_action_steps"] = int(a.split("=", 1)[1])
    if opts.get("diffusion_steps"):
        overrides["diffusion_steps"] = int(opts["diffusion_steps"])
    if opts.get("rtc"):
        overrides["rtc"] = {"horizon": int(body.get("rtc_horizon") or 10), "weight": float(body.get("rtc_weight") or 10.0)}
    env = dict(async_env)
    if not env:  # synchronous: the whole chunk, then the next one
        cfg = model_config(str(local_model))
        if cfg.get("type") == "diffusion":
            # its 8 actions (0.27 s) run out before the next ones come back over the network: take what the horizon covers
            chunk = int(cfg.get("horizon", 16)) - int(cfg.get("n_obs_steps", 2)) + 1
            overrides["n_action_steps"] = chunk
        else:
            chunk = int(cfg.get("n_action_steps") or cfg.get("chunk_size") or 50)
        # ask for the next chunk about 0.5 s (one network round trip) before this one runs out, so the arm does not
        # stop at every chunk boundary waiting for the GPU laptop
        threshold = min(0.9, 15 / chunk)
        env = {"OMX_ASYNC": "1", "OMX_ASYNC_CHUNK": str(chunk), "OMX_ASYNC_THRESHOLD": f"{threshold:.2f}",
               "OMX_ASYNC_AGG": "latest_only", "OMX_FPS": "30", "OMX_RTC": "0"}
    env.update(OMX_REMOTE_URL=remote.infer_url(), OMX_REMOTE_MODEL=gpu_model, OMX_REMOTE_OVERRIDES=json.dumps(overrides))
    return env


def check_model_fits(cfg: dict) -> None:
    """A model drives the robot it was trained on: 6 joints = one arm, 12 = a cooperative pair; and the same
    cameras by name."""
    shape = ((cfg.get("output_features") or {}).get("action") or {}).get("shape") or [None]
    want = 12 if ROBOT_MODE == "bimanual" else 6
    if shape[0] and shape[0] != want:
        kind = {6: "단일 로봇(팔 1개)", 12: "협동 로봇(팔 2개)"}
        raise RuntimeError(f"이 모델은 {kind.get(shape[0], f'관절 {shape[0]}개')}용입니다. 지금 구성은 "
                           f"{kind[want]}입니다. ① 탭 '로봇 구성'을 맞추거나 다른 모델을 고르세요.")
    cams = {k.split(".")[-1] for k in (cfg.get("input_features") or {}) if k.startswith("observation.images.")}
    if cams and not any(c.startswith("camera") for c in cams) and not cams <= set(CAMERAS):
        raise RuntimeError(f"이 모델은 카메라 {', '.join(sorted(cams))}로 학습했습니다. 지금 구성의 카메라: "
                           f"{', '.join(CAMERAS)}. ① 탭 '로봇 구성'에서 같은 이름으로 지정하세요.")


def find_model(model_path: str) -> dict:
    """A trained policy from the team index (or any policy folder under CKPT_HOME)."""
    if CONTROL:
        return find_model_remote(model_path)
    path = Path(model_path).resolve()
    if CKPT_HOME.resolve() not in path.parents or not (path / "model.safetensors").exists() or not (path / "config.json").exists():
        raise RuntimeError("학습 기록에 있는 모델을 고르세요.")
    runs = train_runs(False).get("runs", [])
    for r in runs:
        if Path(r["model_path"]).resolve() == path:
            return r
    # an intermediate save: <run>/checkpoints/012500/pretrained_model -> that run, at that step
    step = names.checkpoint_step(str(path))
    if step is not None:
        rdir = path.parent.parent.parent
        for r in runs:
            m = Path(r["model_path"]).resolve()
            if (m.parent if m.name == "pretrained_model" else m) == rdir:
                return dict(r, model_path=str(path), step=step)
    return {"id": path.name, "model_path": str(path), "tasks": []}


# CPU threads for policy runs. PyTorch defaults to one thread per core (24 here); its thread pool then
# splits every small CPU op of the pre/post-processing 24 ways and spin-waits, using ~2200% CPU and
# dropping the 30 Hz control loop to ~21 Hz. 1-8 threads all held 30 Hz in the benchmark (2026-10-22).
EVAL_THREADS = 4


def thread_env(threads: int) -> dict:
    n = str(threads)
    return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "OMX_TORCH_THREADS": n}


EVAL_HOME_FILE = LOG_DIR / "eval_home_pose.json"  # start pose of the running eval (for omx_return_home.py)
FREEZE_WAIT_S = 3.0  # 비상 정지: force-quit when the eval does not confirm the stop by then
ABORT_WAIT_S = 5.0  # 즉시 중단: kill and use omx_return_home.py if the safe stop has not started by then


RUN_HOME_FILE = LOG_DIR / "run_home_pose.json"  # the standard home pose of the run starting now (omx_safe_start)


def stop_home_env() -> dict:
    """Teleop / recording without a task home pose: the safe stop takes the follower slowly back to where
    the run started (30% output), opens the gripper only if it started more open, then fades out.
    A saved rest pose (끝 자세) keeps priority."""
    import omx_safe_start
    if (omx_safe_start.load_pose("rest") or {}).get("joints"):
        return {}
    return {"OMX_STOP_POSE": "home", "OMX_HOME_GRIPPER": "open"}


def task_home_env(task: str | None, info: dict) -> dict:
    """Teleop / recording with the standard home pose on (config "home_pose"): start at the task's home
    pose and end there (safe stop: home, gripper, fade out, torque off)."""
    if not omx_home.enabled():
        return {}
    code = names.task_code([task], names.load_codes()) if task else None
    home = omx_home.task_home(code) or omx_home.task_home()
    if not home:
        info["home_note"] = "작업 홈 자세가 없어 지금 자세에서 시작합니다 (① 탭에서 [다시 계산])"
        return {}
    info["home_pose"] = home
    omx_home.write_run_file(home, RUN_HOME_FILE)
    return {"OMX_HOME_POSE": str(RUN_HOME_FILE), "OMX_STOP_POSE": "home", "OMX_HOME_GRIPPER": "open"}


def home_status(model_path: str = "") -> dict:
    """Panel: the standard home pose (of a model, else the task default) and the setting."""
    out = {"enabled": omx_home.enabled(), "eval_enabled": omx_home.eval_enabled(), "freeze_return_after_s": float(omx_config.CFG.get("freeze_return_after_s") or 0),
           "tasks": omx_home.load_tasks()}
    if model_path:
        try:
            path = Path(model_path).resolve()
            if CKPT_HOME.resolve() not in path.parents:
                raise RuntimeError("학습 기록에 있는 모델이 아닙니다.")
            out["home"] = omx_home.model_home(path)
        except Exception as exc:
            out["error"] = f"홈 자세를 계산하지 못했습니다: {exc}"
    return out


def home_current() -> dict:
    """Read the follower's position now (bus only, no torque change) to compare with the home pose."""
    if JOB.running() or other_robot_processes():
        raise RuntimeError("로봇 프로그램이 실행 중이라 팔 위치를 따로 읽지 않습니다.")
    current = {}
    sides = ["left", "right"] if ROBOT_MODE == "bimanual" else [None]
    for side, (_, port) in zip(sides, FOLLOWERS):
        r = subprocess.run([str(VENV_BIN / "python"), "-c",
                            "import json,sys; sys.path.insert(0, sys.argv[1]); import omx_home; print(json.dumps(omx_home.read_follower(sys.argv[2])))",
                            str(TOOLS_DIR), port], capture_output=True, text=True, timeout=30, cwd=HOME)
        if r.returncode != 0:
            raise RuntimeError(f"팔로워({port}) 위치를 읽지 못했습니다. USB 연결을 확인하세요.")
        pos = json.loads(r.stdout.strip().splitlines()[-1])
        current.update({f"{side}_{k}": v for k, v in pos.items()} if side else pos)
    return {"current": current}


def home_return() -> dict:
    """After 비상 정지 (arm frozen in place): the user's [홈 위치로 이동] - back to the run's start pose
    (the standard home pose when "home_pose" is on), gripper, then the safe stop."""
    if JOB.running() and JOB.kind == "eval" and JOB.info.get("frozen") and not JOB.info.get("released"):
        JOB.send("home")  # the run is holding the arm and waiting for this
        JOB.info["phase"] = "홈 위치로 이동 중"
        JOB.info["released"] = True
        return {"ok": True, "mode": "release"}
    if JOB.running() or other_robot_processes():
        raise RuntimeError("로봇 프로그램이 아직 실행 중입니다.")
    if not EVAL_HOME_FILE.exists():
        raise RuntimeError("돌아갈 홈 자세 기록이 없습니다.")
    data = json.loads(EVAL_HOME_FILE.read_text())
    JOB.start("home", [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_return_home.py"), str(EVAL_HOME_FILE),
                       data.get("repo_id", "")],
              {"phase": "홈 위치로 이동 중", "repo_id": data.get("repo_id", "")},
              need_leader=False, env_extra={"OMX_HOME_GRIPPER": "open"})
    return {"ok": True, "mode": "return_home"}


def home_env(reset_home: bool, open_gripper: bool, repo_id: str) -> dict:
    """Eval only: on stop (and optionally in every reset period) go back to the pose the run started in."""
    return {"OMX_STOP_POSE": "home", "OMX_RESET_HOME": "1" if reset_home else "0",
            "OMX_HOME_GRIPPER": "open" if open_gripper else "keep",
            "OMX_HOME_FILE": str(EVAL_HOME_FILE), "OMX_REPO_ID": repo_id}


def model_config(model_path: str) -> dict:
    try:
        return json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def policy_eval_options(model_path: str, body: dict) -> tuple[list[str], dict, dict]:
    """Policy-specific omx_run.py arguments, the async/RTC environment, and what was used.

    - ACT: smooth = temporal ensembling (every step), only without async execution.
    - Diffusion: inference steps (DDIM with fewer denoising steps, no retraining).
    - SmolVLA: our camera names -> the model's camera1..3 (as in training), RTC with async.
    - async (any policy): omx_async.py keeps executing queued actions while the next chunk is inferred.
    """
    cfg = model_config(model_path)
    ptype = cfg.get("type", "")
    args, env = [], {}
    use_async = bool(body.get("async_mode"))
    smooth = bool(body.get("smooth")) and ptype == "act" and not use_async
    opts = {"policy_type": ptype, "smooth": smooth, "async_mode": use_async}
    if smooth:
        # ACT temporal ensembling: predict every step and average overlapping chunks (no jumps every 100 steps)
        args += ["--policy.n_action_steps=1", "--policy.temporal_ensemble_coeff=0.01"]
    if ptype == "diffusion" and body.get("diffusion_steps"):
        steps = int(body["diffusion_steps"])
        if not 1 <= steps <= int(cfg.get("num_train_timesteps") or 100):
            raise RuntimeError("추론 단계 수는 1~100 사이로 정하세요.")
        args += ["--policy.noise_scheduler_type=DDIM", f"--policy.num_inference_steps={steps}"]
        opts["diffusion_steps"] = steps
    feats = cfg.get("input_features") or {}
    cams = [k for k in feats if k.startswith("observation.images.camera")]
    if cams and not any(k.endswith((".front", ".wrist")) for k in feats):
        # fine-tuned SmolVLA keeps the pretrained input names: feed our cameras the same way as in training
        order = sorted(CAMERAS, key=lambda n: ({"front": 0, "wrist": 1, "wrist_left": 1, "wrist_right": 2}.get(n, 9), n))
        rename = {f"observation.images.{n}": f"observation.images.camera{i + 1}" for i, n in enumerate(order[:3])}
        args.append("--dataset.rename_map=" + json.dumps(rename))
    if use_async:
        chunk = int(body.get("actions_per_chunk") or 0)
        limit = int(cfg.get("chunk_size") or cfg.get("horizon") or 100)
        if ptype == "diffusion":
            # Diffusion returns n_action_steps actions per inference; allow up to what its horizon covers
            limit = int(cfg.get("horizon", 16)) - int(cfg.get("n_obs_steps", 2)) + 1
            chunk = min(chunk or limit, limit)
            args.append(f"--policy.n_action_steps={chunk}")
        chunk = min(chunk or limit, limit)
        threshold = float(body.get("chunk_size_threshold", 0.5))
        agg = body.get("aggregate_fn") or "weighted_average"
        if not 0 <= threshold <= 1 or agg not in ("weighted_average", "average", "latest_only"):
            raise RuntimeError("chunk_size_threshold는 0~1, 합치는 방법은 weighted_average·average·latest_only 중 하나입니다.")
        rtc = bool(body.get("rtc")) and ptype in ("smolvla", "pi0", "pi05")
        env = {"OMX_ASYNC": "1", "OMX_ASYNC_CHUNK": str(chunk), "OMX_ASYNC_THRESHOLD": str(threshold),
               "OMX_ASYNC_AGG": agg, "OMX_FPS": "30", "OMX_RTC": "1" if rtc else "0",
               "OMX_RTC_HORIZON": str(int(body.get("rtc_horizon") or 10)),
               "OMX_RTC_WEIGHT": str(float(body.get("rtc_weight") or 10.0))}
        opts.update(actions_per_chunk=chunk, chunk_size_threshold=threshold, aggregate_fn=agg, rtc=rtc)
    # output filter (any policy): low-pass on the joint targets before they are sent and recorded
    level = body.get("output_filter") or ""
    if level:
        if level not in FILTER_ALPHA:
            raise RuntimeError("출력 부드럽게 세기는 약·중·강 중 하나입니다.")
        env.update(OMX_FILTER_ALPHA=str(FILTER_ALPHA[level]), OMX_FILTER_GRIPPER="light" if body.get("filter_gripper") else "off")
        opts.update(output_filter=level, filter_alpha=FILTER_ALPHA[level], filter_gripper=bool(body.get("filter_gripper")))
    return args, env, opts


# output filter strength -> moving-average weight of the new target (lag about 0.05 / 0.1 / 0.2 s at 30 Hz)
FILTER_ALPHA = {"약": 0.4, "중": 0.25, "강": 0.14}


# ---- Evaluation protocol v2 (position presets, per-attempt records); see omx_protocol.py ----
def update_note(repo_id: str, fn) -> dict:
    """Read-modify-write the eval notes under the protocol lock (the reader thread and API share it)."""
    with proto.LOCK:
        path = notes_path(repo_id)
        try:
            note = json.loads(path.read_text())
        except (OSError, ValueError):
            note = {}
        fn(note)
        path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
    if CONTROL:
        push_note_later(path)
    return note


_PUSH: dict = {}


def push_note_later(path: Path) -> None:
    """Control PC: copy an edited note to the GPU laptop a moment later (several marks in a row: one copy)."""
    _PUSH[str(path)] = time.time()

    def run(key=str(path)):
        time.sleep(2)
        if time.time() - _PUSH.get(key, 0) >= 1.9:
            remote.sync_note(Path(key))
    threading.Thread(target=run, daemon=True).start()


def protocol_target(job) -> None:
    """What the panel draws: the current attempt's target while running, otherwise the next free one."""
    note = read_notes(job.info["repo_id"])
    if job.info.get("phase") == "실행 중":
        a = note.get("attempts", {}).get(str(job.info.get("episode")))
        if a:
            job.info["proto_target"] = {"id": a["position_id"], "x": a["target_px"][0], "y": a["target_px"][1], "now": True}
            return
    free = proto.free_positions(note)
    # not "target": that key is the number of attempts the panel shows ("시도 3 / 10")
    job.info["proto_target"] = {"id": free[0]["id"], "x": free[0]["x"], "y": free[0]["y"], "now": False} if free else None
    job.info["proto_left"] = len(free)


def protocol_attempt_start(job, episode: int) -> None:
    started = time.time()
    note = update_note(job.info["repo_id"], lambda n: proto.begin_attempt(n, episode, None))
    a = note.get("attempts", {}).get(str(episode))
    if a:
        job.info.setdefault("attempt_map", {})[str(episode)] = a["position_id"]

    def capture():
        # the start position from a picture taken after this attempt began (not a frame left over
        # from before); replaced later by the saved episode's first frame (omx_protocol.fill_auto)
        for _ in range(30):
            d = job.info.get("detect") or {}
            if d.get("t", 0) > started + 0.3:
                px = d.get("px")
                update_note(job.info["repo_id"], lambda n: proto.set_start(n, episode, px, "live"))
                return
            time.sleep(0.1)
    if job.info.get("protocol", {}).get("detector", "none") != "none":
        threading.Thread(target=capture, daemon=True).start()
    if str(episode) not in note.get("attempts", {}):
        job.lines.append("[omx] protocol: 남은 위치가 없어 평가를 끝냅니다")
        threading.Thread(target=lambda: job.send("stop"), daemon=True).start()
    protocol_target(job)
    fill_auto_later(job.info["repo_id"])  # the previous episode is saved by now


def protocol_attempt_end(job) -> None:
    protocol_target(job)
    if job.info.get("proto_left") == 0 and not job.info.get("protocol_done"):
        # every position has an attempt (void ones were given back): end after this one is saved
        job.info["protocol_done"] = True
        job.lines.append("[omx] protocol: 모든 위치를 시도했습니다. 저장하고 끝냅니다 (무효로 표시하면 그 위치를 다시 할 수 있게 다음 평가에서 남음)")
        threading.Thread(target=lambda: job.send("stop"), daemon=True).start()


def fill_auto_later(repo_id: str, delay: float = 2.0) -> None:
    """Estimate failure stages for attempts whose episode is saved (in the background)."""
    def run():
        time.sleep(delay)
        try:
            note = read_notes(repo_id)
            if proto.fill_auto(note, DATA_HOME, repo_id):
                def merge(n):
                    for ep, a in note.get("attempts", {}).items():
                        cur = n.setdefault("attempts", {}).setdefault(ep, {})
                        for k in ("grasp", "fail_stage_auto", "train_rec", "train_rec_priority", "train_rec_reason",
                                  "train_rec_thresholds", "start_px", "start_offset_px", "start_px_source"):
                            if k in a:
                                cur[k] = a.get(k)
                        cur.setdefault("train_candidate", False)
                        if cur.get("result") == "fail" and cur.get("fail_stage_source") != "user":
                            cur["fail_stage"], cur["fail_stage_source"] = a.get("fail_stage_auto"), "auto"
                update_note(repo_id, merge)
        except Exception as exc:  # estimates are optional
            print(f"[omx] protocol: 실패 단계 추정 건너뜀 ({exc})", flush=True)
    threading.Thread(target=run, daemon=True).start()


def detect_loop(job, proc, detector: str, camera: str, roi: list | None = None) -> None:
    """Where the object is in the live view (0.5 s), for the target overlay and the attempt start."""
    import omx_vision

    live = Path(f"{LIVE}{camera}.jpg")
    recent: collections.deque = collections.deque(maxlen=3)  # median of the last 3 readings: steady display
    while job.proc is proc and proc.poll() is None:
        try:
            if live.exists() and time.time() - live.stat().st_mtime < 2:
                px = omx_vision.detect_jpeg(detector, live.read_bytes(), roi)
                recent.append(px)
                found = [p for p in recent if p]
                med = [int(statistics.median(p[0] for p in found)), int(statistics.median(p[1] for p in found))] \
                    if len(found) * 2 > len(recent) else None
                job.info["detect"] = {"px": med, "t": time.time()}
        except Exception:
            pass
        time.sleep(0.25)


def remove_unstarted_eval(repo_id: str, lines: list | None = None) -> bool:
    """A run that failed before its first episode (model loading, cameras) leaves an eval folder with no
    data in it: move that folder to ~/il_ws/omx_trash/<date>/, keep the note marked start_failed."""
    root = DATA_HOME / repo_id
    if root.is_symlink() or not root.is_dir() or not root.name.startswith("eval_") or DATA_HOME.resolve() not in root.resolve().parents:
        return False
    if any(p.is_file() for sub in ("data", "videos", "images") for p in (root / sub).rglob("*")):
        return False  # something was recorded: keep it
    # team rule: nothing under the data folders is deleted by the panel; it goes to the trash, where a
    # person deletes it after checking
    trash = HOME / "il_ws" / "omx_trash" / dt.datetime.now().strftime("%Y%m%d") / repo_id
    trash.parent.mkdir(parents=True, exist_ok=True)
    if trash.exists():
        trash = trash.with_name(f"{trash.name}_{dt.datetime.now():%H%M%S}")
    shutil.move(str(root), str(trash))
    try:
        update_note(repo_id, lambda n: n.update(start_failed=True))
    except Exception:
        pass
    if lines is not None:
        lines.append(f"[omx] 첫 시도 전에 끝나 빈 평가 폴더를 휴지통으로 옮겼습니다: {trash}")
    return True


# ---- armed: a control-tower (MQTT) command may start an evaluation only while a person has armed the panel
ARMED = {"on": False, "since": None, "off_reason": "패널 시작"}  # off after every panel restart
EPISODE_END_FILE = LOG_DIR / "episode_end.json"  # omx_run.py: the arms at the end of each episode (eval)
EVAL_HOME_POSE: dict = {}  # the running / last eval's model home pose
LAST_RESULT: dict = {}  # how the last evaluation ended (for the control tower), until the next one ends


def set_armed(on: bool, reason: str = "") -> dict:
    """Turned on only from the panel screen; any stop, abnormal end or restart turns it off."""
    if on and not ARMED["on"]:
        ARMED.update(on=True, since=dt.datetime.now().isoformat(timespec="seconds"), off_reason="")
    elif not on and ARMED["on"]:
        ARMED.update(on=False, since=None, off_reason=reason or "패널에서 끔")
    return armed_status()


def armed_status() -> dict:
    return {"armed": ARMED["on"], "armed_since": ARMED["since"], "armed_off_reason": ARMED["off_reason"]}


def arm_states(state: dict, home: dict | None) -> dict:
    """Per arm (set name): joints at the end of the last episode, gripper, and whether it is at the model's
    home pose (omx_home.diff, gripper left out). None where it was not measured: never a guess."""
    out = {}
    for side, name in ARM_NAMES.items():
        pre = f"{side}_" if side else ""
        joints = {k[len(pre):-4]: v for k, v in state.items() if k.startswith(pre) and k.endswith(".pos")}
        g = joints.get("gripper")
        grip = None if g is None else "open" if g >= omx_home.GRIP_OPEN else "closed" if g < omx_home.GRIP_CLOSED else "half"
        at_home, far = None, []
        hj = {k: v for k, v in ((home or {}).get("joints") or {}).items() if k.startswith(pre) and not k.endswith("gripper")}
        if hj and joints:
            rows = omx_home.diff({"joints": hj, "std": home.get("std") or {}}, {k: state.get(k + ".pos") for k in hj})
            if all(r["current"] is not None for r in rows):
                far = [r["joint"][len(pre):] for r in rows if r["flag"]]
                at_home = not far
        out[name] = {"side": side or None, "joints": joints or None, "gripper": g, "gripper_state": grip,
                     "gripper_open": None if grip is None else grip == "open", "at_home": at_home, "far_joints": far}
    return out


def eval_result(job) -> dict:
    """/api/status last_result: how the evaluation that just ended went, and where the arms were when its last
    episode ended (before the safe stop moves them to the start pose and closes the grippers)."""
    info = job.info
    try:
        end = json.loads(EPISODE_END_FILE.read_text())
        if end.get("repo_id") != info.get("repo_id"):
            end = None
    except (OSError, ValueError):
        end = None
    home = EVAL_HOME_POSE.get("home") if EVAL_HOME_POSE.get("repo_id") == info.get("repo_id") else None
    stopped = ("즉시 중단" if info.get("abort_sent") else "비상 정지" if info.get("frozen")
               else "강제 종료" if info.get("killed") else "중지" if info.get("stopping") else None)
    rc, err = info.get("returncode"), info.get("err")
    arms = arm_states(end.get("state") or {}, home) if end else {}
    checks = [a["at_home"] for a in arms.values()]
    return {"kind": "eval", "source": info.get("source", "panel"), "repo_id": info.get("repo_id"),
            "started_at": dt.datetime.fromtimestamp(job.started).isoformat(timespec="seconds"),
            "ended_at": dt.datetime.now().isoformat(timespec="seconds"), "returncode": rc,
            "exit_ok": rc == 0 and not err and stopped is None, "stopped_by": stopped, "err": err,
            "target_marker": info.get("target_marker"),
            "last_episode": {k: end.get(k) for k in ("episode", "at", "aborted")} if end else None,
            "arms": arms or None,
            "home_check": "unknown" if not checks or None in checks else "ok" if all(checks) else "far"}


def start_eval(body: dict) -> dict:
    run = find_model(body.get("model_path", ""))
    task = str(body.get("task") or (run.get("tasks") or [""])[0]).strip()
    if not task:
        raise RuntimeError("작업 문장을 입력하세요. 학습 데이터와 같은 문장을 씁니다.")
    if CONTROL:
        preset = remote_preset(body["preset_id"]) if body.get("preset_id") else None
    else:
        preset = proto.load_preset(body["preset_id"]) if body.get("preset_id") else None
    episodes = int(body.get("episodes") or (len(proto.eval_positions(preset)) if preset else 10))
    episode_s = float(body.get("episode_s", 30))
    reset_s = float(body.get("reset_s", 15))
    if not 1 <= episodes <= 100 or not 3 <= episode_s <= 600 or not 0 <= reset_s <= 300:
        raise RuntimeError("에피소드 1~100개, 시도 3~600초, 원위치 0~300초 사이로 정하세요.")
    threads = int(body.get("threads") or EVAL_THREADS)
    reset_home, open_gripper = bool(body.get("reset_home", True)), bool(body.get("open_gripper", True))
    if not 1 <= threads <= (os.cpu_count() or 24):
        raise RuntimeError(f"CPU 스레드 수는 1~{os.cpu_count() or 24} 사이로 정하세요.")
    camera_gate()
    local_model = remote.fetch_model(run["model_path"]) if CONTROL else Path(run["model_path"])
    check_model_fits(model_config(str(local_model)))
    extra_args, async_env, opts = policy_eval_options(str(local_model), body)
    if CONTROL:
        async_env = remote_eval_settings(local_model, run["model_path"], extra_args, async_env, opts, body)
        extra_args = [a for a in extra_args if a.startswith("--dataset.")] + ["--policy.device=cpu"]
        opts["remote_inference"] = remote.HOST
    home = None
    if omx_home.eval_enabled():  # first move to the model's home pose (median of its training data's first frames)
        try:
            if CONTROL:
                home = remote.gpu_json("/api/home?model=" + urllib.parse.quote(run["model_path"])).get("home")
                if not home:
                    raise RuntimeError("GPU 노트북이 홈 자세를 보내지 않았습니다")
            else:
                home = omx_home.model_home(run["model_path"])
        except Exception as exc:
            if omx_home.enabled():
                raise RuntimeError(f"모델의 홈 자세를 계산하지 못했습니다({exc}). 설정 home_pose를 끄면 지금처럼 시작합니다.")
            home = None  # e.g. the training data is gone: start from where the arm is, as before
    # eval_{model display name}__{p10 | free}_{run settings}_{MMDD_HHMM}; _2, _3 ... when the name is taken
    # (a second start in the same minute, or a folder left by a run that failed to start)
    step = run.get("step")  # set when an intermediate save was chosen
    base = names.eval_name(run.get("display") or names.display_name(run["id"]), preset, opts,
                           f"{dt.datetime.now():%m%d_%H%M}", step)
    repo_id, k = f"{USER}/{base}", 2
    while (DATA_HOME / repo_id).exists() or notes_path(repo_id).exists():
        repo_id, k = f"{USER}/{base}_{k}", k + 1
    path = notes_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "eval", "model": run["model_path"], "run_id": run["id"],
                                "model_display": names.model_label(run.get("display") or names.display_name(run["id"]), step, run.get("steps")),
                                "run_number": run.get("number"), "task": task, "results": {},
                                "train_step": step or run.get("steps"), "final_steps": run.get("steps"),
                                "smooth": bool(body.get("smooth")), "threads": threads,
                                "reset_home": reset_home, "open_gripper": open_gripper,
                                "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2))
    PREVIEW.stop()  # the policy run needs the cameras itself
    fix_wrist_fps()
    EVAL_HOME_FILE.unlink(missing_ok=True)  # written again by this run before its first move
    EPISODE_END_FILE.unlink(missing_ok=True)  # written by this run at the end of each episode
    cmd = [
        str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_run.py"),
        *ROBOT_ARGS, f"--robot.cameras={CAMS_ARG}",
        f"--policy.path={local_model}",
        f"--dataset.repo_id={repo_id}", f"--dataset.single_task={task}",
        f"--dataset.num_episodes={episodes}", f"--dataset.episode_time_s={episode_s}",
        f"--dataset.reset_time_s={reset_s}", "--dataset.push_to_hub=false",
        *display_args(body.get("viewer_ip", "").strip()),
    ]
    cmd += extra_args
    note = json.loads(path.read_text())
    note.update(opts)
    if home:
        note["home_pose"] = home
    if preset:
        note["protocol"] = proto.protocol_snapshot(preset)
        note["attempts"] = {}
        # a few spare episodes so void attempts can be retried; the run stops once every position is done
        spare = min(max(int(body.get("spare", 3)), 0), 10)
        cmd = [a if not a.startswith("--dataset.num_episodes=") else f"--dataset.num_episodes={episodes + spare}" for a in cmd]
    eval_info = {"phase": "모델 불러오는 중", "repo_id": repo_id, "target": episodes,
                 "episode_s": episode_s, "reset_s": reset_s, "model": run["id"],
                 "smooth": opts["smooth"], "threads": threads, "reset_home": reset_home,
                 "open_gripper": open_gripper, **opts,
                 "protocol": note.get("protocol"), "source": "mqtt" if body.get("source") == "mqtt" else "panel"}
    tenv = target_env(body, eval_info)
    if tenv:
        note["target_marker"] = eval_info["target_marker"]
    path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
    JOB.start("eval", cmd, eval_info,
              need_leader=False, env_extra=thread_env(threads) | home_env(reset_home, open_gripper, repo_id) | async_env
              | ({"OMX_HOME_POSE": str(omx_home.write_run_file(home, RUN_HOME_FILE))} if home else {}) | tenv
              | {"OMX_EPISODE_END": str(EPISODE_END_FILE)})
    EVAL_HOME_POSE.clear()
    EVAL_HOME_POSE.update(repo_id=repo_id, home=home)
    if preset:
        protocol_target(JOB)
        if preset["detector"] != "none":
            threading.Thread(target=detect_loop, args=(JOB, JOB.proc, preset["detector"], preset.get("camera", "front"),
                                                       preset.get("detect_roi")), daemon=True).start()
    return {"ok": True, "repo_id": repo_id}


def eval_list() -> list[dict]:
    """Policy runs (eval_ datasets) of every account, newest first, with marked results."""
    items = []
    aliases = names.load_aliases()
    steps = names.run_steps()
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
                      "model_display": names.model_label(names.display_name(n.get("run_id"), aliases), names.train_amount(n, steps),
                                                         n.get("final_steps") or steps.get(n.get("run_id"))) if n.get("run_id") else "",
                      "start_failed": bool(n.get("start_failed")),
                      "train_step": names.train_amount(n, steps), "final_steps": n.get("final_steps") or steps.get(n.get("run_id")),
                      "threads": n.get("threads"), "smooth": n.get("smooth"), "loop": n.get("loop"),
                      "policy_type": n.get("policy_type"), "async_mode": n.get("async_mode"), "rtc": n.get("rtc"),
                      "diffusion_steps": n.get("diffusion_steps"),
                      "protocol": n.get("protocol"), "attempts": n.get("attempts"),
                      "voids": sum(1 for a in (n.get("attempts") or {}).values() if a.get("result") == "void"),
                      "aborted": sum(1 for v in res.values() if v == "aborted"),
                      "aborted_info": n.get("aborted") or {}, "output_filter": n.get("output_filter"),
                      "chunk_size_threshold": n.get("chunk_size_threshold"),
                      "task": n.get("task"), "started": n.get("started"), "episodes": total, "results": res,
                      "success": sum(1 for v in res.values() if v == "success"),
                      "marked": sum(1 for v in res.values() if v in ("success", "fail"))})
    return sorted(items, key=lambda x: x.get("started") or "", reverse=True)


def eval_mark(body: dict) -> None:
    repo = body.get("repo_id", "")
    if not notes_path(repo).exists():
        raise RuntimeError("평가 기록을 찾을 수 없습니다.")
    kwargs = {k: body[k] for k in ("result", "fail_stage", "train_candidate") if k in body}
    if "note" in body:
        kwargs["text"] = body["note"]
    note = update_note(repo, lambda n: proto.mark(n, int(body.get("episode")), **kwargs))
    if JOB.running() and JOB.kind == "eval" and JOB.info.get("repo_id") == repo and note.get("protocol"):
        protocol_target(JOB)  # a void attempt gives its position back
    if "result" in body:
        # refresh the team table in the background so the "평가" column follows the marks
        subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_index.py")], cwd=HOME,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                         env=dict(os.environ, HF_LEROBOT_HOME=str(DATA_HOME)))
        if body.get("result") == "fail":
            fill_auto_later(repo, delay=0)


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


REF_TIME_S = 1.0  # the training view: 1 s into each episode (the first frames can still be dark)
REF_EPISODES = 7  # median over this many episodes, spread over the dataset: arms and objects drop out
REF_CACHE: dict = {}  # (repo, model, cam) -> {"jpg", "kp", "des", "edges"}
ALIGN_PTS = ((320, 240), (40, 40), (600, 40), (40, 440), (600, 440))  # centre and corners of a 640x480 view


def real_episodes(root: Path, episodes: list[int]) -> list[int]:
    """The real (recorded) ones among these episodes. A real + sim mix says which are real (meta/omx_mix.json
    "real_episodes": "0..28" or a list) or which are sim (the episode keys of meta/omx_sim_physics.json); a
    name with "sim" in it and neither file is refused: its sim views are renders, not this room."""
    try:
        r = json.loads((root / "meta/omx_mix.json").read_text())["real_episodes"]
        if isinstance(r, list):
            real = {int(x) for x in r}
        else:
            a, _, b = str(r).partition("..")
            real = set(range(int(a), int(b or a) + 1))
        return [e for e in episodes if e in real]
    except (OSError, ValueError, KeyError):
        pass
    try:
        sim = {int(k) for k in json.loads((root / "meta/omx_sim_physics.json").read_text()) if str(k).isdigit()}
        return [e for e in episodes if e not in sim]
    except (OSError, ValueError):
        pass
    if "sim" in root.name.lower():
        raise RuntimeError(f"{root.name}: 실물 에피소드를 알 수 없어(가상 데이터가 섞였을 수 있음) 기준 화면을 만들지 않습니다. 실물 녹화 데이터셋을 기준으로 고르세요.")
    return episodes


def reference_frame(repo: str, cam: str = "front", model: str = "") -> bytes:
    """The training view of one camera as JPEG: the median of the frame REF_TIME_S into up to REF_EPISODES
    episodes of a dataset, or of the dataset a model was trained on (its episodes only). Cached on disk."""
    import av
    import numpy as np
    import pandas as pd

    episodes = None
    if model:
        d = json.loads((Path(model) / "train_config.json").read_text())["dataset"]
        repo, episodes = d["repo_id"], d.get("episodes")
    root = dataset_dir(repo)
    files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    rows = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
    if episodes is not None:
        rows = rows[rows.episode_index.isin(episodes)]
    rows = rows[rows.episode_index.isin(real_episodes(root, [int(e) for e in rows.episode_index]))]
    if not len(rows):
        raise RuntimeError("기준 장면을 만들 실물 에피소드가 없습니다.")
    pick = sorted({int(round(k)) for k in np.linspace(0, len(rows) - 1, min(REF_EPISODES, len(rows)))})
    used = "-".join(str(int(e)) for e in rows.iloc[pick].episode_index)
    stamp = int(max(f.stat().st_mtime for f in files))
    cache = LOG_DIR / "ref_cache" / f"{repo.replace('/', '__')}__{cam}__ep{used}_{stamp}.jpg"
    if cache.exists():
        return cache.read_bytes()
    info = json.loads((root / "meta/info.json").read_text())
    key, frames = f"observation.images.{cam}", []
    for _, row in rows.iloc[pick].iterrows():
        path = root / info["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                                file_index=int(row[f"videos/{key}/file_index"]))
        t = float(row[f"videos/{key}/from_timestamp"]) + min(REF_TIME_S, float(row["length"]) / 30 - 0.1)
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            c.seek(int(max(t - 1, 0) / st.time_base), stream=st, backward=True)
            for fr in c.decode(st):
                if fr.time is not None and fr.time >= t - 1e-3:
                    frames.append(fr.to_ndarray(format="bgr24"))
                    break
    if not frames:
        raise RuntimeError("기준 장면을 읽지 못했습니다.")
    if len(frames) >= 3:  # keep the views that agree with the most typical one (drops odd or augmented episodes)
        v = np.stack([camauto.view_vector(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames])
        sim = v @ v.T / v.shape[1]
        mid = int(np.argmax(np.median(sim, axis=1)))
        frames = [f for f, x in zip(frames, sim[mid]) if x >= 0.5]
    ok, jpg = cv2.imencode(".jpg", np.median(np.stack(frames), axis=0).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(jpg.tobytes())
    return jpg.tobytes()


def reference_entry(repo: str, model: str, cam: str) -> dict:
    """The training view with its ORB keypoints and red edge overlay, fetched once (from the GPU laptop on a
    control PC, where the training data is)."""
    import numpy as np

    k = (repo, model, cam)
    if k not in REF_CACHE:
        if CONTROL and (model or not (DATA_HOME / repo).exists()):
            q = urllib.parse.urlencode({"repo": repo, "model": model, "cam": cam})
            status, _, body = remote.gpu_request("GET", "/api/reference?" + q, None, timeout=120)
            if status != 200:
                raise RuntimeError(json.loads(body or b"{}").get("error") or "GPU 노트북에서 기준 장면을 받지 못했습니다.")
        else:
            body = reference_frame(repo, cam, model)
        gray = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
        kp, des = cv2.ORB_create(3000).detectAndCompute(gray, None)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
        n, lab, st, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        keep = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 40])  # drop specks: outlines only
        line = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # 3 px wide
        halo = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        rgba = np.zeros(gray.shape + (4,), np.uint8)
        rgba[halo] = (0, 0, 0, 200)  # BGRA: a dark rim so the red reads on white shelves too
        rgba[line] = (30, 30, 255, 255)
        # long straight edges (shelf ledges, table and tape edges): thick cyan, the easiest to line up by eye
        segs = cv2.HoughLinesP(keep.astype(np.uint8) * 255, 1, np.pi / 180, 60, minLineLength=70, maxLineGap=6)
        for x1, y1, x2, y2 in (segs[:, 0] if segs is not None else []):
            cv2.line(rgba, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 230), 9, cv2.LINE_AA)
        for x1, y1, x2, y2 in (segs[:, 0] if segs is not None else []):
            cv2.line(rgba, (int(x1), int(y1)), (int(x2), int(y2)), (255, 230, 0, 255), 5, cv2.LINE_AA)
        REF_CACHE[k] = {"jpg": body, "kp": kp, "des": des, "edges": cv2.imencode(".png", rgba)[1].tobytes()}
    return REF_CACHE[k]


def alignment(repo: str, model: str = "", cam: str = "front") -> dict:
    """How far the live view of one camera sits from its training view (ORB + RANSAC similarity): centre shift,
    rotation, scale, largest corner shift (px), the shift of the centre and the four corners, matches kept."""
    import numpy as np

    live = PREVIEW.raw.get(cam) or PREVIEW.frames.get(cam)
    if not live:
        raise RuntimeError("카메라를 먼저 켜세요.")
    ref = reference_entry(repo, model, cam)
    gray = cv2.imdecode(np.frombuffer(live, np.uint8), cv2.IMREAD_GRAYSCALE)
    kp, des = cv2.ORB_create(3000).detectAndCompute(gray, None)
    if ref["des"] is None or des is None or len(kp) < 20:
        return {"ok": False, "msg": "특징점이 부족합니다. 조명이나 화면을 가린 물체를 확인하세요."}
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(ref["des"], des), key=lambda x: x.distance)[:700]
    if len(m) < 20:
        return {"ok": False, "msg": "학습 화면과 겹치는 부분이 거의 없습니다. 겹쳐 보이는 빨간 윤곽을 보며 카메라를 크게 옮기세요."}
    pa = np.float32([ref["kp"][x.queryIdx].pt for x in m])
    pb = np.float32([kp[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, ransacReprojThreshold=2.0)
    if M is None:
        return {"ok": False, "msg": "위치를 계산하지 못했습니다."}
    pts = np.float32(ALIGN_PTS)
    d = pts @ M[:, :2].T + M[:, 2] - pts
    return {"ok": True, "dx": round(float(d[0, 0]), 1), "dy": round(float(d[0, 1]), 1),
            "rot": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2), "scale": round(float(np.hypot(M[0, 0], M[1, 0])), 3),
            "corner": round(float(np.abs(d).max()), 1), "inliers": int(inl.sum()),
            "vectors": [[int(p[0]), int(p[1]), round(float(v[0]), 1), round(float(v[1]), 1)] for p, v in zip(pts, d)]}


def save_view_refs() -> dict:
    """The live views of the fixed cameras become the camera check's reference pictures (cam_refs), as if the
    setup had just been saved: after aligning to the training view, the daily check follows that view."""
    import numpy as np

    saved = []
    for name in CAMERAS:
        live = PREVIEW.raw.get(name) or PREVIEW.frames.get(name)
        if name.startswith("wrist") or not live:
            continue
        gray = cv2.imdecode(np.frombuffer(live, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray.mean() < camauto.DARK:
            continue
        CAM_REFS.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(camauto.ref_path(CAM_REFS, ROBOT_MODE, name)), cv2.resize(gray, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(name)
    if not saved:
        raise RuntimeError("카메라를 먼저 켜세요.")
    return {"ok": True, "saved": saved}


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
        "ports": {p: Path(p).exists() for _, p in FOLLOWERS + LEADERS},
        "cameras": {n: Path(p).exists() for n, p in CAMERAS.items()},
        "mode": ROBOT_MODE,
        "arm_labels": {n: l for (n, _, _), l in zip(arms_to_check(), [l for l, _ in FOLLOWERS + LEADERS])},
        "camera_labels": CAMERA_LABELS,
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

    def _forward(self, method: str, raw: bytes | None = None) -> None:
        """Control PC: pass the request to the GPU laptop's panel and its answer back."""
        try:
            status, ctype, body = remote.gpu_request(method, self.path, raw, timeout=120)
        except Exception as exc:
            return self._json({"error": str(exc)}, 502)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        CLIENT["seen"] = time.time()
        if CONTROL:
            if remote.proxied("GET", self.path):
                return self._forward("GET")
            if self.path.startswith(("/api/dataset?", "/api/reference?")):  # /api/align: the cameras are here
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                repo = q.get("repo", [""])[0]
                if q.get("model", [""])[0] or repo and not (DATA_HOME / repo).exists():  # training data: GPU laptop
                    return self._forward("GET")
            if self.path.startswith("/api/home?") and "model=" in self.path:
                # the model's home pose is computed there (training data); whether to use it is this PC's setting
                try:
                    out = remote.gpu_json(self.path)
                except Exception as exc:
                    out = {"error": str(exc)}
                return self._json(home_status() | {k: v for k, v in out.items() if k in ("home", "error")})
        if self.path == "/api/datasets":
            return self._json({"datasets": list_datasets(), "legacy_note": data_notes.load().get("legacy_note", "")})
        if self.path == "/api/errors":
            return self._json({"codes": omx_errors.all_codes()})
        if self.path == "/api/robot_setup":
            return self._json(setup_status())
        if self.path.startswith("/api/camera_probe?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body = camera_probe(q.get("path", [""])[0])
            except Exception as exc:
                return self._json({"error": str(exc)}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if self.path == "/":
            body = (TOOLS_DIR / "omx_panel.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/motor_output":
            self._json(motor_output())
        elif self.path == "/api/status":
            self._json({
                "user": USER,
                "version": VERSION,
                "gpu_name": CFG.get("gpu_name"),
                "ckpt_home": str(CKPT_HOME),
                "preview": PREVIEW.running,
                "markers": PREVIEW.markers if PREVIEW.running else {},
                **armed_status(),
                "last_result": LAST_RESULT or None,
                "camera_watch": camera_watch(),
                "live": JOB.running() and JOB.kind in ("record", "eval"),
                "brightness": PREVIEW.brightness,
                "cam_stats": camera_stats(),
                "robot_mode": ROBOT_MODE,
                "cameras": [{"name": n, "label": CAMERA_LABELS.get(n, n)} for n in CAMERAS],
                "job": JOB.status(),
                "cpu": cpu_status(),
                "log": list(JOB.lines)[-120:],
                "datasets": merged_datasets() if CONTROL else list_datasets(),
                "data_legacy_note": remote.gpu_legacy_note() if CONTROL else data_notes.load().get("legacy_note", ""),
                "data_home": str(DATA_HOME),
                "role": "control" if CONTROL else "gpu",
                "sync": sync_status() if CONTROL else None,
                "gpu_link": remote.TUNNEL.status() if CONTROL else None,
                "hf_user": UPLOADER.hf_login(),
                "upload": UPLOADER.status(),
            })
        elif self.path.startswith("/stream/"):
            self._stream(self.path.split("?", 1)[0].rsplit("/", 1)[-1])
        elif self.path.startswith("/api/reference?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body = reference_frame(q.get("repo", [""])[0], q.get("cam", ["front"])[0], q.get("model", [""])[0])
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
                self._json(alignment(q.get("repo", [""])[0], q.get("model", [""])[0], q.get("cam", ["front"])[0]))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/reference_edges?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body = reference_entry(q.get("repo", [""])[0], q.get("model", [""])[0], q.get("cam", ["front"])[0])["edges"]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=300")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/home?") or self.path == "/api/home":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            self._json(home_status(q.get("model", [""])[0]))
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
        elif self.path == "/api/presets":
            self._json({"presets": proto.list_presets(), "stages": proto.STAGE_KO})
        elif self.path.startswith("/api/eval/compare?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                import omx_compare
                self._json(omx_compare.compare(q.get("preset", [""])[0], data_home=DATA_HOME))
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/detect?"):
            # object position in the camera view before an evaluation starts (placing the first object)
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                import omx_vision
                preset = proto.load_preset(q.get("preset", [""])[0])
                cam = preset.get("camera", "front")
                frame = PREVIEW.frames.get(cam) if PREVIEW.running else None
                px = omx_vision.detect_jpeg(preset["detector"], frame, preset.get("detect_roi")) if frame else None
                self._json({"px": list(px) if px else None, "camera_on": bool(frame), "detector": preset["detector"]})
            except Exception as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path.startswith("/api/snapshot/"):
            self._snapshot(self.path.rsplit("/", 1)[-1].split("?", 1)[0])
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

    def _snapshot(self, name: str) -> None:
        """One camera frame for the position editor: camera view, or the running program's live frame."""
        frame = PREVIEW.frames.get(name) if PREVIEW.running else None
        live = Path(f"{LIVE}{name}.jpg")
        if frame is None and live.exists() and time.time() - live.stat().st_mtime < 5:
            frame = live.read_bytes()
        if not frame or name not in CAMERAS:
            self._json({"error": "영상이 없습니다. ① 탭에서 카메라 보기를 켜면 배경에 나옵니다."}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(frame)

    def _stream(self, name: str) -> None:
        if name not in CAMERAS:
            self._json({"error": "unknown camera"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        live = Path(f"{LIVE}{name}.jpg")
        # keep the connection through short gaps (preview stops just before a record/eval job starts)
        idle_since = None
        last_key = None
        try:
            while True:
                if PREVIEW.running or JOB.running():
                    idle_since = None
                elif idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since > 5:
                    break
                frame, key = (PREVIEW.frames.get(name), None) if PREVIEW.running else (None, None)
                if frame is not None:
                    key = ("p", id(frame))
                elif JOB.running() and live.exists():
                    st = live.stat()
                    if time.time() - st.st_mtime < 2:
                        key = ("l", st.st_mtime_ns)
                        if key != last_key:
                            frame = live.read_bytes()
                # only new pictures, at display size: the remote link (Tailscale) carries ~0.6 MB/s, and
                # two full-size 640x480 streams with repeated frames did not fit (slow, laggy view)
                if frame and key != last_key:
                    small = display_jpeg(name, key, frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(small)}\r\n\r\n".encode())
                    self.wfile.write(small + b"\r\n")
                    last_key = key
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        CLIENT["seen"] = time.time()
        if self.path == "/api/bye":  # the panel window was closed (or is reloading)
            CLIENT["bye"] = time.time()
            return self._json({"ok": True})
        if CONTROL and remote.proxied("POST", self.path):
            return self._forward("POST", raw)
        body = json.loads(raw or b"{}")
        try:
            mqtt = body.get("source") == "mqtt"  # a control-tower command (the MQTT worker on this PC)
            if mqtt and self.path in ("/api/teleop", "/api/record", "/api/armed"):
                raise RuntimeError("관제 명령으로는 평가 시작만 할 수 있습니다. armed는 패널 화면에서 사람이 켭니다.")
            if mqtt and self.path == "/api/eval/start" and not ARMED["on"]:
                raise RuntimeError("armed가 꺼져 있어 관제 명령을 받지 않습니다. 패널 화면에서 사람이 켜야 합니다.")
            if self.path in ("/api/teleop", "/api/record", "/api/eval/start") and not body.get("confirm") and not mqtt:
                raise RuntimeError("로봇 옆에 사람이 있는지 확인하고 체크해주세요.")
            if self.path == "/api/check":
                self._json(run_check())
            elif self.path == "/api/robot_setup":
                self._json(save_setup(body))
            elif self.path == "/api/camera_detect":
                self._json(detect_cameras(bool(body.get("apply"))))
            elif self.path == "/api/sync" and CONTROL:
                self._json(sync_retry(str(body.get("repo_id") or "")))
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
                set_armed(False, "중지 버튼")
                JOB.stop()
                self._json({"ok": True})
            elif self.path == "/api/armed":
                if body.get("on") and not body.get("confirm"):
                    raise RuntimeError("로봇 옆에 사람이 있는지 확인하고 체크해주세요.")
                self._json(set_armed(bool(body.get("on"))))
            elif self.path == "/api/cam_refs/save":
                self._json(save_view_refs())
            elif self.path == "/api/start_pose/save":
                self._json(save_start_pose(body.get("kind", "start")))
            elif self.path == "/api/eval/start":
                self._json(start_eval(body))
            elif self.path == "/api/target":
                self._json(set_target(body))
            elif self.path == "/api/marker_view":
                t = str(body.get("target", "")).strip()
                self._json(PREVIEW.set_marker_view(bool(body.get("show")), int(t) if t else None))
            elif self.path == "/api/eval/report":
                start_report(body.get("repo_id", ""))
                self._json({"ok": True})
            elif self.path == "/api/presets/save":
                self._json({"ok": True, "preset": proto.save_preset(body, USER)})
            elif self.path == "/api/presets/delete":
                proto.delete_preset(body.get("id", ""))
                self._json({"ok": True})
            elif self.path == "/api/eval/compare_exclude":
                update_note(body.get("repo_id", ""), lambda n: n.update(compare_exclude=bool(body.get("exclude"))))
                self._json({"ok": True})
            elif self.path == "/api/eval/candidates":
                if body.get("mode") not in ("rec_all", "rec1", "clear"):
                    raise RuntimeError("알 수 없는 선택입니다.")
                update_note(body.get("repo_id", ""), lambda n: proto.select_candidates(n, body["mode"]))
                self._json({"ok": True})
            elif self.path == "/api/eval/mark":
                eval_mark(body)
                self._json({"ok": True})
            elif self.path == "/api/train/start":
                self._json(train_start(body))
            elif self.path == "/api/train/name":
                self._json(train_name(body))
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
            elif self.path == "/api/home/current":
                self._json(home_current())
            elif self.path == "/api/home/task":
                if JOB.running() and body.get("apply"):
                    raise RuntimeError("작업이 실행 중일 때는 홈 자세를 바꾸지 않습니다.")
                self._json({"home": omx_home.compute_task(str(body.get("task") or "straw"), bool(body.get("apply"))),
                            "tasks": omx_home.load_tasks()})
            elif self.path == "/api/home/return":
                self._json(home_return())
            elif self.path == "/api/kill":
                set_armed(False, {"home": "즉시 중단", "freeze": "비상 정지"}.get(body.get("mode"), "강제 종료"))
                if body.get("mode") == "home":  # 즉시 중단 (eval): stop and return to the start pose
                    self._json({"ok": True, "result": JOB.abort()})
                elif body.get("mode") == "freeze":  # 비상 정지: hold in place, keep the records, wait for [홈 위치로 이동]
                    self._json({"ok": True, "result": JOB.freeze()})
                else:  # 비상 정지 / 강제 종료: SIGKILL, the follower stays where it is with the torque on
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


# Control PC: closing the panel window stops the server. The page sends /api/bye when it goes away; if no
# page asks for anything within CLOSE_GRACE_S (a reload asks again in a second or two) the server stops as
# with Ctrl+C - but only once no robot program, GPU transfer, upload or eval report is running.
CLOSE_GRACE_S = 10.0
CLIENT = {"seen": time.time(), "bye": 0.0}


def busy_reason() -> str:
    if JOB.running():
        return {"record": "녹화가", "eval": "평가가", "teleop": "텔레옵이", "home": "처음 자세 복귀가"}.get(JOB.kind, "작업이") + " 끝나면"
    if getattr(JOB, "reader", None) and JOB.reader.is_alive():
        return "작업 마무리(저장·전송 시작)가 끝나면"
    if SYNC["busy"]:
        return "GPU 노트북 전송이 끝나면"
    if UPLOADER.running():
        return "Hugging Face 업로드가 끝나면"
    if any(p.poll() is None for p in REPORTS.values()):
        return "평가 보고서가 만들어지면"
    return ""


def close_watch() -> None:
    told, idle = "", 0
    while True:
        time.sleep(2)
        bye = CLIENT["bye"]
        if not bye or CLIENT["seen"] > bye or time.time() - bye < CLOSE_GRACE_S:
            told, idle = "", 0
            continue
        why = busy_reason()
        if why:
            if why != told:
                print(f"[omx] 제어판 창이 닫혔습니다. {why} 서버를 끕니다", flush=True)
            told, idle = why, 0
            continue
        idle += 1
        if idle >= 2:  # idle on two checks in a row: a job's follow-up (transfer, upload) has had time to start
            print("[omx] 제어판 창이 닫혀 서버를 끕니다 (다시 켜려면 바탕화면 아이콘 또는 omx-panel-v3)", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def start_inference_server() -> None:
    """GPU laptop (v2): this account's inference server for control PCs (omx_infer.py, 127.0.0.1 only),
    restarted if it ends; loads a model only when a control PC asks."""
    import omx_infer

    port = omx_infer.default_port()

    def keep():
        while True:
            if not _port_in_use(port):
                log = open(LOG_DIR / "infer_server.log", "a")
                p = subprocess.Popen([str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_infer.py"), "serve", "--port", str(port)],
                                     cwd=HOME, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                p.wait()
            time.sleep(5)
    threading.Thread(target=keep, daemon=True).start()


def _port_in_use(port: int) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    # Started in the background, so SIGINT may be ignored; make INT and TERM both shut down cleanly.
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    if CONTROL:
        remote.TUNNEL.start()
        threading.Thread(target=sync_loop, daemon=True).start()
        threading.Thread(target=close_watch, daemon=True).start()
        print(f"[omx] control PC: training and inference on {remote.HOST} (SSH tunnel)", flush=True)
    else:
        start_inference_server()
    print(f"OMX panel {VERSION} on http://{HOST}:{PORT} (user {USER}, settings: {CFG['_source']})", flush=True)
    try:
        server.serve_forever()
    finally:
        JOB.stop()
        PREVIEW.stop()
        if CONTROL:
            remote.TUNNEL.stop()


if __name__ == "__main__":
    main()
