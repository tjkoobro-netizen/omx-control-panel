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
import hashlib
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
from concurrent.futures import ThreadPoolExecutor
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
import omx_host  # noqa: E402  (control tower MQTT Host, started when armed turns on)
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
CAMERA_ROLES = {"single": ["front", "front_2", "front_3", "front_4", "wrist"],
                "bimanual": ["front", "front_2", "front_3", "front_4", "wrist_left", "wrist_right"]}
# 3.19.0: front_3 / front_4 are optional extra global cameras. A record keeps every chosen camera; an evaluation opens
# them only for a model trained with them (a model of the usual cameras runs as before, at the same camera load).
EXTRA_CAMERA_ROLES = ("front_3", "front_4")
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
                "front_3": f"{pair} · 글로벌 front 3", "front_4": f"{pair} · 글로벌 front 4",
                "wrist_left": f"{left} · 왼팔 wrist", "wrist_right": f"{right} · 오른팔 wrist"}.get(name, name)
    robot = setup["single"]["set"]
    return {"front": f"{robot} · 글로벌 front", "front_2": f"{robot} · 글로벌 front 2", "front_3": f"{robot} · 글로벌 front 3",
            "front_4": f"{robot} · 글로벌 front 4", "wrist": f"{robot} · 팔로워 wrist"}.get(name, name)


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


def capture_time(cam, frame) -> float | None:
    """When the camera took this frame (epoch s, 3.20.0): LeRobot's read thread stamps each frame (perf_counter) and
    async_read hands out that same array, so it is paired by identity. None when a newer frame replaced it."""
    lock = getattr(cam, "frame_lock", None)
    if lock is None or frame is None:
        return None
    with lock:
        if getattr(cam, "latest_frame", None) is not frame or getattr(cam, "latest_timestamp", None) is None:
            return None
        ts = cam.latest_timestamp
    return time.time() - (time.perf_counter() - ts)


def list_cameras() -> list[dict]:
    """Every camera plugged into this PC, by USB port (identical cameras share serials, so the port is the id).
    The laptop's own camera is left out unless a role already uses it."""
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
        if camauto.BUILTIN.search(model) and not used:  # the laptop's own camera: never a robot camera (3.20.2)
            continue
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


HOSTNAME = os.uname().nodename


def robot_here() -> bool:
    """An arm board of this setup is plugged into this PC (the robot can move between the two PCs by one USB hub)."""
    return any(Path(p).exists() for _, p in FOLLOWERS + LEADERS)


_GPU_ROBOT = {"t": 0.0, "busy": False, "data": None}


def _refresh_gpu_robot() -> None:
    try:
        _GPU_ROBOT["data"] = remote.gpu_json("/api/robot_here", timeout=3) if remote.TUNNEL.status()["ok"] else None
    except Exception:
        _GPU_ROBOT["data"] = None  # an older GPU panel (no such request) or the link is down: unknown
    finally:
        _GPU_ROBOT["t"], _GPU_ROBOT["busy"] = time.time(), False


def gpu_panel() -> dict:
    """Control PC: the GPU laptop's panel through the tunnel, and whether the robot is plugged in there. Refreshed in
    the background every 5 s so the status poll never waits on the link."""
    if time.time() - _GPU_ROBOT["t"] > 5 and not _GPU_ROBOT["busy"]:
        _GPU_ROBOT["busy"] = True
        threading.Thread(target=_refresh_gpu_robot, daemon=True).start()
    data = _GPU_ROBOT["data"] or {}
    return {"port": remote.TUNNEL_PANEL, "ok": remote.TUNNEL.status()["ok"],
            "robot_here": data.get("robot_here"), "host": data.get("host", "")}


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
    # 3.22.1: all at once (one after another 2.7 s for 4 cameras, at once 0.6 s: most of the start's answer time)
    paths = [p for p in cameras.values() if Path(p).exists()]
    with ThreadPoolExecutor(max_workers=max(1, len(paths))) as pool:
        frames = dict(zip(paths, pool.map(camauto.grab_gray, paths)))
    wrong = camauto.swapped(cameras, refs, frames)
    if wrong:
        raise RuntimeError(f"{', '.join(wrong)} 화면이 다른 카메라의 기준 화면과 일치합니다. 케이블이 바뀐 것 같습니다. "
                           "로봇 구성에서 화면을 확인하고 저장한 뒤 시작하세요.")


DROP_WATCH_FILE = LOG_DIR / "drop_watch.json"  # omx_run's drop watch settings (OMX_DROP_WATCH, 3.23.0)
DROP_FRAMES = LOG_DIR / "drop_frames"  # the top frame at each confirmed drop


def drop_env(eval_info: dict, arm: str | None, cams: dict, body: dict | None = None) -> dict:
    """3.23.0: the box drop watch of a cooperative evaluation, record mode (never ends the run, never moves an arm).
    Its own switch, not tied to end_on_return. cfg "drop_watch": false turns it off; "drop_tray_roi" [v0, v1, u0, u1]
    replaces the tray region of the top camera.
    3.26.0: one arm too (DropWatch 1125a, record mode): a loading mission (end_on_place) checks on the top camera
    (slot 2A / 2C), an unloading (pinky_check) on that arm's side camera (front_3 / front_4); a panel-screen one-arm
    eval (no mission) records the gripper signal only."""
    if ROBOT_MODE != "bimanual" or CFG.get("drop_watch", True) is False:
        return {}
    body = body or {}
    box = eval_info.get("target_marker")
    if box is None and str(body.get("target_marker", "")).strip().isdigit():
        box = int(body["target_marker"])
    if arm:
        plan = eval_info.get("pinky_check") or {}
        if plan.get("one_arm"):
            op, slot, cam = "OUTBOUND", plan["slot"], {"A": "front_3", "C": "front_4"}[plan["slot"][1:]]
        elif body.get("end_on_place"):
            op, slot, cam = "INBOUND", command_slot(body), "front"
        else:
            op, slot, cam = None, None, None
        conf = {"box_ids": [int(box)] if box is not None else None, "arm": arm, "operation": op,
                "slot": slot, "frame_dir": str(DROP_FRAMES),
                "repo_id": eval_info.get("repo_id")}
        if cam is not None and cam not in cams:
            conf["operation"] = None  # its camera is not open: stage 1 only
        _write_json(DROP_WATCH_FILE, {k: v for k, v in conf.items() if v is not None})
        return {"OMX_DROP_WATCH": str(DROP_WATCH_FILE)}
    if "front" not in cams:
        return {}
    conf = {"box_ids": [int(box)] if box is not None else None, "camera": "front", "frame_dir": str(DROP_FRAMES),
            "repo_id": eval_info.get("repo_id")}
    if CFG.get("drop_tray_roi"):
        conf["tray_roi"] = CFG["drop_tray_roi"]
    _write_json(DROP_WATCH_FILE, {k: v for k, v in conf.items() if v is not None})
    return {"OMX_DROP_WATCH": str(DROP_WATCH_FILE)}


TRAY_JUDGE_FILE = LOG_DIR / "tray_judge.json"  # omx_run's unloading tray judge settings (OMX_TRAY_JUDGE, 3.24.0)
TRAY_PREFIX = {"placed": "[omx] tray_placed ", "record": "[omx] tray_judge ", "reentry": "[omx] tray_reentry ",
               "recheck": "[omx] tray_recheck "}


def tray_settings() -> dict:
    """cfg "tray_judge": {"mode": "record" | "on" | "off", "hold_s": 1.0}. Default record: the judge only records when
    it would have judged; "on" ends the unloading eval there and lets the Host report it (TK 11/24)."""
    c = CFG.get("tray_judge")
    c = c if isinstance(c, dict) else {"mode": "off"} if c is False else {}
    return {"mode": c.get("mode", "record"), "hold_s": float(c.get("hold_s", 1.0))}


def tray_env(eval_info: dict, arm: str | None, cams: dict) -> dict:
    """3.24.0: the "placed on the Pinky tray" judge of a control-tower unloading (pinky_check) on the cooperative robot."""
    plan = eval_info.get("pinky_check")
    st = tray_settings()
    if arm and plan and plan.get("one_arm"):
        # 3.25.0: one arm, its side top camera (front_3 left / front_4 right); cfg "tray_judge_one_arm" (default on:
        # the eval ends on the judge, the tray is checked again with the arm home, one report after that)
        c = CFG.get("tray_judge_one_arm") or {}
        mode, hold = c.get("mode", "on"), float(c.get("hold_s", 1.0))
        cam = {"A": "front_3", "C": "front_4"}[plan["slot"][1:]]
        if mode not in ("record", "on") or not {cam, "front_2"} <= set(cams) or not (TOOLS_DIR / "tray_judge.py").exists():
            raise RuntimeError(f"한팔 출고 쟁반 판정에는 {cam}·front_2 카메라와 tray_judge.py가 필요합니다.")
        _write_json(TRAY_JUDGE_FILE, {"box": int(plan["box"]), "slot": plan["slot"], "mode": mode, "hold_s": hold,
                                      "tray_ref_file": None, "rate_hz": 6, "max_age_s": 3.0, "one_arm": True, "arm": arm})
        eval_info["tray_judge"] = {"mode": mode, "hold_s": hold, "one_arm": True, "camera": cam}
        return {"OMX_TRAY_JUDGE": str(TRAY_JUDGE_FILE)}
    if not plan or ROBOT_MODE != "bimanual" or arm or st["mode"] not in ("record", "on") \
            or not {"front", "front_2"} <= set(cams) or not (TOOLS_DIR / "tray_judge.py").exists():
        return {}
    _write_json(TRAY_JUDGE_FILE, {"box": int(plan["box"]), "slot": plan["slot"], "mode": st["mode"], "hold_s": st["hold_s"],
                                  "tray_ref_file": (plan.get("tray") or {}).get("tray_ref_file"), "rate_hz": 6,
                                  "max_age_s": 3.0})
    eval_info["tray_judge"] = {"mode": st["mode"], "hold_s": st["hold_s"]}
    return {"OMX_TRAY_JUDGE": str(TRAY_JUDGE_FILE)}


def tray_line(job, kind: str, line: str) -> None:
    """omx_run's tray judge lines (3.24.0). placed: job.info.tray_placed at once (the Host reports the placement and the
    Pinky may leave). reentry hold: the result says "hold" right away (the run waits, torque on, for [홈 위치로 이동])."""
    try:
        d = json.loads(line[len(TRAY_PREFIX[kind]):])
    except ValueError:
        return
    if kind == "placed":
        job.info["tray_placed"] = d
    elif kind == "record":
        job.info["tray_judge_record"] = d
    elif kind == "recheck":
        job.info["tray_recheck"] = d  # 3.25.0: one-arm unloading, the tray seen again with the arm home
    else:
        job.info.setdefault("tray_reentry", []).append(d)
        if d.get("event") == "hold":
            hold = {"reason": "tray_reentry", "side": d.get("side"), "joints": d.get("joints"), "at": d.get("at"),
                    "after_s": d.get("after_s")}
            job.info["tray_hold"] = hold
            LAST_RESULT.clear()
            LAST_RESULT.update(kind="eval", source=job.info.get("source", "panel"), repo_id=job.info.get("repo_id"),
                               stage="hold", final=False, hold=hold, placement=job.info.get("tray_placed"),
                               exit_ok=None, err=None)


WRIST_CHECK_PREFIX = "[omx] wrist_check "  # omx_run.py / omx_wristcheck.py (3.22.0)


def wrist_check_on() -> bool:
    """3.22.0: before a cooperative evaluation starts, each gripper opens a little and the wrist camera whose picture
    changed is that arm's (both wrist cameras are the same model with the same serial). cfg "wrist_check": false
    turns it off."""
    return ROBOT_MODE == "bimanual" and CFG.get("wrist_check", True) is not False


def wrist_swap_setup(chk: dict) -> str:
    """The evaluation's wrist check found wrist_left / wrist_right exchanged and swapped them in that run: robot.json
    follows at once (the alignment runner reads it after the eval). No person needed, the evidence is the gripper move
    (TK 11/23). Only when the saved paths still are the ones the run started with."""
    setup = load_setup() or {}
    by = {c.get("name"): c for c in (setup.get("cameras") or {}).get("bimanual") or []}
    left, right, was = by.get("wrist_left"), by.get("wrist_right"), chk.get("was") or {}
    if not left or not right or (left.get("path"), right.get("path")) != (was.get("left"), was.get("right")):
        return "kept: 저장된 손목 경로가 실행을 시작할 때와 달라 고치지 않음"
    left["path"], right["path"] = right["path"], left["path"]
    SETUP_FILE.write_text(json.dumps(setup, ensure_ascii=False, indent=2))
    apply_setup(setup)
    return "fixed"


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
        if WARM_FLAG in args and not Path(WARM_BUSY.format(pid=pid)).exists():
            continue  # a ready evaluation process (3.22.2) that has not taken a run: it holds no robot
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
        self.frame_ts: dict[str, float] = {}  # when each frame was read (epoch s): /api/snapshot X-Frame-Ts
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
                read_at = capture_time(cam, frame) or time.time()  # 3.20.0: when the camera took it
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
                    self.frame_ts[name] = read_at
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
            self.cams, self.frames, self.raw, self.brightness, self.frame_ts = {}, {}, {}, {}, {}


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
        info.setdefault("base_phase", info.get("phase"))  # a runner's own words, kept through its safe start
        self.slow = collections.deque(maxlen=2000)  # (time, Hz) of "Record loop is running slower" warnings
        self.slow_total = 0
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMX_PORT=str(PORT), **(env_extra or {}))  # same LIVE files
        warm = WARM.take(cmd, env) if kind == "eval" else None
        if warm is not None:
            self.proc = warm
            info["warm"] = True
            self.log_file.write(f"$ (미리 띄운 평가 프로그램 PID {warm.pid}에 위 인자를 넘김)\n")
        else:
            self.proc = subprocess.Popen(
                cmd, cwd=LEROBOT_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def adopt(self, kind: str, standby: "Standby", info: dict) -> None:
        """Take a process that is already running (the alignment runner that waited during the eval, 3.18.0) as
        the robot job: its output, stop buttons and end handling from now on, as if started here."""
        if self.running():
            raise RuntimeError("이미 실행 중인 작업이 있습니다.")
        self._acquire_lock(kind)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = open(LOG_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{kind}.log", "w")
        self.log_file.write("$ (평가 중 대기하던 실행기를 넘겨받음) " + " ".join(standby.cmd or []) + "\n")
        self.lines.clear()
        for line in list(standby.lines):
            self.lines.append(line)
            self.log_file.write(line + "\n")
        self.kind, self.info, self.started = kind, info, standby.started
        info.setdefault("base_phase", info.get("phase"))
        self.slow = collections.deque(maxlen=2000)
        self.slow_total = 0
        self.proc = proc = standby.proc
        self.reader = standby.collector  # running() lasts until it has run the end handling below
        standby.attach(self._handle_line, lambda: self._finish(proc))

    def _handle_line(self, raw: str) -> None:
        line = ANSI_RE.sub("", raw).rstrip()
        if not line:
            return
        if line.startswith("Teleop loop time"):
            self.info["loop"] = line.split(":", 1)[1].strip()
            return
        if TABLE_RE.match(line):
            return
        if m := SLOW_RE.search(line):
            # omx_run.py warns on every slow loop iteration: count them instead of flooding the log view
            self.slow.append((time.time(), float(m.group(1))))
            self.slow_total += 1
            if self.log_file:
                self.log_file.write(line + "\n")
            return
        self._parse(line)
        self.lines.append(line)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def _read_output(self) -> None:
        proc = self.proc
        for raw in output_lines(proc):
            self._handle_line(raw)
        self._finish(proc)

    def _finish(self, proc) -> None:
        proc.wait()
        Path(WARM_BUSY.format(pid=proc.pid)).unlink(missing_ok=True)
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
            LAST_RESULT.update(eval_result(self), stage="done", final=True, align=None, pinky=None, placement=None,
                               handover=None)
            if self.info.get("tray_placed") and not (self.info.get("pinky_check") or {}).get("one_arm"):
                LAST_RESULT["placement"] = self.info["tray_placed"]  # 3.24.0: the unloading ended on the tray judge
            if self.info.get("one_arm_place"):  # 3.25.0: one-arm loading, the judged placement is the evidence
                LAST_RESULT["placement"] = one_arm_placement(self.info, _episode_end_now(self))
            if self.info.get("tray_hold"):
                LAST_RESULT["hold"] = dict(self.info["tray_hold"], released_at=now_iso())
            if self.info.get("handover"):
                handover_after_eval(self, proc)  # 3.18.0: placed -> the waiting runner; otherwise the usual alignment
            elif self.info.get("shelf_align"):
                align_after_eval(self.info["shelf_align"], proc)
            if (self.info.get("pinky_check") or {}).get("one_arm"):
                # 3.25.0: one-arm unloading: the tray was checked again in the run with the arm home
                LAST_RESULT["pinky"] = one_arm_pinky(self.info, self.info["pinky_check"])
            elif self.info.get("pinky_check"):
                pinky_after_eval(self.info["pinky_check"], proc)
        if self.kind in ("align", "align_release"):
            align_ended(self, proc)
        if self.kind == "home" and self.info.get("handover_fallback"):
            handover_home_ended(self, proc)
        if self.kind == "shelf_read":
            shelf_read_ended(self, proc)
        if self.kind == "pinky_read":
            pinky_read_ended(self, proc)
        if self.kind == "eval" and self.info.get("protocol") and self.info.get("record", True):
            fill_auto_later(self.info["repo_id"], delay=0)  # reads the recorded videos
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
                if not self.info.get("record", True):  # nothing recorded: the attempts that ran, for marking results
                    last = LAST_RESULT.get("last_episode") or {}
                    note["episodes_run"] = int(last["episode"]) + 1 if last.get("episode") is not None else 0
                    if proc.returncode != 0 and not note["episodes_run"]:
                        note["start_failed"] = True
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
        elif self.kind == "eval" and line.startswith("[omx] tray_"):
            for kind, prefix in TRAY_PREFIX.items():
                if line.startswith(prefix):
                    tray_line(self, kind, line)
        elif line.startswith(WRIST_CHECK_PREFIX) and self.kind == "eval":
            try:
                chk = json.loads(line[len(WRIST_CHECK_PREFIX):])
            except ValueError:
                chk = {"ok": False, "why": "결과를 읽지 못함"}
            if chk.get("ok") and chk.get("swapped") and chk.get("mode") == "pair":
                chk["setup"] = wrist_swap_setup(chk)
            self.info["wrist_check"] = chk
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
            if self.kind not in ("record", "eval", "teleop"):
                # the shelf / Pinky runner grabbing the arms: keep its own words ("정렬 중"), not "녹화 중" (11/17)
                base = self.info.get("base_phase") or "실행 중"
                self.info["phase"] = base if done else f"{base} · 팔 잡는 중"
            elif self.kind == "teleop":
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
        if self.kind == "align":
            self._align_stop("abort")
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
        if self.running() and self.kind == "align":
            return self._align_stop("abort")
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
        if self.kind == "align":
            return self._align_stop("freeze")
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

    def _align_stop(self, mode: str) -> str:
        """Shelf alignment runner: SIGTERM = it stops where it is with the torque on and exits. 즉시 중단 / Stop
        then lands the arms (runner --release), 비상 정지 waits for [홈 위치로 이동]. A second press, or no exit
        within ALIGN_STOP_WAIT_S, force-quits it (the torque stays on, the landing still follows)."""
        again = bool(self.info.get("abort_sent") or self.info.get("frozen"))
        if mode == "freeze":
            self.info["frozen"] = time.time()
            self.info["land_after"] = False
            self.info["phase"] = "비상 정지 · 제자리에 멈추는 중"
        else:
            self.info.setdefault("abort_sent", time.time())
            self.info["land_after"] = not self.info.get("frozen")
            self.info["phase"] = "즉시 중단 · 멈춘 뒤 내려놓습니다"
        if again:
            self.kill()
            return "kill"
        os.kill(self.proc.pid, signal.SIGTERM)
        proc = self.proc

        def watchdog() -> None:
            end = time.time() + ALIGN_STOP_WAIT_S
            while time.time() < end and proc.poll() is None:
                time.sleep(0.2)
            if self.proc is proc and proc.poll() is None:
                self.lines.append("[omx] align: 응답이 없어 강제로 끝냅니다 (팔은 토크를 켠 채 제자리)")
                self.kill()

        threading.Thread(target=watchdog, daemon=True).start()
        return mode

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
            "started_at": round(self.started, 3),  # 3.26.0: tells one job from the next (the "미션 접수" voice cue)
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
    try:
        runs = gpu_runs().get("runs", [])
    except Exception:  # no tunnel and no copy here yet: the UDP model check below decides
        runs = []
    for r in runs:
        if r["model_path"] == model_path:
            return r
        for c in r.get("checkpoints") or []:
            if c["model_path"] == model_path:
                return dict(r, model_path=model_path, step=c["step"])
    if not model_path.rstrip("/").endswith("pretrained_model"):
        raise RuntimeError("GPU 노트북의 학습 기록에 있는 모델을 고르세요.")
    return run_from_path(model_path)  # the UDP model check (fetch_model_udp) confirms it exists there


def remote_preset(pid: str) -> dict:
    for p in remote.gpu_json("/api/presets").get("presets", []):
        if p.get("id") == pid:
            return p
    raise RuntimeError(f"평가 방식 {pid}을 GPU 노트북에서 찾지 못했습니다.")


# Control PC: the AI inference (the GPU laptop's models) is reached only over UDP: wire RPI1 (infer_link) straight to
# the GPU laptop's UDP server on the same router, with a shared key file (hex, chmod 600; never in a repository).
# The evaluation's preparation (model files, home pose) goes the same way, so a mission needs no SSH tunnel.
# ~/.config/omx/v3/infer_link.json, set on the panel.
INFER_LINK_FILE = SETUP_FILE.parent / "infer_link.json"
INFER_LINK_DEFAULT = {"udp_server": "", "udp_key_file": "~/.config/roport/omx-udp.key"}
UDP_BASE_PORT = 5600  # + this account's offset of its GPU panel port from 8865, so accounts never share a port


def udp_port_hint() -> int:
    return UDP_BASE_PORT + max(0, remote.REMOTE_PANEL - 8865)


def infer_link() -> dict:
    out = dict(INFER_LINK_DEFAULT)
    try:
        out.update({k: v for k, v in json.loads(INFER_LINK_FILE.read_text()).items() if k in out})
    except (OSError, ValueError):
        pass
    return out


def check_udp(server: str, key_file: str) -> None:
    import omx_infer

    if not server:
        raise RuntimeError(f"③ 평가 설정의 AI 추론 통신에 GPU 노트북 UDP 서버 주소를 적으세요 (예: 192.168.x.x:{udp_port_hint()})")
    try:
        omx_infer.udp_peer(server)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
    try:
        key = omx_infer.read_udp_key(key_file)
    except FileNotFoundError:
        raise RuntimeError(f"키 파일이 없습니다: {key_file} (③ 평가 설정의 '키 만들기·GPU에 복사')") from None
    except ValueError as exc:
        raise RuntimeError(f"키 파일을 쓸 수 없습니다 ({exc})") from None
    if len(key) < 16:
        raise RuntimeError("키가 너무 짧습니다 (16바이트 이상, hex)")


def save_infer_link(body: dict) -> dict:
    """Panel: the GPU laptop's UDP server address and the key file. Takes effect from the next evaluation."""
    if JOB.running() and JOB.kind == "eval":
        raise RuntimeError("평가가 끝난 뒤 바꾸세요.")
    cur = infer_link()
    server = str(body.get("udp_server", cur["udp_server"])).strip()
    key_file = str(body.get("udp_key_file", cur["udp_key_file"])).strip() or INFER_LINK_DEFAULT["udp_key_file"]
    check_udp(server, key_file)
    INFER_LINK_FILE.parent.mkdir(parents=True, exist_ok=True)
    INFER_LINK_FILE.write_text(json.dumps({"udp_server": server, "udp_key_file": key_file},
                                          ensure_ascii=False, indent=2) + "\n")
    UDP_LINK.reset()
    return infer_link_status()


def provision_udp_key() -> dict:
    """Make the shared key if this PC has none, and put the same key on the GPU laptop (over SSH, once, if it has
    none there). A different key already on the GPU laptop is never overwritten: that is reported instead."""
    import omx_infer
    import omx_udp_keeper

    if JOB.running() and JOB.kind == "eval":
        raise RuntimeError("평가가 끝난 뒤 하세요.")
    path = Path(infer_link()["udp_key_file"]).expanduser()
    made = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(os.urandom(32).hex() + "\n")
        made = True
    try:
        key = omx_infer.read_udp_key(str(path))
    except ValueError as exc:
        raise RuntimeError(f"이 PC의 키 파일을 쓸 수 없습니다 ({exc})") from None
    fp = omx_udp_keeper.key_fingerprint(key)
    script = ('f="$HOME/.config/roport/omx-udp.key"; if [ -e "$f" ]; then cat >/dev/null; echo EXISTS; '
              'python3 -c "import hashlib,sys; print(hashlib.sha256(bytes.fromhex(open(sys.argv[1]).read().strip())).hexdigest()[:12])" "$f"; '
              'else umask 077; mkdir -p "$HOME/.config/roport"; cat > "$f"; chmod 600 "$f"; echo CREATED; fi')
    r = subprocess.run(["ssh", *remote.SSH_OPTS, remote.HOST, script], input=key.hex() + "\n",
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"GPU 노트북({remote.HOST})에 SSH로 연결하지 못했습니다: {r.stderr.strip()[-160:]}")
    out = r.stdout.split()
    if out[:1] == ["EXISTS"] and out[1:2] != [fp]:
        raise RuntimeError("GPU 노트북에 다른 키가 이미 있습니다 (~/.config/roport/omx-udp.key). 덮어쓰지 않았습니다. "
                           "같은 키를 쓰려면 두 PC 중 한쪽 키 파일을 다른 쪽과 같게 맞추세요.")
    UDP_LINK.reset()
    gpu = "GPU 노트북에 복사함" if out[:1] == ["CREATED"] else "GPU 노트북 키와 같음"
    return {"ok": True, "made": made, "fingerprint": fp,
            "message": ("이 PC에 새 키를 만들고 " if made else "이 PC의 키가 ") + gpu + f" (지문 {fp})"}


# ---- the GPU laptop's training index, kept here so a mission (Host: GET /api/train/runs, then eval start) does
# not need the SSH tunnel; refreshed whenever the tunnel answers
RUNS_CACHE_FILE = SETUP_FILE.parent / "gpu_runs_cache.json"


def gpu_runs() -> dict:
    """{"runs": [...]} from the GPU laptop, or the last copy here (with "cached_at") when the tunnel is down."""
    try:
        out = remote.gpu_json("/api/train/runs", timeout=15)
    except Exception as exc:
        try:
            cached = json.loads(RUNS_CACHE_FILE.read_text())
        except (OSError, ValueError):
            raise RuntimeError(str(exc)) from None
        return cached | {"stale": True, "stale_reason": str(exc)[:200]}
    try:
        RUNS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        RUNS_CACHE_FILE.write_text(json.dumps(out | {"cached_at": time.time()}, ensure_ascii=False))
    except OSError:
        pass
    return out


# ---- model files and home pose over UDP (LOAD with overrides {"_info_only": true}: nothing is loaded on the GPU)
def udp_ready() -> dict:
    """Before an evaluation: the settings, and a fresh PING when the last check failed (refuse at once, with where)."""
    link = infer_link()
    check_udp(link["udp_server"], link["udp_key_file"])
    st = UDP_LINK.state
    if st.get("ok") is False:
        try:
            UDP_LINK.probe(link)
        except Exception:
            raise RuntimeError("AI 추론 서버와 UDP 연결이 안 되어 평가를 시작하지 않습니다: " + (st.get("error") or "응답 없음")
                               + (f" ({st['hint']})" if st.get("hint") else "")
                               + " · ③ 평가 설정 맨 위 AI 추론 통신 표시를 보세요") from None
        UDP_LINK.wake.set()
    return link


def udp_model_info(model_path: str, timeout: float = 10.0) -> dict:
    import omx_infer
    from infer_link.client import InferenceClient, InferenceError, InferenceTimeout

    cfg = infer_link()
    check_udp(cfg["udp_server"], cfg["udp_key_file"])
    client = InferenceClient(omx_infer.udp_peer(cfg["udp_server"]), omx_infer.read_udp_key(cfg["udp_key_file"]))
    try:
        out = client.load(model_path, {"_info_only": True}, timeout=timeout, attempts=2)
    except InferenceTimeout:
        raise RuntimeError("GPU 노트북 UDP 서버가 모델 정보 요청에 답하지 않습니다. ③ 평가 설정 맨 위 AI 추론 통신 표시를 보세요.") from None
    except InferenceError as exc:
        if getattr(exc, "code", "") == "MODEL_NOT_ALLOWED":
            raise RuntimeError(f"GPU 노트북에 이 모델이 없습니다: {model_path}") from None
        raise RuntimeError(f"GPU 노트북이 모델 정보 요청을 거절했습니다 ({exc})") from None
    finally:
        client.close()
    if "files" not in out:
        raise RuntimeError("GPU 노트북 제어판이 옛 버전이라 UDP로 모델 정보를 주지 않습니다 (3.13.0 이상으로 업데이트)")
    return out


def model_local(model_path: str) -> Path:
    """This PC's copy of a model's config files and home pose (the weights stay on the GPU laptop)."""
    return remote.MODEL_CACHE / hashlib.sha1(model_path.encode()).hexdigest()[:16]


def fetch_model_udp(model_path: str) -> Path:
    """Local config.json/train_config.json (omx_run.py reads the policy settings from them), from the GPU laptop
    over UDP the first time; the home pose comes in the same reply and is kept too."""
    local = model_local(model_path)
    if all((local / n).exists() for n in ("config.json", "train_config.json")):
        return local
    info = udp_model_info(model_path)
    local.mkdir(parents=True, exist_ok=True)
    for name, text in info["files"].items():
        if name in ("config.json", "train_config.json"):
            (local / name).write_text(text)
    if not (local / "config.json").exists():
        raise RuntimeError(f"GPU 노트북이 모델 설정 파일(config.json)을 보내지 않았습니다: {model_path}")
    if info.get("home") and not (local / "home_pose.json").exists():
        (local / "home_pose.json").write_text(json.dumps(info["home"], ensure_ascii=False))
    (local / "SOURCE").write_text(f"{remote.HOST}:{model_path}\n")
    return local


def run_from_path(model_path: str) -> dict:
    """A run entry made from the model path alone (the GPU index was never reached): .../<run>/pretrained_model or
    .../<run>/checkpoints/<step>/pretrained_model."""
    p = Path(model_path)
    parts = p.parts
    step = names.checkpoint_step(model_path)
    run = parts[-4] if step is not None and len(parts) >= 4 else (p.parent.name if p.name == "pretrained_model" else p.name)
    return {"id": run, "model_path": model_path, "step": step}


class UdpLink:
    """Every few seconds: while the SSH tunnel is up, ask the GPU laptop's panel to keep its UDP server running (it
    remembers the request and starts the server by itself after a restart, so the tunnel is not needed later);
    then PING the server straight over UDP. When it fails, say where: settings, the GPU panel, the server, or the
    network."""

    EVERY_S = 5.0

    def __init__(self):
        self.state = self._blank()
        self.client, self.key = None, None
        self.wake = threading.Event()
        self.thread = None

    @staticmethod
    def _blank() -> dict:
        return {"ok": None, "round_ms": None, "error": "", "hint": "", "stage": "", "loaded": [], "at": None,
                "server": None, "peer": ""}

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def reset(self) -> None:
        self.key = None
        self.state = self._blank()
        self.wake.set()

    @staticmethod
    def own_address(host: str, port: int) -> str:
        """This PC's address on the way to the GPU laptop (the one its UDP server must allow)."""
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((host, port))
            return s.getsockname()[0]
        finally:
            s.close()

    def _ask_gpu(self, cfg: dict) -> tuple[dict | None, str]:
        import omx_infer
        import omx_udp_keeper

        host, port = omx_infer.udp_peer(cfg["udp_server"])
        peer = self.own_address(host, port)
        self.state["peer"] = peer
        body = {"action": "ensure", "bind": host, "port": port, "peer": peer,
                "key_fp": omx_udp_keeper.key_fingerprint(omx_infer.read_udp_key(cfg["udp_key_file"]))}
        if not remote.TUNNEL.status()["ok"]:
            return None, "SSH 터널 끊김 (" + (remote.TUNNEL.error or "연결 중") + ")"
        try:
            return remote.gpu_json("/api/udp_server", body, timeout=8.0), ""
        except Exception as exc:
            msg = str(exc)
            if msg == "not found":
                msg = "GPU 노트북 제어판이 옛 버전이라 UDP 서버를 자동으로 켜지 못합니다 (3.13.0 이상 필요: GPU에 새 버전 설치)"
            return None, msg

    @staticmethod
    def probe(cfg: dict) -> None:
        """One PING on its own socket (the watch thread keeps its own client)."""
        import omx_infer
        from infer_link.client import InferenceClient

        client = InferenceClient(omx_infer.udp_peer(cfg["udp_server"]), omx_infer.read_udp_key(cfg["udp_key_file"]))
        try:
            client.ping(timeout=1.5)
        finally:
            client.close()

    def _ping(self, cfg: dict) -> dict:
        import omx_infer
        from infer_link.client import InferenceClient

        key = (cfg["udp_server"], cfg["udp_key_file"])
        if self.key != key:
            if self.client is not None:
                self.client.close()
            self.client = None
            self.client = InferenceClient(omx_infer.udp_peer(cfg["udp_server"]),
                                          omx_infer.read_udp_key(cfg["udp_key_file"]))
            self.key = key
        return self.client.ping(timeout=1.0)

    @staticmethod
    def diagnose(exc: Exception, srv: dict | None, gpu_err: str, cfg: dict, peer: str) -> tuple[str, str, str, bool | None]:
        """(stage, error, hint, ok) for a PING that failed; ok None = still starting (not an error yet)."""
        from infer_link.client import InferenceError, InferenceTimeout

        port = cfg["udp_server"].rsplit(":", 1)[-1]
        if not isinstance(exc, (InferenceTimeout, InferenceError)):
            return "settings", str(exc), "③ 평가 설정의 UDP 서버 주소와 키 파일을 확인하세요", False
        if isinstance(exc, InferenceError):
            return "server", f"GPU 노트북의 UDP 서버가 거절했습니다 ({exc})", "", False
        if srv is None:
            if gpu_err:
                return "gpu_panel", f"UDP 응답 없음. GPU 노트북 제어판에도 닿지 않아 서버 상태를 알 수 없습니다: {gpu_err}", \
                    "GPU 노트북이 켜져 있는지, 그 제어판이 켜져 있는지(제어판이 UDP 서버를 켬), 두 PC가 같은 공유기인지 확인하세요", False
            return "network", "UDP 응답 없음", "", False
        if srv.get("error"):
            return "server", srv["error"], srv.get("hint", ""), False
        if srv.get("starting"):
            return "starting", "GPU 노트북에서 UDP 서버를 켜는 중입니다 (처음 몇십 초)", "", None
        if srv.get("external"):
            return "network", "UDP 응답 없음 (GPU 노트북에서 직접 켠 서버를 쓰는 중)", srv.get("hint", ""), False
        if not srv.get("running"):
            return "server", "GPU 노트북의 UDP 서버가 꺼져 있습니다", "잠시 뒤 다시 켭니다. 계속되면 아래 서버 기록을 보세요", False
        if (srv.get("self_ping") or {}).get("ok") is False:
            return "server", "GPU 노트북의 UDP 서버가 응답하지 않습니다 (모델을 불러오는 중이거나 멈춤)", \
                "계속되면 아래 서버 기록을 보세요", False
        if peer and peer not in (srv.get("peers") or []):
            return "network", f"GPU 노트북 서버가 이 PC 주소({peer})를 허용 목록에 아직 넣지 않았습니다", "잠시 뒤 다시 확인합니다", None
        return "network", "GPU 노트북의 UDP 서버는 켜져 있지만 이 PC에서 닿지 않습니다 (방화벽 또는 다른 공유기)", \
            f"GPU 노트북 터미널에서: sudo ufw allow from {peer or '이 PC 주소'} to any port {port} proto udp " \
            "· 두 PC가 같은 공유기(같은 Wi-Fi)인지도 확인", False

    def _run(self) -> None:
        while True:
            cfg = infer_link()
            srv, gpu_err = None, ""
            try:
                srv, gpu_err = self._ask_gpu(cfg)
            except Exception as exc:  # settings: address or key file
                gpu_err = str(exc)
            busy = JOB.running() and JOB.kind == "eval"
            if busy and self.state.get("ok"):
                # the evaluation is using the server; its own log shows the round trips
                self.state = dict(self.state, server=srv, at=time.time())
            else:
                try:
                    out = self._ping(cfg)
                    self.state = {**self._blank(), "ok": True, "round_ms": self.client.last.get("round_ms"),
                                  "stage": "ok", "server": srv, "peer": self.state.get("peer", ""),
                                  "hint": (srv or {}).get("hint", "") if srv and srv.get("external") else
                                  (f"추론 통신은 정상입니다. GPU 노트북 서버 상태만 확인하지 못했습니다: {gpu_err}" if gpu_err else ""),
                                  "loaded": [Path(m).parent.name if m.endswith("pretrained_model") else Path(m).name
                                             for m in out.get("loaded") or []], "at": time.time()}
                except Exception as exc:
                    stage, err, hint, ok = self.diagnose(exc, srv, gpu_err, cfg, self.state.get("peer", ""))
                    self.state = {**self._blank(), "ok": ok, "stage": stage, "error": err, "hint": hint,
                                  "server": srv, "peer": self.state.get("peer", ""), "at": time.time()}
            self.wake.wait(self.EVERY_S)
            self.wake.clear()


UDP_LINK = UdpLink()


def infer_link_status() -> dict:
    cfg = infer_link()
    return {"transport": "udp", "udp_server": cfg["udp_server"], "udp_key_file": cfg["udp_key_file"],
            "port_hint": udp_port_hint(), "udp": UDP_LINK.state}


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
    if opts.get("constant_inputs"):  # 3.25.0: e.g. observation.environment_state of the one-arm right models
        overrides["constant_inputs"] = opts["constant_inputs"]
    if opts.get("rtc"):
        overrides["rtc"] = {"horizon": int(body.get("rtc_horizon") or 10), "weight": float(body.get("rtc_weight") or 10.0)}
    env = dict(async_env)  # may hold only the output filter's settings
    if "OMX_ASYNC" not in env:  # synchronous: the whole chunk, then the next one
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
        env.update({"OMX_ASYNC": "1", "OMX_ASYNC_CHUNK": str(chunk), "OMX_ASYNC_THRESHOLD": f"{threshold:.2f}",
                    "OMX_ASYNC_AGG": "latest_only", "OMX_FPS": "30", "OMX_RTC": "0"})
    env.update(OMX_REMOTE_MODEL=gpu_model, OMX_REMOTE_OVERRIDES=json.dumps(overrides))
    link = udp_ready()
    env.update(OMX_REMOTE_URL=f"udp://{link['udp_server']}",
               OMX_UDP_KEY=str(Path(link["udp_key_file"]).expanduser()))
    return env


ARM_KO = {"left": "왼팔", "right": "오른팔"}
SLOT_ARM = {"A": "left", "C": "right"}  # shelf column -> the arm that works it alone (control tower SLOTS, 11/24)
ONE_ARM_ASYNC = {"async_mode": True, "actions_per_chunk": 100, "chunk_size_threshold": 0.2, "aggregate_fn": "weighted_average"}


def one_arm(body: dict) -> str | None:
    """3.21.0: the person runs a one-arm (6-joint) model on one arm of the cooperative robot ("arm": "left" | "right").
    That arm alone is driven as a single OMX (its own port, the shared calibration); the other arm is not touched.
    Panel evaluations only: a control-tower mission (Pinky check, shelf alignment, handover) still needs both arms."""
    arm = str(body.get("arm") or "").strip()
    if arm in ("", "both"):
        return None
    if arm not in ARM_KO:
        raise RuntimeError("실행할 팔은 left(왼팔) 또는 right(오른팔)입니다.")
    if ROBOT_MODE != "bimanual":
        raise RuntimeError("한 팔 선택은 협동 로봇(팔 2개) 구성에서만 씁니다. 지금은 단일 로봇 구성이라 팔을 고르지 않고 시작하세요.")
    if body.get("source") == "mqtt" or body.get("pinky_check") or body.get("end_on_place") or body.get("end_on_return"):
        # 3.25.0 (TK 11/24): a control-tower one-arm mission (arm_action_command): column A = OMX 1 (left), C = OMX 2
        # (right), level 2 (the judges know 2A / 2C). Loading ends on the placement judge, unloading on the tray judge
        # with the camera check after the arm is home; no shelf alignment, no hand-over.
        slot = command_slot(body) or ""
        if SLOT_ARM.get(slot[1:]) != arm:
            raise RuntimeError(f"한팔 관제 미션은 칸 A(왼팔)·C(오른팔)를 그 팔로 보냅니다 (받은 칸 {slot or '없음'}, 팔 {ARM_KO[arm]}).")
        if slot[:1] != "2":
            raise RuntimeError(f"한팔 관제 미션은 아직 2층(2A·2C)만 됩니다 (받은 칸 {slot}).")
        if body.get("shelf_align") or body.get("place_shadow"):
            raise RuntimeError("한팔 미션에는 선반 정렬·넘겨받기가 없습니다. shelf_align·place_shadow 없이 보내세요.")
        if bool(body.get("pinky_check")) == bool(body.get("end_on_place")):
            raise RuntimeError("한팔 관제 미션은 입고면 end_on_place, 출고면 pinky_check 하나만 보냅니다.")
    return arm


def one_arm_robot_args(arm: str) -> list:
    """That arm as a single OMX: same port and calibration file (omx_follower_arm) as in the pair, so the joint
    values match the per-arm training data cut from two-arm recordings."""
    return ["--robot.type=omx_follower", f"--robot.port={FOLLOWERS[0 if arm == 'left' else 1][1]}",
            "--robot.id=omx_follower_arm"]


def model_camera_map(cfg: dict, arm: str | None = None) -> tuple[list, dict]:
    """3.25.0: the panel cameras a model reads, and the names it learned them under when those differ (rename_map
    panel name -> model name). The right-arm models trained on the one-arm datasets (TK 11/24, #36 / #37) use
    shelf = front_2, pinky_side = the side top view of that arm (front_4 right, front_3 left), wrist = that arm's
    wrist camera. Raises when a model camera has no panel camera."""
    names = [k.split(".")[-1] for k in (cfg.get("input_features") or {}) if k.startswith("observation.images.")]
    used, rename = [], {}
    for n in names:
        if n in CAMERAS or n.startswith("camera"):
            used.append(n)
            continue
        alias = {"shelf": "front_2", "top": "front",
                 "wrist": f"wrist_{arm}" if arm else ("wrist" if "wrist" in CAMERAS else None),
                 "pinky_side": {"right": "front_4", "left": "front_3"}.get(arm or "")}.get(n)
        if not alias or alias not in CAMERAS:
            raise RuntimeError(f"이 모델의 카메라 '{n}'에 맞는 카메라가 지금 구성에 없습니다. 지금 구성의 카메라: {', '.join(CAMERAS)}.")
        used.append(alias)
        rename[f"observation.images.{alias}"] = f"observation.images.{n}"
    return used, rename


def constant_inputs(cfg: dict, body: dict) -> dict:
    """3.25.0: model inputs that are not read from the robot. observation.environment_state [2] of the one-arm right
    models (#36 / #37) is the shelf level one-hot: [1, 0] level 1, [0, 1] level 2 (it was [0, 1] all through their
    training, Guide 11/24). The level comes from the commanded slot ("2A" -> level 2), else level 2."""
    feat = (cfg.get("input_features") or {}).get("observation.environment_state")
    if not feat:
        return {}
    shape = feat.get("shape") or []
    if shape != [2]:
        raise RuntimeError(f"이 모델의 observation.environment_state 모양 {shape}을 패널이 채울 수 없습니다 (층 one-hot [2]만 지원).")
    slot = str(body.get("slot") or "").upper()
    level = int(slot[0]) if slot[:1] in ("1", "2") else 2
    return {"observation.environment_state": [1.0, 0.0] if level == 1 else [0.0, 1.0]}


def check_model_fits(cfg: dict, arm: str | None = None) -> None:
    """A model drives the robot it was trained on: 6 joints = one arm, 12 = a cooperative pair; and the same
    cameras by name. On a cooperative robot a 6-joint model runs on the arm the person chose (3.21.0)."""
    shape = ((cfg.get("output_features") or {}).get("action") or {}).get("shape") or [None]
    cams = {k.split(".")[-1] for k in (cfg.get("input_features") or {}) if k.startswith("observation.images.")}
    if arm:  # 3.25.0: a one-arm model's own camera names map to the panel's (model_camera_map)
        cams = set(model_camera_map(cfg, arm)[0])
    want = 12 if ROBOT_MODE == "bimanual" and not arm else 6
    if shape[0] == 6 and want == 12:
        side = "left" if "wrist_left" in cams and "wrist_right" not in cams else "right" if "wrist_right" in cams and "wrist_left" not in cams else ""
        raise RuntimeError("이 모델은 한 팔(관절 6개)용입니다. 평가 설정의 '실행할 팔'에서 "
                           + (f"{ARM_KO[side]}을 고르세요 (손목 카메라 wrist_{side}로 학습)." if side else "왼팔 또는 오른팔을 고르세요."))
    if shape[0] == 12 and arm:
        raise RuntimeError("이 모델은 두 팔(관절 12개)용입니다. '실행할 팔'을 두 팔로 두세요.")
    if shape[0] and shape[0] != want:
        kind = {6: "단일 로봇(팔 1개)", 12: "협동 로봇(팔 2개)"}
        raise RuntimeError(f"이 모델은 {kind.get(shape[0], f'관절 {shape[0]}개')}용입니다. 지금 구성은 "
                           f"{kind[want]}입니다. ① 탭 '로봇 구성'을 맞추거나 다른 모델을 고르세요.")
    if cams and not any(c.startswith("camera") for c in cams) and not cams <= set(CAMERAS):
        raise RuntimeError(f"이 모델은 카메라 {', '.join(sorted(cams))}로 학습했습니다. 지금 구성의 카메라: "
                           f"{', '.join(CAMERAS)}. ① 탭 '로봇 구성'에서 같은 이름으로 지정하세요.")
    other = {"left": "wrist_right", "right": "wrist_left"}.get(arm or "")
    if other and other in cams:
        raise RuntimeError(f"이 모델은 {ARM_KO['right' if arm == 'left' else 'left']} 손목 카메라({other})로 학습했습니다. "
                           f"'실행할 팔'을 {ARM_KO['right' if arm == 'left' else 'left']}로 고르세요.")


def eval_cameras(cfg: dict, arm: str | None = None) -> dict:
    """The cameras an evaluation opens: all of the setup, except an extra global camera (front_3 / front_4) the model
    was not trained with (3.19.0). The usual cameras stay open as before: the judges read front / front_2.
    One arm (3.21.0): only the model's cameras (no judges run; the other arm's wrist camera stays closed)."""
    used = {k.split(".")[-1] for k in (cfg.get("input_features") or {}) if k.startswith("observation.images.")}
    if arm:
        # 3.25.0 (TK 11/24): every global camera (the judges read front / front_2 / front_3 / front_4) and this arm's
        # wrist camera; the model is given only the ones it learned (OMX_POLICY_CAMERAS, one_arm_env)
        return {n: p for n, p in CAMERAS.items() if not n.startswith("wrist") or n == f"wrist_{arm}"}
    return {n: p for n, p in CAMERAS.items() if n not in EXTRA_CAMERA_ROLES or n in used}


def one_arm_env(cfg: dict, arm: str | None, body: dict) -> dict:
    """3.25.0: a one-arm run feeds its model only the cameras it learned (the others are open for the judges), and the
    model's constant inputs (environment_state) - this process; the GPU laptop gets them in the overrides."""
    if not arm:
        return {}
    used, _ = model_camera_map(cfg, arm)
    env = {"OMX_POLICY_CAMERAS": json.dumps(used)}
    consts = constant_inputs(cfg, body)
    if consts:
        env["OMX_CONST_INPUTS"] = json.dumps(consts)
    return env


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


WARM_FLAG = "--omx-warm"  # omx_run.py: wait, imports done, for the next evaluation (3.22.2)
WARM_BUSY = "/dev/shm/omx_warm_busy_{pid}"  # written by that process when it takes a run (it holds the robot then)
# read when torch / numpy / LeRobot are imported: a ready process started with other values is not used
WARM_IMPORT_KEYS = re.compile(r"^(OMP_|MKL_|OPENBLAS_|HF_|CUDA_|PYTORCH_|TORCH_|LEROBOT|PYTHON)")
WARM_IDLE_S = 5.0  # nothing ran here this long: start the next ready process (never during a run: its imports use CPU)
WARM_RETRY_S = 30.0  # a ready process that ended by itself is started again only after this


class Warm:
    """3.22.2: one omx_run.py kept ready for the next evaluation (python, torch, LeRobot and the panel's modules
    imported, about 2-3 s of each start). It holds no robot, camera or model, so it is not a job: /api/status
    job.running stays false until an evaluation takes it. An evaluation hands it its arguments and environment on
    stdin (omx_run.warm_wait); anything that does not fit (another python, other thread settings) starts as before.
    It ends by itself when the panel ends (its stdin closes). cfg "warm_eval": false turns it off."""

    def __init__(self) -> None:
        self.proc, self.cmd, self.env, self.started, self.error = None, None, None, 0.0, None
        self.idle_since = time.time()
        self.lock = threading.Lock()

    def enabled(self) -> bool:
        return CFG.get("warm_eval", True) is not False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self) -> None:
        """From the status poll: one ready process while nothing runs here and an arm of this setup is plugged in."""
        if JOB.running() or STANDBY.alive():
            self.idle_since = time.time()
            return
        if not self.enabled() or not robot_here() or time.time() - self.idle_since < WARM_IDLE_S:
            return
        with self.lock:
            if self.alive() or (self.proc is not None and time.time() - self.started < WARM_RETRY_S):
                return
            cmd = [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_run.py"), WARM_FLAG]
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMX_PORT=str(PORT), **thread_env(EVAL_THREADS))
            self.proc = subprocess.Popen(cmd, cwd=LEROBOT_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self.cmd, self.env, self.started, self.error = cmd, env, time.time(), None

    def take(self, cmd: list, env: dict):
        """The ready process, now running this evaluation; None: start it as before."""
        with self.lock:
            proc, self.proc = self.proc, None
            self.idle_since = time.time()
            if proc is None:
                return None
            same = {k: v for k, v in self.env.items() if WARM_IMPORT_KEYS.match(k)} == \
                   {k: v for k, v in env.items() if WARM_IMPORT_KEYS.match(k)}
            if proc.poll() is not None or list(cmd[:2]) != self.cmd[:2] or not same:
                self._end(proc)
                return None
            try:
                proc.stdin.write(json.dumps({"argv": list(cmd[2:]), "env": env}, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except OSError:
                self._end(proc)
                return None
            return proc

    @staticmethod
    def _end(proc) -> None:
        if proc.poll() is None:
            proc.terminate()  # it waits on stdin and holds nothing: nothing to land

    def stop(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
        if proc is not None:
            self._end(proc)

    def status(self) -> dict:
        return {"on": self.enabled(), "ready": self.alive(), "pid": self.proc.pid if self.alive() else None,
                "age_s": round(time.time() - self.started) if self.alive() else None, "error": self.error}


WARM = Warm()


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


HOLD_REASON_KO = {"NO_HOLD": "Host에 HOLD가 없습니다", "HOLD_CHANGED": "그사이 다른 HOLD로 바뀌었습니다. 화면을 다시 보세요",
                  "CHECKS_MISSING": "확인 항목이 빠졌습니다", "TASK_ACTIVE": "Host가 다른 작업을 진행 중입니다",
                  "EXECUTOR_BUSY": "로봇 작업이 실행 중입니다", "ARMS_NOT_HOME": "두 팔이 홈 자세가 아니거나 그리퍼를 놓지 않았습니다",
                  "STALE_REQUEST": "요청이 오래되었거나 형식이 맞지 않습니다",
                  "NO_RESPONSE": "Host가 10초 안에 답하지 않아 요청을 취소했습니다. HOLD는 그대로입니다. Host가 실행 중인지, 브로커에 연결되어 있는지(위 관제 연결 표시) 확인한 뒤 다시 누르세요"}


def host_hold_release(body: dict) -> dict:
    """The person's [HOLD 해제]: checks on screen, no robot job, arms readable; then the Host decides (it checks the
    arms itself and never turns armed on)."""
    st = omx_host.hold_state()
    hold = st.get("hold")
    if not st.get("available"):
        raise RuntimeError("Host가 HOLD 상태 파일을 쓰지 않습니다 (옛 Host).")
    if not hold:
        raise RuntimeError("Host에 HOLD가 없습니다.")
    if str(body.get("hold") or "") != str(hold.get("command_id") or ""):
        raise RuntimeError("그사이 HOLD가 바뀌었습니다. 화면의 내용을 다시 확인하세요.")
    checks = body.get("checks") or {}
    missing = [k for k in omx_host.HOLD_CHECKS if checks.get(k) is not True]
    if missing:
        raise RuntimeError("확인 항목을 모두 체크하세요.")
    if JOB.running() or other_robot_processes():
        raise RuntimeError("로봇 작업이 실행 중입니다. 끝난 뒤 해제하세요.")
    arms = arms_now()
    if arms.get("error") or not arms.get("arms"):
        raise RuntimeError(f"팔 위치를 읽지 못했습니다: {arms.get('error') or '응답 없음'}")
    res = omx_host.request_hold_release(str(hold.get("command_id")), checks)
    res["reason_ko"] = HOLD_REASON_KO.get(res.get("reason") or "", res.get("reason") or "")
    return res


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
    if ALIGN.get("hold"):  # a shelf alignment stopped in place: its runner lands the arms
        start_align(dict(ALIGN), release=True)
        return {"ok": True, "mode": "align_release"}
    if not EVAL_HOME_FILE.exists():
        raise RuntimeError("돌아갈 홈 자세 기록이 없습니다.")
    data = json.loads(EVAL_HOME_FILE.read_text())
    info = {"phase": "홈 위치로 이동 중", "repo_id": data.get("repo_id", "")}
    if HANDOVER.pop("hold", None):  # a handover the runner did not take: the arms wait with the torque on (3.18.0)
        info["handover_fallback"] = "hold_release"
    JOB.start("home", [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_return_home.py"), str(EVAL_HOME_FILE),
                       data.get("repo_id", "")],
              info, need_leader=False, env_extra={"OMX_HOME_GRIPPER": "open"})
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
MQTT_HOST = omx_host.Host(LOG_DIR)  # armed on → started if configured; keeps running when armed goes off
EPISODE_END_FILE = LOG_DIR / "episode_end.json"  # omx_run.py: the arms at the end of each episode (eval)
EVAL_HOME_POSE: dict = {}  # the running / last eval's model home pose
LAST_RESULT: dict = {}  # how the last evaluation ended (for the control tower), until the next one ends


def set_armed(on: bool, reason: str = "") -> dict:
    """Turned on only from the panel screen; any stop, abnormal end or restart turns it off."""
    if on and not ARMED["on"]:
        ARMED.update(on=True, since=dt.datetime.now().isoformat(timespec="seconds"), off_reason="")
    if on:
        MQTT_HOST.ensure()  # also when already armed: restarts a Host that exited
    elif not on and ARMED["on"]:
        ARMED.update(on=False, since=None, off_reason=reason or "패널에서 끔")
    return armed_status()


def armed_status() -> dict:
    return {"armed": ARMED["on"], "armed_since": ARMED["since"], "armed_off_reason": ARMED["off_reason"]}


def arm_states(state: dict, home: dict | None, arm: str | None = None) -> dict:
    """Per arm (set name): joints at the end of the last episode, gripper, and whether it is at the model's
    home pose (omx_home.diff, gripper left out). None where it was not measured: never a guess.
    arm (3.25.0): a one-arm model / run on the cooperative robot - its unprefixed joint names are that arm's."""
    if arm in ("left", "right"):
        def side_names(d):
            return {(k if k.startswith(("left_", "right_")) else f"{arm}_{k}"): v for k, v in (d or {}).items()}
        state = side_names(state)
        if home:
            home = dict(home, joints=side_names(home.get("joints")), std=side_names(home.get("std")))
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


ARMS_NOW = {"t": 0.0, "current": None, "error": None, "lock": threading.Lock()}
ARMS_NOW_TTL_S = 10.0  # one bus read (about 3.5 s for two arms) serves readiness polls this long
MODEL_HOMES: dict = {}  # model_path -> (time, home pose)
MODEL_HOME_TTL_S = 300.0


def model_home_pose(model_path: str) -> dict | None:
    """The model's home pose (where an evaluation starts and its safe stop returns), cached."""
    hit = MODEL_HOMES.get(model_path)
    if hit and time.time() - hit[0] < MODEL_HOME_TTL_S:
        return hit[1]
    if CONTROL:
        home = control_home_pose(model_path)
    else:
        home = omx_home.model_home(model_path)
    MODEL_HOMES[model_path] = (time.time(), home)
    return home


def control_home_pose(model_path: str) -> dict | None:
    """Control PC: the copy kept here (written once, like the GPU laptop's home_pose.json), else ask over UDP."""
    f = model_local(model_path) / "home_pose.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        pass
    if UDP_LINK.state.get("ok") is False:  # do not keep a caller (the Host's readiness poll) waiting on a dead link
        raise RuntimeError("AI 추론 서버와 UDP 연결 안 됨: " + (UDP_LINK.state.get("error") or ""))
    info = udp_model_info(model_path, timeout=4.0)
    home = info.get("home")
    if home:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(home, ensure_ascii=False))
        for name, text in (info.get("files") or {}).items():  # the same reply has the config files
            if name in ("config.json", "train_config.json") and not (f.parent / name).exists():
                (f.parent / name).write_text(text)
    elif info.get("home_error"):
        raise RuntimeError(f"GPU 노트북이 홈 자세를 만들지 못했습니다 ({info['home_error']})")
    return home


def arms_now(model_path: str = "", arm: str | None = None) -> dict:
    """/api/arms: where the arms are now, for the control tower's readiness check (MQTT Host).
    Read from the bus (no torque change) only while no robot program runs; otherwise unknown.
    One read is reused for ARMS_NOW_TTL_S unless a robot program ran since.
    at_home compares with the model's home pose when a model is given. Unknown stays None."""
    out = {"source": None, "at": None, "arms": None, "error": None}
    if JOB.running() or other_robot_processes():
        return out | {"source": "busy", "error": "로봇 프로그램이 실행 중이라 팔 위치를 따로 읽지 않습니다."}
    with ARMS_NOW["lock"]:
        # a robot program that started after the last read may have moved the arms: read again
        moved = (getattr(JOB, "started", 0) or 0) > ARMS_NOW["t"]
        if moved or time.time() - ARMS_NOW["t"] >= ARMS_NOW_TTL_S:
            try:
                ARMS_NOW.update(current=home_current()["current"], error=None)
            except Exception as exc:
                ARMS_NOW.update(current=None, error=str(exc)[:200])
            ARMS_NOW["t"] = time.time()
        current, error, at = ARMS_NOW["current"], ARMS_NOW["error"], ARMS_NOW["t"]
    out.update(source="bus", at=dt.datetime.fromtimestamp(at).isoformat(timespec="seconds"), error=error)
    if current is None:
        return out
    home = None
    if model_path:
        try:
            home = model_home_pose(model_path)
        except Exception as exc:
            out["error"] = f"모델 홈 자세를 알 수 없습니다: {exc}"[:200]
    out["arms"] = arm_states({f"{k}.pos": v for k, v in current.items()}, home, arm if arm in ("left", "right") else None)
    out["home_known"] = bool(home)
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
    arms = arm_states(end.get("state") or {}, home, info.get("arm")) if end else {}
    checks = [a["at_home"] for a in arms.values()]
    return {"kind": "eval", "source": info.get("source", "panel"), "repo_id": info.get("repo_id"),
            "recorded": info.get("record", True),
            "started_at": dt.datetime.fromtimestamp(job.started).isoformat(timespec="seconds"),
            "ended_at": dt.datetime.now().isoformat(timespec="seconds"), "returncode": rc,
            "exit_ok": rc == 0 and not err and stopped is None, "stopped_by": stopped, "err": err,
            "target_marker": info.get("target_marker"), "target_drawn": info.get("target_drawn"),
            "last_episode": {k: end.get(k) for k in ("episode", "at", "aborted", "returned", "elapsed_s", "end_reason",
                                                     "judged_at", "loop_ms")} if end else None,
            "arms": arms or None, "wrist_check": info.get("wrist_check"),
            "tray_judge": (end or {}).get("tray_judge"),
            "drop_watch": (end or {}).get("drop_watch"),
            "home_check": "unknown" if not checks or None in checks else "ok" if all(checks) else "far"}


# ---------------------------------------------------------------- 선반 정렬 (관제 평가 뒤 이어서)
ALIGN_RESULT_FILE = LOG_DIR / "align_result.json"  # the runner's result (--result)
# the runner writes this as soon as it has read the shelf, before it aligns (3.17.0): "the box is on its slot"
PLACEMENT_FILE = ALIGN_RESULT_FILE.with_suffix(".placement.json")
# the eval's model home pose, handed to the runner (--home-pose): after aligning it goes straight there (3.17.1)
ALIGN_HOME_FILE = LOG_DIR / "align_home_pose.json"
SLOT_RE = re.compile(r"[1-9][A-Z]")
PINKY_RE = re.compile(r"pinky(\d+)", re.IGNORECASE)
ALIGN_DELAY_S = 0.3  # after the eval let go of the ports and cameras (the runner waits for its own start time)
PINKY_RESULT_FILE = LOG_DIR / "pinky_result.json"  # the runner's --step pinky_read result
PINKY: dict = {}  # the running pinky_read: proc, timed_out
ALIGN_STOP_WAIT_S = 5.0  # 즉시 중단 / 비상 정지: force-quit the runner when it has not exited by then
ALIGN: dict = {}  # box and slot of the last alignment; "hold" while its arms wait for [홈 위치로 이동]
ALIGN_REST_TOL = 5.0  # joint units: after the alignment the arms are back where it found them (gripper left out)


def align_settings() -> dict | None:
    """config "shelf_align": {"cmd": [python, real_runner.py], "models": [part of the model path, ...],
    "slot": "2B", "draw_target": false}. Missing or without cmd = off."""
    s = omx_config.CFG.get("shelf_align") or {}
    cmd = [str(Path(c).expanduser()) for c in s.get("cmd") or []]
    return dict(s, cmd=cmd) if cmd else None


def read_arms() -> dict | None:
    """Both followers' positions now, read over the bus (no torque change, no motion). None if unreadable."""
    try:
        return home_current()["current"]
    except Exception:
        return None


def rest_check(before: dict | None, after: dict | None) -> dict:
    """Are the arms back at the pose the alignment started from (the eval's landed pose)? Per arm, every joint
    but the gripper within ALIGN_REST_TOL; "unknown" when either reading is missing."""
    arms = {}
    for side, name in ARM_NAMES.items():
        pre = f"{side}_" if side else ""
        joints = {k[len(pre):]: v for k, v in (after or {}).items() if k.startswith(pre)} or None
        g = (joints or {}).get("gripper")
        far = sorted(k[len(pre):] for k, v in (after or {}).items() if k.startswith(pre) and not k.endswith("gripper")
                     and (before or {}).get(k) is not None and abs(v - before[k]) > ALIGN_REST_TOL)
        arms[name] = {"side": side or None, "joints": joints, "gripper": g,
                      "gripper_state": None if g is None else "open" if g >= omx_home.GRIP_OPEN else "closed" if g < omx_home.GRIP_CLOSED else "half",
                      "far_joints": far}
    check = "unknown" if not before or not after else "far" if any(a["far_joints"] for a in arms.values()) else "ok"
    return {"check": check, "tolerance": ALIGN_REST_TOL, "arms": arms}


def align_done_later(proc) -> None:
    """When the runner (or its landing) has let go of the ports: read where the arms ended, then final."""
    def run() -> None:
        end = time.time() + 30
        while JOB.proc is proc and JOB.running() and time.time() < end:
            time.sleep(0.2)
        time.sleep(0.5)
        a = LAST_RESULT.get("align") or {}
        a["rest"] = rest_check(ALIGN.get("before"), read_arms())
        LAST_RESULT.update(stage="done", final=True)

    threading.Thread(target=run, daemon=True).start()


def now_iso() -> str:
    """Local time with milliseconds and the time zone, e.g. 2026-11-17T18:30:01.123+09:00 (the Host compares)."""
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def command_slot(body: dict) -> str | None:
    """The control tower's slot ("2B"), or None when the command has none (the settings' slot is used)."""
    raw = body.get("slot")
    if raw is None or str(raw).strip() == "":
        return None
    slot = str(raw).strip().upper()
    if not SLOT_RE.fullmatch(slot):
        raise RuntimeError(f"칸(slot)은 2B처럼 층 숫자 하나와 칸 글자 하나로 보내세요 (받은 값 {raw!r}).")
    return slot


def command_pinky(body: dict) -> dict:
    """"pinky_id": "pinky1" -> the runner's --pinky 1 (that Pinky's tray marker reference). Missing or another
    form: no --pinky (the common reference), and the value is kept to show why."""
    raw = body.get("pinky_id")
    m = PINKY_RE.fullmatch(str(raw).strip()) if raw is not None else None
    out = {"pinky_id": raw, "pinky_no": int(m.group(1)) if m else None}
    if raw is not None and not m:
        out["pinky_id_ignored"] = True
    return out


def take_placement(source: str) -> bool:
    """The runner's placement file -> last_result.placement (once per alignment / shelf read). A handover's first
    report from the eval's judge (source "eval_judge", 3.18.0) is replaced by the runner's own shelf read, kept as "first"."""
    prev = LAST_RESULT.get("placement")
    if prev and prev.get("source") != "eval_judge":
        return True
    try:
        d = json.loads(PLACEMENT_FILE.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(d, dict):
        return False
    d.setdefault("on_shelf", None)
    LAST_RESULT["placement"] = dict(d, source=source, reported_at=now_iso(), eval_end=eval_end_evidence(),
                                    **({"first": {k: prev.get(k) for k in ("source", "on_shelf", "observed_at", "reported_at")}} if prev else {}))
    return True


def watch_placement(proc, source: str) -> None:
    """While the runner works, pick up its placement file at once (the Host reports COMPLETE from it)."""
    def run() -> None:
        while JOB.proc is proc and JOB.running():
            if take_placement(source):
                return
            time.sleep(0.2)

    threading.Thread(target=run, daemon=True).start()


def placement_from_result(source: str, r: dict, slot: str | None, box) -> None:
    """The runner ended without a placement file (an older runner, a stop): its final result, when it has one."""
    if take_placement(source):
        return
    if (LAST_RESULT.get("placement") or {}).get("source") == "eval_judge" and r.get("on_shelf") is None:
        return  # the runner read nothing itself: keep the judge's report
    if "on_shelf" in r or "shelf" in r:
        LAST_RESULT["placement"] = {"on_shelf": r.get("on_shelf"), "shelf": r.get("shelf"), "slot": r.get("slot") or slot,
                                    "box": box, "observed_at": r.get("observed_at"), "observed_ts": r.get("observed_ts"),
                                    "observed_src": r.get("observed_src"), "source": source, "from_result": True,
                                    "eval_end": eval_end_evidence(), "reported_at": now_iso()}
    else:
        placement_skipped("결과 없음", slot, box)


def judge_evidence(end: dict, judge: dict) -> dict:
    """The eval's placed judgement (3.20.0, 11/19 TK + 관제): observed_* = when the top camera took the frame the
    judge used (camera read thread's stamp; the control step's time when unknown, observed_clock tells which);
    tips_clear = both fingertips behind the box front / shelf edge minus 3 cm (FK), with tip_y / y_out (m) and the
    grippers at that moment."""
    frame_ts = judge.get("frame_ts")
    ts = frame_ts or judge.get("judged_ts")
    ty = judge.get("tip_y") if isinstance(judge.get("tip_y"), list) else []
    y_out = judge.get("y_out")
    joints = judge.get("joints") or {}
    sides = ("left", "right")
    return {"observed_at": judge.get("frame_at") or end.get("judged_at"), "observed_ts": ts, "observed_src": "front",
            "observed_clock": "capture" if frame_ts else "control_step",
            "judged_at": end.get("judged_at"), "judged_ts": judge.get("judged_ts"),
            "tips_clear": bool(max(ty) < y_out) if len(ty) == 2 and y_out is not None else None,
            "tip_y": dict(zip(sides, ty)) if len(ty) == 2 else None, "y_out": y_out,
            "gripper": {s: joints.get(f"{s}_gripper") for s in sides}}


def eval_end_evidence() -> dict | None:
    """Where the arms were when the eval that just ran ended (3.20.0): its last measured pose, per arm at the model's
    home or not (gripper left out) and the gripper, with the measurement's time. None when not measured."""
    try:
        end = json.loads(EPISODE_END_FILE.read_text())
    except (OSError, ValueError):
        return None
    if end.get("repo_id") != LAST_RESULT.get("repo_id"):
        return None
    arms = {(a.get("side") or name): {"at_home": a.get("at_home"), "gripper": a.get("gripper"),
                                      "gripper_state": a.get("gripper_state")}
            for name, a in (LAST_RESULT.get("arms") or {}).items()}
    homes = [a["at_home"] for a in arms.values()]
    return {"at": end.get("state_at"), "ts": end.get("state_ts"), "end_reason": end.get("end_reason"),
            "returned": end.get("returned"), "aborted": end.get("aborted"), "arms": arms or None,
            "all_at_home": None if not homes or None in homes else all(homes)}


def placement_skipped(why: str, slot: str | None, box) -> None:
    LAST_RESULT["placement"] = {"on_shelf": None, "skipped": why, "slot": slot, "box": box, "observed_at": None,
                                "observed_ts": None, "reported_at": now_iso()}


def episode_end_pose() -> dict | None:
    """Where the eval's last episode ended (omx_run.py's episode_end.json), in read_arms() form."""
    try:
        end = json.loads(EPISODE_END_FILE.read_text())
    except (OSError, ValueError):
        return None
    state = end.get("state") if isinstance(end.get("state"), dict) else None
    return {k.removesuffix(".pos"): v for k, v in state.items()} if state else None


RUNNER_POSE_RE = re.compile(r"현재 자세 \[([-0-9., ]+)\]")
MOTOR_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def runner_start_pose(r: dict, lines) -> tuple[dict | None, str | None]:
    """Where the runner found the arms when it connected (it goes back there after aligning): its result's
    start_pose, else its log line "현재 자세 [...]" (left arm then right arm, motor order)."""
    sp = r.get("start_pose")
    if isinstance(sp, dict) and sp:
        return {k.removesuffix(".pos"): float(v) for k, v in sp.items()}, "runner"
    for line in lines:
        m = RUNNER_POSE_RE.search(line)
        if not m:
            continue
        try:
            vals = [float(x) for x in m.group(1).split(",")]
        except ValueError:
            return None, None
        sides = [s for s in ARM_NAMES] if ROBOT_MODE == "bimanual" else [""]
        names = [f"{s}_{j}" if s else j for s in sides for j in MOTOR_ORDER]
        if len(vals) == len(names):
            return dict(zip(names, vals)), "runner_log"
        return None, None
    return None, None


def align_plan(body: dict, run: dict) -> dict | None:
    """A control-tower eval with a target marker is followed by the shelf alignment of that box when the
    model is in the config list (or the command says "shelf_align": true / false)."""
    s = align_settings()
    marker = str(body.get("target_marker", "")).strip()
    if not s or body.get("source") != "mqtt" or not marker:
        return None
    want = body.get("shelf_align")
    if want is None:
        want = any(m and m in run["model_path"] for m in s.get("models") or [])
    if not want:
        return None
    return {"box": int(marker), "slot": command_slot(body) or str(s.get("slot") or "2B"),
            "draw_target": bool(s.get("draw_target", False))}


def align_after_eval(plan: dict, eval_proc) -> None:
    """Called when the eval has ended: only a clean eval with armed still on goes on to the alignment.
    last_result stays final=False until the alignment (and the landing after a stop) is over."""
    align = {"ran": False, "ok": None, "box": plan["box"], "slot": plan["slot"]}
    if not LAST_RESULT.get("exit_ok"):  # the Host fails the mission on the eval error itself: nobody looks
        LAST_RESULT["align"] = dict(align, ok=False, skipped="평가가 정상으로 끝나지 않음", on_shelf=None)
        placement_skipped("평가가 정상으로 끝나지 않음", plan["slot"], plan["box"])
        # 3.21.1: final here too. A handover eval that ended in an error (not placed) had already set stage "align"
        # before coming here, and nothing closed it: the Host waited for a final result (HOST_BUSY, 11/21 14:02)
        LAST_RESULT.update(stage="done", final=True)
        return
    LAST_RESULT.update(stage="align", final=False, align=align)

    def start_later() -> None:
        end = time.time() + 30
        while JOB.proc is eval_proc and JOB.running() and time.time() < end:
            time.sleep(0.2)
        time.sleep(ALIGN_DELAY_S)
        if JOB.proc is not eval_proc or JOB.running():
            # another job holds the robot and the cameras: nobody can look at the shelf (Host: PANEL_ALIGN_SKIPPED)
            LAST_RESULT["align"].update(ok=False, skipped="다른 작업이 먼저 시작됨", on_shelf=None)
            placement_skipped("다른 작업이 먼저 시작됨", plan["slot"], plan["box"])
            LAST_RESULT.update(stage="done", final=True)
            return
        why = f"armed 꺼짐 ({ARMED['off_reason']})" if not ARMED["on"] else None
        if not why:
            try:
                # the rest check's reference: where the runner finds the arms (its start pose, after it ends);
                # until then the eval's last pose. No bus read here: it cost about 2 s before the runner started
                ALIGN["before"] = episode_end_pose()
                start_align(plan)
                return
            except Exception as exc:
                why = f"시작하지 못함: {exc}"
                set_armed(False, "선반 정렬 " + why)
        start_shelf_read(plan, why)

    threading.Thread(target=start_later, daemon=True).start()


def pinky_plan(body: dict, run: dict, arm: str | None = None) -> dict | None:
    """Unloading (OUTBOUND, 11/16): a control-tower eval with "pinky_check": true is followed by a camera-only check
    (real_runner.py --step pinky_read: front = box on the Pinky tray, front_2 = left on the shelf). Checked at the
    start of the eval, so the Host hears of a wrong request before anything moves."""
    if body.get("pinky_check") is not True:
        return None
    if body.get("source") != "mqtt":
        raise RuntimeError("상차 확인(pinky_check)은 관제 명령으로 시작한 평가에서만 씁니다.")
    marker = str(body.get("target_marker", "")).strip()
    if not marker.isdigit():
        raise RuntimeError("상차 확인(pinky_check)에는 상자 번호(target_marker)가 필요합니다.")
    if arm:
        # 3.25.0: one arm. Its Pinky stands outside the arm (N21 / N22): the tray is seen by that side's top camera
        # (tray_judge), not by the runner's front camera. The tray judge ends the eval and checks the tray again once
        # the arm is home (omx_run TrayWatch.recheck), while the Pinky still waits (one report after the arm is home).
        return {"box": int(marker), "slot": command_slot(body), "one_arm": True, "arm": arm, **command_pinky(body)}
    if align_plan(body, run):
        raise RuntimeError("선반 정렬(shelf_align)과 상차 확인(pinky_check)은 함께 쓸 수 없습니다. 하차 명령은 \"shelf_align\": false로 보내세요.")
    a = omx_config.CFG.get("shelf_align") or {}
    s = omx_config.CFG.get("pinky_check") or {}
    cmd = [str(Path(c).expanduser()) for c in s.get("cmd") or a.get("cmd") or []]
    if not cmd:
        raise RuntimeError("설정 pinky_check(cmd) 또는 shelf_align(cmd)이 없어 상차 확인을 할 수 없습니다.")
    return {"box": int(marker), "slot": command_slot(body) or str(s.get("slot") or a.get("slot") or "2B"), "cmd": cmd,
            "timeout_s": float(s.get("timeout_s") or 20), **command_pinky(body)}


TRAY_RESULT_FILE = LOG_DIR / "tray_check_result.json"  # the runner's --step tray_check result (3.20.0)
TRAY_LOCK = threading.Lock()
TRAY_WHY = {"not_empty": "쟁반이 비어 있지 않습니다", "no_frame": "위 카메라(front) 사진을 찍지 못했습니다",
            "camera_moved": "위 카메라가 움직여 쟁반 위치를 믿을 수 없습니다",
            "tray_marker_present": "쟁반 마커가 붙어 있습니다(마커를 뗀 쟁반이 기준)", "tray_not_found": "쟁반을 찾지 못했습니다",
            "timeout": "쟁반 확인이 시간 안에 끝나지 않았습니다", "no_result": "쟁반 확인 결과가 없습니다"}


class Refused(RuntimeError):
    """A refusal the Host can read (3.20.0): the message, plus keys merged into the 400 answer."""

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.extra = extra


def tray_check_before(plan: dict) -> dict:
    """Unloading (11/19 TK): right before the eval, the runner's --step tray_check photographs the Pinky tray into a
    file of this mission only (--tray-ref <logs>/tray_pre_<time>.json; never the shared or another Pinky's photo).
    Only "ok" with tray_reference "pre_mission" and its reference_at starts the eval; the pinky_read after the eval
    gets the same --tray-ref and compares with that photo only. Anything else, a missing photo or a runner without
    --tray-ref included, refuses the start (400, "refused": "tray_check")."""
    if not runner_takes(plan["cmd"], "--tray-ref"):
        raise Refused("하차 시작 전 쟁반 확인: 실행기가 작업 직전 쟁반 사진(--tray-ref)을 지원하지 않습니다. 평가를 시작하지 않았습니다.",
                      refused="tray_check", tray_check={"tray_check": "runner_unsupported"})
    if JOB.running():
        raise RuntimeError("다른 작업이 실행 중입니다.")
    if not TRAY_LOCK.acquire(blocking=False):
        raise RuntimeError("쟁반 확인이 이미 진행 중입니다.")
    try:
        TRAY_RESULT_FILE.unlink(missing_ok=True)
        ref = LOG_DIR / f"tray_pre_{dt.datetime.now():%Y%m%d_%H%M%S_%f}.json"  # this mission's own photo
        args = ["--box", str(plan["box"]), "--slot", plan["slot"], "--step", "tray_check", "--auto",
                "--result", str(TRAY_RESULT_FILE), "--tray-ref", str(ref)]
        if plan.get("pinky_no") is not None:
            args += ["--pinky", str(plan["pinky_no"])]
        t0, rc, timed_out = time.time(), None, False
        try:
            rc = subprocess.run(plan["cmd"] + args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=plan["timeout_s"], cwd=str(HOME),
                                env=dict(os.environ, OMX_PANEL_URL=f"http://127.0.0.1:{PORT}")).returncode
        except subprocess.TimeoutExpired:
            timed_out = True  # subprocess.run has killed it
        try:
            res = json.loads(TRAY_RESULT_FILE.read_text())
            r = res.get("result") if isinstance(res.get("result"), dict) else {}
        except (OSError, ValueError, AttributeError):
            r = {}
        status = r.get("tray_check") or ("timeout" if timed_out else "no_result")
        out = {"tray_check": status, "tray_reference": r.get("tray_reference"), "tray_ref_file": str(ref),
               "reference_at": r.get("reference_at") or r.get("tray_reference_at"), "observed_at": r.get("observed_at"),
               "not_empty_why": r.get("tray_not_empty_why"), "tray_empty_verified": r.get("tray_empty_verified"),
               "observed_ts": r.get("observed_ts"), "observed_src": r.get("observed_src"),
               "box_on_tray_ids": r.get("box_on_tray_ids"), "tray_changed": r.get("tray_changed"),
               "photo_top": r.get("photo_top"), "returncode": rc, "checked_at": now_iso(),
               "elapsed_s": round(time.time() - t0, 1)}
        if status != "ok" or out["tray_reference"] != "pre_mission" or not out["reference_at"]:
            why = TRAY_WHY.get(status, status) if status != "ok" else "작업 직전 쟁반 사진을 기준으로 남기지 못했습니다"
            raise Refused(f"하차 시작 전 쟁반 확인: {why}. 평가를 시작하지 않았습니다.", refused="tray_check", tray_check=out)
        return out
    finally:
        TRAY_LOCK.release()


def pinky_after_eval(plan: dict, eval_proc) -> None:
    """Called when the eval has ended. Every end but 비상 정지 (arms held in place) and 강제 종료 goes on to the
    camera check; last_result stays final=False until it is over. 3.24.0: an unloading that ended on the tray judge
    has no camera check after it (the Pinky has left by then: the tray would read empty); the judged values are it."""
    placed = LAST_RESULT.get("placement") if (LAST_RESULT.get("placement") or {}).get("source") == "eval_judge" \
        and (LAST_RESULT.get("placement") or {}).get("operation") == "OUTBOUND" else None
    if placed:
        LAST_RESULT["pinky"] = {"ran": False, "skipped": "tray_placed", "source": "eval_judge", "box": plan["box"],
                                "slot": plan["slot"], "pinky_id": plan.get("pinky_id"), "pinky_no": plan.get("pinky_no"),
                                **{k: placed.get(k) for k in ("on_pinky", "slot_empty", "tray_marker", "tray_changed",
                                                              "tray_reference", "tray_reference_state", "pre_mission_match",
                                                              "observed_ts", "observed_clock", "judged_at")},
                                "on_shelf": False if placed.get("slot_empty") else None, "reported_at": now_iso(),
                                **({"pre_mission": plan["tray"]} if plan.get("tray") else {})}
        return
    pinky = {"ran": False, "on_pinky": None, "on_shelf": None, "box": plan["box"], "slot": plan["slot"],
             "pinky_id": plan.get("pinky_id"), "pinky_no": plan.get("pinky_no"), "returncode": None, "skipped": None}
    if plan.get("pinky_id_ignored"):
        pinky["pinky_id_ignored"] = True
    if plan.get("tray"):  # 3.20.0: the empty-tray photo taken right before the eval
        pinky["pre_mission"] = plan["tray"]
    stopped = LAST_RESULT.get("stopped_by")
    if stopped in ("비상 정지", "강제 종료"):
        LAST_RESULT["pinky"] = dict(pinky, skipped=stopped, reported_at=now_iso())
        return
    LAST_RESULT.update(stage="pinky_check", final=False, pinky=pinky)

    def start_later() -> None:
        end = time.time() + 30
        while JOB.proc is eval_proc and JOB.running() and time.time() < end:
            time.sleep(0.2)
        time.sleep(ALIGN_DELAY_S)
        if JOB.proc is not eval_proc or JOB.running():
            LAST_RESULT["pinky"].update(skipped="다른 작업이 먼저 시작됨", reported_at=now_iso())
            LAST_RESULT.update(stage="done", final=True)
            return
        try:
            start_pinky_read(plan)
        except Exception as exc:
            LAST_RESULT["pinky"].update(skipped=f"시작하지 못함: {str(exc)[:160]}", reported_at=now_iso())
            LAST_RESULT.update(stage="done", final=True)

    threading.Thread(target=start_later, daemon=True).start()


def start_pinky_read(plan: dict) -> None:
    PREVIEW.stop()  # the runner opens the cameras itself
    PINKY_RESULT_FILE.unlink(missing_ok=True)
    args = ["--box", str(plan["box"]), "--slot", plan["slot"], "--step", "pinky_read", "--auto", "--result", str(PINKY_RESULT_FILE)]
    if plan.get("pinky_no") is not None:
        args += ["--pinky", str(plan["pinky_no"])]
    if (plan.get("tray") or {}).get("tray_ref_file"):  # 3.20.0: compare with this mission's photo only
        args += ["--tray-ref", plan["tray"]["tray_ref_file"]]
    JOB.start("pinky_read", plan["cmd"] + args,
              {"phase": "상차 확인 중 (카메라만)", "box": plan["box"], "eval_repo_id": LAST_RESULT.get("repo_id")},
              need_leader=False, env_extra={"OMX_PANEL_URL": f"http://127.0.0.1:{PORT}"})
    proc = JOB.proc
    PINKY.clear()
    PINKY.update(proc=proc, timed_out=False)

    def watchdog() -> None:  # upper limit: a hung camera must not keep the Host waiting
        end = time.time() + plan["timeout_s"]
        while JOB.proc is proc and JOB.running() and time.time() < end:
            time.sleep(0.2)
        if JOB.proc is proc and JOB.running():
            PINKY["timed_out"] = True
            JOB.kill()

    threading.Thread(target=watchdog, daemon=True).start()


def pinky_read_ended(job, proc) -> None:
    """The runner's result object as it is, plus the panel's own keys; on_pinky / on_shelf null unless it said so."""
    try:
        res = json.loads(PINKY_RESULT_FILE.read_text())
    except (OSError, ValueError):
        res = {}
    r = res.get("result") if isinstance(res.get("result"), dict) else {}
    p = LAST_RESULT.get("pinky") or {"box": job.info.get("box")}
    rc = job.info.get("returncode")
    timed_out = PINKY.get("proc") is proc and PINKY.get("timed_out")
    p.update({k: v for k, v in r.items() if k not in ("ran", "box", "returncode", "skipped", "pinky_id", "pinky_no")})
    p.update(ran=True, returncode=rc, log=res.get("log"), reported_at=now_iso())
    if timed_out or job.info.get("killed"):
        p.update(on_pinky=None, on_shelf=None, skipped="시간 초과" if timed_out else "강제 종료")
    elif res.get("early_exit"):
        p["skipped"] = "실행기가 중간에 끝남"
    elif not r:
        p["skipped"] = f"결과 없음 (종료 코드 {rc})"
    p.setdefault("on_pinky", None)
    p.setdefault("on_shelf", None)
    pre = p.get("pre_mission") or {}
    if pre:  # 3.20.0: was the read compared with the photo taken right before this mission? (runner's own word + time)
        used_at = r.get("reference_at") or r.get("tray_reference_at")
        same = r.get("tray_reference") == "pre_mission" and bool(pre.get("reference_at")) and used_at == pre.get("reference_at")
        p.update(reference_at=used_at, pre_mission_match=same)
    LAST_RESULT["pinky"] = p
    LAST_RESULT.update(stage="done", final=True)


def runner_takes(cmd: list, flag: str) -> bool:
    """Does the runner script (the .py in the command) know this option? Read from its source, not by running it.
    The whole option name only: --tray-ref-reset is not --tray-ref (3.20.0)."""
    pat = re.compile(re.escape(flag) + r"(?![\w-])")
    for c in cmd:
        if c.endswith(".py"):
            try:
                return bool(pat.search(Path(c).read_text(errors="replace")))
            except OSError:
                return False
    return False


def align_home_args(cmd: list) -> list:
    """--home-pose for the alignment: the home pose of the eval that just ran (the model's, as its safe stop
    uses), when the runner knows the option and there is one. Otherwise the runner keeps its old rule."""
    home = EVAL_HOME_POSE.get("home") if EVAL_HOME_POSE.get("repo_id") == LAST_RESULT.get("repo_id") else None
    ALIGN_HOME_FILE.unlink(missing_ok=True)
    if not home or not (home.get("joints") or {}) or not runner_takes(cmd, "--home-pose"):
        return []
    ALIGN_HOME_FILE.write_text(json.dumps(home, ensure_ascii=False, indent=2))
    return ["--home-pose", str(ALIGN_HOME_FILE)]


def start_align(plan: dict, release: bool = False) -> None:
    """The shelf alignment runner as the robot job (it opens the cameras itself), or with release=True its
    landing only (back to its start pose, torque off)."""
    s = align_settings()
    if not s:
        raise RuntimeError("설정 shelf_align(cmd)이 없습니다.")
    PREVIEW.stop()
    before = ALIGN.get("before")
    ALIGN.clear()
    ALIGN.update(box=plan["box"], slot=plan["slot"], before=before)
    args = ["--box", str(plan["box"]), "--auto"]
    if release:
        args.append("--release")
    else:
        ALIGN_RESULT_FILE.unlink(missing_ok=True)
        PLACEMENT_FILE.unlink(missing_ok=True)
        args += ["--slot", plan["slot"], "--step", "shelf_align", "--result", str(ALIGN_RESULT_FILE)]
        args += align_home_args(s["cmd"])
    JOB.start("align_release" if release else "align", s["cmd"] + args,
              {"phase": "로봇 내려놓는 중" if release else "정렬 중", "box": plan["box"], "slot": plan["slot"],
               "eval_repo_id": LAST_RESULT.get("repo_id")},
              need_leader=False, env_extra={"OMX_PANEL_URL": f"http://127.0.0.1:{PORT}"})
    if not release:
        watch_placement(JOB.proc, "align")


def start_shelf_read(plan: dict, why: str) -> None:
    """The alignment is skipped (armed off, it could not start): is the box on its shelf slot? The runner reads the
    cameras only (--step shelf_read: no robot connection, no torque). Without the cameras on_shelf stays null."""
    LAST_RESULT["align"].update(ok=False, skipped=why)
    s = align_settings()
    try:
        if not s:
            raise RuntimeError("설정 shelf_align(cmd)이 없습니다")
        PREVIEW.stop()  # the runner opens the cameras itself
        ALIGN_RESULT_FILE.unlink(missing_ok=True)
        PLACEMENT_FILE.unlink(missing_ok=True)
        JOB.start("shelf_read", s["cmd"] + ["--box", str(plan["box"]), "--slot", plan["slot"], "--step", "shelf_read",
                                           "--auto", "--result", str(ALIGN_RESULT_FILE)],
                  {"phase": "선반 확인 중 (카메라만)", "box": plan["box"], "slot": plan["slot"], "skipped": why,
                   "eval_repo_id": LAST_RESULT.get("repo_id")},
                  need_leader=False, env_extra={"OMX_PANEL_URL": f"http://127.0.0.1:{PORT}"})
        watch_placement(JOB.proc, "shelf_read")
    except Exception as exc:
        LAST_RESULT["align"].update(on_shelf=None, shelf_read_error=str(exc)[:200])
        placement_skipped(f"선반 확인을 시작하지 못함: {str(exc)[:160]}", plan["slot"], plan["box"])
        LAST_RESULT.update(stage="done", final=True)


def runner_result() -> tuple[dict, dict]:
    try:
        res = json.loads(ALIGN_RESULT_FILE.read_text())
    except (OSError, ValueError):
        res = {}
    return res, res.get("result") or {}


def shelf_read_ended(job, proc) -> None:
    res, r = runner_result()
    a = LAST_RESULT.get("align") or {}
    # the runner's values as they are (shelf "ok" etc., on_shelf true/false/null); the panel's own keys stay
    a.update({k: v for k, v in r.items() if k not in ("ran", "ok", "skipped", "box", "slot")})
    a.setdefault("on_shelf", None)
    a["shelf_read"] = {"returncode": job.info.get("returncode"), "log": res.get("log")}
    a["reported_at"] = now_iso()
    LAST_RESULT["align"] = a
    placement_from_result("shelf_read", r, job.info.get("slot"), job.info.get("box"))
    LAST_RESULT.update(stage="done", final=True)


def align_ended(job, proc) -> None:
    """The alignment (or its landing) has ended: last_result.align, then land the arms after a stop."""
    info = job.info
    if job.kind == "align_release":
        if LAST_RESULT.get("align"):
            LAST_RESULT["align"]["released"] = info.get("returncode") == 0
        if info.get("returncode") == 3:  # no recorded way back from here: it did not move, the torque is still on
            ALIGN["hold"] = True
            LAST_RESULT.update(stage="hold", final=False)
            job.lines.append("[omx] align: 되짚어 돌아갈 길이 없어 움직이지 않았습니다. 팔은 토크가 켜진 채 제자리입니다. "
                             "팔을 받친 채 기록된 길 가까이 옮긴 뒤 [홈 위치로 이동]을 다시 누르거나, 받친 채 강제 종료로 내려 주세요")
            return
        ALIGN.pop("hold", None)
        align_done_later(proc)
        return
    if info.get("handover") and handover_align_ended(job, proc):
        return
    res, r = runner_result()
    rc = info.get("returncode")
    stopped = ("즉시 중단" if info.get("abort_sent") else "비상 정지" if info.get("frozen")
               else "강제 종료" if info.get("killed") else None)
    LAST_RESULT["align"] = {"ran": True, "ok": rc == 0 and stopped is None and r.get("shelf") == "ok" and r.get("align") == "aligned",
                            "box": info.get("box"), "slot": info.get("slot"), "returncode": rc, "stopped_by": stopped,
                            "err": info.get("err"), "shelf": r.get("shelf"), "align": r.get("align"), "log": res.get("log"),
                            "ended_at": dt.datetime.now().isoformat(timespec="seconds"),
                            "released": rc == 0 if stopped is None else None, "rest": None}
    # every other result key as the runner wrote it (on_shelf, yaw_start, yaw_end, taps, align_s, ...): the Host uses them
    LAST_RESULT["align"].update({k: v for k, v in r.items() if k not in LAST_RESULT["align"]})
    LAST_RESULT["align"].setdefault("on_shelf", None)
    LAST_RESULT["align"]["reported_at"] = now_iso()
    placement_from_result("align", r, info.get("slot"), info.get("box"))
    start_pose, ref = runner_start_pose(r, job.lines)
    home = r.get("home_pose") if r.get("rest_target") == "home" else None
    if isinstance(home, dict) and home:  # 3.17.1: the runner went to the model's home pose: compare with that
        ALIGN["before"] = {k.removesuffix(".pos"): float(v) for k, v in home.items()}
        ref = "home"
    elif start_pose:  # the runner goes back to where it found the arms: compare with that, not the eval's end
        ALIGN["before"] = start_pose
    LAST_RESULT["align"]["rest_ref"] = ref or ("eval_end" if ALIGN.get("before") else None)
    if info.get("handover") and r.get("handover_align") == "hold":
        # 3.21.1: after taking the arms the runner found no safe way home and stopped in place with the torque on
        # (exit 2): the arms wait for a person, as when the handover was not taken. [홈 위치로 이동] on the screen
        # (omx_return_home) after checking; a control-tower home return is refused while this holds
        ho = LAST_RESULT.get("handover") or {}
        ho["fallback"] = "hold"
        HANDOVER["hold"] = True
        LAST_RESULT.update(stage="hold", final=False)
        job.lines.append("[omx] 정렬 실행기가 안전한 홈 경로를 찾지 못해 팔을 토크가 켜진 채 제자리에 두었습니다. "
                         "상자와 팔을 확인한 뒤 팔을 받친 채 [홈 위치로 이동]을 누르세요")
        return
    if info.get("land_after"):
        LAST_RESULT.update(stage="release", final=False)

        def land() -> None:
            end = time.time() + 30
            while JOB.proc is proc and JOB.running() and time.time() < end:
                time.sleep(0.2)
            time.sleep(1.0)
            try:
                start_align(dict(ALIGN), release=True)
            except Exception as exc:
                ALIGN["hold"] = True
                LAST_RESULT.update(stage="hold", final=False)
                job.lines.append(f"[omx] align: 내려놓기를 시작하지 못했습니다 ({exc}). [홈 위치로 이동]을 누르세요")

        threading.Thread(target=land, daemon=True).start()
    elif info.get("frozen"):
        ALIGN["hold"] = True  # 비상 정지: the arms stay in place until [홈 위치로 이동]
        LAST_RESULT.update(stage="hold", final=False)
    else:
        align_done_later(proc)


# ---------------------------------------------------------------- 넘겨받기 (3.18.0): 놓은 순간 평가를 끝내고 정렬에 넘김
HANDOVER_JUDGE_FILE = LOG_DIR / "place_judge.json"  # omx_run's judge settings (OMX_PLACE_JUDGE)
HANDOVER_HOLD_EXIT_FILE = LOG_DIR / "handover_hold_exit.json"  # omx_run: "I closed the ports with the torque on"
HANDOVER_READY = LOG_DIR / "handover_ready.json"  # the waiting runner's heartbeat (pid, ts, ready_at)
HANDOVER_STOP = LOG_DIR / "handover_stop"  # panel -> runner: do not take the arms
HANDOVER_EVAL_DONE = LOG_DIR / "handover_eval_done"  # panel -> runner: the eval process has ended (hold_exit, end_reason)
HANDOVER_STATE = LOG_DIR / "handover_state.json"  # runner -> panel: taken | regs_mismatch | stopped | not_placed
HANDOVER_WATCH_S = 8.0  # the runner must have taken the arms this long after the eval ended, else the panel lands them
HANDOVER_JUDGE_HZ = 6
HANDOVER: dict = {}  # the running / last handover: mode, box, slot, repo_id; "hold" while the arms wait for [홈 위치로 이동]


class Standby:
    """The shelf alignment runner started together with the eval (--handover-wait): it prepares while the policy runs
    and opens no port or camera until the panel says the eval has let go of them. Its output is kept here until
    JOB.adopt() makes it the robot job. Started with nice 10 and one math thread so the eval loop keeps its rate."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.cmd: list | None = None
        self.started = 0.0
        self.lines: collections.deque[str] = collections.deque(maxlen=400)
        self.lock = threading.Lock()
        self.sink = self.on_end = None
        self.ended = False
        self.collector: threading.Thread | None = None
        self.log_file = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list, env_extra: dict) -> None:
        if self.alive():
            self.terminate()
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMX_PORT=str(PORT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", **env_extra)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.lines.clear()
        self.sink = self.on_end = None
        self.ended = False
        self.cmd, self.started = cmd, time.time()
        self.log_file = open(LOG_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_align_standby.log", "w")
        self.log_file.write("$ " + " ".join(cmd) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=LEROBOT_DIR, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                                     preexec_fn=lambda: os.nice(10))
        self.collector = threading.Thread(target=self._collect, args=(self.proc,), daemon=True)
        self.collector.start()

    def _collect(self, proc) -> None:
        for raw in output_lines(proc):
            with self.lock:
                sink = self.sink
            if sink is not None:
                sink(raw)
                continue
            line = ANSI_RE.sub("", raw).rstrip()
            if line:
                self.lines.append(line)
                if self.log_file:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
        proc.wait()
        with self.lock:
            self.ended, end = True, self.on_end
            if self.log_file:
                self.log_file.close()
                self.log_file = None
        if end is not None:
            end()

    def attach(self, sink, on_end) -> None:
        """From now on every line goes to sink and on_end runs when the process has ended (now, if it already has)."""
        with self.lock:
            self.sink, self.on_end, ended = sink, on_end, self.ended
            if self.log_file:
                self.log_file.close()
                self.log_file = None
        if ended:
            on_end()

    def wait(self, timeout: float) -> bool:
        try:
            self.proc.wait(timeout=timeout)
            return True
        except (subprocess.TimeoutExpired, AttributeError):
            return self.proc is None

    def terminate(self, grace_s: float = 3.0) -> None:
        if not self.alive():
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        if not self.wait(grace_s):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass


STANDBY = Standby()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def handover_plan(body: dict, plan: dict | None, episodes: int) -> dict | None:
    """Control-tower command "end_on_place": true (3.18.0): end the eval the moment the box is placed and hand the arms,
    torque on, to the alignment runner that waited during the eval. "place_shadow": true runs the same judge but only
    records what it saw (no handover). Both need the shelf alignment to follow; end_on_place needs one attempt."""
    want, shadow = bool(body.get("end_on_place")), bool(body.get("place_shadow"))
    if not (want or shadow):
        return None
    if not plan:
        raise RuntimeError("end_on_place·place_shadow는 선반 정렬이 이어지는 관제 평가(상자 번호, 정렬 모델)에서만 씁니다.")
    if want and episodes != 1:
        raise RuntimeError("end_on_place는 시도 1번(episodes 1) 평가에서만 씁니다.")
    ho = {"mode": "handover" if want else "shadow", "box": plan["box"], "slot": plan["slot"], "off_reason": None}
    s = align_settings() or {}
    if want and not runner_takes(s.get("cmd") or [], "--handover-wait"):
        ho.update(mode="shadow", off_reason="정렬 실행기가 넘겨받기(--handover-wait)를 모름: 판정만 기록")
    return ho


ONE_ARM_PLACE_FILE = LOG_DIR / "place_judge_one_arm.json"  # 3.25.0


def place_one_arm_env(body: dict, arm: str | None, eval_info: dict, episodes: int, cams: dict) -> dict:
    """3.25.0: a one-arm loading mission ("end_on_place") ends the moment the judge (handover_judge 2A / 2C) sees the
    box placed in its slot with the hand clear; then the arm goes home as usual (no alignment, no hand-over) and the
    judged placement is the evidence (last_result.placement)."""
    if not arm or not body.get("end_on_place"):
        return {}
    marker = str(body.get("target_marker", "")).strip()
    if not marker.isdigit():
        raise RuntimeError("한팔 입고(end_on_place)에는 상자 번호(target_marker)가 필요합니다.")
    if episodes != 1:
        raise RuntimeError("end_on_place는 시도 1번(episodes 1) 평가에서만 씁니다.")
    if "front" not in cams:
        raise RuntimeError("한팔 입고 놓음 판정에는 위 카메라 front가 필요합니다.")
    slot = command_slot(body)
    _write_json(ONE_ARM_PLACE_FILE, {"box_ids": [int(marker)], "camera": "front", "rate_hz": HANDOVER_JUDGE_HZ,
                                     "shadow": False, "slot": slot, "arm": arm, "independent": True})
    eval_info["one_arm_place"] = {"box": int(marker), "slot": slot, "arm": arm}
    eval_info.setdefault("target_marker", int(marker))
    return {"OMX_PLACE_JUDGE": str(ONE_ARM_PLACE_FILE)}


def one_arm_placement(info: dict, end: dict | None) -> dict:
    """last_result.placement of a one-arm loading: the judge's moment (placed) or why there is none."""
    plan = info["one_arm_place"]
    out = {"operation": "INBOUND", "slot": plan["slot"], "box": plan["box"], "source": "eval_judge", "one_arm": True,
           "arm": plan["arm"], "on_shelf": None}
    judge = (end or {}).get("judge") or {}
    if (end or {}).get("end_reason") == "placed" and judge:
        ts = judge.get("frame_ts")
        out.update(on_shelf=True, judged_at=(end or {}).get("judged_at"), judge_version=judge.get("version"),
                   tips_clear=True, observed_ts=ts, observed_clock="capture" if ts else None,
                   tip_y=judge.get("tip_y"))
    else:
        out["not_placed"] = (end or {}).get("end_reason") or "no_episode_end"
    return out


def one_arm_pinky(info: dict, plan: dict) -> dict:
    """last_result.pinky of a one-arm unloading: the tray checked again with the arm home (TrayWatch.recheck)."""
    r = info.get("tray_recheck") or {}
    base = {"box": plan["box"], "slot": plan["slot"], "pinky_id": plan.get("pinky_id"), "pinky_no": plan.get("pinky_no"),
            "one_arm": True, "arm": plan["arm"], "source": "tray_recheck", "reported_at": now_iso()}
    if not r:
        return base | {"ran": False, "on_pinky": None, "on_shelf": None, "skipped": "다시 확인한 결과 없음"}
    return base | {"ran": True, **{k: r.get(k) for k in ("on_pinky", "slot_empty", "tray_marker", "tray_changed",
                                                          "tray_reference", "tray_reference_state", "observed_ts",
                                                          "observed_clock", "camera", "error", "placed_at")},
                   "on_shelf": None if r.get("slot_empty") is None else not r.get("slot_empty")}


def handover_env(ho: dict) -> dict:
    """omx_run settings: the judge (shadow unless this is a real handover) and where the handover files live."""
    for f in (HANDOVER_STOP, HANDOVER_EVAL_DONE, HANDOVER_STATE, HANDOVER_HOLD_EXIT_FILE):
        f.unlink(missing_ok=True)
    _write_json(HANDOVER_JUDGE_FILE, {"box_ids": [int(ho["box"])], "camera": "front", "rate_hz": HANDOVER_JUDGE_HZ,
                                      "shadow": ho["mode"] != "handover"})
    env = {"OMX_PLACE_JUDGE": str(HANDOVER_JUDGE_FILE)}
    if ho["mode"] == "handover":
        env |= {"OMX_HANDOVER_DIR": str(LOG_DIR), "OMX_HOLD_EXIT_FILE": str(HANDOVER_HOLD_EXIT_FILE)}
    return env


def handover_standby_start(ho: dict, plan: dict, repo_id: str, home: dict | None) -> None:
    """Right after the eval has started: the alignment runner in its waiting mode."""
    HANDOVER.clear()
    HANDOVER.update(mode=ho["mode"], box=plan["box"], slot=plan["slot"], repo_id=repo_id)
    s = align_settings()
    try:
        ALIGN_RESULT_FILE.unlink(missing_ok=True)
        PLACEMENT_FILE.unlink(missing_ok=True)
        ALIGN_HOME_FILE.unlink(missing_ok=True)
        args = ["--box", str(plan["box"]), "--auto", "--slot", plan["slot"], "--step", "shelf_align",
                "--result", str(ALIGN_RESULT_FILE), "--handover-wait", str(EPISODE_END_FILE), "--repo-id", repo_id,
                "--handover-dir", str(LOG_DIR)]
        if home and (home.get("joints") or {}) and runner_takes(s["cmd"], "--home-pose"):
            ALIGN_HOME_FILE.write_text(json.dumps(home, ensure_ascii=False, indent=2))
            args += ["--home-pose", str(ALIGN_HOME_FILE)]
        STANDBY.start(s["cmd"] + args, {"OMX_PANEL_URL": f"http://127.0.0.1:{PORT}"})
        JOB.lines.append(f"[omx] 넘겨받기: 정렬 실행기를 대기 모드로 띄웠습니다 (PID {STANDBY.proc.pid})")
    except Exception as exc:  # no runner: omx_run sees no heartbeat and lands the arms as usual
        HANDOVER["standby_error"] = str(exc)[:200]
        JOB.lines.append(f"[omx] 넘겨받기: 정렬 실행기를 띄우지 못해 평소처럼 진행합니다 ({exc})")


def _episode_end_now(job) -> dict:
    try:
        end = json.loads(EPISODE_END_FILE.read_text())
        return end if end.get("repo_id") == job.info.get("repo_id") else {}
    except (OSError, ValueError):
        return {}


def handover_status() -> dict:
    """/api/status: the waiting runner (3.18.0) and a handover hold."""
    out = {"standby": STANDBY.alive(), "hold": bool(HANDOVER.get("hold")), "ready_age_s": None}
    try:
        out["ready_age_s"] = round(time.time() - float(json.loads(HANDOVER_READY.read_text())["ts"]), 1) if out["standby"] else None
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return out


def handover_after_eval(job, proc) -> None:
    """The eval with a judge has ended. Shadow: record what the judge saw, then the usual alignment. Handover: if the
    eval closed its ports with the torque on (box placed, runner ready, nothing stopped), tell the runner and make it
    the alignment job; otherwise tell it not to take the arms (they were landed as usual) and run the usual alignment."""
    ho, plan = job.info["handover"], job.info["shelf_align"]
    end = _episode_end_now(job)
    judge = end.get("judge") or {}
    seen = {"mode": ho["mode"], "end_reason": end.get("end_reason"), "judged_at": end.get("judged_at"),
            "judge": {k: judge.get(k) for k in ("box", "version", "judged_ts", "tip_y", "marker_seen") if k in judge} or None,
            "judge_watch": {k: v for k, v in (end.get("judge_watch") or {}).items() if k in ("version", "updates", "errors", "hit")} or None,
            "loop_ms": end.get("loop_ms"), "off_reason": ho.get("off_reason"), "state": None, "fallback": None}
    LAST_RESULT["handover"] = seen
    if ho["mode"] != "handover":
        seen["state"] = "off"
        align_after_eval(plan, proc)
        return
    try:
        hold_exit = HANDOVER_HOLD_EXIT_FILE.stat().st_mtime >= job.started
    except OSError:
        hold_exit = False
    _write_json(HANDOVER_EVAL_DONE, {"at": now_iso(), "ts": time.time(), "hold_exit": hold_exit,
                                     "end_reason": end.get("end_reason")})
    if not hold_exit:
        HANDOVER_STOP.write_text(now_iso())  # the arms were landed: the runner must not take them
        seen["state"] = "not_placed" if end.get("end_reason") != "placed" else "not_ready"
        LAST_RESULT.update(stage="align", final=False, align={"ran": False, "ok": None, "box": plan["box"], "slot": plan["slot"]})

        def later() -> None:  # the waiting runner ends on its own; then the alignment as before
            if not STANDBY.wait(5.0):
                STANDBY.terminate()
            align_after_eval(plan, proc)

        threading.Thread(target=later, daemon=True).start()
        return
    # placed and the ports were closed with the torque on: the box is on its slot (seen from above)
    LAST_RESULT["placement"] = {"on_shelf": True, "shelf": "ok", "slot": plan["slot"], "box": plan["box"],
                                **judge_evidence(end, judge), "source": "eval_judge", "judge_version": judge.get("version"),
                                "reported_at": now_iso()}
    LAST_RESULT.update(stage="align", final=False,
                       align={"ran": False, "ok": None, "box": plan["box"], "slot": plan["slot"], "handover": True})
    seen["state"] = "waiting"
    if not ARMED["on"] or not STANDBY.alive():
        why = f"armed 꺼짐 ({ARMED['off_reason']})" if not ARMED["on"] else "정렬 실행기가 이미 끝남"
        HANDOVER_STOP.write_text(now_iso())
        seen.update(state="stopped", reason=why)
        threading.Thread(target=handover_fallback_home, args=(why, proc), daemon=True).start()
        return
    ALIGN.clear()
    ALIGN.update(box=plan["box"], slot=plan["slot"], before=None)

    def adopt_later() -> None:
        # this runs inside the eval job's end handling: the job counts as running until that has finished
        end = time.time() + 10
        while JOB.proc is proc and JOB.running() and time.time() < end:
            time.sleep(0.02)
        try:
            JOB.adopt("align", STANDBY, {"phase": "정렬 중 · 넘겨받는 중", "box": plan["box"], "slot": plan["slot"],
                                         "eval_repo_id": job.info.get("repo_id"), "handover": True})
        except Exception as exc:
            HANDOVER_STOP.write_text(now_iso())
            seen.update(state="stopped", reason=f"넘겨받지 못함: {exc}"[:200])
            handover_fallback_home(seen["reason"])
            return
        watch_placement(JOB.proc, "align")
        handover_watch(JOB.proc)

    threading.Thread(target=adopt_later, daemon=True).start()


def handover_state_now(since: float) -> dict:
    try:
        if HANDOVER_STATE.stat().st_mtime < since:
            return {}
        return json.loads(HANDOVER_STATE.read_text())
    except (OSError, ValueError):
        return {}


def handover_watch(proc) -> None:
    """The runner must say "taken" (handover_state.json) within HANDOVER_WATCH_S, or the panel stops it and lands the
    arms itself (lift, synchronized home, safe stop: omx_return_home)."""
    since = HANDOVER_EVAL_DONE.stat().st_mtime if HANDOVER_EVAL_DONE.exists() else time.time()
    end = time.time() + HANDOVER_WATCH_S
    ho = LAST_RESULT.get("handover") or {}
    while time.time() < end:
        st = handover_state_now(since)
        if st.get("state"):
            ho.update(state=st["state"], reason=st.get("reason"), state_at=now_iso())
            if st["state"] == "taken":
                ho["gap_s"] = round(float(st.get("ts", time.time())) - float((ho.get("judge") or {}).get("judged_ts") or since), 2)
            return
        if JOB.proc is not proc or not JOB.running():
            break  # it ended without saying: align_ended decides from its result
        time.sleep(0.1)
    if JOB.proc is proc and JOB.running() and not handover_state_now(since).get("state"):
        ho.update(state="failed", reason=f"{HANDOVER_WATCH_S:.0f}초 안에 팔을 잡지 못함")
        JOB.lines.append(f"[omx] 넘겨받기 실패: 정렬 실행기가 {HANDOVER_WATCH_S:.0f}초 안에 팔을 잡지 못해 멈추고 패널이 홈으로 보냅니다")
        HANDOVER_STOP.write_text(now_iso())
        JOB.info["handover_failed"] = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass


def handover_align_ended(job, proc) -> bool:
    """align_ended for an adopted runner. True when handled here (it never took the arms); False: the usual path."""
    res, r = runner_result()
    ho = LAST_RESULT.get("handover") or {}
    since = HANDOVER_EVAL_DONE.stat().st_mtime if HANDOVER_EVAL_DONE.exists() else job.started
    st = handover_state_now(since).get("state") or r.get("handover")
    if job.info.get("handover_failed"):
        st = "failed"
    if st == "taken":
        ho.setdefault("state", "taken")
        return False
    ho["state"] = st or "failed"
    ho.setdefault("reason", r.get("handover_reason") or ("결과 없음" if not r else None))
    LAST_RESULT["align"] = dict(LAST_RESULT.get("align") or {}, ran=False, ok=False, returncode=job.info.get("returncode"),
                                skipped=f"넘겨받지 못함 ({ho['state']})", log=res.get("log"), reported_at=now_iso())
    if ho["state"] in ("regs_mismatch", "stopped") or job.info.get("frozen"):
        # the arms stay where they are with the torque on: a person checks, then [홈 위치로 이동] (omx_return_home)
        ho["fallback"] = "hold"
        HANDOVER["hold"] = True
        LAST_RESULT.update(stage="hold", final=False)
        job.lines.append("[omx] 넘겨받기 안 됨: 팔은 토크가 켜진 채 제자리입니다. 상자와 팔을 확인한 뒤 [홈 위치로 이동]을 누르세요")
        return True
    threading.Thread(target=handover_fallback_home, args=(ho.get("reason") or ho["state"], proc), daemon=True).start()
    return True


def handover_fallback_home(why: str, after_proc=None) -> None:
    """Nobody took the arms (failed, runner gone, armed off): omx_return_home lifts them, goes home synchronized and
    lands them. If it cannot start, the arms wait with the torque on for [홈 위치로 이동]. No alignment follows."""
    ho0 = LAST_RESULT.get("handover") or {}
    LAST_RESULT["align"] = dict(LAST_RESULT.get("align") or {}, ran=False, ok=False,
                                skipped=(LAST_RESULT.get("align") or {}).get("skipped") or f"넘겨받지 못함 ({ho0.get('state')}: {why})"[:200],
                                reported_at=now_iso())
    end = time.time() + 30
    while after_proc is not None and JOB.proc is after_proc and JOB.running() and time.time() < end:
        time.sleep(0.2)
    if not STANDBY.wait(5.0):
        STANDBY.terminate()
    time.sleep(0.5)
    ho = LAST_RESULT.get("handover") or {}
    ho["fallback"] = "home"
    LAST_RESULT.update(stage="release", final=False)
    try:
        data = json.loads(EVAL_HOME_FILE.read_text())
        JOB.start("home", [str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_return_home.py"), str(EVAL_HOME_FILE),
                           data.get("repo_id", "")],
                  {"phase": "넘겨받기 안 됨 · 홈으로 이동 중", "repo_id": data.get("repo_id", ""), "handover_fallback": "home",
                   "why": why},
                  need_leader=False, env_extra={"OMX_HOME_GRIPPER": "open"})
    except Exception as exc:
        ho.update(fallback="hold", fallback_error=str(exc)[:200])
        HANDOVER["hold"] = True
        LAST_RESULT.update(stage="hold", final=False)


def handover_home_ended(job, proc) -> None:
    ho = LAST_RESULT.get("handover") or {}
    ho["fallback_ok"] = job.info.get("returncode") == 0
    if job.info.get("frozen") or job.info.get("killed"):
        HANDOVER["hold"] = True
        LAST_RESULT.update(stage="hold", final=False)
        return
    LAST_RESULT.update(stage="done", final=True)



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
    arm = one_arm(body)  # 3.21.0: a one-arm model on the arm the person chose (refused for missions)
    if arm and CONTROL and "async_mode" not in body:
        # 3.25.0 (TK 11/24): a one-arm run from this control PC (GPU laptop over UDP) uses these async settings unless
        # the request names its own: 100 actions per chunk, the next chunk at 20 % left, overlaps weighted 0.3/0.7.
        # cfg "one_arm_async" replaces them.
        body = {**body, **ONE_ARM_ASYNC, **(CFG.get("one_arm_async") or {})}
    command_slot(body)  # a wrong slot is refused before anything moves
    pinky = pinky_plan(body, run, arm)  # a wrong unloading request is refused before anything moves
    camera_gate()
    if CONTROL:
        udp_ready()
    local_model = fetch_model_udp(run["model_path"]) if CONTROL else Path(run["model_path"])
    model_cfg = model_config(str(local_model))
    check_model_fits(model_cfg, arm)
    eval_cams = eval_cameras(model_cfg, arm)
    extra_args, async_env, opts = policy_eval_options(str(local_model), body)
    if arm:  # 3.25.0: the model's own camera names (e.g. shelf = front_2), for the policy's rename processor
        _, rename = model_camera_map(model_cfg, arm)
        if rename and not any(a.startswith("--dataset.rename_map=") for a in extra_args):
            extra_args.append("--dataset.rename_map=" + json.dumps(rename))
            opts["camera_names"] = {k.split(".")[-1]: v.split(".")[-1] for k, v in rename.items()}
    if constant_inputs(model_cfg, body):
        opts["constant_inputs"] = constant_inputs(model_cfg, body)
    if CONTROL:
        async_env = remote_eval_settings(local_model, run["model_path"], extra_args, async_env, opts, body)
        extra_args = [a for a in extra_args if a.startswith("--dataset.")] + ["--policy.device=cpu"]
        opts["remote_inference"] = remote.HOST
        opts["transport"] = "udp"
    home = None
    if omx_home.eval_enabled():  # first move to the model's home pose (median of its training data's first frames)
        try:
            if CONTROL:
                home = model_home_pose(run["model_path"])
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
    if pinky and not pinky.get("one_arm"):
        pinky["tray"] = tray_check_before(pinky)  # 3.20.0: an unloading starts only on an empty tray (this photo)
    step = run.get("step")  # set when an intermediate save was chosen
    base = names.eval_name(run.get("display") or names.display_name(run["id"]), preset, opts,
                           f"{dt.datetime.now():%m%d_%H%M}", step)
    repo_id, k = f"{USER}/{base}", 2
    while (DATA_HOME / repo_id).exists() or notes_path(repo_id).exists():
        repo_id, k = f"{USER}/{base}_{k}", k + 1
    path = notes_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # recording is off unless asked for ("record": true, the panel's checkbox): a control-tower mission or a quick check
    # only runs the policy; a recorded evaluation keeps videos and data for review, reports and training
    record = bool(body.get("record"))
    path.write_text(json.dumps({"kind": "eval", "model": run["model_path"], "run_id": run["id"],
                                "model_display": names.model_label(run.get("display") or names.display_name(run["id"]), step, run.get("steps")),
                                "run_number": run.get("number"), "task": task, "results": {},
                                "train_step": step or run.get("steps"), "final_steps": run.get("steps"),
                                "smooth": bool(body.get("smooth")), "threads": threads,
                                "reset_home": reset_home, "open_gripper": open_gripper, "recorded": record,
                                **({"arm": arm, "arm_set": ARM_NAMES.get(arm)} if arm else {}),
                                "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=2))
    PREVIEW.stop()  # the policy run needs the cameras itself
    fix_wrist_fps()
    EVAL_HOME_FILE.unlink(missing_ok=True)  # written again by this run before its first move
    EPISODE_END_FILE.unlink(missing_ok=True)  # written by this run at the end of each episode
    cmd = [
        str(VENV_BIN / "python"), str(TOOLS_DIR / "omx_run.py"),
        *(one_arm_robot_args(arm) if arm else ROBOT_ARGS), f"--robot.cameras={omx_config.cameras_arg(warmup_s=3, cameras=eval_cams)}",
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
    end_on_return = bool(body.get("end_on_return"))
    # the shelf alignment follows: when the attempt ends by return detection, the short safe stop (omx_safe_start)
    fast_release = end_on_return and bool(body.get("fast_release"))
    eval_info = {"phase": "모델 불러오는 중", "repo_id": repo_id, "target": episodes, "end_on_return": end_on_return,
                 "fast_release": fast_release, "record": record,
                 "episode_s": episode_s, "reset_s": reset_s, "model": run["id"],
                 "smooth": opts["smooth"], "threads": threads, "reset_home": reset_home,
                 "open_gripper": open_gripper, **opts,
                 "protocol": note.get("protocol"), "source": "mqtt" if body.get("source") == "mqtt" else "panel",
                 **({"arm": arm} if arm else {}),
                 "task": task}  # the panel's voice cue names the object from it (box, strawberry, ...)
    plan = None if arm else align_plan(body, run)
    marker = str(body.get("target_marker", "")).strip()
    if plan and not plan["draw_target"]:  # the box number is for the alignment; the model never saw a circle
        tenv = {}
        eval_info["target_marker"] = plan["box"]
    elif body.get("draw_target") is False:
        # the command keeps the number (Pinky check, the record) but no circle goes into the front image:
        # a model trained without one (unloading, 11/17) must not see one
        tenv = {}
        if marker:
            eval_info["target_marker"] = int(marker)
    else:
        tenv = target_env(body, eval_info)
    eval_info["target_drawn"] = bool(tenv)
    if plan:
        eval_info["shelf_align"] = plan
    ho = None if arm else handover_plan(body, plan, episodes)
    if ho:
        eval_info["handover"] = ho
    if pinky:
        eval_info["pinky_check"] = pinky
        eval_info.setdefault("target_marker", pinky["box"])
    if eval_info.get("target_marker") is not None:
        note["target_marker"] = eval_info["target_marker"]
        note["target_drawn"] = bool(tenv)
    path.write_text(json.dumps(note, ensure_ascii=False, indent=2))
    JOB.start("eval", cmd, eval_info,
              need_leader=False, env_extra=thread_env(threads) | home_env(reset_home, open_gripper, repo_id) | async_env
              | ({"OMX_HOME_POSE": str(omx_home.write_run_file(home, RUN_HOME_FILE))} if home else {}) | tenv
              | {"OMX_EPISODE_END": str(EPISODE_END_FILE)} | ({"OMX_END_ON_RETURN": "1"} if end_on_return else {})
              | ({"OMX_FAST_RELEASE": "1"} if fast_release else {}) | ({} if record else {"OMX_NO_RECORD": "1"})
              | (handover_env(ho) if ho else {}) | ({"OMX_ARM_SIDE": arm} if arm else {})
              | ({"OMX_WRIST_CHECK": "1"} if wrist_check_on() else {}) | drop_env(eval_info, arm, eval_cams, body)
              | tray_env(eval_info, arm, eval_cams) | one_arm_env(model_cfg, arm, body)
              | place_one_arm_env(body, arm, eval_info, episodes, eval_cams))
    EVAL_HOME_POSE.clear()
    EVAL_HOME_POSE.update(repo_id=repo_id, home=home)
    if ho and ho["mode"] == "handover":
        handover_standby_start(ho, plan, repo_id, home)
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
        total = json.loads(info.read_text()).get("total_episodes", 0) if info.exists() else n.get("episodes_run", 0)
        res = n.get("results", {})
        items.append({"repo_id": repo, "model": n.get("model"), "run_id": n.get("run_id"), "run_number": n.get("run_number"),
                      "model_display": names.model_label(names.display_name(n.get("run_id"), aliases), names.train_amount(n, steps),
                                                         n.get("final_steps") or steps.get(n.get("run_id"))) if n.get("run_id") else "",
                      "start_failed": bool(n.get("start_failed")), "recorded": n.get("recorded", True),
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


class NoCameraRef(RuntimeError):
    """No real recording has this camera yet (3.20.1): where the camera is, its setup picture stands in."""
    extra = {"no_camera_ref": True}


def camera_ref_repos(cam: str, owner: str = "") -> list[str]:
    """Real recordings that have this camera, the chosen reference's owner first, then the newest (3.20.1): a
    reference without front_3/front_4 (a two-arm model) still shows those cameras' outlines from data that has them.
    Eval runs and names with "sim" are left out (their views can be renders)."""
    key, found = f"observation.images.{cam}", []
    for info in DATA_HOME.glob("*/*/meta/info.json"):
        repo = info.parent.parent
        if repo.name.startswith("eval_") or "sim" in repo.name.lower():
            continue
        try:
            if key in (json.loads(info.read_text()).get("features") or {}):
                found.append((repo.parent.name == owner, info.stat().st_mtime, f"{repo.parent.name}/{repo.name}"))
        except (OSError, ValueError):
            continue
    return [r for *_, r in sorted(found, reverse=True)]


def reference_view(repo: str, cam: str = "front", model: str = "") -> tuple[bytes, str]:
    """reference_frame, and where it came from: "" = the chosen model or dataset, else the recording used because
    the chosen one has no such camera (3.20.1). NoCameraRef when no recording has it."""
    episodes = None
    if model:
        d = json.loads((Path(model) / "train_config.json").read_text())["dataset"]
        repo, episodes = d["repo_id"], d.get("episodes")
    root = dataset_dir(repo)
    if f"observation.images.{cam}" in (json.loads((root / "meta/info.json").read_text()).get("features") or {}):
        return reference_frame(repo, cam, episodes=episodes), ""
    for other in camera_ref_repos(cam, repo.split("/")[0])[:5]:
        try:
            return reference_frame(other, cam), other
        except Exception:
            continue
    raise NoCameraRef(f"{cam}: 고른 기준에 이 카메라가 없고, 이 카메라로 녹화한 실물 데이터셋도 없습니다.")


def reference_frame(repo: str, cam: str = "front", model: str = "", episodes: list | None = None) -> bytes:
    """The training view of one camera as JPEG: the median of the frame REF_TIME_S into up to REF_EPISODES
    episodes of a dataset, or of the dataset a model was trained on (its episodes only). Cached on disk."""
    import av
    import numpy as np
    import pandas as pd

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
        try:
            if CONTROL and (model or not (DATA_HOME / repo).exists()):
                q = urllib.parse.urlencode({"repo": repo, "model": model, "cam": cam})
                status, ctype, body = remote.gpu_request("GET", "/api/reference?" + q, None, timeout=120)
                if status != 200:
                    err = json.loads(body or b"{}")
                    raise (NoCameraRef if err.get("no_camera_ref") else RuntimeError)(err.get("error") or "GPU 노트북에서 기준 장면을 받지 못했습니다.")
                m = re.search(r'source="([^"]*)"', ctype or "")
                source = m.group(1) if m else ""
            else:
                body, source = reference_view(repo, cam, model)
            gray = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
        except NoCameraRef:
            # no recording with this camera yet: the view saved with the robot setup (cam_refs, 320x240 grey)
            gray = cv2.imread(str(camauto.ref_path(CAM_REFS, ROBOT_MODE, cam)), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise
            gray = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_LINEAR)
            body, source = cv2.imencode(".jpg", gray)[1].tobytes(), "setup"
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
        REF_CACHE[k] = {"jpg": body, "kp": kp, "des": des, "edges": cv2.imencode(".png", rgba)[1].tobytes(), "source": source}
    return REF_CACHE[k]


def alignment(repo: str, model: str = "", cam: str = "front") -> dict:
    """How far the live view of one camera sits from its training view (ORB + RANSAC similarity): centre shift,
    rotation, scale, largest corner shift (px), the shift of the centre and the four corners, matches kept."""
    live = PREVIEW.raw.get(cam) or PREVIEW.frames.get(cam)
    if not live:
        raise RuntimeError("카메라를 먼저 켜세요.")
    ref = reference_entry(repo, model, cam)
    return dict(_alignment(ref, live), ref_source=ref["source"])


def _alignment(ref: dict, live: bytes) -> dict:
    import numpy as np

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
            if self.path == "/api/train/runs":  # also the Host's model check: answered from the copy when the tunnel is down
                try:
                    return self._json(gpu_runs())
                except Exception as exc:
                    return self._json({"error": str(exc)}, 502)
            if remote.proxied("GET", self.path):
                return self._forward("GET")
            if self.path.startswith(("/api/dataset?", "/api/reference?")):  # /api/align: the cameras are here
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                repo = q.get("repo", [""])[0]
                if q.get("model", [""])[0] or repo and not (DATA_HOME / repo).exists():  # training data: GPU laptop
                    return self._forward("GET")
            if self.path.startswith("/api/home?") and "model=" in self.path:
                # the model's home pose is computed there (training data), kept here; whether to use it is this PC's setting
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                try:
                    out = {"home": model_home_pose(q.get("model", [""])[0])}
                except Exception as exc:
                    out = {"error": str(exc)}
                return self._json(home_status() | out)

        if self.path == "/api/datasets":
            return self._json({"datasets": list_datasets(), "legacy_note": data_notes.load().get("legacy_note", "")})
        if self.path == "/api/errors":
            return self._json({"codes": omx_errors.all_codes()})
        if self.path == "/api/robot_setup":
            return self._json(setup_status())
        if self.path == "/api/robot_here":  # light: the control PC asks this every 5 s
            return self._json({"robot_here": robot_here(), "host": HOSTNAME, "role": "control" if CONTROL else "gpu",
                               "version": VERSION})
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
        elif self.path == "/tips":  # 학습 TIP (top bar): a self-contained page, also saved as an HTML file from the page
            body = (TOOLS_DIR / "omx_train_tips.html").read_text().replace("__OMX_VERSION__", VERSION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/motor_output":
            self._json(motor_output())
        elif self.path == "/api/status":
            try:
                WARM.ensure()
            except Exception as exc:
                WARM.error = str(exc)[:200]
            self._json({
                "warm": WARM.status(),
                "user": USER,
                "version": VERSION,
                "gpu_name": CFG.get("gpu_name"),
                "ckpt_home": str(CKPT_HOME),
                "preview": PREVIEW.running,
                "markers": PREVIEW.markers if PREVIEW.running else {},
                **armed_status(),
                "mqtt_host": MQTT_HOST.status(),
                "host_hold": omx_host.hold_state(),
                "last_result": LAST_RESULT or None,
                "camera_watch": camera_watch(),
                "live": JOB.running() and JOB.kind in ("record", "eval"),
                "brightness": PREVIEW.brightness,
                "cam_stats": camera_stats(),
                "robot_mode": ROBOT_MODE,
                "cameras": [{"name": n, "label": CAMERA_LABELS.get(n, n)} for n in CAMERAS],
                "job": JOB.status(),
                "handover": handover_status(),
                "cpu": cpu_status(),
                "log": list(JOB.lines)[-120:],
                "datasets": merged_datasets() if CONTROL else list_datasets(),
                "data_legacy_note": remote.gpu_legacy_note() if CONTROL else data_notes.load().get("legacy_note", ""),
                "data_home": str(DATA_HOME),
                "role": "control" if CONTROL else "gpu",
                "host": HOSTNAME,
                "robot_here": robot_here(),
                "gpu_panel": gpu_panel() if CONTROL else None,
                "sync": sync_status() if CONTROL else None,
                "gpu_link": dict(remote.TUNNEL.status(), infer=infer_link_status()) if CONTROL else None,
                "hf_user": UPLOADER.hf_login(),
                "upload": UPLOADER.status(),
            })
        elif self.path.startswith("/stream/"):
            self._stream(self.path.split("?", 1)[0].rsplit("/", 1)[-1])
        elif self.path.startswith("/api/reference?"):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body, source = reference_view(q.get("repo", [""])[0], q.get("cam", ["front"])[0], q.get("model", [""])[0])
                self.send_response(200)
                self.send_header("Content-Type", f'image/jpeg; source="{source}"' if source else "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=300")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._json({"error": str(exc), **(getattr(exc, "extra", None) or {})}, 400)
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
        elif self.path.startswith("/api/arms?") or self.path == "/api/arms":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            self._json(arms_now(q.get("model", [""])[0], q.get("arm", [""])[0] or None))  # arm: a one-arm model (3.25.0)
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
            parts = urllib.parse.urlsplit(self.path)
            self._snapshot(parts.path.rsplit("/", 1)[-1], urllib.parse.parse_qs(parts.query))
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

    def _snapshot(self, name: str, query: dict | None = None) -> None:
        """One camera frame for the position editor and the shelf / Pinky runner: camera view, or the running
        program's live frame. X-Frame-Ts = when it was read (epoch s), X-Frame-Age = seconds since;
        ?raw=1: the camera view's frame without the marker check's drawing; ?max_age=S: 410 when older."""
        query = query or {}
        raw_wanted = (query.get("raw") or [""])[0] in ("1", "true")
        try:
            max_age = float((query.get("max_age") or [""])[0] or "nan")
        except ValueError:
            max_age = float("nan")
        frame, ts, drawn = None, None, False
        if PREVIEW.running:
            raw = PREVIEW.raw.get(name) if raw_wanted else None
            frame = raw or PREVIEW.frames.get(name)
            ts = PREVIEW.frame_ts.get(name)
            drawn = raw is None and name in PREVIEW.raw  # the marker check drew on the frame sent
        live = Path(f"{LIVE}{name}.jpg")
        if frame is None and live.exists():
            mtime = live.stat().st_mtime
            if time.time() - mtime < 5:
                frame, ts = live.read_bytes(), mtime
                drawn = name == "front" and bool(JOB.running() and JOB.info.get("target_drawn"))
        if not frame or name not in CAMERAS:
            self._json({"error": "영상이 없습니다. ① 탭에서 카메라 보기를 켜면 배경에 나옵니다."}, 404)
            return
        age = None if ts is None else max(0.0, time.time() - ts)
        if max_age == max_age and (age is None or age > max_age):  # max_age given (not nan)
            self._json({"error": f"{name} 영상이 {max_age:g}초보다 오래됐습니다",
                        "age_s": None if age is None else round(age, 3)}, 410)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        if ts is not None:
            self.send_header("X-Frame-Ts", f"{ts:.3f}")
            self.send_header("X-Frame-Age", f"{age:.3f}")
        self.send_header("X-Frame-Raw", "0" if drawn else "1")
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
            if mqtt and self.path in ("/api/teleop", "/api/record", "/api/armed", "/api/infer_link", "/api/infer_link/key"):
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
                    if JOB.running() and JOB.kind in ("align", "align_release", "shelf_read") or LAST_RESULT.get("stage") == "align":
                        raise RuntimeError("선반 정렬 중에는 정렬 프로그램이 카메라를 씁니다.")
                    if JOB.running() and JOB.kind == "pinky_read" or LAST_RESULT.get("stage") == "pinky_check":
                        raise RuntimeError("상차 확인 중에는 확인 프로그램이 카메라를 씁니다.")
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
            elif self.path == "/api/infer_link" and CONTROL:
                self._json(save_infer_link(body))
            elif self.path == "/api/infer_link/key" and CONTROL:
                self._json(provision_udp_key())
            elif self.path == "/api/udp_server" and not CONTROL:  # GPU laptop: a control PC wants UDP (or not)
                if UDP_KEEPER is None:
                    raise RuntimeError("이 PC는 UDP 추론 서버를 켜지 않습니다.")
                if body.get("action") == "release":
                    self._json(UDP_KEEPER.release(str(body.get("peer", ""))))
                else:
                    self._json(UDP_KEEPER.ensure(body))
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
            elif self.path == "/api/host/hold_release":
                self._json(host_hold_release(body))
            elif self.path == "/api/home/return":
                # a control-tower (Host) request moves the arms only while armed is on (TK 11/16); the screen button as before
                if body.get("source") == "mqtt" and not ARMED["on"]:
                    raise RuntimeError(f"armed 꺼짐: 관제 홈 복귀 거절 ({ARMED.get('off_reason') or '꺼짐'})")
                tray_hold = JOB.running() and JOB.kind == "eval" and JOB.info.get("tray_hold")
                if body.get("source") == "mqtt" and (ALIGN.get("hold") or HANDOVER.get("hold") or tray_hold):
                    # 3.21.1: the arms were left in place with the torque on (no safe way back): a person checks first
                    # 3.24.0: also a hand held over the tray zone after an unloading
                    raise Refused("팔이 토크가 켜진 채 제자리에서 기다리는 중이라 관제 홈 복귀를 거절합니다. "
                                  "현장에서 상자와 팔을 확인한 뒤 화면의 [홈 위치로 이동]을 누르세요", refused="hold",
                                  hold="align" if ALIGN.get("hold") else "handover" if HANDOVER.get("hold") else "tray")
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
            self._json({"error": str(exc), **(getattr(exc, "extra", None) or {})}, 400)


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


UDP_KEEPER = None


def start_udp_keeper() -> None:
    """GPU laptop: the UDP inference server, started when a control PC chooses UDP (omx_udp_keeper.py)."""
    global UDP_KEEPER
    import omx_udp_keeper

    UDP_KEEPER = omx_udp_keeper.UdpServerKeeper(VENV_BIN / "python", TOOLS_DIR, LOG_DIR,
                                                 state_path=omx_config.CONF_DIR / "udp_server.json")
    UDP_KEEPER.start()


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
        UDP_LINK.start()
        threading.Thread(target=sync_loop, daemon=True).start()
        threading.Thread(target=close_watch, daemon=True).start()
        print(f"[omx] control PC: training and inference on {remote.HOST} (SSH tunnel)", flush=True)
    else:
        start_inference_server()
        start_udp_keeper()
    print(f"OMX panel {VERSION} on http://{HOST}:{PORT} (user {USER}, settings: {CFG['_source']})", flush=True)
    try:
        server.serve_forever()
    finally:
        MQTT_HOST.stop()
        JOB.stop()
        PREVIEW.stop()
        if CONTROL:
            remote.TUNNEL.stop()
        elif UDP_KEEPER is not None:
            UDP_KEEPER.stop()


if __name__ == "__main__":
    main()
