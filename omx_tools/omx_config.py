#!/usr/bin/env python3
"""Per-machine settings of the OMX panel, shared by every tool in this folder.

The installer (install_gpu.sh) writes them to ~/.config/omx/config.json for each account on the
GPU laptop. Lookup order: $OMX_CONFIG, ~/.config/omx/config.json, /etc/omx/config.json; missing
keys fall back to DEFAULTS. Print the effective settings with:  python omx_config.py
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
from pathlib import Path

VERSION = "1.6.1"

DEFAULTS: dict = {
    # shown in the panel header
    "gpu_name": socket.gethostname(),
    # LeRobot home shared by the team (datasets + calibration) and the trained-model folder
    "data_home": "/srv/il/lerobot_data",
    "checkpoints": "/srv/il/checkpoints",
    # folder for the robot / training locks; must be writable by every account using the robot
    "shared_root": "/srv/il",
    # group that gets read/write on recordings and models ("" = do not change permissions)
    "team_group": "ilteam",
    # robot serial ports (udev symlinks or /dev/serial/by-id/... paths)
    "follower_port": "/dev/omx_follower",
    "leader_port": "/dev/omx_leader",
    "follower_ids": [11, 12, 13, 14, 15, 16],
    "leader_ids": [1, 2, 3, 4, 5, 6],
    "baudrate": 1_000_000,
    # cameras: name -> /dev/v4l/by-id/... path (names become observation.images.<name>)
    "cameras": {
        "front": "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0",
        "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
    },
    # camera whose exposure_dynamic_framerate is switched off (keeps 30 fps in dim light); "" = none
    "fixed_fps_camera": "wrist",
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    # Python environment with LeRobot v0.4.4
    "venv": "~/venv/il",
    # one panel server per account: port = port_base + uid - uid_base
    "port_base": 8765,
    "uid_base": 1001,
}


def config_paths() -> list[Path]:
    paths = []
    if os.environ.get("OMX_CONFIG"):
        paths.append(Path(os.environ["OMX_CONFIG"]).expanduser())
    paths += [Path("~/.config/omx/config.json").expanduser(), Path("/etc/omx/config.json")]
    return paths


def load() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for path in config_paths():
        if path.is_file():
            try:
                cfg.update(json.loads(path.read_text()))
                cfg["_source"] = str(path)
            except ValueError as exc:
                raise SystemExit(f"[omx] {path} is not valid JSON: {exc}")
            break
    cfg.setdefault("_source", "defaults")
    return cfg


CFG = load()


def path(key: str) -> Path:
    return Path(str(CFG[key])).expanduser()


def venv_bin() -> Path:
    return path("venv") / "bin"


def port(uid: int | None = None) -> int:
    uid = os.getuid() if uid is None else uid
    if os.environ.get("OMX_PORT"):
        return int(os.environ["OMX_PORT"])
    return int(CFG["port_base"]) + max(0, uid - int(CFG["uid_base"]))


def lock_dir() -> Path:
    """Shared folder for the robot/training locks; falls back to /tmp when it is missing or read-only."""
    root = path("shared_root")
    if root.is_dir() and os.access(root, os.W_OK):
        return root
    tmp = Path("/tmp/omx-shared")
    tmp.mkdir(exist_ok=True)
    try:
        os.chmod(tmp, 0o1777)
    except PermissionError:
        pass
    return tmp


def share_with_team(target: Path) -> None:
    """Give the team group read/write on a recording or model (same rule as the shared folder default)."""
    group = CFG.get("team_group") or ""
    if not group or not Path(target).exists():
        return
    subprocess.run(["chmod", "-R", "g+rwX", str(target)], capture_output=True)
    subprocess.run(["setfacl", "-R", "-m", f"g::rwX,g:{group}:rwX,m::rwX", str(target)], capture_output=True)


def cameras_arg(warmup_s: int = 3) -> str:
    """--robot.cameras value for lerobot scripts."""
    w, h, fps = CFG["camera_width"], CFG["camera_height"], CFG["camera_fps"]
    return "{" + ", ".join(
        f"{name}: {{type: opencv, index_or_path: '{dev}', width: {w}, height: {h}, fps: {fps}, fourcc: MJPG, warmup_s: {warmup_s}}}"
        for name, dev in CFG["cameras"].items()
    ) + "}"


if __name__ == "__main__":
    shown = {"version": VERSION, **CFG}
    shown["_port_for_this_account"] = port()
    shown["_lock_dir"] = str(lock_dir())
    print(json.dumps(shown, ensure_ascii=False, indent=2))
