#!/usr/bin/env python3
"""The control tower's MQTT Host for this PC, started by the panel when a person turns armed on.

The Host (RO:PORT services/roport-omx scripts/cooperative_host.py --executor panel) subscribes to the
control tower's shelf commands and asks this panel to start an evaluation; the panel accepts those
only while armed is on. Turning armed on starts the Host if it is not running. Turning armed off
(or a stop) leaves it running: it still has to report the stopped run and answer status requests,
and it rejects new tasks with NOT_ARMED by itself. The Host stops with the panel.

Settings: ~/.config/roport/omx-host.json (absent → the panel never starts a Host)
  {"repo": "~/kernel-robot-repo-1/services/roport-omx",
   "link_config": "~/roport-baseline/run/trial-omx-link.hardware.json",   # site/control/binding
   "broker_env": "~/roport-baseline/run/broker.env",                      # export ROPORT_MQTT_*=...
   "routes": "~/.config/roport/shelf-01-routes.json",
   "box_marker": 16,
   "journal": "~/.local/state/roport/shelf-01-hardware.json",
   "client_id": "roport-shelf-01-host"}
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

CONFIG = Path(os.environ.get("OMX_HOST_CONFIG") or "~/.config/roport/omx-host.json").expanduser()
SCRIPT = "scripts/cooperative_host.py"
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?(ROPORT_MQTT_[A-Z_]+)=(.*)$")


def _path(value) -> Path:
    return Path(str(value)).expanduser()


def load_config() -> dict:
    """Host settings with site/control/binding taken from the trial link file. Raises on a bad file."""
    raw = json.loads(CONFIG.read_text())
    need = ("repo", "link_config", "broker_env", "routes", "box_marker", "journal")
    missing = [k for k in need if raw.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{CONFIG}: {', '.join(missing)} 항목이 없습니다")
    link = json.loads(_path(raw["link_config"]).read_text())
    omx = link.get("omx") or {}
    if omx.get("execution_mode") != "hardware":
        raise ValueError(f"{raw['link_config']}: execution_mode가 hardware가 아닙니다 (mock 설정으로 실물 Host를 띄우지 않음)")
    repo = _path(raw["repo"])
    if not (repo / SCRIPT).is_file():
        raise ValueError(f"{repo / SCRIPT} 이 없습니다")
    return {
        "repo": repo,
        "site": link["site_id"],
        "control": link["control_id"],
        "binding": omx["binding_id"],
        "broker_env": _path(raw["broker_env"]),
        "routes": _path(raw["routes"]),
        "box_marker": int(raw["box_marker"]),
        "journal": _path(raw["journal"]),
        "client_id": str(raw.get("client_id") or "roport-shelf-01-host"),
    }


def broker_env(path: Path) -> dict:
    """ROPORT_MQTT_* from a broker.env written as `export KEY=VALUE` lines (no shell is run)."""
    env = {}
    for line in path.read_text().splitlines():
        m = _ENV_LINE.match(line)
        if m:
            env[m.group(1)] = m.group(2).strip().strip("'\"")
    if "ROPORT_MQTT_HOST" not in env:
        raise ValueError(f"{path}: ROPORT_MQTT_HOST가 없습니다")
    return env


def running_hosts() -> list[int]:
    """PIDs of cooperative_host.py processes of this user (one started by hand counts too)."""
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        argv = args.split()
        # the program itself is python or uv running the script; a shell or editor naming it is not
        prog = Path(argv[0]).name if argv else ""
        if (prog.startswith("python") or prog == "uv") and any(a.endswith(SCRIPT.split("/")[-1]) for a in argv[1:4]):
            pids.append(int(pid))
    return pids


class Host:
    def __init__(self, log_dir: Path):
        self.log_file = log_dir / "omx_host.log"
        self.proc: subprocess.Popen | None = None
        self.since: str | None = None
        self.error: str | None = None
        self.lines: collections.deque = collections.deque(maxlen=40)
        self.lock = threading.Lock()

    def ensure(self) -> dict:
        """Start the Host unless one already runs. Never raises: the reason is shown on the panel."""
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return self.status()
            if not CONFIG.exists():
                self.error = None
                return self.status()
            others = running_hosts()
            if others:
                self.error = None
                self.lines.append(f"[omx] 이미 실행 중인 Host 사용 (PID {', '.join(map(str, others))})")
                return self.status()
            try:
                cfg = load_config()
                env = dict(os.environ, **broker_env(cfg["broker_env"]), PYTHONUNBUFFERED="1")
                uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
                cmd = [uv, "run", "python", SCRIPT, "--executor", "panel",
                       "--routes", str(cfg["routes"]), "--box-marker", str(cfg["box_marker"]),
                       "--site", cfg["site"], "--control", cfg["control"], "--binding", cfg["binding"],
                       "--client-id", cfg["client_id"], "--journal", str(cfg["journal"])]
                cfg["journal"].parent.mkdir(parents=True, exist_ok=True)
                log = open(self.log_file, "a")
                log.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} armed → Host 시작\n")
                log.flush()
                # Own session: Ctrl+C in the panel's terminal must not stop it mid-report; stop() does.
                self.proc = subprocess.Popen(cmd, cwd=cfg["repo"], env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, start_new_session=True)
                self.since, self.error = dt.datetime.now().isoformat(timespec="seconds"), None
                self.lines.clear()
                threading.Thread(target=self._pump, args=(self.proc, log), daemon=True).start()
            except Exception as exc:  # noqa: BLE001 - armed stays on; the panel shows why the Host is off
                self.proc, self.error = None, f"Host를 시작하지 못했습니다: {exc}"[:300]
            return self.status()

    def _pump(self, proc: subprocess.Popen, log) -> None:
        with log:
            for line in proc.stdout:
                line = line.rstrip()
                self.lines.append(line)
                log.write(line + "\n")
                log.flush()
            code = proc.wait()
            self.lines.append(f"[omx] Host 종료 (코드 {code})")
            log.write(f"[omx] Host 종료 (코드 {code})\n")

    def stop(self, timeout: float = 10.0) -> None:
        """With the panel: SIGINT lets the Host close its journal; kill only if it does not exit."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def status(self) -> dict:
        mine = self.proc is not None and self.proc.poll() is None
        others = [] if mine or not CONFIG.exists() else running_hosts()
        text = "\n".join(self.lines)
        subscribed = mine and "subscribed" in text.rsplit("broker link lost", 1)[-1]
        return {
            "configured": CONFIG.exists(),
            "config": str(CONFIG),
            "state": "running" if mine else "external" if others else "off",
            "pid": self.proc.pid if mine else (others[0] if others else None),
            "since": self.since if mine else None,
            "subscribed": subscribed if mine else None,
            "error": self.error,
            "log": list(self.lines)[-8:],
            "exit_code": self.proc.returncode if self.proc is not None and not mine else None,
        }
