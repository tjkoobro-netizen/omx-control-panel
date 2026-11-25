#!/usr/bin/env python3
"""The control tower's MQTT Host for this PC, started by the panel when a person turns armed on.

The Host (RO:PORT services/roport-omx scripts/cooperative_host.py --executor panel) subscribes to the
control tower's shelf commands and asks this panel to start an evaluation; the panel accepts those
only while armed is on. Turning armed on starts the Host if it is not running. Turning armed off
(or a stop) leaves it running: it still has to report the stopped run and answer status requests,
and it rejects new tasks with NOT_ARMED by itself. The Host stops with the panel.

Settings: ~/.config/roport/omx-host.json (absent → the panel never starts a Host)
  {"repo": "~/kernel-robot-repo-1/services/roport-omx",
   "protocol": "v2",                                                      # v2 (default) or v1
   "control": "control-01",                                               # v2: control id (default)
   "link_config": "~/roport-baseline/run/trial-omx-link.hardware.json",   # v1 only: site/control/binding
   "broker_env": "~/roport-baseline/run/broker.env",                      # export ROPORT_MQTT_*=...
   "routes": "~/.config/roport/shelf-01-routes.json",
   "box_marker": 16,
   "journal": "~/.local/state/roport/shelf-01-hardware.json",             # v1 journal
   "journal_v2": "~/.local/state/roport/shelf-01-hardware-v2.json",       # v2 (default: <journal>-v2.json)
   "client_id": "roport-shelf-01-host"}

3.21.2: the field contract with the control tower is roport-control-v2 (RO:PORT #135), and the Host
starts v2 by default. Each protocol keeps its own journal (the Host refuses the other's), so v2 uses
journal_v2. v1 (the earlier OmxTrial test link) only with "protocol": "v1". A Host that does not
know --protocol (before #135) is v1 only: v2 is refused, v1 starts as before.
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


PROTOCOLS = ("v2", "v1")


def journal_path(raw: dict) -> Path:
    """The journal of the configured protocol: v2 has its own (a v1 journal is refused by the Host)."""
    journal = _path(raw["journal"])
    if (raw.get("protocol") or "v2") != "v2":
        return journal
    if raw.get("journal_v2"):
        return _path(raw["journal_v2"])
    return journal.with_name(journal.stem + "-v2" + journal.suffix)


def host_knows_protocol(repo: Path) -> bool:
    """The Host takes --protocol (RO:PORT #135); an older Host speaks v1 only."""
    try:
        return '"--protocol"' in (repo / SCRIPT).read_text()
    except OSError:
        return False


def load_config() -> dict:
    """Host settings; v1 takes site/control/binding from the trial link file. Raises on a bad file."""
    raw = json.loads(CONFIG.read_text())
    protocol = raw.get("protocol") or "v2"
    if protocol not in PROTOCOLS:
        raise ValueError(f"{CONFIG}: protocol은 v2 또는 v1입니다 (받은 값 {protocol!r})")
    need = ("repo", "broker_env", "routes", "box_marker", "journal") + (("link_config",) if protocol == "v1" else ())
    missing = [k for k in need if raw.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{CONFIG}: {', '.join(missing)} 항목이 없습니다")
    repo = _path(raw["repo"])
    if not (repo / SCRIPT).is_file():
        raise ValueError(f"{repo / SCRIPT} 이 없습니다")
    knows = host_knows_protocol(repo)
    cfg = {
        "repo": repo,
        "protocol": protocol,
        "protocol_flag": knows,
        "broker_env": _path(raw["broker_env"]),
        "routes": _path(raw["routes"]),
        "box_marker": int(raw["box_marker"]),
        "journal": journal_path(raw),
        "client_id": str(raw.get("client_id") or "roport-shelf-01-host"),
    }
    if protocol == "v2":
        if not knows:
            raise ValueError(f"{repo / SCRIPT}: 이 Host는 v2(--protocol)를 모릅니다. Host를 갱신하거나 \"protocol\": \"v1\"로 두세요")
        cfg["control"] = str(raw.get("control") or "control-01")
        return cfg
    link = json.loads(_path(raw["link_config"]).read_text())
    omx = link.get("omx") or {}
    if omx.get("execution_mode") != "hardware":
        raise ValueError(f"{raw['link_config']}: execution_mode가 hardware가 아닙니다 (mock 설정으로 실물 Host를 띄우지 않음)")
    return cfg | {"site": link["site_id"], "control": link["control_id"], "binding": omx["binding_id"]}


def host_command(cfg: dict, uv: str) -> list[str]:
    """cooperative_host.py arguments for the configured protocol."""
    cmd = [uv, "run", "python", SCRIPT]
    if cfg["protocol_flag"]:
        cmd += ["--protocol", cfg["protocol"]]
    cmd += ["--executor", "panel", "--routes", str(cfg["routes"]), "--box-marker", str(cfg["box_marker"]),
            "--control", cfg["control"]]
    if cfg["protocol"] == "v1":
        cmd += ["--site", cfg["site"], "--binding", cfg["binding"]]
    return cmd + ["--client-id", cfg["client_id"], "--journal", str(cfg["journal"])]


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
                cmd = host_command(cfg, uv)
                cfg["journal"].parent.mkdir(parents=True, exist_ok=True)
                log = open(self.log_file, "a")
                log.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} armed → Host 시작 ({cfg['protocol']})\n")
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


# ---- HOLD release (Host PR #103): files next to the Host journal <stem>.json --------------------------------
# Host writes <stem>.host-state.json {"boot_id", "active", "hold": null | {"command_id", "error_code", "detail",
# "mission_result", "last_step", "since"}, "updated_at"}; the panel writes <stem>.hold-request.json and the Host
# answers in <stem>.hold-result.json with the same request_id. The Host checks the arms itself before clearing,
# and clearing never turns armed on.
HOLD_CHECKS = ("arms_safe", "package_checked", "pinky_checked")


def hold_files() -> dict | None:
    """The three file paths from the journal in the Host settings; None without settings."""
    try:
        raw = json.loads(CONFIG.read_text())
        journal = journal_path(raw)  # the running protocol's journal: its Host writes these files
    except (OSError, ValueError, KeyError, TypeError):
        return None
    base = journal.parent / journal.stem
    return {"state": Path(f"{base}.host-state.json"), "request": Path(f"{base}.hold-request.json"),
            "result": Path(f"{base}.hold-result.json")}


def hold_state() -> dict:
    """For /api/status: available = this Host writes its state file (an older Host does not: no button)."""
    files = hold_files()
    if not files or not files["state"].exists():
        return {"available": False, "hold": None}
    try:
        st = json.loads(files["state"].read_text())
    except (OSError, ValueError):
        return {"available": False, "hold": None, "error": "Host 상태 파일을 읽지 못했습니다"}
    hold = st.get("hold") if isinstance(st.get("hold"), dict) else None
    return {"available": True, "hold": hold, "active": st.get("active"), "updated_at": st.get("updated_at"),
            "boot_id": st.get("boot_id")}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def request_hold_release(hold_id: str, checks: dict, wait_s: float = 10.0) -> dict:
    """Write the request and wait for the Host's answer with the same request_id (10 s: while the Host is
    connecting to the broker, up to 5 s, it does not look at requests). The result file stays (the Host
    overwrites it; the last one is a site record)."""
    import uuid

    files = hold_files()
    if not files:
        raise RuntimeError("Host 설정(omx-host.json)이 없습니다.")
    request_id = str(uuid.uuid4())
    _write_json(files["request"], {"request_id": request_id, "hold": hold_id,
                                   "checks": {k: bool(checks.get(k)) for k in HOLD_CHECKS},
                                   "requested_at": time.time(), "by": "panel"})
    def answer(until: float) -> dict | None:
        while time.time() < until:
            try:
                res = json.loads(files["result"].read_text())
                if res.get("request_id") == request_id:
                    return res
                # the Host keeps recent answers by request_id (RO:PORT #103): a newer request's
                # answer on top must not hide ours
                mine = (res.get("results") or {}).get(request_id)
                if isinstance(mine, dict):
                    return mine
            except (OSError, ValueError, AttributeError):
                pass
            time.sleep(0.1)
        return None

    res = answer(time.time() + wait_s)
    if res:
        return res
    # no answer: cancel, so a late release never happens without the person seeing it (the Host only acts on
    # a request file it finds). If the file is already gone the Host took it just now: wait for that answer.
    try:
        if json.loads(files["request"].read_text()).get("request_id") == request_id:
            files["request"].unlink()
    except FileNotFoundError:
        res = answer(time.time() + 3.0)
        if res:
            return res
    except (OSError, ValueError):
        pass
    return {"request_id": request_id, "hold": hold_id, "cleared": None, "reason": "NO_RESPONSE"}
