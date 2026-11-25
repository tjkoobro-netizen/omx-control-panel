#!/usr/bin/env python3
"""The mission loss monitor (omx_live_monitor.py next to this file, by Guide; docs/LIVE_MONITOR.md) from the panel
(3.27.0, control PC only).

The monitor reads counters only (cameras, CPU, network, UDP, the eval log) and writes its records to a folder of its
own (OMX_LOSS_DIR, default ~/il_ws/omx_loss_tracker/live: outside the panel folder, so a reinstall keeps them); it never
touches the robot, the panel or an evaluation. The panel only starts it (its own session: setsid, so a panel restart
does not stop it), stops it (by PID, after checking that process's arguments) and reads its files for the screen.
A monitor is "the" monitor when it writes to the same record folder (also one started by hand from another copy).

Settings: cfg "loss_monitor" {"live_dir": "~/il_ws/omx_loss_tracker/live", "gpu": <the GPU laptop ssh name, default
gpu_host>, "auto": true}. "auto" (start it when the panel starts and it is not running) can be switched on the panel;
that choice is kept in loss_monitor.json next to robot.json.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

SCRIPT = "omx_live_monitor.py"
HERE = Path(__file__).resolve().parent
DEFAULT = {"live_dir": "~/il_ws/omx_loss_tracker/live", "gpu": None, "auto": True, "script": None, "panel_logs": None}
STALE_S = 60.0  # status.json is rewritten every 1 s (eval) or 30 s (idle): older than this = stopped
ONE_SHOT = ("--collect", "--gpu-stop", "--report", "--watch")  # the monitor's other commands (not the running one)
REPORT: dict = {"running": False}
_LOCK = threading.Lock()


def settings(cfg: dict, state_file: Path | None = None) -> dict:
    s = dict(DEFAULT)
    s.update({k: v for k, v in (cfg.get("loss_monitor") or {}).items() if k in DEFAULT})
    if state_file is not None and state_file.exists():
        try:
            saved = json.loads(state_file.read_text())
            if isinstance(saved.get("auto"), bool):
                s["auto"] = saved["auto"]
        except (OSError, ValueError):
            pass
    s["live_dir"] = os.path.realpath(Path(str(s["live_dir"])).expanduser())
    s["script"] = str(Path(str(s["script"])).expanduser()) if s.get("script") else str(HERE / SCRIPT)
    return s


def available(s: dict) -> bool:
    return Path(s["script"]).is_file()


def _argv(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def is_monitor(argv: list[str]) -> bool:
    """The running monitor: a python with omx_live_monitor.py as its script and none of the one-shot commands."""
    if len(argv) < 2 or "python" not in os.path.basename(argv[0]):
        return False
    script = next((a for a in argv[1:] if not a.startswith("-")), "")
    return os.path.basename(script) == SCRIPT and not any(a in ONE_SHOT for a in argv)


def _out_dir(pid: int) -> str:
    """Where that monitor writes: its OMX_LOSS_DIR, else the monitor's default."""
    try:
        env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0") if "=" in x)
    except OSError:
        return ""
    return os.path.realpath(Path(env.get("OMX_LOSS_DIR") or DEFAULT["live_dir"]).expanduser())


def pids(s: dict) -> list[int]:
    """This account's running monitors writing to this record folder (normally one)."""
    me, out, home = os.getuid(), [], s["live_dir"]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != me:
                continue
        except OSError:
            continue
        if is_monitor(_argv(int(p.name))) and _out_dir(int(p.name)) == home:
            out.append(int(p.name))
    return sorted(out)


def _cmd(s: dict, *extra: str) -> list[str]:
    return ["python3", s["script"], *(["--gpu", s["gpu"]] if s.get("gpu") else []), *extra]


def _env(s: dict) -> dict:
    env = dict(os.environ, OMX_LOSS_DIR=s["live_dir"], PYTHONUNBUFFERED="1")
    if s.get("panel_logs"):
        env["OMX_PANEL_LOGS"] = str(s["panel_logs"])
    return env


def start(s: dict) -> dict:
    """Start it in its own session (setsid -f: it outlives the panel), unless one runs already."""
    with _LOCK:
        running = pids(s)
        if running:
            return {"started": False, "pids": running}
        live = Path(s["live_dir"])
        live.mkdir(parents=True, exist_ok=True)
        with open(live / "monitor.out", "ab") as out:
            p = subprocess.Popen(["setsid", "-f", "nice", "-n", "10", *_cmd(s)], cwd=str(live), env=_env(s),
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, close_fds=True)
            p.wait(timeout=10)  # setsid -f returns at once; the monitor is not this panel's child
        for _ in range(20):
            if pids(s):
                break
            time.sleep(0.1)
        return {"started": True, "pids": pids(s)}


def stop(s: dict) -> dict:
    """Stop the running monitor(s): SIGTERM by PID after reading its arguments again, SIGKILL after 3 s. The GPU
    side's own small recorder is left running (Guide: 0.06 %, it is not started twice)."""
    with _LOCK:
        done = []
        for pid in pids(s):
            if not is_monitor(_argv(pid)):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                done.append(pid)
            except ProcessLookupError:
                pass
        end = time.time() + 3
        while time.time() < end and any(p in pids(s) for p in done):
            time.sleep(0.1)
        for pid in done:
            if pid in pids(s) and is_monitor(_argv(pid)):
                os.kill(pid, signal.SIGKILL)
        return {"stopped": done, "pids": pids(s)}


def _count_lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _last_line(p: Path) -> dict | None:
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 8192))
            lines = [x for x in f.read().splitlines() if x.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, IndexError):
        return None


def status(s: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    if not available(s):
        return {"available": False}
    live = Path(s["live_dir"])
    day = dt.datetime.fromtimestamp(now).strftime("%Y%m%d")
    st, age = {}, None
    try:
        st = json.loads((live / "status.json").read_text())
        age = round(now - (live / "status.json").stat().st_mtime, 1)
    except (OSError, ValueError):
        pass
    running = pids(s)
    run = _last_line(live / f"{day}_runs.jsonl")
    keep = ("start", "dur_s", "cam_fps_min_med", "cpu_core_max", "resent_total", "pc_udp_drops", "pc_net_drops",
            "loop_slow_lines", "warn_n", "infer_rtt_ms_last")
    return {
        "available": True, "on": bool(running), "pids": running, "auto": bool(s["auto"]), "gpu": s.get("gpu"),
        "live_dir": s["live_dir"],
        "status_at": st.get("at"), "status_age_s": age, "stale": bool(running) and (age is None or age > STALE_S),
        "eval_active": st.get("eval_active"), "warn_now": st.get("warn") or [],
        "warn_today": _count_lines(live / f"{day}_warn.log"), "day": day,
        "last_run": {k: run.get(k) for k in keep if k in run} if isinstance(run, dict) else None,
        "report": dict(REPORT),
    }


def report(s: dict, day: str | None = None) -> dict:
    """After a mission: fetch the GPU side's records (--collect, rsync over ssh) and write live/report_<day>.md
    (--report), in a thread; REPORT holds the state for the screen. The caller refuses it while a run is going."""
    day = day or dt.datetime.now().strftime("%Y%m%d")
    with _LOCK:
        if REPORT.get("running"):
            raise RuntimeError("보고서를 이미 만드는 중입니다.")
        REPORT.clear()
        REPORT.update(running=True, day=day, started=time.time())

    def work() -> None:
        out = {"running": False, "day": day, "ok": False}
        try:
            steps = ([_cmd(s, "--collect")] if s.get("gpu") else []) + [["python3", s["script"], "--report", day]]
            for cmd in steps:
                r = subprocess.run(["nice", "-n", "10", *cmd], cwd=s["live_dir"], env=_env(s), capture_output=True,
                                   text=True, timeout=600)
                if r.returncode != 0:
                    out["error"] = f"{' '.join(cmd[-2:])}: {(r.stderr or r.stdout).strip()[-300:]}"
                    break
            else:
                f = Path(s["live_dir"]) / f"report_{day}.md"
                out.update(ok=f.exists(), file=str(f))
                if not f.exists():
                    out["error"] = "보고서 파일이 생기지 않았습니다."
        except (OSError, subprocess.SubprocessError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["done_at"] = dt.datetime.now().strftime("%H:%M:%S")
        with _LOCK:
            REPORT.clear()
            REPORT.update(out)

    threading.Thread(target=work, daemon=True, name="lossmon-report").start()
    return dict(REPORT)


def report_text(s: dict, day: str) -> str:
    if not day.isdigit() or len(day) != 8:
        raise RuntimeError("날짜는 YYYYMMDD입니다.")
    f = Path(s["live_dir"]) / f"report_{day}.md"
    if not f.is_file():
        raise RuntimeError(f"{day} 보고서가 아직 없습니다.")
    return f.read_text(errors="replace")[:200_000]
