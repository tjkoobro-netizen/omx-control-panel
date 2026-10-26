#!/usr/bin/env python3
"""Control-PC side of the panel (v2, config "role": "control").

A team member's PC holds the robot (leader, follower, cameras) and runs teleop, recording and the
eval loop with its safety; the GPU laptop trains, serves inference and keeps the team's shared data.
They talk over one SSH connection (Tailscale + the key the installer set up), with two forwards:
  127.0.0.1:<panel port + 30>  -> the GPU laptop's panel of this person's account (training, models,
                                  comparisons, reports, presets, home poses)
  127.0.0.1:<panel port + 31>  -> that account's inference server (omx_infer.py), for evaluations
Nothing is exposed on the network. Recordings and evaluations are made here and then copied to the
shared data folder on the GPU laptop (never over a dataset that came from somewhere else; nothing is
deleted there).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import omx_config

CFG = omx_config.CFG
HOST = CFG.get("gpu_host") or "omx-gpu"
TUNNEL_PANEL = omx_config.port() + 30
TUNNEL_INFER = omx_config.port() + 31
REMOTE_PANEL = int(CFG.get("gpu_panel_port") or 8765)
REMOTE_INFER = REMOTE_PANEL + 25  # omx_infer.default_port() of that account
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
MODEL_CACHE = Path("~/.cache/omx/models").expanduser()

# Requests the GPU laptop answers (everything else is handled here, next to the robot)
PROXY_GET = ("/api/train/", "/api/eval/compare", "/api/eval/report", "/report/", "/api/presets")
PROXY_POST = ("/api/train/", "/api/eval/compare", "/api/eval/report", "/api/presets/", "/api/home/task")


def gpu_user() -> str:
    return CFG.get("gpu_user") or ""


def infer_url() -> str:
    return f"http://127.0.0.1:{TUNNEL_INFER}"


# ---------------------------------------------------------------- the SSH tunnel
class Tunnel:
    """Keeps `ssh -N -L ... -L ...` running; reconnects after a drop (Wi-Fi, sleep)."""

    def __init__(self):
        self.proc = None
        self.ok = False
        self.error = ""
        self.since = None

    def start(self) -> None:
        self._clear_stale()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    @staticmethod
    def _clear_stale() -> None:
        """A tunnel left by a panel that did not shut down cleanly still holds the ports: end it (ours only)."""
        out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if any(f"127.0.0.1:{p} " in line for p in (TUNNEL_PANEL, TUNNEL_INFER)) and '"ssh"' in line:
                for pid in {int(x) for x in __import__("re").findall(r"pid=(\d+)", line)}:
                    try:
                        if Path(f"/proc/{pid}").stat().st_uid == __import__("os").getuid():
                            __import__("os").kill(pid, 15)
                    except OSError:
                        pass
        time.sleep(0.5)

    def _run(self) -> None:
        while True:
            cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", *SSH_OPTS,
                   "-L", f"127.0.0.1:{TUNNEL_PANEL}:127.0.0.1:{REMOTE_PANEL}",
                   "-L", f"127.0.0.1:{TUNNEL_INFER}:127.0.0.1:{REMOTE_INFER}", HOST]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE, text=True)
                time.sleep(2)
                if self.proc.poll() is None:
                    self.ok, self.error, self.since = True, "", time.time()
                err = self.proc.stderr.read()  # until ssh ends
                self.error = (err.strip().splitlines() or ["연결이 끊겼습니다"])[-1][:200]
            except Exception as exc:
                self.error = str(exc)
            self.ok = False
            time.sleep(3)

    def status(self) -> dict:
        return {"ok": self.ok and port_open(TUNNEL_PANEL), "host": HOST, "error": "" if self.ok else self.error}


TUNNEL = Tunnel()


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- the GPU laptop's panel
def gpu_request(method: str, path: str, body: bytes | None = None, ctype: str = "application/json",
                timeout: float = 30.0) -> tuple[int, str, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", TUNNEL_PANEL, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers={"Content-Type": ctype} if body is not None else {})
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type", "application/json"), r.read()
    except OSError as exc:
        raise RuntimeError(f"GPU 노트북({HOST})에 연결되지 않았습니다. 네트워크(Tailscale)를 확인하세요. ({exc})")
    finally:
        conn.close()


def gpu_json(path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    status, _, body = gpu_request("POST" if payload is not None else "GET", path,
                                  json.dumps(payload).encode() if payload is not None else None, timeout=timeout)
    out = json.loads(body or b"{}")
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(out["error"])
    return out


def proxied(method: str, path: str) -> bool:
    return path.startswith(PROXY_GET if method == "GET" else PROXY_POST)


_DATASETS = {"t": 0.0, "items": []}


def gpu_datasets() -> list[dict]:
    """The team's datasets on the GPU laptop (cached 15 s for the status poll)."""
    if time.time() - _DATASETS["t"] > 15:
        try:
            _DATASETS["items"] = gpu_json("/api/datasets", timeout=10).get("datasets", [])
            _DATASETS["t"] = time.time()
        except Exception:
            pass  # not connected yet: ask again at the next status poll
    return _DATASETS["items"]


# ---------------------------------------------------------------- models (configs only, the weights stay there)
def ssh_read(path: str, timeout: float = 30.0) -> bytes:
    r = subprocess.run(["ssh", *SSH_OPTS, HOST, "cat", "--", path], capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"GPU 노트북에서 {path} 를 읽지 못했습니다: {r.stderr.decode(errors='replace').strip()[:160]}")
    return r.stdout


def fetch_model(model_path: str) -> Path:
    """Local copy of a model's config files (config.json, train_config.json): lerobot-record reads the
    policy settings from them; the weights are used only on the GPU laptop."""
    local = MODEL_CACHE / hashlib.sha1(model_path.encode()).hexdigest()[:16]
    local.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "train_config.json"):
        if not (local / name).exists():
            (local / name).write_bytes(ssh_read(f"{model_path.rstrip('/')}/{name}"))
    (local / "SOURCE").write_text(f"{HOST}:{model_path}\n")
    return local


# ---------------------------------------------------------------- copying runs to the shared folder
def origin_tag() -> str:
    return f"{socket.gethostname()}"


def sync_dataset(local_root: Path, lines: list | None = None) -> bool:
    """Copy a recording / eval dataset made here to <gpu data>/<gpu user>/<name> on the GPU laptop.
    Additions and changes go over (a resumed recording grows); nothing there is deleted, and a dataset of
    the same name that did not come from this PC is never touched."""
    user, name = gpu_user(), local_root.name
    if not user:
        raise RuntimeError("설정에 gpu_user(GPU 노트북 계정)가 없습니다.")
    remote = f"{CFG.get('gpu_data_home', '/srv/il/lerobot_data')}/{user}/{name}"
    r = subprocess.run(["ssh", *SSH_OPTS, HOST,
                        f"if test -e '{remote}'; then echo EXISTS; cat '{remote}/.omx_origin' 2>/dev/null; else echo NEW; fi; true"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"GPU 노트북 확인 실패: {r.stderr.strip()[:160]}")
    out = r.stdout.split()
    if out[:1] == ["EXISTS"] and out[1:2] != [origin_tag()]:
        msg = f"GPU 노트북에 같은 이름의 데이터({user}/{name})가 이미 있어 보내지 않았습니다 (다른 곳에서 만든 데이터 보호)."
        if lines is not None:
            lines.append(f"[omx] sync: {msg}")
        raise RuntimeError(msg)
    (local_root / ".omx_origin").write_text(origin_tag())
    r = subprocess.run(["rsync", "-a", "--chmod=Dug+rwx,Fug+rw", "-e", "ssh " + " ".join(SSH_OPTS),
                        f"{local_root}/", f"{HOST}:{remote}/"], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"데이터 전송 실패: {r.stderr.strip()[-200:]}")
    if lines is not None:
        lines.append(f"[omx] sync: GPU 노트북 공유 폴더로 보냈습니다 ({user}/{name})")
    return True


def sync_note(note_path: Path) -> None:
    """The run's notes (results, marks) - this PC is where they are edited; newer copies win."""
    user = gpu_user()
    remote = f"{CFG.get('gpu_data_home', '/srv/il/lerobot_data')}/{user}/.omx_notes/"
    subprocess.run(["rsync", "-a", "--update", "--chmod=Fug+rw", "-e", "ssh " + " ".join(SSH_OPTS),
                    str(note_path), f"{HOST}:{remote}"], capture_output=True, timeout=120)


def sync_later(fn, *args) -> None:
    def run():
        try:
            fn(*args)
        except Exception as exc:
            print(f"[omx] sync: {exc}", flush=True)
    threading.Thread(target=run, daemon=True).start()
