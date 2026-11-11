"""GPU laptop: keep the UDP inference server (infer_link/server.py, wire RPI1) running for control PCs.

Control PCs reach the AI inference only over UDP. Each one asks for the server every few seconds through the SSH
tunnel while the tunnel is up (POST /api/udp_server): the address to listen on, its own address and a fingerprint of
its key. This module checks what it can before starting (key file, same key, address, port), starts
the server (infer_link.server.main), restarts it if it ends and PINGs it from this PC. The last request is remembered
(state file), so when this panel starts again the server starts at once, without waiting for a control PC or the
tunnel; it is stopped only when the panel closes (or a control PC releases it). Every failure is kept as a short
Korean sentence plus a hint, so the control PC can show it.
"""

import errno
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_KEY = "~/.config/roport/omx-udp.key"
CHECK_S = 2.0
PING_EVERY_S = 5.0
START_GRACE_S = 40.0  # importing PyTorch/LeRobot takes a while before the first PING can be answered


def key_fingerprint(key: bytes) -> str:
    """Short, one-way: tells two PCs hold the same key without sending the key."""
    return hashlib.sha256(key).hexdigest()[:12]


def local_addresses() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()
    except (OSError, subprocess.SubprocessError):
        out = []
    return [a for a in out if ":" not in a]


def port_owner(port: int) -> int | None:
    """PID of this account's process holding udp/port (ss shows only our own processes' PIDs without root)."""
    try:
        out = subprocess.run(["ss", "-Hulnp", f"sport = :{port}"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for part in out.split("pid=")[1:]:
        digits = part.split(",")[0]
        if digits.isdigit():
            return int(digits)
    return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


class UdpServerKeeper:
    def __init__(self, python: Path, tools_dir: Path, log_dir: Path, key_file: str | None = None,
                 state_path: Path | None = None):
        self.python, self.tools_dir = Path(python), Path(tools_dir)
        self.log_path, self.pid_path = Path(log_dir) / "udp_server.log", Path(log_dir) / "udp_server.pid"
        self.key_file = Path(os.path.expanduser(key_file or os.environ.get("OMX_UDP_KEY_FILE") or DEFAULT_KEY))
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None  # ours (started here, or adopted after a panel restart)
        self.args: tuple | None = None  # (bind, port, peers) the running server was started with
        self.want: tuple | None = None  # (bind, port) asked for
        self.peers: set[str] = set()
        self.asked_at = 0.0
        self.started_at = 0.0
        self.fails = 0
        self.next_try = 0.0
        self.error, self.hint = "", ""
        self.external: int | None = None
        self.ping = {"ok": None, "round_ms": None, "at": None}
        self.thread: threading.Thread | None = None
        self.state_path = Path(state_path) if state_path else None
        self._adopt()
        if self.want is None:
            self._restore()

    def _restore(self) -> None:
        """The last control PC request, kept across panel restarts: serve it again without waiting for the tunnel."""
        if self.state_path is None:
            return
        try:
            st = json.loads(self.state_path.read_text())
            self.want, self.peers = (str(st["bind"]), int(st["port"])), {str(p) for p in st["peers"]}
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.asked_at = time.time()
        with self.lock:
            if self._check(*self.want, ""):
                self._keep_locked()

    def _remember(self) -> None:
        if self.state_path is None:
            return
        try:
            if self.want is None:
                self.state_path.unlink(missing_ok=True)
            else:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                self.state_path.write_text(json.dumps({"bind": self.want[0], "port": self.want[1],
                                                       "peers": sorted(self.peers)}) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------ requests from control PCs
    def ensure(self, body: dict) -> dict:
        try:
            bind = str(ipaddress.IPv4Address(str(body.get("bind", "")).strip()))
            peer = str(ipaddress.IPv4Address(str(body.get("peer", "")).strip()))
            port = int(body.get("port") or 5600)
        except (ValueError, TypeError):
            raise RuntimeError("UDP 서버 주소, 포트, 제어 PC 주소가 올바르지 않습니다.") from None
        if not 1024 <= port <= 65535:
            raise RuntimeError("UDP 포트는 1024~65535 사이여야 합니다.")
        with self.lock:
            self.asked_at = time.time()
            if self.want != (bind, port) or peer not in self.peers:
                self.next_try = 0.0  # a new request: try now, not after the back-off
                self.fails = 0
            changed = self.want != (bind, port) or peer not in self.peers
            self.want = (bind, port)
            self.peers.add(peer)
            if changed:
                self._remember()
            if not self._check(bind, port, str(body.get("key_fp") or "")):
                self._stop_locked("설정 확인 실패")
            else:
                if not self.fails:
                    self.error, self.hint = "", ""  # an earlier settings problem is fixed
                self._keep_locked()
        self.start()
        return self.status()

    def release(self, peer: str) -> dict:
        with self.lock:
            self.peers.discard(str(peer))
            if not self.peers:
                self.want = None
            self._remember()
            if self.want is None:
                self._stop_locked("제어 PC가 서버를 놓음")
                self.error, self.hint = "", ""
        return self.status()

    def status(self) -> dict:
        with self.lock:
            running = self._alive()
            return {
                "running": running,
                "external": self.external,
                "pid": self.pid if running else self.external,
                "bind": (self.args or self.want or (None, None))[0],
                "port": (self.args or self.want or (None, None))[1],
                "peers": sorted(self.peers),
                "since": self.started_at if running else None,
                "starting": running and self.ping["ok"] is not True and time.time() - self.started_at < START_GRACE_S,
                "self_ping": dict(self.ping),
                "error": self.error,
                "hint": self.hint,
                "key_file": str(self.key_file),
                "log": self._tail(6),
            }

    # ------------------------------------------------------------ checks before starting
    def _fail(self, error: str, hint: str = "") -> bool:
        self.error, self.hint = error, hint
        return False

    def _check(self, bind: str, port: int, key_fp: str) -> bool:
        from omx_infer import read_udp_key

        try:
            key = read_udp_key(str(self.key_file))
        except FileNotFoundError:
            return self._fail(f"GPU 노트북에 키 파일이 없습니다: {self.key_file}",
                              "제어 PC의 키 파일을 같은 경로로 복사하고 chmod 600")
        except ValueError as exc:
            return self._fail(f"GPU 노트북의 키 파일을 쓸 수 없습니다 ({exc})", f"chmod 600 {self.key_file}")
        if key_fp and key_fp != key_fingerprint(key):
            return self._fail("제어 PC와 GPU 노트북의 키가 서로 다릅니다 (지문 불일치)",
                              "두 PC의 키 파일 내용을 같게 맞추세요")
        if ipaddress.IPv4Address(bind).is_unspecified or ipaddress.IPv4Address(bind).is_loopback:
            return self._fail(f"UDP 서버 주소 {bind}는 쓸 수 없습니다", "GPU 노트북의 공유기(LAN) 주소를 적으세요")
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind((bind, 0))
        except OSError:
            return self._fail(f"GPU 노트북에 주소 {bind}가 없습니다 (지금 주소: {', '.join(local_addresses()) or '없음'})",
                              "③ 평가 설정의 UDP 서버 주소를 GPU 노트북의 지금 주소로 고치거나, 공유기에서 주소를 고정하세요")
        finally:
            probe.close()
        return True

    # ------------------------------------------------------------ the server process
    def _alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self.pid is not None:
            return "infer_link.server" in _cmdline(self.pid)
        return False

    def _adopt(self) -> None:
        """A server this panel started before it was restarted: keep using it (and stop it later)."""
        try:
            pid, bind, port, peers = self.pid_path.read_text().split("\t")
            pid = int(pid)
        except (OSError, ValueError):
            return
        if "infer_link.server" in _cmdline(pid):
            self.pid, self.args = pid, (bind, int(port), tuple(sorted(peers.split(","))))
            self.want, self.peers, self.asked_at = (bind, int(port)), set(peers.split(",")), time.time()
            self.started_at = time.time()

    def _keep_locked(self) -> None:
        if self.want is None:
            return
        bind, port = self.want
        args = (bind, port, tuple(sorted(self.peers)))
        if self._alive():
            if self.args == args:
                return
            self._stop_locked("설정 바뀜")  # restart with the new address or another control PC
        if time.time() < self.next_try:
            return
        owner = port_owner(port)
        busy = owner is not None
        if not busy:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.bind((bind, port))
            except OSError as exc:
                busy = exc.errno == errno.EADDRINUSE
            finally:
                probe.close()
        if busy:
            self.external = owner or -1
            cmd = _cmdline(owner) if owner else ""
            who = f"pid {owner}" + (" · 직접 실행한 omx_udp.server" if "omx_udp" in cmd else "") if owner else "다른 계정"
            self.error, self.hint = "", (f"UDP {port}번을 이미 다른 프로그램({who})이 쓰고 있어 그 서버를 그대로 씁니다. "
                                         "패널이 관리하게 하려면 그 프로그램을 끄세요 (Ctrl+C)")
            return
        self.external = None
        peers = sorted(set(self.peers) | {bind})  # this PC too, for the PING that tells "server alive" from "network"
        # not "-m infer_link.server": that runs the module as __main__ while panel_policy imports infer_link.server
        # again, so its PolicyError is another class and a refusal (MODEL_NOT_ALLOWED) came back as INFER_FAILED
        cmd = [str(self.python), "-c", "import sys; from infer_link.server import main; sys.exit(main())",
               "--bind", bind, "--port", str(port),
               "--key-file", str(self.key_file), "--panel-tools", str(self.tools_dir)]
        for p in peers:
            cmd += ["--peer", p]
        log = open(self.log_path, "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start {bind}:{port} peers {','.join(peers)}\n")
        log.flush()
        try:
            self.proc = subprocess.Popen(cmd, cwd=self.tools_dir, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except OSError as exc:
            self._fail(f"UDP 서버를 시작하지 못했습니다 ({exc})")
            return
        finally:
            log.close()
        self.pid, self.args, self.started_at = self.proc.pid, args, time.time()
        self.ping = {"ok": None, "round_ms": None, "at": None}
        self.error, self.hint = "", ""
        try:
            self.pid_path.write_text(f"{self.pid}\t{bind}\t{port}\t{','.join(self.peers)}")
        except OSError:
            pass

    def _stop_locked(self, why: str) -> None:
        if not self._alive():
            self.proc, self.pid, self.args = None, None, None
            return
        pid = self.pid
        try:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(2)
            elif pid and "infer_link.server" in _cmdline(pid):
                os.kill(pid, 15)
        except OSError:
            pass
        try:
            with open(self.log_path, "a") as log:
                log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} stopped ({why})\n")
        except OSError:
            pass
        self.proc, self.pid, self.args = None, None, None
        self.ping = {"ok": None, "round_ms": None, "at": None}
        try:
            self.pid_path.unlink()
        except OSError:
            pass

    def stop(self) -> None:
        with self.lock:
            self._stop_locked("제어판 종료")

    def _tail(self, n: int) -> list[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4000))
                lines = f.read().decode(errors="replace").splitlines()
        except OSError:
            return []
        return [ln[-200:] for ln in lines if ln.strip()][-n:]

    # ------------------------------------------------------------ watching it
    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _ended_locked(self) -> None:
        code = self.proc.returncode if self.proc is not None else None
        self.proc, self.pid, self.args = None, None, None
        self.fails += 1
        self.next_try = time.time() + (5 if self.fails < 3 else 60)
        last = next((ln for ln in reversed(self._tail(12)) if not ln.startswith("===")), "")
        if "Address already in use" in last:
            self._fail("UDP 포트를 다른 프로그램이 쓰고 있습니다", "GPU 노트북에서 직접 켠 UDP 서버가 있으면 끄세요")
        else:
            self._fail(f"GPU 노트북의 UDP 서버가 꺼졌습니다 (종료 코드 {code}, {self.fails}번째)" + (f": {last}" if last else ""),
                       "아래 서버 기록을 확인하세요" + (". 1분 뒤 다시 시도합니다" if self.fails >= 3 else ". 5초 뒤 다시 켭니다"))

    def _self_ping(self, bind: str, port: int) -> None:
        from infer_link.client import InferenceClient
        from omx_infer import read_udp_key

        client = None
        try:
            client = InferenceClient((bind, port), read_udp_key(str(self.key_file)), bind_host=bind)
            client.ping(timeout=1.0)
            self.ping = {"ok": True, "round_ms": client.last.get("round_ms"), "at": time.time()}
        except Exception:
            self.ping = {"ok": False, "round_ms": None, "at": time.time()}
        finally:
            if client is not None:
                client.close()

    def _run(self) -> None:
        last_ping = 0.0
        while True:
            time.sleep(CHECK_S)
            with self.lock:
                if self.proc is not None and self.proc.poll() is not None:
                    self._ended_locked()
                if self.external is not None and self.want and port_owner(self.want[1]) is None:
                    self.external = None  # the hand-started server was closed: start ours
                if self.error == "" or self.next_try:
                    self._keep_locked()
                running, args = self._alive(), self.args
            if running and args and time.time() - last_ping >= PING_EVERY_S:
                last_ping = time.time()
                self._self_ping(args[0], args[1])
                if self.ping["ok"]:
                    self.fails = 0
