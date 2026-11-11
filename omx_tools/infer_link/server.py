# Copied from the team's AI server repo (services/roport-ai-server omx_udp); keep the code identical there and here.
"""OMX inference UDP server (GPU side): observations in, action chunks out; never drives a robot.

The robot loop stays on the host PC (OMX panel): it sends the latest observation when its action
queue runs low and keeps the arms safe if no reply comes. This server answers each complete,
authenticated request once. Before computing, it drains the socket and keeps only the newest
complete request per session, so a backlog of older observations is skipped, not computed.
Retries always use a new request_id (LOAD is cheap to repeat: the policy keeps loaded models).
"""

import argparse
import ipaddress
import os
import socket
import stat
import sys
import time
from pathlib import Path
from typing import Protocol

from .wire import (
    ACTIONS,
    ECHO_FIELDS,
    ERROR,
    INFER,
    LOAD,
    LOADED,
    PING,
    PONG,
    SCHEMA,
    Message,
    Peer,
    Reassembler,
    check_key,
    pack,
)

DEFAULT_PORT = 5600
RCVBUF_BYTES = 8 * 1024 * 1024


class PolicyError(Exception):
    """A refusal with a stable code, e.g. MODEL_NOT_ALLOWED or NOT_LOADED."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class Policy(Protocol):
    def load(self, model: str, overrides: dict) -> dict:
        """Load (or reuse) a model; reply fields such as image_size and n_obs."""

    def infer(self, header: dict, data: bytes) -> dict:
        """One chunk from header["obs"] and its blobs: {"actions", "original", ...}."""

    def status(self) -> dict:
        """Cheap facts for PING, e.g. loaded models."""


class InferenceServer:
    def __init__(
        self,
        sock: socket.socket,
        policy: Policy,
        key: bytes,
        *,
        peers: set[str],
        timeout: float = 1.0,
    ):
        if not peers:
            raise ValueError("allow at least one host PC address")
        self.sock, self.policy, self.key = sock, policy, check_key(key)
        self.reassembler = Reassembler(key, peers=peers, timeout=timeout)
        self.stats = self.reassembler.stats

    def _drain(self, first: tuple[bytes, Peer]) -> list[tuple[Message, Peer]]:
        """Feed every datagram already queued; newest complete request per session."""
        latest: dict[tuple, tuple[Message, Peer]] = {}
        packet, peer = first
        self.sock.setblocking(False)
        try:
            while True:
                message = self.reassembler.feed(packet, peer)
                if message is not None:
                    stream = (peer, message.session)
                    if stream in latest:
                        self.stats["skipped_backlog"] += 1
                    latest[stream] = (message, peer)
                try:
                    packet, peer = self.sock.recvfrom(65535)
                except BlockingIOError:
                    break
        finally:
            self.sock.setblocking(True)
        return list(latest.values())

    def serve_once(self, timeout: float | None = None) -> int:
        """Wait for datagrams and answer the requests they complete; returns replies sent."""
        self.sock.settimeout(timeout)
        try:
            first = self.sock.recvfrom(65535)
        except TimeoutError:
            self.reassembler.expire()
            return 0
        replies = 0
        for message, peer in self._drain(first):
            reply_kind, header = self._answer(message)
            if reply_kind is None:
                continue
            reply = Message(reply_kind, message.session, message.request_id, header)
            for datagram in pack(reply, self.key):
                self.sock.sendto(datagram, peer)
            replies += 1
        return replies

    def _answer(self, message: Message) -> tuple[int | None, dict]:
        try:
            if message.kind == PING:
                return PONG, {"ok": True, **self.policy.status()}
            if message.kind == LOAD:
                model = message.header.get("model")
                if not isinstance(model, str) or not model:
                    raise PolicyError("BAD_REQUEST", "model is required")
                overrides = message.header.get("overrides") or {}
                return LOADED, self.policy.load(model, overrides)
            if message.kind == INFER:
                header = message.header
                if header.get("schema_version") != SCHEMA:
                    raise PolicyError(
                        "BAD_REQUEST", f"schema_version {SCHEMA} required"
                    )
                if (
                    not isinstance(header.get("inference_id"), str)
                    or header.get("observation_seq") != message.request_id
                    or not isinstance(header.get("model"), str)
                    or not isinstance(header.get("obs"), list)
                ):
                    raise PolicyError("BAD_REQUEST", "inference_id, seq, model, obs")
                started = time.perf_counter()
                reply = self.policy.infer(header, message.data)
                infer_ms = round((time.perf_counter() - started) * 1000, 1)
                return ACTIONS, {"infer_ms": infer_ms} | reply | {
                    **{k: header.get(k) for k in ECHO_FIELDS},
                    "model_version": header.get("model_version") or header["model"],
                    "valid_until": header.get("deadline"),
                }
            self.stats["unexpected_kind"] += 1
            return None, {}
        except PolicyError as exc:
            self.stats["refused"] += 1
            return ERROR, {"code": exc.code, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - report, never hide a failed inference
            self.stats["failed"] += 1
            print(f"[omx-udp] {message.kind=} failed: {exc!r}", file=sys.stderr)
            return ERROR, {"code": "INFER_FAILED", "message": f"{type(exc).__name__}"}

    def serve_forever(self):
        while True:
            self.serve_once(1.0)


def read_key(path: Path) -> bytes:
    """Shared key file (hex), owner-only permissions; never commit it."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError(f"{path} must be readable by its owner only (chmod 600)")
    return check_key(bytes.fromhex(path.read_text().strip()))


def bind_address(host: str) -> str:
    address = ipaddress.ip_address(host)
    if address.is_unspecified:
        raise ValueError("bind to the LAN address of this PC, not 0.0.0.0")
    return host


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="this GPU PC's LAN address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--peer", action="append", required=True, help="host PC IP")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument(
        "--panel-tools",
        type=Path,
        default=os.environ.get("OMX_PANEL_TOOLS"),
        help="OMX panel omx_tools directory (LeRobot policy runner)",
    )
    args = parser.parse_args(argv)
    try:
        key = read_key(args.key_file)
        host = bind_address(args.bind)
        peers = {str(ipaddress.ip_address(p)) for p in args.peer}
    except (OSError, ValueError) as exc:
        print(f"[omx-udp] {exc}", file=sys.stderr)
        return 2
    if args.panel_tools is None:
        print("[omx-udp] --panel-tools or OMX_PANEL_TOOLS is required", file=sys.stderr)
        return 2
    from .panel_policy import PanelPolicy

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind((host, args.port))
    server = InferenceServer(sock, PanelPolicy(args.panel_tools), key, peers=peers)
    print(f"[omx-udp] serving {host}:{args.port} for {sorted(peers)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
