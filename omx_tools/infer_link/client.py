# Copied from the team's robot repo (services/roport-omx infer_link, wire RPI1); keep the code identical there and here.
"""UDP inference client for the host PC's OMX panel (wire RPI1); stdlib only.

One request at a time. Every call uses a new request_id, and only the reply to that id is
accepted: a late reply to an older observation is dropped. infer() never retries by itself;
on InferenceTimeout the caller sends its newest observation instead and keeps the arms safe
while no actions arrive. load() may be retried because the server keeps loaded models.
"""

import socket
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .wire import (
    ECHO_FIELDS,
    ERROR,
    INFER,
    LOAD,
    PING,
    REPLIES,
    SCHEMA,
    Message,
    Peer,
    Reassembler,
    check_key,
    pack,
)


class InferenceTimeout(TimeoutError):
    pass


class InferenceError(RuntimeError):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class InferenceClient:
    def __init__(self, server: Peer, key: bytes, *, bind_host: str = ""):
        self.server, self.key = server, check_key(key)
        self.session = uuid4()
        self._next_id = 1
        self.reassembler = Reassembler(key, peers={server[0]})
        self.stats = self.reassembler.stats
        self.last: dict = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, 0))

    def close(self):
        self.sock.close()

    def _new_id(self) -> int:
        request_id, self._next_id = self._next_id, self._next_id + 1
        return request_id

    def _call(
        self, kind: int, header: dict, data: bytes, timeout: float, request_id=None
    ) -> dict:
        request_id = request_id or self._new_id()
        message = Message(kind, self.session, request_id, header, data)
        started = time.monotonic()
        sent = 0
        for datagram in pack(message, self.key):
            sent += self.sock.sendto(datagram, self.server)
        deadline = started + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InferenceTimeout(f"no reply to request {request_id}")
            self.sock.settimeout(remaining)
            try:
                packet, peer = self.sock.recvfrom(65535)
            except TimeoutError:
                raise InferenceTimeout(f"no reply to request {request_id}") from None
            if peer != self.server:
                self.stats["other_peer"] += 1
                continue
            reply = self.reassembler.feed(packet, peer)
            if reply is None:
                continue
            if reply.session != self.session or reply.request_id != request_id:
                self.stats["stale_reply"] += 1
                continue
            self.last = {
                "round_ms": round((time.monotonic() - started) * 1000, 1),
                "bytes": sent,
            }
            if reply.kind == ERROR:
                raise InferenceError(
                    str(reply.header.get("code") or "ERROR"),
                    str(reply.header.get("message") or ""),
                )
            if reply.kind != REPLIES[kind]:
                raise InferenceError("BAD_REPLY", f"kind {reply.kind}")
            return reply.header

    def ping(self, timeout: float = 1.0) -> dict:
        return self._call(PING, {}, b"", timeout)

    def load(
        self, model: str, overrides: dict | None = None, *, timeout=180.0, attempts=3
    ) -> dict:
        """Model loading can take tens of seconds the first time."""
        header = {"model": model, "overrides": overrides or {}}
        for _ in range(attempts - 1):
            try:
                return self._call(LOAD, header, b"", timeout)
            except InferenceTimeout:
                pass
        return self._call(LOAD, header, b"", timeout)

    def infer(
        self,
        header: dict,
        data: bytes,
        *,
        timeout: float = 1.0,
        context: dict | None = None,
        captured_at: datetime | None = None,
    ) -> dict:
        """header: the panel's inference header ("model", "obs", "chunk", ...); data: its blobs.

        context carries task_id/command_id/execution_id of the MQTT command being executed
        (None in standalone tests). The reply must echo this attempt's identifiers and model;
        anything else is refused so it never becomes a motor target.
        """
        request_id = self._new_id()
        now = datetime.now(UTC)
        context = context or {}
        request = header | {
            "schema_version": SCHEMA,
            "boot_id": str(self.session),
            "inference_id": str(uuid4()),
            "observation_seq": request_id,
            "captured_at": _iso(captured_at or now),
            "deadline": _iso(now + timedelta(seconds=timeout)),
            **{k: context.get(k) for k in ("task_id", "command_id", "execution_id")},
        }
        reply = self._call(INFER, request, data, timeout, request_id)
        if any(reply.get(k) != request.get(k) for k in ECHO_FIELDS):
            raise InferenceError("RESULT_MISMATCH", "reply is for another attempt")
        if reply.get("model_version") != (
            request.get("model_version") or request.get("model")
        ):
            raise InferenceError("MODEL_MISMATCH", str(reply.get("model_version")))
        return reply


def _iso(moment: datetime) -> str:
    return (
        moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
