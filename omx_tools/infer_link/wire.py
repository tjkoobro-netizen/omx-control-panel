# Copied from the team's AI server repo (services/roport-ai-server/src/omx_udp, task/omx-cooperative c2baaa0; same code as services/roport-omx/src/roport_omx/infer_link/wire.py); keep the code identical there and here.
"""OMX inference UDP wire v1 (stdlib only): message framing, HMAC and reassembly.

A message (one observation request or one action reply) is split into datagrams of at most
DATAGRAM_BYTES. Each carries the whole message's size, CRC32 and HMAC-SHA256, so the receiver
uses a message only when every chunk arrived and both checks pass; nothing is ever patched up.
A newer request_id from the same session supersedes an incomplete older one: an observation is
never waited for once a fresher one exists. The fragment layout follows the team's camera frame
wire (roport-control-server adapters/udp/udp_frame_protocol.py).

Body: 4-byte big-endian JSON length, JSON header, then the binary blobs the header points to
(the OMX panel's inference format: JPEG camera frames for "obs").
"""

import hashlib
import hmac
import json
import math
import struct
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

MAGIC = b"RPI1"  # wire version; incompatible changes bump it
DATAGRAM_BYTES = 1200  # fits a 1280-byte IPv6/VPN MTU without IP fragmentation
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MIN_KEY_BYTES = 16
_HEADER = struct.Struct("!4sB16sQHHII32s")
CHUNK_BYTES = DATAGRAM_BYTES - _HEADER.size

PING, PONG, LOAD, LOADED, INFER, ACTIONS, ERROR = range(1, 8)
KINDS = {PING, PONG, LOAD, LOADED, INFER, ACTIONS, ERROR}
KIND_NAMES = {
    PING: "ping",
    PONG: "pong",
    LOAD: "load",
    LOADED: "loaded",
    INFER: "infer",
    ACTIONS: "actions",
    ERROR: "error",
}
REPLIES = {PING: PONG, LOAD: LOADED, INFER: ACTIONS}

# INFER header identifiers from the team's AI UDP draft (Confluence MFS 10125435): the reply echoes
# them so the Host applies actions only to the matching attempt, observation and model.
SCHEMA = "roport-omx-infer-v1"
ECHO_FIELDS = (
    "inference_id",
    "observation_seq",
    "task_id",
    "command_id",
    "execution_id",
)

Peer = tuple[str, int]


@dataclass(frozen=True)
class Message:
    kind: int
    session: UUID
    request_id: int
    header: dict
    data: bytes = b""


def _no_constant(name: str):
    raise ValueError(f"non-finite number {name} in header")


def encode_body(header: dict, data: bytes = b"") -> bytes:
    """NaN/Infinity are refused (ValueError): a non-finite action must never reach a motor."""
    raw = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("!I", len(raw)) + raw + data


def decode_body(body: bytes) -> tuple[dict, bytes]:
    if len(body) < 4:
        raise ValueError("body too short")
    (n,) = struct.unpack_from("!I", body)
    if 4 + n > len(body):
        raise ValueError("header length out of range")
    header = json.loads(body[4 : 4 + n], parse_constant=_no_constant)
    if not isinstance(header, dict):
        raise TypeError("header must be a JSON object")
    return header, body[4 + n :]


def check_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < MIN_KEY_BYTES:
        raise ValueError(f"shared key requires at least {MIN_KEY_BYTES} bytes")
    return key


def _mac(key: bytes, kind: int, session: UUID, request_id: int, body: bytes) -> bytes:
    signed = struct.pack("!4sB16sQ", MAGIC, kind, session.bytes, request_id) + body
    return hmac.new(key, signed, hashlib.sha256).digest()


def pack(message: Message, key: bytes) -> list[bytes]:
    """Datagrams for one message, in order; the caller sends all of them."""
    if message.kind not in KINDS or not 0 <= message.request_id < 2**64:
        raise ValueError("unknown kind or request_id out of range")
    body = encode_body(message.header, message.data)
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("message too large")
    count = max(1, math.ceil(len(body) / CHUNK_BYTES))
    crc = zlib.crc32(body)
    mac = _mac(check_key(key), message.kind, message.session, message.request_id, body)
    return [
        _HEADER.pack(
            MAGIC,
            message.kind,
            message.session.bytes,
            message.request_id,
            index,
            count,
            len(body),
            crc,
            mac,
        )
        + body[index * CHUNK_BYTES : (index + 1) * CHUNK_BYTES]
        for index in range(count)
    ]


def _parse(packet: bytes):
    if not _HEADER.size < len(packet) <= DATAGRAM_BYTES:
        raise ValueError("invalid datagram size")
    magic, kind, session, request_id, index, count, size, crc, mac = (
        _HEADER.unpack_from(packet)
    )
    if magic != MAGIC or kind not in KINDS or not 1 <= size <= MAX_MESSAGE_BYTES:
        raise ValueError("invalid version/kind/size")
    if count != math.ceil(size / CHUNK_BYTES) or not 0 <= index < count:
        raise ValueError("invalid fragment count/index")
    chunk = packet[_HEADER.size :]
    if len(chunk) != min(CHUNK_BYTES, size - index * CHUNK_BYTES):
        raise ValueError("invalid fragment length")
    return kind, UUID(bytes=session), request_id, index, count, size, crc, mac, chunk


@dataclass
class _Pending:
    request_id: int
    fields: tuple
    deadline: float
    chunks: dict[int, bytes] = field(default_factory=dict)


class Reassembler:
    """Single consumer. One incomplete message per (peer, session); stale ids are dropped.

    The newest id of a stream advances only after a message passes CRC and HMAC, so a forged or
    corrupted datagram can at most hold up a stream until its incomplete message expires
    (timeout), never block it for good. At most max_streams verified streams are remembered.
    """

    def __init__(
        self,
        key: bytes,
        *,
        peers: set[str] | None = None,
        timeout: float = 1.0,
        max_pending: int = 8,
        max_streams: int = 256,
    ):
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or max_pending < 1
            or max_streams < 1
        ):
            raise ValueError("positive finite reassembly limits required")
        self.key = check_key(key)
        self.peers = None if peers is None else set(peers)
        self.timeout, self.max_pending = timeout, max_pending
        self.max_streams = max_streams
        self.stats: Counter = Counter()
        self._pending: dict[tuple, _Pending] = {}
        self._latest: dict[tuple, int] = {}

    def expire(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        for stream, pending in list(self._pending.items()):
            if now >= pending.deadline:
                self._drop(stream, "timeout")

    def feed(self, packet: bytes, peer: Peer) -> Message | None:
        now = time.monotonic()
        self.expire(now)
        if self.peers is not None and peer[0] not in self.peers:
            self.stats["untrusted_peer"] += 1
            return None
        try:
            kind, session, request_id, index, count, size, crc, mac, chunk = _parse(
                packet
            )
        except (ValueError, struct.error):
            self.stats["malformed_datagram"] += 1  # one datagram, not a whole request
            return None
        stream = (peer, session)
        pending = self._pending.get(stream)
        if request_id <= self._latest.get(stream, -1) or (
            pending is not None and request_id < pending.request_id
        ):
            self.stats["late"] += 1
            return None
        if pending is not None and request_id > pending.request_id:
            self._drop(stream, "superseded")
            pending = None
        if pending is None:
            if len(self._pending) >= self.max_pending:
                self.stats["capacity"] += 1
                return None
            fields = (kind, count, size, crc, mac)
            pending = _Pending(request_id, fields, now + self.timeout)
            self._pending[stream] = pending
            self.stats["fragments_expected"] += count
        if pending.fields != (kind, count, size, crc, mac):
            self._drop(stream, "conflict")
            return None
        if index in pending.chunks:
            self.stats["duplicate"] += 1
            return None
        pending.chunks[index] = chunk
        self.stats["fragments_received"] += 1
        if len(pending.chunks) != count:
            return None
        del self._pending[stream]
        body = b"".join(pending.chunks[i] for i in range(count))
        if zlib.crc32(body) != crc:
            self.stats["checksum"] += 1
            return None
        if not hmac.compare_digest(
            mac, _mac(self.key, kind, session, request_id, body)
        ):
            self.stats["bad_mac"] += 1
            return None
        # re-insert so the least recently verified stream is dropped first by the cap
        self._latest.pop(stream, None)
        self._latest[stream] = request_id
        while len(self._latest) > self.max_streams:
            del self._latest[next(iter(self._latest))]
        try:
            header, data = decode_body(body)
        except (ValueError, TypeError):
            self.stats["malformed_body"] += (
                1  # a whole request, verified but unreadable
            )
            return None
        self.stats["completed"] += 1
        self.stats[f"completed_{KIND_NAMES[kind]}"] += 1
        return Message(kind, session, request_id, header, data)

    def _drop(self, stream: tuple, reason: str):
        """Forget an incomplete message; count it and the fragments that never arrived."""
        pending = self._pending.pop(stream)
        count = pending.fields[1]
        self.stats[reason] += 1
        self.stats["fragments_missing"] += count - len(pending.chunks)
        self.stats[f"{reason}_{KIND_NAMES[pending.fields[0]]}"] += 1
