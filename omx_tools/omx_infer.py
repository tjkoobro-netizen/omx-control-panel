#!/usr/bin/env python3
"""Policy inference for the panel, run in-process (GPU laptop) or as a server for control PCs (v2.0).

Control PC (robot, cameras, recording, safety) <-> GPU laptop (this server, the models):
  - the control PC's omx_async engine keeps executing queued actions at 30 Hz and, when the queue
    runs low, sends the latest observation(s) here: joint state + JPEG camera frames + task;
  - this server runs the same steps as the in-process engine (processors, Diffusion's two-frame
    history, RTC guidance for SmolVLA) and returns the chunk (post-processed for the robot, and the
    raw one RTC needs to continue it).
Only observations at inference time cross the network, not every frame.

  python omx_infer.py serve [--port N]     start the server (127.0.0.1 only; control PCs reach it through
                                           an SSH tunnel over Tailscale, so nothing is exposed)

Wire format (POST /infer): 4-byte big-endian header length, JSON header, then the JPEG blobs it points to.
  header: {"model": <pretrained_model dir on this laptop>, "overrides": {...}, "task": str, "robot_type": str,
           "obs": [{"state": [..], "images": {"observation.images.front": [offset, length], ...}}, ...],
           "chunk": int, "rtc": {"delay": int, "left": [[..], ..] | null} | null}
  reply:  {"actions": [[..]], "original": [[..]], "infer_ms": float, "load_ms": float}
overrides: diffusion_steps (DDIM), n_action_steps (Diffusion async), rename_map (SmolVLA cameras),
           rtc {"horizon", "weight"}.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_SIZE = 2  # models kept loaded per server (one GPU, shared by every control PC of this account)


# ---------------------------------------------------------------- the inference itself (shared)
class PolicyRunner:
    """One loaded policy with its processors, and one chunk inference from an observation history."""

    def __init__(self, model_path: str, overrides: dict | None = None, device: str | None = None):
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        o = overrides or {}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = PreTrainedConfig.from_pretrained(model_path)
        cfg.pretrained_path = model_path
        cfg.device = self.device.type
        if cfg.type == "diffusion" and o.get("diffusion_steps"):
            cfg.noise_scheduler_type, cfg.num_inference_steps = "DDIM", int(o["diffusion_steps"])
        if cfg.type == "diffusion" and o.get("n_action_steps"):
            cfg.n_action_steps = int(o["n_action_steps"])
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(model_path, config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": self.device.type},
                                    "rename_observations_processor": {"rename_map": o.get("rename_map") or {}}})
        self.rtc = False
        if o.get("rtc") and cfg.type in ("smolvla", "pi0", "pi05"):
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=int(o["rtc"].get("horizon", 10)),
                                                      max_guidance_weight=float(o["rtc"].get("weight", 10.0)))
            self.policy.init_rtc_processor()
            self.rtc = True
        self.lock = threading.Lock()

    def image_size(self) -> list | None:
        """[height, width] the policy resizes camera frames to first (Diffusion resize_shape), so the control PC
        can send them at that size: the same pixels reach the model, 1/4 of the bytes. None: send as they are."""
        shape = getattr(self.cfg, "resize_shape", None)
        return list(shape) if self.cfg.type == "diffusion" and shape else None

    def infer(self, hist: list, task: str | None, robot_type: str | None, chunk: int,
              rtc_delay: int = 0, rtc_left=None):
        """hist: observation frames (as lerobot-record builds them), oldest first. Returns (processed, original)
        tensors on the CPU, shape (chunk, action_dim)."""
        import torch

        from lerobot.policies.utils import prepare_observation_for_inference

        policy, dev = self.policy, self.device

        def prep(obs):
            return self.pre(prepare_observation_for_inference(copy(obs), dev, task, robot_type))

        amp = torch.autocast(device_type=dev.type) if dev.type == "cuda" and self.cfg.use_amp else nullcontext()
        grad = nullcontext() if self.rtc else torch.inference_mode()  # RTC steers the denoising with gradients
        with self.lock, grad, amp:
            kwargs = {}
            if policy.name == "diffusion":
                # Diffusion stacks its last n_obs_steps observations: rebuild that history here
                from lerobot.policies.utils import populate_queues
                from lerobot.utils.constants import OBS_IMAGES

                policy.reset()
                frames = hist[-2:] if len(hist) > 1 else [hist[-1]] * 2
                for obs in frames:
                    batch = dict(prep(obs))
                    batch.pop("action", None)
                    if policy.config.image_features:
                        batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
                    policy._queues = populate_queues(policy._queues, batch)
            else:
                batch = prep(hist[-1])
                if self.rtc:
                    kwargs["inference_delay"] = int(rtc_delay)
                    kwargs["prev_chunk_left_over"] = (torch.as_tensor(rtc_left, dtype=torch.float32, device=dev)
                                                      if rtc_left is not None and len(rtc_left) else None)
            actions = policy.predict_action_chunk(batch, **kwargs)[:, : chunk]
            original = actions.squeeze(0).detach().clone()
            processed = self.post(actions.detach()).squeeze(0).to("cpu")
        return processed, original.to("cpu")


# ---------------------------------------------------------------- wire format
def encode(header: dict, blobs: list[bytes]) -> bytes:
    h = json.dumps(header).encode()
    return struct.pack(">I", len(h)) + h + b"".join(blobs)


def decode(body: bytes) -> tuple[dict, bytes]:
    n = struct.unpack(">I", body[:4])[0]
    return json.loads(body[4:4 + n]), body[4 + n:]


def pack_observations(hist: list, quality: int = 90, size: list | None = None) -> tuple[list, list[bytes]]:
    """Observation frames -> (json-able list, JPEG blobs). Camera frames are HxWx3 uint8 (RGB); with size
    [h, w] they are resized first (area interpolation, as the policy's own resize would see them)."""
    import cv2
    import numpy as np

    out, blobs, offset = [], [], 0
    for obs in hist:
        item = {"state": None, "images": {}}
        for k, v in obs.items():
            if k.startswith("observation.images."):
                img = cv2.cvtColor(np.asarray(v), cv2.COLOR_RGB2BGR)
                if size and tuple(img.shape[:2]) != tuple(size):
                    img = cv2.resize(img, (int(size[1]), int(size[0])), interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                b = jpg.tobytes()
                item["images"][k] = [offset, len(b)]
                blobs.append(b)
                offset += len(b)
            elif k == "observation.state":
                item["state"] = np.asarray(v, dtype=np.float32).tolist()
        out.append(item)
    return out, blobs


def unpack_observations(items: list, data: bytes) -> list:
    import cv2
    import numpy as np

    hist = []
    for item in items:
        obs = {"observation.state": np.asarray(item["state"], dtype=np.float32)}
        for k, (off, n) in item["images"].items():
            img = cv2.imdecode(np.frombuffer(data[off:off + n], np.uint8), cv2.IMREAD_COLOR)
            obs[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist.append(obs)
    return hist


# ---------------------------------------------------------------- server (GPU laptop)
class ModelCache:
    def __init__(self, size: int = CACHE_SIZE):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, model: str, overrides: dict) -> tuple[PolicyRunner, float]:
        key = (model, json.dumps(overrides, sort_keys=True))
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key], 0.0
        t = time.perf_counter()
        runner = PolicyRunner(model, overrides)
        load_ms = (time.perf_counter() - t) * 1000
        with self.lock:
            self.items[key] = runner
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return runner, load_ms


def allowed_model(path: str) -> bool:
    """Only policies under the team's checkpoints folder are served."""
    import omx_config

    root = Path(os.environ.get("OMX_CHECKPOINTS", str(omx_config.path("checkpoints")))).resolve()
    p = Path(path).resolve()
    return root in p.parents and (p / "config.json").exists() and (p / "model.safetensors").exists()


def serve(port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cache = ModelCache()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one connection per eval run

        def _send(self, code: int, obj: dict) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                import torch

                self._send(200, {"ok": True, "cuda": torch.cuda.is_available(), "loaded": [k[0] for k in cache.items]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                header, data = decode(body)
                if not allowed_model(header.get("model", "")):
                    return self._send(403, {"error": "학습 기록에 있는 모델이 아닙니다"})
                runner, load_ms = cache.get(header["model"], header.get("overrides") or {})
                if self.path == "/load":
                    return self._send(200, {"ok": True, "load_ms": load_ms, "type": runner.cfg.type,
                                            "image_size": runner.image_size(),
                                            "n_obs": int(getattr(runner.cfg, "n_obs_steps", 1) or 1) if runner.cfg.type == "diffusion" else 1})
                hist = unpack_observations(header["obs"], data)
                rtc = header.get("rtc") or {}
                t = time.perf_counter()
                processed, original = runner.infer(hist, header.get("task"), header.get("robot_type"),
                                                   int(header.get("chunk") or 50), rtc.get("delay", 0), rtc.get("left"))
                self._send(200, {"actions": processed.tolist(), "original": original.tolist(),
                                 "infer_ms": round((time.perf_counter() - t) * 1000, 1), "load_ms": round(load_ms, 1)})
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[omx] inference server on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


def default_port() -> int:
    import omx_config

    return omx_config.port() + 25  # next to this account's panel port


# ---------------------------------------------------------------- client (control PC)
class RemoteClient:
    """Keep-alive HTTP client for one eval run (through the SSH tunnel to the GPU laptop)."""

    def __init__(self, url: str, model: str, overrides: dict | None = None, timeout: float = 10.0):
        from urllib.parse import urlsplit

        u = urlsplit(url)
        self.host, self.port, self.timeout = u.hostname, u.port, timeout
        self.model, self.overrides = model, overrides or {}
        self.conn = None
        self.image_size = None
        self.n_obs = 2

    def _post(self, path: str, payload: bytes) -> dict:
        import http.client

        for attempt in (0, 1):  # one reconnect (the tunnel or keep-alive may have dropped)
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request("POST", path, body=payload, headers={"Content-Type": "application/octet-stream"})
                r = self.conn.getresponse()
                out = json.loads(r.read())
                if r.status != 200:
                    raise RuntimeError(out.get("error") or f"HTTP {r.status}")
                return out
            except (OSError, http.client.HTTPException):
                self.conn = None
                if attempt:
                    raise
        return {}

    def load(self) -> dict:
        """Load the model on the GPU laptop before the run (SmolVLA takes tens of seconds the first time).
        The reply says how large the server wants the camera frames (image_size), so less is sent."""
        old, self.timeout = self.timeout, 180.0
        try:
            self.conn = None
            out = self._post("/load", encode({"model": self.model, "overrides": self.overrides, "obs": []}, []))
        finally:
            self.timeout, self.conn = old, None
        self.image_size = out.get("image_size")
        self.n_obs = int(out.get("n_obs") or 1)  # only Diffusion needs the observation before the latest
        return out

    def infer(self, hist: list, task, robot_type, chunk: int, rtc_delay: int = 0, rtc_left=None):
        import torch

        items, blobs = pack_observations(hist[-self.n_obs:], size=self.image_size)
        header = {"model": self.model, "overrides": self.overrides, "task": task, "robot_type": robot_type,
                  "obs": items, "chunk": chunk,
                  "rtc": {"delay": rtc_delay, "left": rtc_left.tolist() if rtc_left is not None else None}}
        t = time.perf_counter()
        out = self._post("/infer", encode(header, blobs))
        self.last = {"round_ms": (time.perf_counter() - t) * 1000, "infer_ms": out.get("infer_ms"),
                     "bytes": sum(len(b) for b in blobs)}
        return (torch.tensor(out["actions"], dtype=torch.float32), torch.tensor(out["original"], dtype=torch.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int)
    a = ap.parse_args()
    serve(a.port or default_port())
