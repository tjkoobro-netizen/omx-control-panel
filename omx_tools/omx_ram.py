#!/usr/bin/env python3
"""RAM and time estimates for a training job (panel training tab and the plan runner).

RAM model for lerobot-train (v0.4.4), fitted to probe runs on team16-PC (RAM 15GiB, 24 cores):
the images in a batch are still the recorded size in float32 (the policy resizes them later on the
GPU), so one batch is
    bb = batch * cameras * frames_per_sample * H * W * 3 * 4 bytes
with frames_per_sample = n_obs_steps (ACT 1, Diffusion 2). Each DataLoader worker keeps batches
ahead (prefetch 2) in shared memory and holds its own dataset copy and video decoders; the main
process holds the batch being trained plus the pinned copy. Peak system RAM (MemAvailable drop):
    A[policy] + B * bb + workers * (PW[policy] + K * bb) + MARGIN
Time per step comes from earlier training logs (updt_s + data_s) with the same policy, batch and
workers, else the same policy and batch, else the calibration runs; otherwise it is unknown.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

GiB = 1024 ** 3
OBS_STEPS = {"act": 1, "diffusion": 2, "smolvla": 1}

# ---- calibration: least-squares fit to the probe runs (within ±0.5GB), plus a margin ----
# The peak comes when the workers first fill their prefetch queue, and again around each epoch end.
A = {"act": 0.395 * GiB, "diffusion": 1.401 * GiB, "smolvla": 0.2 * GiB}   # main process: model, CUDA context, dataset index
B = 8.64                                             # batches held by the main process (training + pinned)
PW = {"act": 0.388 * GiB, "diffusion": 0.762 * GiB, "smolvla": 0.8 * GiB}  # a worker's own dataset copy and video decoders
K = 0.849                                            # shared-memory batches per worker
MARGIN = 0.8 * GiB  # fit error, plus ~0.5GB extra around the end of each epoch (1,750-step run)
# measured peak RAM in GB (MemAvailable drop) and s/step, 2 cameras 640x480, augmentation on,
# 50-episode dataset, 2026-10-22. None = killed at the cap (needs more than the number shown).
CALIBRATION = [
    {"policy": "act", "batch": 8, "workers": 4, "ram_gb": 2.6, "sec": 0.20},
    {"policy": "act", "batch": 8, "workers": 8, "ram_gb": 3.9, "sec": 0.20},
    {"policy": "act", "batch": 16, "workers": 8, "ram_gb": 5.7, "sec": 0.44},
    {"policy": "diffusion", "batch": 8, "workers": 4, "ram_gb": 5.6, "sec": 0.24},
    {"policy": "diffusion", "batch": 16, "workers": 4, "ram_gb": 7.9, "sec": 0.48},  # 1,750 steps, past epoch 1
    {"policy": "diffusion", "batch": 16, "workers": 8, "ram_gb": None, "over": 10.6, "sec": 0.47},
    {"policy": "diffusion", "batch": 32, "workers": 2, "ram_gb": 7.7, "sec": 1.27},
    {"policy": "diffusion", "batch": 32, "workers": 4, "ram_gb": 9.3, "sec": 0.95},
    {"policy": "diffusion", "batch": 32, "workers": 8, "ram_gb": None, "over": 9.7, "sec": None},
]
CALIBRATION += [  # 2026-10-23, 150-step probes; SmolVLA = lerobot/smolvla_base fine-tuning
    {"policy": "smolvla", "batch": 8, "workers": 4, "ram_gb": 4.1, "sec": 0.23},
    {"policy": "smolvla", "batch": 16, "workers": 4, "ram_gb": 5.0, "sec": 0.45},
    {"policy": "smolvla", "batch": 32, "workers": 4, "ram_gb": 6.8, "sec": 0.94},
    {"policy": "smolvla", "batch": 32, "workers": 8, "ram_gb": None, "over": 10.6, "sec": 0.92},
]
# GPU memory of the training itself (GiB) = V0 + V1 * batch, from the same probes (nvidia-smi peak minus
# what was already in use). SmolVLA batch 64 ran out of the 16GB.
VRAM = {"act": (0.95, 0.475), "diffusion": (4.1, 0.256), "smolvla": (1.61, 0.216)}
PROBE_RATES = {(c["policy"], c["batch"], c["workers"]): c["sec"] for c in CALIBRATION if c["sec"]}

YELLOW, RED = 0.70, 0.90                             # share of available RAM


def mem_available() -> tuple[float, float]:
    """(available, total) RAM in bytes from /proc/meminfo."""
    info = Path("/proc/meminfo").read_text()
    get = lambda k: int(re.search(rf"^{k}:\s+(\d+)", info, re.M).group(1)) * 1024  # noqa: E731
    return get("MemAvailable"), get("MemTotal")


def camera_shapes(dataset_roots: list[Path]) -> list[tuple[int, int]]:
    """(H, W) of every camera in the first dataset (merged training data must share them)."""
    for root in dataset_roots:
        try:
            feats = json.loads((root / "meta/info.json").read_text())["features"]
        except (OSError, ValueError, KeyError):
            continue
        shapes = []
        for name, f in feats.items():
            if f.get("dtype") in ("video", "image") and len(f.get("shape", [])) == 3:
                h, w = f["shape"][:2] if f["shape"][2] in (1, 3, 4) else f["shape"][1:]
                shapes.append((int(h), int(w)))
        if shapes:
            return shapes
    return [(480, 640), (480, 640)]


def batch_bytes(policy: str, batch: int, shapes: list[tuple[int, int]]) -> int:
    return batch * OBS_STEPS.get(policy, 1) * sum(h * w * 3 * 4 for h, w in shapes)


def ram_bytes(policy: str, batch: int, workers: int, shapes: list[tuple[int, int]]) -> float:
    bb = batch_bytes(policy, batch, shapes)
    p = policy if policy in A else "act"
    return A[p] + B * bb + workers * (PW[p] + K * bb) + MARGIN


def vram_bytes(policy: str, batch: int) -> float:
    v0, v1 = VRAM.get(policy, VRAM["act"])
    return (v0 + v1 * batch) * GiB


def gpu_memory() -> tuple[float, float] | None:
    """(used by others right now, total) GPU memory in bytes, or None without nvidia-smi."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        used, total = (float(x) * 2**20 for x in out.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def suggest(policy: str, batch: int, workers: int, shapes, limit: float) -> dict | None:
    """A nearby combination under `limit` bytes: fewer workers first (keeps the training the same),
    then a smaller batch."""
    for b in [batch] + [x for x in (batch // 2, batch // 4) if x >= 4]:
        for w in range(min(workers, 16), -1, -1):
            if (b, w) == (batch, workers):
                continue
            if ram_bytes(policy, b, w, shapes) <= limit and (w >= 2 or b == batch):
                return {"batch_size": b, "num_workers": w, "ram_gb": round(ram_bytes(policy, b, w, shapes) / GiB, 1)}
    return None


# ---- time per step from earlier logs ----
_LOG_CACHE: dict[str, tuple[float, dict | None]] = {}
CMD_RE = re.compile(r"lerobot-train .*?--policy\.type=(\w+).*?--batch_size=(\d+)(?:.*?--num_workers=(\d+))?")
STEP_RE = re.compile(r"updt_s:([0-9.]+) data_s:([0-9.]+)")


def _log_rate(path: Path) -> dict | None:
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    hit = _LOG_CACHE.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    out = None
    try:
        text = path.read_text(errors="replace")
        m = CMD_RE.search(text)
        steps = [float(a) + float(b) for a, b in STEP_RE.findall(text)]
        if m and len(steps) >= 5:
            steps = steps[len(steps) // 5:]  # skip the warm-up
            out = {"policy": m.group(1), "batch": int(m.group(2)), "workers": int(m.group(3) or 8),
                   "sec": statistics.median(steps), "log": str(path)}
    except OSError:
        pass
    _LOG_CACHE[str(path)] = (mt, out)
    return out


def step_seconds(policy: str, batch: int, workers: int, ckpt_home: Path) -> tuple[float | None, str]:
    # team runs next to the models, plus this account's own logs (also stopped or failed runs)
    logs = list(ckpt_home.glob("*/*/train.log")) + list((Path.home() / "il_ws/omx_logs").glob("train_*.log"))
    rates = [r for r in (_log_rate(p) for p in logs) if r and r["policy"] == policy]
    same = [r["sec"] for r in rates if r["batch"] == batch and r["workers"] == workers]
    if same:
        return statistics.median(same), f"이전 학습 {len(same)}회 기록"
    if (policy, batch, workers) in PROBE_RATES:
        return PROBE_RATES[(policy, batch, workers)], "시험 측정"
    near = [r["sec"] for r in rates if r["batch"] == batch]
    if near:
        return statistics.median(near), "이전 학습 기록 (로딩 수 다름)"
    return None, ""


def estimate(policy: str, batch: int, workers: int, steps: int, dataset_roots: list[Path], ckpt_home: Path,
             frames: int = 0) -> dict:
    shapes = camera_shapes(dataset_roots)
    avail, total = mem_available()
    ram = ram_bytes(policy, batch, workers, shapes)
    ratio = ram / avail if avail else 9
    level = "bad" if ratio > RED else "warn" if ratio > YELLOW else "ok"
    sec, src = step_seconds(policy, batch, workers, ckpt_home)
    gpu = gpu_memory()
    vram = vram_bytes(policy, batch)
    vram_info = {}
    if gpu:
        used, gtotal = gpu
        vratio = (vram + used) / gtotal
        vram_info = {"vram_gb": round(vram / GiB, 1), "vram_used_gb": round(used / GiB, 1),
                     "vram_total_gb": round(gtotal / GiB, 1), "vram_ratio": round(vratio, 2),
                     "vram_level": "bad" if vratio > 0.95 else "warn" if vratio > 0.85 else "ok"}
        order = {"ok": 0, "warn": 1, "bad": 2}
        if order[vram_info["vram_level"]] > order[level]:
            level = vram_info["vram_level"]
    return {**vram_info,
        "ram_gb": round(ram / GiB, 1), "avail_gb": round(avail / GiB, 1), "total_gb": round(total / GiB, 1),
        "ratio": round(ratio, 2), "level": level, "cameras": len(shapes),
        "resolution": f"{shapes[0][1]}x{shapes[0][0]}" if shapes else "",
        "suggest": suggest(policy, batch, workers, shapes, YELLOW * avail) if level != "ok" else None,
        "sec_per_step": round(sec, 3) if sec else None, "time_source": src,
        "total_s": int(sec * steps) if sec else None,
        "epochs": round(steps * batch / frames, 1) if frames else None,
    }
