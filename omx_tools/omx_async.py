#!/usr/bin/env python3
"""Asynchronous policy execution (and RTC) for the panel's policy evaluation (omx_run.py).

A chunking policy computes many actions at once. Run synchronously, the robot stands still while the
next chunk is inferred (Diffusion DDPM: ~1.2 s every 8 steps; over the network to the GPU laptop also
the round trip). This engine keeps executing queued actions at the control rate while the next chunk is
inferred in a background thread - the same idea as LeRobot's async inference and the rollout RTC engine,
in the same process as the recording, episodes, reset periods and the safe start/stop.

Every control step t (AsyncRunner.step):
  - the observation is remembered (the two latest, for Diffusion's n_obs_steps=2),
  - when the queue holds <= chunk_size_threshold * actions_per_chunk actions and no inference is
    running, a background inference starts from the latest observation (time t0),
  - the action for step t is taken from the queue (if the queue ran dry, the step waits for the
    running inference, or infers right away at the start of an episode).
A finished chunk covers steps t0, t0+1, ...; steps that already passed while it was computed are
dropped, steps still queued are combined with aggregate_fn (weighted_average: 0.3 old + 0.7 new,
as in LeRobot async), the rest is appended.

RTC (Real-Time Chunking, flow-matching policies: SmolVLA, pi0, pi0.5): the new chunk is generated
already knowing the actions that will run during the inference (prev_chunk_left_over and the
expected inference_delay), so it continues them smoothly; it then replaces the queue instead of
being averaged.

The chunk itself comes from a backend with one method,
    infer(hist, task, robot_type, chunk, rtc_delay, rtc_left) -> (processed, original)
omx_infer.RemoteClient (the GPU laptop, control PC) or omx_infer.PolicyRunner (this process, GPU laptop).

Settings come from environment variables set by the panel:
  OMX_ASYNC=1, OMX_ASYNC_CHUNK (actions_per_chunk), OMX_ASYNC_THRESHOLD (0..1),
  OMX_ASYNC_AGG (weighted_average | average | latest_only), OMX_FPS,
  OMX_RTC=1, OMX_RTC_HORIZON (execution_horizon), OMX_RTC_WEIGHT (max_guidance_weight)
Progress lines "[omx] async: ..." go to the log every 10 s.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

AGGREGATE = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "latest_only": lambda old, new: new,
}


def _say(msg: str) -> None:
    import sys

    sys.stdout.write(f"[omx] async: {msg}\n")  # one write: lines from two threads stay apart
    sys.stdout.flush()


class AsyncRunner:
    def __init__(self, backend, chunk: int = 50, threshold: float = 0.5, aggregate: str = "weighted_average",
                 fps: float = 30.0, rtc: bool = False):
        self.backend = backend
        self.chunk, self.threshold, self.fps = max(int(chunk), 1), min(max(float(threshold), 0.0), 1.0), float(fps)
        self.agg_name = aggregate if aggregate in AGGREGATE else "weighted_average"
        self.agg = AGGREGATE[self.agg_name]
        self.rtc = rtc
        self.task, self.robot_type = None, None
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.latencies: deque = deque(maxlen=20)
        self.new_episode()

    # ---------------------------------------------------------------- episode / stats
    def new_episode(self) -> None:
        if self.worker is not None:
            self.worker.join(timeout=10)
        with self.lock:
            self.queue: list[list] = []  # [timestep, processed action (cpu), original action]
            self.history: deque = deque(maxlen=2)  # (timestep, observation frame)
            self.t = -1     # last step asked for an action
            self.done = -1  # last step whose action was handed out
            self.error: BaseException | None = None
        self.stalls = 0
        self.stall_s = 0.0
        self.chunks = 0
        self.last_report = time.monotonic()

    def _report(self) -> None:
        now = time.monotonic()
        if now - self.last_report < 10:
            return
        self.last_report = now
        lat = sorted(self.latencies)
        med = lat[len(lat) // 2] * 1000 if lat else 0
        net = ""
        stats = list(getattr(self.backend, "net", []) or [])
        if stats:
            rt = sorted(n["round_ms"] for n in stats)
            inf = sorted(n["infer_ms"] or 0 for n in stats)
            net = f" · 원격: 왕복 {rt[len(rt) // 2]:.0f}ms (GPU 추론 {inf[len(inf) // 2]:.0f}ms, 보낸 영상 {stats[-1]['bytes'] // 1024}KB)"
        _say(f"추론 {med:.0f}ms (묶음 {self.chunks}개), 큐 {len(self.queue)}, 큐가 비어 기다린 횟수 {self.stalls} ({self.stall_s:.1f}s)"
             + (" · RTC" if self.rtc else f" · {self.agg_name}") + net)

    # ---------------------------------------------------------------- inference
    def _infer(self, t0: int, hist: list) -> None:
        """Compute one chunk from the observations up to step t0 and merge it into the queue."""
        import torch

        started = time.perf_counter()
        try:
            left = None
            if self.rtc:
                with self.lock:
                    rest = [q[2] for q in self.queue if q[0] > self.done]
                left = torch.stack(rest) if rest else None
            lat = max(self.latencies) if self.latencies else 0.0
            processed, original = self.backend.infer([o for _, o in hist], self.task, self.robot_type, self.chunk,
                                                     math.ceil(lat * self.fps) if self.rtc else 0, left)
        except BaseException as exc:  # surface in the control thread (with where it happened, in the log)
            import traceback

            print("[omx] async: 추론 오류\n" + traceback.format_exc(), flush=True)
            self.error = exc
            return
        self.latencies.append(time.perf_counter() - started)
        self.chunks += 1
        self._merge(t0, processed, original)

    def _merge(self, t0: int, processed, original) -> None:
        with self.lock:
            now = self.done + 1  # the next step to execute
            old = {q[0]: q for q in self.queue if q[0] >= now}
            new = []
            for i in range(len(processed)):
                ts = t0 + i
                if ts < now:
                    continue  # already executed while this chunk was being computed
                if ts in old and not self.rtc:
                    new.append([ts, self.agg(old.pop(ts)[1], processed[i]), original[i]])
                else:
                    old.pop(ts, None)
                    new.append([ts, processed[i], original[i]])
            tail = [old[k] for k in sorted(old) if k > (new[-1][0] if new else now - 1)]
            self.queue = new + tail if not self.rtc or not new else new

    def _start(self, t0: int) -> None:
        hist = list(self.history)
        self.worker = threading.Thread(target=self._infer, args=(t0, hist), daemon=True)
        self.worker.start()

    # ---------------------------------------------------------------- one control step
    def step(self, observation: dict):
        """The action (processed, on the CPU) for this control step, from the dataset-style observation frame."""
        with self.lock:
            self.t += 1
            t = self.t
            self.history.append((t, observation))
            self.queue = [q for q in self.queue if q[0] >= t]
            have = len(self.queue)
        busy = self.worker is not None and self.worker.is_alive()
        if have == 0:
            waited = time.perf_counter()
            if busy:
                self.worker.join()
            with self.lock:
                self.queue = [q for q in self.queue if q[0] >= t]
                have = len(self.queue)
            if have == 0:
                self._infer(t, list(self.history))  # start of an episode, or the queue ran dry
            if t > 0:
                self.stalls += 1
                self.stall_s += time.perf_counter() - waited
        elif not busy and have <= self.threshold * self.chunk:
            self._start(t)
        if self.error is not None:
            err, self.error = self.error, None
            raise RuntimeError(f"[omx] async: 추론 실패: {err}") from err
        with self.lock:
            self.queue = [q for q in self.queue if q[0] >= t]
            item = self.queue.pop(0) if self.queue else None
            self.done = t
        self._report()
        if item is None:
            raise RuntimeError("[omx] async: 실행할 행동이 없습니다")
        return item[1]


def from_env(backend) -> AsyncRunner | None:
    """The engine the panel asked for (OMX_ASYNC=1), or None for the synchronous select_action path."""
    if os.environ.get("OMX_ASYNC") != "1":
        return None
    runner = AsyncRunner(backend, chunk=int(os.environ.get("OMX_ASYNC_CHUNK", 50)),
                         threshold=float(os.environ.get("OMX_ASYNC_THRESHOLD", 0.5)),
                         aggregate=os.environ.get("OMX_ASYNC_AGG", "weighted_average"),
                         fps=float(os.environ.get("OMX_FPS", 30)),
                         rtc=os.environ.get("OMX_RTC") == "1" and getattr(backend, "rtc_capable", True))
    if os.environ.get("OMX_RTC") == "1" and not runner.rtc:
        _say("RTC는 SmolVLA·pi0 계열 전용이라 이 모델에서는 끕니다")
    _say(f"비동기 실행: actions_per_chunk {runner.chunk}, chunk_size_threshold {runner.threshold:g}, "
         f"{'RTC' if runner.rtc else runner.agg_name}")
    return runner
