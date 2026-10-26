#!/usr/bin/env python3
"""Asynchronous policy execution (and RTC) inside lerobot-record, for the panel's policy evaluation.

lerobot-record asks the policy for an action every control step (predict_action). A chunking
policy then runs inference only when its action queue is empty, and the robot stands still for the
whole inference (Diffusion DDPM: ~1.2 s every 8 steps). install() replaces predict_action with an
engine that keeps executing queued actions at the control rate while the next chunk is inferred in
a background thread - the same idea as LeRobot's async inference (policy_server/robot_client), but
in the same process, so dataset recording, episodes, reset periods and the panel's safe start/stop
keep working unchanged.

Every control step t:
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
from contextlib import nullcontext
from copy import copy

AGGREGATE = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "latest_only": lambda old, new: new,
}


def _say(msg: str) -> None:
    print(f"[omx] async: {msg}", flush=True)


class AsyncRunner:
    def __init__(self, chunk: int = 50, threshold: float = 0.5, aggregate: str = "weighted_average",
                 fps: float = 30.0, rtc: bool = False, rtc_horizon: int = 10, rtc_weight: float = 10.0):
        self.chunk, self.threshold, self.fps = max(int(chunk), 1), min(max(float(threshold), 0.0), 1.0), float(fps)
        self.agg_name = aggregate if aggregate in AGGREGATE else "weighted_average"
        self.agg = AGGREGATE[self.agg_name]
        self.rtc, self.rtc_horizon, self.rtc_weight = rtc, int(rtc_horizon), float(rtc_weight)
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.ctx: dict | None = None
        self.latencies: deque = deque(maxlen=20)
        self.net: deque = deque(maxlen=20)  # remote: round trip, server inference time, bytes sent
        self.rtc_ready = False
        self.remote = None  # omx_infer.RemoteClient: the chunk is computed on the GPU laptop (control PC mode)
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
        if self.net:
            rt = sorted(n["round_ms"] for n in self.net)
            inf = sorted(n["infer_ms"] or 0 for n in self.net)
            net = f" · 원격: 왕복 {rt[len(rt) // 2]:.0f}ms (GPU 추론 {inf[len(inf) // 2]:.0f}ms, 보낸 영상 {self.net[-1]['bytes'] // 1024}KB)"
        _say(f"추론 {med:.0f}ms (묶음 {self.chunks}개), 큐 {len(self.queue)}, 큐가 비어 기다린 횟수 {self.stalls} ({self.stall_s:.1f}s)"
             + (" · RTC" if self.rtc else f" · {self.agg_name}") + net)

    # ---------------------------------------------------------------- inference
    def _infer(self, t0: int, hist: list) -> None:
        """Compute one chunk from the observations up to step t0 and merge it into the queue."""
        import torch

        c = self.ctx
        policy, pre, post = c["policy"], c["preprocessor"], c["postprocessor"]
        started = time.perf_counter()
        if self.remote is not None:
            try:
                left = None
                if self.rtc:
                    with self.lock:
                        rest = [q[2] for q in self.queue if q[0] > self.done]
                    left = torch.stack(rest) if rest else None
                lat = max(self.latencies) if self.latencies else 0.0
                processed, original = self.remote.infer([o for _, o in hist], c["task"], c["robot_type"], self.chunk,
                                                        math.ceil(lat * self.fps) if self.rtc else 0, left)
                self.net.append(self.remote.last)
            except BaseException as exc:
                print(f"[omx] async: 원격 추론 오류 ({exc})", flush=True)
                self.error = exc
                return
            self.latencies.append(time.perf_counter() - started)
            self.chunks += 1
            self._merge(t0, processed, original)
            return
        try:
            from lerobot.policies.utils import prepare_observation_for_inference

            def prep(obs):
                return pre(prepare_observation_for_inference(copy(obs), c["device"], c["task"], c["robot_type"]))

            amp = torch.autocast(device_type=c["device"].type) if c["device"].type == "cuda" and c["use_amp"] else nullcontext()
            # RTC steers the denoising with gradients, which inference_mode would forbid
            grad = nullcontext() if self.rtc else torch.inference_mode()
            with grad, amp:
                kwargs = {}
                if policy.name == "diffusion":
                    # Diffusion stacks its last n_obs_steps observations: rebuild that history here
                    from lerobot.policies.utils import populate_queues
                    from lerobot.utils.constants import OBS_IMAGES

                    policy.reset()
                    frames = [o for _, o in hist] if len(hist) > 1 else [hist[-1][1]] * 2
                    for obs in frames:
                        batch = dict(prep(obs))
                        batch.pop("action", None)  # the processors carry an empty action (as in select_action)
                        if policy.config.image_features:
                            batch = dict(batch)
                            batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
                        policy._queues = populate_queues(policy._queues, batch)
                else:
                    batch = prep(hist[-1][1])
                    if self.rtc:
                        lat = max(self.latencies) if self.latencies else 0.0
                        kwargs["inference_delay"] = math.ceil(lat * self.fps)
                        with self.lock:
                            left = [q[2] for q in self.queue if q[0] > self.done]
                        kwargs["prev_chunk_left_over"] = torch.stack(left) if left else None
                actions = policy.predict_action_chunk(batch, **kwargs)[:, : self.chunk]
                original = actions.squeeze(0).clone()
                actions = actions.detach()
                original = original.detach()
                processed = post(actions).squeeze(0).to("cpu")
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

    # ---------------------------------------------------------------- lerobot-record hook
    def predict(self, observation, policy, device, preprocessor, postprocessor, use_amp, task=None, robot_type=None):
        if self.ctx is None or self.ctx["policy"] is not policy:
            self.ctx = {"policy": policy, "device": device, "preprocessor": preprocessor,
                        "postprocessor": postprocessor, "use_amp": use_amp, "task": task, "robot_type": robot_type}
            if self.rtc and not self.rtc_ready and self.remote is None:  # remote: the server sets RTC up
                self._enable_rtc(policy)
        self.ctx["task"] = task
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

    def _enable_rtc(self, policy) -> None:
        if policy.name not in ("smolvla", "pi0", "pi05"):
            _say(f"RTC는 SmolVLA·pi0 계열 전용이라 {policy.name}에서는 끕니다")
            self.rtc = False
            return
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=self.rtc_horizon,
                                             max_guidance_weight=self.rtc_weight)
        policy.init_rtc_processor()
        self.rtc_ready = True
        _say(f"RTC 켬 (execution_horizon {self.rtc_horizon}, max_guidance_weight {self.rtc_weight:g})")


RUNNER: AsyncRunner | None = None


def install(lerobot_record) -> AsyncRunner | None:
    """Replace lerobot_record.predict_action when the panel asked for asynchronous execution."""
    global RUNNER
    if os.environ.get("OMX_ASYNC") != "1":
        return None
    RUNNER = AsyncRunner(chunk=int(os.environ.get("OMX_ASYNC_CHUNK", 50)),
                         threshold=float(os.environ.get("OMX_ASYNC_THRESHOLD", 0.5)),
                         aggregate=os.environ.get("OMX_ASYNC_AGG", "weighted_average"),
                         fps=float(os.environ.get("OMX_FPS", 30)),
                         rtc=os.environ.get("OMX_RTC") == "1",
                         rtc_horizon=int(os.environ.get("OMX_RTC_HORIZON", 10)),
                         rtc_weight=float(os.environ.get("OMX_RTC_WEIGHT", 10.0)))
    if os.environ.get("OMX_REMOTE_URL"):
        import json

        import omx_infer

        RUNNER.remote = omx_infer.RemoteClient(os.environ["OMX_REMOTE_URL"], os.environ["OMX_REMOTE_MODEL"],
                                               json.loads(os.environ.get("OMX_REMOTE_OVERRIDES") or "{}"))
        _say(f"원격 추론: {os.environ['OMX_REMOTE_URL']} · 모델 {os.environ['OMX_REMOTE_MODEL']}")
    lerobot_record.predict_action = RUNNER.predict
    _say(f"비동기 실행: actions_per_chunk {RUNNER.chunk}, chunk_size_threshold {RUNNER.threshold:g}, "
         f"{'RTC' if RUNNER.rtc else RUNNER.agg_name}")
    return RUNNER
