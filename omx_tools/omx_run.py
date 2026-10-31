#!/usr/bin/env python3
"""One robot run for the panel: teleop, recording or policy evaluation (LeRobot 0.6.1).

v3 runs its own control loop. It no longer patches lerobot-record, which in 0.6.1 records with a
leader only (policies moved to lerobot-rollout). LeRobot supplies the parts: the robot and leader
classes (plus omx_bimanual's pair), the cameras, LeRobotDataset (standard v3.0 datasets, videos
encoded while recording) and the policies (omx_infer.py). This file supplies what the panel adds
around them: the safe start/stop, 즉시 중단 / 비상 정지, aborted attempts, the reset period, the
live view, the loop rate and the remote / asynchronous inference.

  teleop      --robot.* --teleop.*                                     (no dataset)
  recording   --robot.* --robot.cameras=... --teleop.* --dataset.*     [--resume=true]
  evaluation  --robot.* --robot.cameras=... --policy.path=... --dataset.*

The arguments keep lerobot-record's names, and the output lines the panel reads are the same as
before ("Recording episode N", "Reset the environment", "Stop recording", "[omx] ...").

Commands while running, one per line on stdin (the panel writes them; in a terminal type them + Enter):
  next      finish the current episode / reset early
  rerecord  discard and re-record the current episode
  stop      stop and save the finished episodes
  home      after 비상 정지: return slowly to the start pose
Signals: SIGUSR1 즉시 중단 (evaluation: land now), SIGUSR2 비상 정지 (hold in place, keep the records),
SIGINT stop (a second SIGINT or SIGTERM: stop at once).

Environment (set by the panel): OMX_REMOTE_URL / OMX_REMOTE_MODEL / OMX_REMOTE_OVERRIDES (policy on the
GPU laptop), OMX_ASYNC* / OMX_RTC* (omx_async.py), OMX_FILTER_ALPHA / OMX_FILTER_GRIPPER (output filter),
OMX_RESET_HOME=1 (return to the start pose in the reset period), OMX_TORCH_THREADS.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# The video encoders run in worker processes. Forking this multi-threaded process (camera readers,
# stdin listener) can deadlock them, so start workers with "spawn".
multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draccus  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402,F401  (registers --robot.cameras type opencv)
from lerobot.robots import RobotConfig, make_robot_from_config, omx_follower  # noqa: E402,F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, omx_leader  # noqa: E402,F401
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

import omx_bimanual  # noqa: E402,F401  (registers bi_omx_follower / bi_omx_leader)
import omx_config  # noqa: E402
import omx_safe_start  # noqa: E402

OBS, ACT = "observation", "action"


# ---------------------------------------------------------------- arguments
@dataclass
class DatasetArgs:
    repo_id: str = ""
    single_task: str = ""
    num_episodes: int = 50
    episode_time_s: float = 60
    reset_time_s: float = 60
    fps: int = 30
    root: str | None = None
    video: bool = True
    push_to_hub: bool = False
    # SmolVLA fine-tuned from the pretrained model keeps its camera names (camera1..3)
    rename_map: dict = field(default_factory=dict)
    # videos are encoded while recording (no PNG frames on disk, no long save after each episode);
    # 2 threads per camera: with LeRobot's default (all cores each) four encoders starting together stall the
    # first frames of an episode (4 cameras, 15 s: 441 of 450 frames; with 2: 449)
    streaming_encoding: bool = True
    encoder_threads: int | None = 2


@dataclass
class PolicyArgs:
    path: str = ""
    device: str | None = None
    n_action_steps: int | None = None
    temporal_ensemble_coeff: float | None = None
    noise_scheduler_type: str | None = None
    num_inference_steps: int | None = None


@dataclass
class RunConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig | None = None
    dataset: DatasetArgs = field(default_factory=DatasetArgs)
    policy: PolicyArgs = field(default_factory=PolicyArgs)
    resume: bool = False
    fps: int = 60  # teleop without a dataset
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None


def _say(msg: str) -> None:
    # one write per line: print() writes the text and the newline separately, and lines from two threads
    # (stdin commands, the safe stop) could run together, which the panel reads line by line
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- 즉시 중단 / 비상 정지 / stop
EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
ABORT = {"on": False, "robot": None, "in_loop": False,
         "phase": None,    # "record" (an episode with the dataset) or "reset" (the reset period between them)
         "during": None,   # the phase when 즉시 중단 / 비상 정지 came
         "mode": None}     # "abort" (즉시 중단: land now) or "freeze" (비상 정지: hold in place, land on [홈 위치로 이동])
ABORTED_MIN_S = 1.0  # an attempt cut by 즉시 중단 / 비상 정지 is kept from this long on (shorter: dropped)


def _stop_now() -> None:
    EVENTS["stop_recording"] = True
    EVENTS["exit_early"] = True


def _land_now() -> None:
    robot = ABORT["robot"]
    if robot is not None and robot.is_connected:
        _say("[omx] abort: returning the follower now")
        try:
            robot.disconnect()  # patched by omx_safe_start: soft landing first
        except Exception as exc:
            _say(f"[omx] abort: landing failed ({exc})")


def _on_freeze(signum, frame) -> None:
    """비상 정지 (SIGUSR2): stop sending commands at once and hold where the arm is (torque on), keep
    what was recorded (the attempt in progress as aborted) and finalize the dataset, then wait for the
    panel's [홈 위치로 이동] ("home" on stdin) before the slow return and the safe stop. The panel kills
    the run (as before) only if this line does not come out within a few seconds."""
    omx_safe_start.FREEZE.set()  # also stops a slow move that is running (safe start / reset return)
    if ABORT["on"] and ABORT["mode"] == "freeze":
        return
    first = not ABORT["on"]
    ABORT["on"], ABORT["mode"] = True, "freeze"
    if first:
        ABORT["during"] = ABORT["phase"]
    _say("[omx] freeze: 비상 정지 · 제자리에서 멈춤 (토크 유지). 기록을 저장하고 [홈 위치로 이동]를 기다립니다")
    _stop_now()


def _on_abort(signum, frame) -> None:
    if ABORT["on"]:
        if ABORT["mode"] == "freeze":  # 즉시 중단 after 비상 정지 = go back now
            omx_safe_start.RELEASE.set()
        return
    ABORT["on"], ABORT["mode"] = True, "abort"
    ABORT["during"] = ABORT["phase"]
    _say("[omx] abort: 즉시 중단 (policy stopped)" + (" · 원위치 시간: 끝난 시도는 저장" if ABORT["during"] == "reset" else ""))
    _stop_now()
    if not ABORT["in_loop"]:
        # saving or between loops: the bus is idle, so land from a thread right away
        threading.Thread(target=_land_now, daemon=True).start()
    if ABORT["during"] == "record":
        _say("[omx] abort: 진행 중인 시도는 1초 이상이면 '중단됨'으로 저장합니다")


def _on_interrupt(signum, frame) -> None:
    if EVENTS["stop_recording"] or signum == signal.SIGTERM:
        raise KeyboardInterrupt  # second request: leave at once (the finally blocks still land the arm)
    _say("[omx] control: stop")
    _stop_now()


def _listen_stdin() -> None:
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "next":
            EVENTS["exit_early"] = True
        elif cmd == "rerecord":
            EVENTS["rerecord_episode"] = True
            EVENTS["exit_early"] = True
        elif cmd == "stop":
            _stop_now()
        elif cmd == "home":  # [홈 위치로 이동] after 비상 정지
            omx_safe_start.RELEASE.set()
        else:
            continue
        _say(f"[omx] control: {cmd}")


def _install_controls() -> None:
    signal.signal(signal.SIGUSR1, _on_abort)
    signal.signal(signal.SIGUSR2, _on_freeze)
    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)
    # LeRobot 0.6.1's keyboard listener is not used: it is global on a desktop and also takes the letters
    # n / r / q, so typing in another window (the browser) would end an episode or the run
    threading.Thread(target=_listen_stdin, daemon=True).start()


# ---------------------------------------------------------------- output filter
class ActionFilter:
    """Low-pass filter on the joint targets just before they are sent (and recorded). OMX_FILTER_ALPHA:
    exponential moving average weight of the new target (0.4 / 0.25 / 0.14 = about 0.05 / 0.1 / 0.2 s of
    lag at 30 Hz); off when unset. OMX_FILTER_GRIPPER: off (default: the grasp timing stays exact) or light
    (alpha 0.5, ~1 frame). Restarts with every episode (the first target of an episode is taken as it is)."""

    def __init__(self, alpha: float, gripper: str):
        self.alpha, self.gripper = alpha, gripper
        self.prev = None
        self.gripper_index = None

    def reset(self, names: list | None) -> None:
        self.prev = None
        if names:
            self.gripper_index = next((i for i, n in enumerate(names) if str(n).startswith("gripper")), None)

    def __call__(self, action):
        import torch

        x = action.detach().clone() if isinstance(action, torch.Tensor) else action
        if self.prev is None or getattr(self.prev, "shape", None) != getattr(x, "shape", None):
            self.prev = x
            return x
        a = torch.full_like(x, self.alpha)
        if self.gripper_index is not None:
            a[..., self.gripper_index] = 1.0 if self.gripper == "off" else max(self.alpha, 0.5)
        y = self.prev + a * (x - self.prev)
        self.prev = y
        return y


def _make_filter() -> ActionFilter | None:
    if float(os.environ.get("OMX_FILTER_ALPHA") or 0) <= 0:
        return None
    f = ActionFilter(min(float(os.environ["OMX_FILTER_ALPHA"]), 1.0), os.environ.get("OMX_FILTER_GRIPPER", "off"))
    _say(f"[omx] filter: 출력 부드럽게 alpha {f.alpha:g} (지연 약 {(1 - f.alpha) / f.alpha / 30:.2f}s), "
         f"그리퍼 {'안 씀' if f.gripper == 'off' else '약하게'}")
    return f


# ---------------------------------------------------------------- policy
class Policy:
    """Where the actions come from in an evaluation: the GPU laptop (control PC) or this process, and
    asynchronously (omx_async) or one select_action per step (the policy's own queue)."""

    def __init__(self, cfg: RunConfig):
        import omx_async
        import omx_infer

        if os.environ.get("OMX_TORCH_THREADS"):
            # the default of one thread per core makes the 30 Hz loop drop to ~21 Hz (see EVAL_THREADS in omx_web.py)
            import torch

            torch.set_num_threads(int(os.environ["OMX_TORCH_THREADS"]))
            _say(f"[omx] torch CPU threads: {torch.get_num_threads()}")
        t = time.perf_counter()
        if os.environ.get("OMX_REMOTE_URL"):
            self.backend = omx_infer.RemoteClient(os.environ["OMX_REMOTE_URL"], os.environ["OMX_REMOTE_MODEL"],
                                                  json.loads(os.environ.get("OMX_REMOTE_OVERRIDES") or "{}"))
            _say(f"[omx] async: 원격 추론: {os.environ['OMX_REMOTE_URL']} · 모델 {os.environ['OMX_REMOTE_MODEL']}")
            _say("[omx] remote: GPU 노트북에서 모델을 불러옵니다 (처음엔 SmolVLA 10~30초)")
            info = self.backend.load()
            _say(f"[omx] remote: 모델 준비 완료 ({info.get('type')}, GPU 불러오기 {info.get('load_ms', 0):.0f}ms, "
                 f"전체 {(time.perf_counter() - t) * 1000:.0f}ms)")
        else:
            p = cfg.policy
            overrides = {"rename_map": cfg.dataset.rename_map or {}}
            if p.n_action_steps:
                overrides["n_action_steps"] = p.n_action_steps
            if p.temporal_ensemble_coeff is not None:
                overrides["temporal_ensemble_coeff"] = p.temporal_ensemble_coeff
            if (p.noise_scheduler_type or "").upper() == "DDIM" and p.num_inference_steps:
                overrides["diffusion_steps"] = p.num_inference_steps
            if os.environ.get("OMX_RTC") == "1":
                overrides["rtc"] = {"horizon": int(os.environ.get("OMX_RTC_HORIZON", 10)),
                                    "weight": float(os.environ.get("OMX_RTC_WEIGHT", 10.0))}
            if overrides["rename_map"]:
                _say(f"[omx] policy: camera names {overrides['rename_map']}")
            self.backend = omx_infer.PolicyRunner(p.path, overrides, p.device)
            _say(f"[omx] policy: {self.backend.cfg.type} 불러옴 ({(time.perf_counter() - t) * 1000:.0f}ms, {self.backend.device})")
        self.runner = omx_async.from_env(self.backend)
        if self.runner is None and os.environ.get("OMX_REMOTE_URL"):
            raise RuntimeError("[omx] 원격 추론에는 비동기 엔진(OMX_ASYNC=1)이 필요합니다")

    def new_episode(self, task: str, robot_type: str) -> None:
        self.task, self.robot_type = task, robot_type
        if self.runner is not None:
            self.runner.task, self.runner.robot_type = task, robot_type
            self.runner.new_episode()  # fresh queue: the first action of an episode comes from its own observation
        else:
            self.backend.reset()

    def action(self, obs_frame: dict):
        if self.runner is not None:
            return self.runner.step(obs_frame)
        return self.backend.select(obs_frame, self.task, self.robot_type)


# ---------------------------------------------------------------- dataset
def open_dataset(cfg: RunConfig, robot):
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

    d = cfg.dataset
    root = Path(d.root) if d.root else Path(os.environ.get("HF_LEROBOT_HOME") or HF_LEROBOT_HOME) / d.repo_id
    features = combine_feature_dicts(hw_to_dataset_features(robot.action_features, ACT, d.video),
                                     hw_to_dataset_features(robot.observation_features, OBS, d.video))
    cams = len(getattr(robot, "cameras", {}) or {})
    writers = 0 if d.streaming_encoding else 4 * cams  # PNG frames only without streaming encoding
    common = dict(streaming_encoding=d.streaming_encoding, encoder_threads=d.encoder_threads,
                  image_writer_threads=writers)
    if cfg.resume:
        from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility

        dataset = LeRobotDataset.resume(d.repo_id, root=root, **common)
        sanity_check_dataset_robot_compatibility(dataset, robot, d.fps, features)
    else:
        dataset = LeRobotDataset.create(d.repo_id, d.fps, features=features, root=root, robot_type=robot.name,
                                        use_videos=d.video, **common)
    return dataset


def _buffer_size(dataset) -> int:
    buf = getattr(dataset.writer, "episode_buffer", None)
    return buf.get("size", 0) if buf is not None else 0


def save_episode(dataset) -> bool:
    """Save the episode in the buffer, with the panel's rules: an attempt cut by 즉시 중단 / 비상 정지 is
    kept (marked aborted) from ABORTED_MIN_S on; an empty buffer is skipped; every saved episode also
    gets a small joint-data backup."""
    size = _buffer_size(dataset)
    index = dataset.num_episodes
    if ABORT["on"] and ABORT["during"] != "reset":
        secs = size / max(dataset.fps, 1)
        if secs < ABORTED_MIN_S:
            _say(f"[omx] abort: the episode in progress is not saved ({secs:.1f}s, under {ABORTED_MIN_S:g}s)")
            dataset.clear_episode_buffer()
            return False
        _say(f"[omx] abort: saving the episode in progress as aborted: episode {index} at {secs:.1f}s "
             f"mode {ABORT['mode']}")
    elif ABORT["on"]:
        # 즉시 중단 in the reset period: the attempt before it had finished, so keep it
        _say("[omx] abort: saving the attempt that finished before the reset period")
    if size == 0:
        _say("[omx] skipped an empty episode (stopped before any frame was recorded)")
        return False
    _backup_episode(dataset, dataset.writer.episode_buffer, index)
    dataset.save_episode()
    return True


def _backup_episode(dataset, buffer: dict, index: int) -> None:
    """Joint data of every saved episode, also as a small file outside LeRobot's parquet
    (<owner>/.omx_backup/<dataset>/episode_000003.npz): a run that had to be force-quit leaves its
    parquet without a footer (unreadable); the panel's analysis then reads these instead."""
    try:
        import numpy as np

        out = dataset.root.parent / ".omx_backup" / dataset.root.name
        out.mkdir(parents=True, exist_ok=True)
        arrays = {k.replace(".", "__"): np.asarray(buffer[k]) for k in ("observation.state", "action", "timestamp", "frame_index")
                  if k in buffer}
        names = {k: (dataset.features.get(k) or {}).get("names") for k in ("observation.state", "action")}
        np.savez_compressed(out / f"episode_{index:06d}.npz", **arrays,
                            meta=np.array(json.dumps({"fps": dataset.fps, "names": names, "episode_index": index,
                                                      "aborted": ABORT["on"] and ABORT["during"] != "reset",
                                                      "abort_mode": ABORT["mode"] if ABORT["on"] else None})))
    except Exception as exc:  # the backup must never stop the real save
        _say(f"[omx] backup: episode {index} not written ({exc})")


# ---------------------------------------------------------------- live view (panel)
LIVE_LATEST: dict = {}  # camera -> latest frame
ROBOT_CAMS: dict = {}


def install_live_view(robot) -> None:
    """While this process owns the cameras the panel cannot open them, so share the frames the robot
    already reads: a writer thread JPEG-encodes the latest frame of each camera about 10 times a second
    into /dev/shm (atomic rename), with the real camera fps and brightness. The control loop only stores
    a reference, so its timing is not affected."""
    try:
        import cv2
    except Exception:
        return
    import omx_camstats

    omx_camstats.install()
    ROBOT_CAMS.update(getattr(robot, "cameras", {}) or {})
    if not ROBOT_CAMS:
        return
    prefix = omx_config.live_prefix()

    def writer() -> None:
        last_stats = 0.0
        while True:
            time.sleep(0.1)
            if time.time() - last_stats > 0.5:  # fps and brightness for the panel
                last_stats = time.time()
                try:
                    tmp = f"{prefix}stats.tmp"
                    with open(tmp, "w") as f:
                        json.dump(omx_camstats.stats(ROBOT_CAMS, dict(LIVE_LATEST)), f)
                    os.replace(tmp, f"{prefix}stats.json")
                except Exception:
                    pass
            for cam, frame in list(LIVE_LATEST.items()):
                try:
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        tmp = f"{prefix}{cam}.tmp"
                        with open(tmp, "wb") as f:
                            f.write(jpg.tobytes())
                        os.replace(tmp, f"{prefix}{cam}.jpg")
                except Exception:
                    pass

    threading.Thread(target=writer, daemon=True).start()


def _show(obs: dict) -> None:
    for cam in ROBOT_CAMS:
        if cam in obs:
            LIVE_LATEST[cam] = obs[cam]


def _feed_live_view(robot, stop: threading.Event) -> None:
    """Camera frames only (no motor bus) about 10 times a second, while the robot is not read."""
    while not stop.is_set():
        for name, cam in list(ROBOT_CAMS.items()):
            try:
                LIVE_LATEST[name] = cam.async_read(timeout_ms=200)
            except Exception:
                pass
        stop.wait(0.1)


# ---------------------------------------------------------------- motor output (panel)
class MotorOutput:
    """Each follower joint's output as a share of its maximum torque, for the panel's 모터 출력 view: read
    in the control loop (the only thread on the motor bus) about 10 times a second, and only while the view
    is open (the panel touches <prefix>torque.want while it polls), so a closed view costs nothing.
    Address 126 is Present_Load (0.1 % of max torque) on the XL430 and Present_Current (mA) on the XL330;
    LeRobot calls both Present_Current. XL330 current is taken against its stall current (1.47 A at 5 V)."""
    PERIOD = 0.1
    STALL_MA = {"xl330-m288": 1470, "xl330-m077": 1470}
    LOAD = {"xl430-w250"}

    def __init__(self, robot):
        self.arms = [(getattr(a, "_omx_side", "") or "", a.bus) for a in (getattr(robot, "arms", None) or [robot])]
        self.prefix = omx_config.live_prefix() + "torque"
        self.due = 0.0

    def __call__(self) -> None:
        now = time.perf_counter()
        if now < self.due:
            return
        self.due = now + self.PERIOD
        try:
            if time.time() - os.stat(self.prefix + ".want").st_mtime > 3:
                return
        except OSError:
            return
        arms = []
        for side, bus in self.arms:
            try:
                raw = bus.sync_read("Present_Current", normalize=False)
            except Exception as exc:  # shown on the panel; the control loop goes on
                arms.append({"side": side, "joints": [], "error": str(exc).splitlines()[0][:160]})
                continue
            joints = []
            for name, value in raw.items():
                model = bus.motors[name].model
                if model in self.LOAD:
                    pct = value / 10
                elif model in self.STALL_MA:
                    pct = value / self.STALL_MA[model] * 100
                else:
                    continue
                joints.append({"name": name, "model": model, "pct": round(pct, 1), "raw": int(value)})
            arms.append({"side": side, "joints": joints})
        try:
            with open(self.prefix + ".tmp", "w") as f:
                json.dump({"t": time.time(), "arms": arms}, f)
            os.replace(self.prefix + ".tmp", self.prefix + ".json")
        except OSError:
            pass


MOTOR_OUT: MotorOutput | None = None


# ---------------------------------------------------------------- control loops
class Viewer:
    """Rerun on another PC (the panel's viewer address), as lerobot-record --display_data."""

    def __init__(self, cfg: RunConfig):
        self.on = bool(cfg.display_data)
        if self.on:
            from lerobot.utils.visualization_utils import init_visualization

            init_visualization("rerun", session_name="omx", ip=cfg.display_ip, port=cfg.display_port)

    def log(self, obs, action) -> None:
        if self.on:
            from lerobot.utils.visualization_utils import log_visualization_data

            log_visualization_data("rerun", observation=obs, action=action, compress_images=True)


def control_loop(robot, fps: int, seconds: float, *, teleop=None, policy: Policy | None = None, dataset=None,
                 task: str = "", filt: ActionFilter | None = None, viewer: Viewer | None = None,
                 loop_times: list | None = None) -> None:
    """One episode (with a dataset) or a reset period / warm-up (without): read, decide, send, record."""
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.feature_utils import build_dataset_frame

    interval = 1 / fps
    start = due = time.perf_counter()  # frames on a fixed schedule (start + k / fps): no drift over an episode
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        if EVENTS["exit_early"]:
            EVENTS["exit_early"] = False
            break
        obs = robot.get_observation()
        _show(obs)
        if MOTOR_OUT is not None:
            MOTOR_OUT()
        if loop_times is not None:
            loop_times.append(t0)
        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS) if dataset is not None else None
        if policy is not None:
            act = policy.action(obs_frame)
            if filt is not None:
                act = filt(act)
            action = make_robot_action(act, dataset.features)
        elif teleop is not None:
            action = teleop.get_action()
        else:
            due += interval
            precise_sleep(max(due - time.perf_counter(), 0.0))
            continue
        robot.send_action(action)
        if dataset is not None:
            dataset.add_frame({**obs_frame, **build_dataset_frame(dataset.features, action, prefix=ACT), "task": task})
        if viewer is not None:
            viewer.log(obs, action)
        dt = time.perf_counter() - t0
        if dt > interval:
            logging.warning(f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({fps} Hz).")
        due += interval
        if time.perf_counter() > due + interval:  # fell behind (a stall): start the schedule again from now
            due = time.perf_counter()
        precise_sleep(max(due - time.perf_counter(), 0.0))


def _report_loop_rate(index: int, times: list) -> None:
    """Control rate of a policy episode (the panel keeps it with the attempt)."""
    if len(times) < 10:
        return
    hz = sorted(1.0 / max(b - a, 1e-6) for a, b in zip(times, times[1:]))
    pick = lambda q: hz[min(len(hz) - 1, int(q * len(hz)))]  # noqa: E731
    _say(f"[omx] loop: episode {index} median {pick(0.5):.1f} p5 {pick(0.05):.1f} min {hz[0]:.1f} Hz over {len(hz)} steps")


def warm_up(robot, teleop, fps: int) -> None:
    """With a leader arm, run the safe start before the first recorded episode: the follower catches up
    with the leader (every arm of a cooperative robot) without recording, so the episode starts with the
    leader's real targets instead of the slow approach."""
    states = [getattr(a, "_omx_ss", None) for a in omx_safe_start.arms_of(robot)]
    if not states or None in states or all(st["phase"] == "normal" for st in states):
        return
    _say("[omx] safe start before recording: the follower catches up with the leader first")
    # short slices until every arm has caught up (or stop): next / rerecord have no episode to act on yet
    deadline = time.perf_counter() + omx_safe_start.settings()["timeout_s"] + 30
    while any(st["phase"] != "normal" for st in states) and not EVENTS["stop_recording"] \
            and time.perf_counter() < deadline:
        EVENTS["exit_early"] = EVENTS["rerecord_episode"] = False
        control_loop(robot, fps, 0.2, teleop=teleop)
    EVENTS["exit_early"] = EVENTS["rerecord_episode"] = False


def reset_period(robot, cfg: RunConfig, teleop, viewer: Viewer | None) -> None:
    """Between episodes: with a leader the follower keeps following it; in an evaluation the arm returns to
    its start pose (OMX_RESET_HOME=1) or holds, while the object is put back."""
    seconds = cfg.dataset.reset_time_s
    if teleop is not None:
        control_loop(robot, cfg.dataset.fps, seconds, teleop=teleop, viewer=viewer)
        return
    end = time.perf_counter() + seconds
    feeding = threading.Event()  # keep the panel's live view (and its object detection) current
    threading.Thread(target=_feed_live_view, args=(robot, feeding), daemon=True).start()
    try:
        if os.environ.get("OMX_RESET_HOME") == "1":
            _say(f"[omx] reset period: {seconds}s (the follower returns to its starting pose, put the object back)")
            try:
                omx_safe_start.go_home_all(robot, "reset", EVENTS)  # both arms of a cooperative robot together
            except Exception as exc:  # never break the run over the reset move
                _say(f"[omx] safe reset: 건너뜀 ({exc})")
            finally:
                omx_safe_start.set_pwm_all(robot, 1.0)  # full output again for the next episode
        else:
            _say(f"[omx] reset period: {seconds}s (robot holds its pose, put the object back)")
        while time.perf_counter() < end and not EVENTS["exit_early"]:
            time.sleep(0.05)
        EVENTS["exit_early"] = False
    finally:
        feeding.set()


def run_episodes(cfg: RunConfig, robot, teleop, policy: Policy | None, dataset, viewer: Viewer) -> None:
    d = cfg.dataset
    filt = _make_filter() if policy is not None else None
    if teleop is not None:
        ABORT["phase"] = "reset"
        warm_up(robot, teleop, d.fps)
    recorded = 0
    while recorded < d.num_episodes and not EVENTS["stop_recording"]:
        index = dataset.num_episodes
        logging.info(f"Recording episode {index}")
        times: list = []
        if policy is not None:
            policy.new_episode(d.single_task, getattr(robot, "robot_type", robot.name))
        if filt is not None:
            filt.reset((dataset.features.get(ACT) or {}).get("names"))
        ABORT["phase"], ABORT["in_loop"] = "record", True
        try:
            control_loop(robot, d.fps, d.episode_time_s, teleop=teleop, policy=policy, dataset=dataset,
                         task=d.single_task, filt=filt, viewer=viewer, loop_times=times if policy else None)
        finally:
            ABORT["in_loop"] = False
            if ABORT["on"] and ABORT["mode"] == "abort":
                _land_now()
        if policy is not None:
            _report_loop_rate(index, times)
        if not EVENTS["stop_recording"] and (recorded < d.num_episodes - 1 or EVENTS["rerecord_episode"]):
            logging.info("Reset the environment")
            ABORT["phase"], ABORT["in_loop"] = "reset", teleop is not None
            try:
                reset_period(robot, cfg, teleop, viewer)
            finally:
                ABORT["in_loop"] = False
                if ABORT["on"] and ABORT["mode"] == "abort":
                    _land_now()
        if EVENTS["rerecord_episode"]:
            logging.info("Re-record episode")
            EVENTS["rerecord_episode"] = False
            EVENTS["exit_early"] = False
            dataset.clear_episode_buffer()
            continue
        if save_episode(dataset):
            recorded += 1
        elif ABORT["on"]:
            break


def run_teleop(cfg: RunConfig, robot, teleop) -> None:
    viewer = Viewer(cfg)
    teleop.connect()
    robot.connect()
    interval, last, n, spent = 1 / cfg.fps, time.perf_counter(), 0, 0.0
    start = time.perf_counter()
    try:
        while not EVENTS["stop_recording"]:
            if cfg.teleop_time_s is not None and time.perf_counter() - start > cfg.teleop_time_s:
                break
            t0 = time.perf_counter()
            obs = robot.get_observation()
            _show(obs)
            if MOTOR_OUT is not None:
                MOTOR_OUT()
            action = teleop.get_action()
            robot.send_action(action)
            viewer.log(obs, action)
            dt = time.perf_counter() - t0
            precise_sleep(max(interval - dt, 0.0))
            n, spent = n + 1, spent + (time.perf_counter() - t0)
            if time.perf_counter() - last >= 1.0 and n:
                ms = spent / n * 1000
                _say(f"Teleop loop time: {ms:.2f}ms ({1000 / ms:.0f} Hz)")
                last, n, spent = time.perf_counter(), 0, 0.0
    finally:
        if robot.is_connected:
            robot.disconnect()  # safe stop (omx_safe_start)
        if teleop.is_connected:
            teleop.disconnect()


def run(cfg: RunConfig) -> None:
    omx_safe_start.install()
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    ABORT["robot"] = robot
    install_live_view(robot)
    global MOTOR_OUT
    try:
        MOTOR_OUT = MotorOutput(robot)
    except Exception as exc:  # the view is optional: never stop a run over it
        _say(f"[omx] 모터 출력 보기 꺼짐 ({exc})")
    _install_controls()
    if not cfg.dataset.repo_id:
        if teleop is None:
            raise SystemExit("[omx] 텔레옵에는 --teleop.* 이, 녹화·평가에는 --dataset.repo_id 가 필요합니다")
        run_teleop(cfg, robot, teleop)
        return
    policy = Policy(cfg) if cfg.policy.path or os.environ.get("OMX_REMOTE_URL") else None
    if policy is None and teleop is None:
        raise SystemExit("[omx] 녹화에는 --teleop.* 이, 평가에는 --policy.path 가 필요합니다")
    if ABORT["on"]:  # 즉시 중단 while the model was loading
        return
    dataset = open_dataset(cfg, robot)
    viewer = Viewer(cfg)
    try:
        # the leader first, so the follower is not left idle while the leader connects
        if teleop is not None:
            teleop.connect()
        robot.connect()
        run_episodes(cfg, robot, teleop, policy, dataset, viewer)
    finally:
        logging.info("Stop recording")
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()  # safe stop; after 비상 정지 it waits here for [홈 위치로 이동]
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if cfg.dataset.push_to_hub and dataset.num_episodes > 0:
            dataset.push_to_hub()
        logging.info("Exiting")


def _share_dataset_with_team(cfg: RunConfig) -> None:
    """Group-read the recorded dataset (videos are saved as 600 and skip the folder's default ACL)."""
    import omx_config

    if not cfg.dataset.repo_id:
        return
    home = os.environ.get("HF_LEROBOT_HOME") or str(omx_config.path("data_home"))
    path = Path(cfg.dataset.root) if cfg.dataset.root else Path(home) / cfg.dataset.repo_id
    group = omx_config.CFG.get("team_group")
    if path.exists() and group:
        omx_config.share_with_team(path)
        _say(f"[omx] shared {path} with group {group} (read/write)")


def main() -> None:
    init_logging()
    cfg = draccus.parse(RunConfig, args=sys.argv[1:])
    try:
        run(cfg)
    finally:
        _share_dataset_with_team(cfg)
    # Saved and disconnected: skip interpreter teardown, where a camera or live-view thread still inside
    # OpenCV can abort the process ("terminate called...", code -6) and the panel shows a failed run
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
