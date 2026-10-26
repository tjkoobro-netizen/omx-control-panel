#!/usr/bin/env python3
"""lerobot-record wrapper: same arguments, plus fixes for hangs and remote control.

Over SSH or from the web panel there is no display, so LeRobot disables its
arrow-key listener. This wrapper also feeds the same events from stdin lines
(at a desktop terminal the arrow keys keep working as well):
  next      -> finish the current episode / reset early   (right arrow)
  rerecord  -> discard and re-record the current episode  (left arrow)
  stop      -> stop recording and save finished episodes  (Esc)
All other arguments are passed to lerobot-record unchanged.
"""

import json
import multiprocessing
import os
import sys
import threading
import time

# save_episode() encodes each camera's video in a ProcessPoolExecutor. Forking this
# multi-threaded process (camera readers, stdin listener) can deadlock the workers and
# hang the recording forever after the last episode, so start workers with "spawn".
multiprocessing.set_start_method("spawn", force=True)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.scripts import lerobot_record  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import omx_safe_start  # noqa: E402
import omx_bimanual  # noqa: E402,F401  (registers the cooperative robot: --robot.type=bi_omx_follower)

# Policy runs: limit PyTorch's CPU thread pool (the panel passes OMX_TORCH_THREADS with OMP_NUM_THREADS).
# The default of one thread per core makes the 30 Hz loop drop to ~21 Hz; see EVAL_THREADS in omx_web.py.
if os.environ.get("OMX_TORCH_THREADS"):
    import torch  # noqa: E402

    torch.set_num_threads(int(os.environ["OMX_TORCH_THREADS"]))
    print(f"[omx] torch CPU threads: {torch.get_num_threads()}", flush=True)

# Follower starts with a slow, low-power lift -> close gripper -> approach (no full-power jump)
omx_safe_start.install()

# Policy runs with OMX_ASYNC=1: keep executing queued actions while the next chunk is inferred
# (and RTC for SmolVLA). Replaces lerobot_record.predict_action; see omx_async.py.
import omx_async  # noqa: E402

ASYNC = omx_async.install(lerobot_record)


class ActionFilter:
    """Low-pass filter on the joint targets just before they are sent (and recorded: lerobot-record
    records the predict_action output it sends). OMX_FILTER_ALPHA: exponential moving average weight
    of the new target (0.4 / 0.25 / 0.14 = about 0.05 / 0.1 / 0.2 s of lag at 30 Hz); off when unset.
    OMX_FILTER_GRIPPER: off (default: the grasp timing stays exact) or light (alpha 0.5, ~1 frame).
    Restarts with every episode (the first target of an episode is taken as it is)."""

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


FILTER = None
if float(os.environ.get("OMX_FILTER_ALPHA") or 0) > 0:
    FILTER = ActionFilter(min(float(os.environ["OMX_FILTER_ALPHA"]), 1.0), os.environ.get("OMX_FILTER_GRIPPER", "off"))
    _predict = lerobot_record.predict_action

    def _filtered_predict(*args, **kwargs):
        return FILTER(_predict(*args, **kwargs))

    lerobot_record.predict_action = _filtered_predict
    print(f"[omx] filter: 출력 부드럽게 alpha {FILTER.alpha:g} (지연 약 {(1 - FILTER.alpha) / FILTER.alpha / 30:.2f}s), "
          f"그리퍼 {'안 씀' if FILTER.gripper == 'off' else '약하게'}", flush=True)

# A fine-tuned SmolVLA keeps the pretrained camera names (camera1..3); the panel passes
# --dataset.rename_map (front -> camera1, wrist -> camera2), but lerobot-record v0.4.4 hands it only to
# the processors and calls make_policy(cfg.policy, ds_meta=...) without it, so the feature check
# fails ("Missing camera1/2/3, Extra front/wrist"). With the rename map make_policy skips that check,
# as lerobot-train does; a camera the model has but the robot lacks (camera3) is left out by SmolVLA
# itself, the same as in training (empty_cameras=0).
_original_make_policy = lerobot_record.make_policy


def _rename_map_arg() -> dict:
    import json

    for a in sys.argv[1:]:
        if a.startswith("--dataset.rename_map="):
            return json.loads(a.split("=", 1)[1])
    return {}


class _RemotePolicy:
    """Control PC (v2): the policy runs on the GPU laptop (omx_infer.py); lerobot-record only needs these
    few attributes, and omx_async's engine sends the observations there."""
    name = "remote"

    def __init__(self, cfg):
        self.config = cfg
        cfg.device = "cpu"

    def reset(self):
        pass


class _Identity:
    """Pre/post-processing happens on the GPU laptop together with the policy."""

    def __call__(self, x):
        return x

    def reset(self):
        pass


def _make_remote_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    if ASYNC is None or ASYNC.remote is None:
        raise RuntimeError("[omx] 원격 추론에는 비동기 엔진(OMX_ASYNC=1)이 필요합니다")
    t = time.perf_counter()
    print("[omx] remote: GPU 노트북에서 모델을 불러옵니다 (처음엔 SmolVLA 10~30초)", flush=True)
    info = ASYNC.remote.load()
    print(f"[omx] remote: 모델 준비 완료 ({info.get('type')}, GPU 불러오기 {info.get('load_ms', 0):.0f}ms, "
          f"전체 {(time.perf_counter() - t) * 1000:.0f}ms)", flush=True)
    return _RemotePolicy(cfg)


def _make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    rename_map = rename_map or _rename_map_arg() or None
    if rename_map:
        print(f"[omx] policy: camera names {rename_map}", flush=True)
    return _original_make_policy(cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)


lerobot_record.make_policy = _make_policy
if os.environ.get("OMX_REMOTE_URL"):
    lerobot_record.make_policy = _make_remote_policy
    lerobot_record.make_pre_post_processors = lambda *a, **k: (_Identity(), _Identity())

# Stopping during the reset period makes lerobot-record call save_episode() on an empty
# buffer, which raises and marks the run as failed although the finished episodes are fine.
_original_save_episode = LeRobotDataset.save_episode


ABORTED_MIN_S = 1.0  # an attempt cut by 즉시 중단 / 비상 정지 is kept from this long on (shorter: dropped)


def _save_episode_if_frames(self, episode_data=None, *args, **kwargs):
    buffer = episode_data if episode_data is not None else self.episode_buffer
    size = buffer.get("size", 0) if buffer is not None else 0
    index = self.meta.total_episodes
    if ABORT["on"] and ABORT["during"] != "reset":
        # the attempt in progress was cut: keep what was recorded (marked "aborted" in the panel's
        # notes), unless it is too short to tell anything
        secs = size / max(self.fps, 1)
        if secs < ABORTED_MIN_S:
            print(f"[omx] abort: the episode in progress is not saved ({secs:.1f}s, under {ABORTED_MIN_S:g}s)", flush=True)
            return
        print(f"[omx] abort: saving the episode in progress as aborted: episode {index} at {secs:.1f}s "
              f"mode {ABORT['mode']}", flush=True)
    elif ABORT["on"]:
        # 즉시 중단 in the reset period: the attempt before it had finished (LeRobot saves an episode
        # only after its reset period), so keep it; the arm has already been landed
        print("[omx] abort: saving the attempt that finished before the reset period", flush=True)
    if buffer is not None and size == 0:
        print("[omx] skipped an empty episode (stopped before any frame was recorded)", flush=True)
        return
    _backup_episode(self, buffer, index)
    return _original_save_episode(self, episode_data, *args, **kwargs)


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
        print(f"[omx] backup: episode {index} not written ({exc})", flush=True)


LeRobotDataset.save_episode = _save_episode_if_frames

# 즉시 중단 from the panel (SIGUSR1): stop the policy at the next loop step, skip saving the episode
# in progress (an attempt that already finished, i.e. abort in the reset period, is saved), and land the follower right away (safe stop: home or rest pose, fade out, torque off)
# instead of after the episode is encoded and the dataset finalized. Finished episodes stay saved.
import signal  # noqa: E402

ABORT = {"on": False, "events": None, "robot": None, "in_loop": False,
         "phase": None,    # "record" (an episode with the dataset) or "reset" (the reset period between them)
         "during": None,   # the phase when 즉시 중단 / 비상 정지 came
         "mode": None}     # "abort" (즉시 중단: land now) or "freeze" (비상 정지: hold in place, land on [홈 위치로 이동])


def _land_now() -> None:
    robot = ABORT["robot"]
    if robot is not None and robot.is_connected:
        print("[omx] abort: returning the follower now", flush=True)
        try:
            robot.disconnect()  # patched by omx_safe_start: soft landing first
        except Exception as exc:
            print(f"[omx] abort: landing failed ({exc})", flush=True)


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
    print("[omx] freeze: 비상 정지 · 제자리에서 멈춤 (토크 유지). 기록을 저장하고 [홈 위치로 이동]를 기다립니다", flush=True)
    events = ABORT["events"]
    if events is not None:
        events["stop_recording"] = True
        events["exit_early"] = True


signal.signal(signal.SIGUSR2, _on_freeze)


def _on_abort(signum, frame) -> None:
    if ABORT["on"]:
        if ABORT["mode"] == "freeze":  # 즉시 중단 after 비상 정지 = go back now
            omx_safe_start.RELEASE.set()
        return
    ABORT["on"], ABORT["mode"] = True, "abort"
    ABORT["during"] = ABORT["phase"]
    print("[omx] abort: 즉시 중단 (policy stopped)" + (" · 원위치 시간: 끝난 시도는 저장" if ABORT["during"] == "reset" else ""), flush=True)
    events = ABORT["events"]
    if events is not None:
        events["stop_recording"] = True
        events["exit_early"] = True
    if not ABORT["in_loop"]:
        # saving or between loops: the bus is idle, so land from a thread right away
        threading.Thread(target=_land_now, daemon=True).start()
    if ABORT["during"] == "record":
        print("[omx] abort: 진행 중인 시도는 1초 이상이면 '중단됨'으로 저장합니다", flush=True)


signal.signal(signal.SIGUSR1, _on_abort)

_original_init_keyboard_listener = lerobot_record.init_keyboard_listener


def _listen_stdin(events: dict) -> None:
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "next":
            events["exit_early"] = True
        elif cmd == "rerecord":
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif cmd == "stop":
            events["stop_recording"] = True
            events["exit_early"] = True
        elif cmd == "home":  # [홈 위치로 이동] after 비상 정지
            omx_safe_start.RELEASE.set()
        else:
            continue
        print(f"[omx] control: {cmd}", flush=True)


def _init_listeners():
    # Keep LeRobot's arrow-key listener when a display is available (it returns None when headless),
    # and also accept the same commands as lines on stdin (web panel, or typed + Enter in a terminal).
    listener, events = _original_init_keyboard_listener()
    threading.Thread(target=_listen_stdin, args=(events,), daemon=True).start()
    return listener, events


lerobot_record.init_keyboard_listener = _init_listeners

# Policy evaluation has no teleop, so the reset period runs record_loop with neither policy nor
# teleop. In LeRobot v0.4.4 that branch `continue`s before the timer is updated, so the reset
# never ends and the robot stays frozen in its last pose. Wait out the reset time here instead.
_original_record_loop = lerobot_record.record_loop


def _record_loop(*args, **kwargs):
    ABORT["events"], ABORT["robot"] = kwargs.get("events"), kwargs.get("robot")
    if ABORT["on"]:  # aborted before the first loop (model loading): end the run (즉시 중단: land now)
        kwargs["events"]["stop_recording"] = kwargs["events"]["exit_early"] = True
        if ABORT["mode"] == "abort":
            _land_now()
        return None
    ABORT["in_loop"] = True
    ABORT["phase"] = "record" if kwargs.get("dataset") is not None else "reset"
    try:
        return _record_loop_inner(*args, **kwargs)
    finally:
        ABORT["in_loop"] = False
        if ABORT["on"] and ABORT["mode"] == "abort":
            _land_now()


def _record_loop_inner(*args, **kwargs):
    if kwargs.get("policy") is None and kwargs.get("teleop") is None and kwargs.get("dataset") is None:
        events = kwargs["events"]
        end = time.perf_counter() + kwargs["control_time_s"]
        robot = kwargs.get("robot")
        # keep the panel's live view (and its object detection) current while the object is put back:
        # without this the picture stays frozen at the end of the previous attempt
        feeding = threading.Event()
        if robot is not None:
            threading.Thread(target=_feed_live_view, args=(robot, feeding), daemon=True).start()
        if os.environ.get("OMX_RESET_HOME") == "1" and robot is not None:
            print(f"[omx] reset period: {kwargs['control_time_s']}s (the follower returns to its starting pose, put the object back)", flush=True)
            try:
                omx_safe_start.go_home_all(robot, "reset", events)  # both arms of a cooperative robot together
            except Exception as exc:  # never break the run over the reset move
                print(f"[omx] safe reset: 건너뜀 ({exc})", flush=True)
            finally:
                omx_safe_start.set_pwm_all(robot, 1.0)  # full output again for the next episode
        else:
            print(f"[omx] reset period: {kwargs['control_time_s']}s (robot holds its pose, put the object back)", flush=True)
        while time.perf_counter() < end:
            if events["exit_early"]:
                events["exit_early"] = False
                break
            time.sleep(0.05)
        feeding.set()
        return None
    _finish_safe_start(kwargs)
    if ASYNC is not None and kwargs.get("policy") is not None:
        ASYNC.new_episode()  # fresh queue: the first action of an episode comes from its own observation
    if FILTER is not None and kwargs.get("policy") is not None:
        ds = kwargs.get("dataset")
        FILTER.reset(((ds.features.get("action") or {}).get("names")) if ds is not None else None)
    if kwargs.get("policy") is None or kwargs.get("dataset") is None:
        return _original_record_loop(*args, **kwargs)
    LOOP_TIMES.clear()
    try:
        return _original_record_loop(*args, **kwargs)
    finally:
        _report_loop_rate(kwargs["dataset"])


# Control rate of each policy episode (the panel keeps it with the attempt): one timestamp per
# robot observation, i.e. per loop iteration.
LOOP_TIMES: list = []


def _report_loop_rate(dataset) -> None:
    t = list(LOOP_TIMES)
    if len(t) < 10:
        return
    hz = sorted(1.0 / max(b - a, 1e-6) for a, b in zip(t, t[1:]))
    pick = lambda q: hz[min(len(hz) - 1, int(q * len(hz)))]  # noqa: E731
    print(f"[omx] loop: episode {dataset.num_episodes} median {pick(0.5):.1f} p5 {pick(0.05):.1f} "
          f"min {hz[0]:.1f} Hz over {len(hz)} steps", flush=True)


def _finish_safe_start(kwargs) -> None:
    """With a leader arm, run the safe start before the first recorded episode.

    Otherwise its first seconds would hold the slow approach, while the recorded actions are the
    leader's targets. The warm-up runs the normal teleop loop without a dataset until the follower
    has caught up with the leader.
    """
    robot, teleop = kwargs.get("robot"), kwargs.get("teleop")
    states = [getattr(a, "_omx_ss", None) for a in omx_safe_start.arms_of(robot)] if robot is not None else []
    if kwargs.get("dataset") is None or teleop is None or not states or None in states \
            or all(st["phase"] == "normal" for st in states):
        return
    events = kwargs["events"]
    print("[omx] safe start before recording: the follower catches up with the leader first", flush=True)

    def watch() -> None:  # every arm (both of a cooperative robot) has caught up with its leader
        while any(st["phase"] != "normal" for st in states):
            time.sleep(0.05)
        events["exit_early"] = True

    threading.Thread(target=watch, daemon=True).start()
    warm = dict(kwargs, dataset=None, control_time_s=omx_safe_start.settings()["timeout_s"] + 30)
    _original_record_loop(**warm)
    events["exit_early"] = False


lerobot_record.record_loop = _record_loop


LIVE_LATEST: dict = {}  # camera -> latest frame for the panel's live view


# Live view for the web panel: while this process owns the cameras the panel cannot open them,
# so share the frames the robot already reads. A writer thread JPEG-encodes the latest frame of
# each camera about 10 times a second into /dev/shm (atomic rename); the robot loop only stores
# a reference, so control timing is not affected.
def _install_live_view() -> None:
    import os

    try:
        import cv2
        from lerobot.robots.omx_follower import omx_follower
    except Exception:
        return
    import json

    import omx_bimanual
    import omx_camstats

    omx_camstats.install()  # real camera fps for the panel
    latest = LIVE_LATEST
    prefix = f"/dev/shm/omx_live_{os.getuid()}_"
    robot_cams: dict = {}

    def hook(cls):
        original = cls.get_observation

        def get_observation(self, *args, **kwargs):
            if getattr(self, "_omx_side", None) is None:  # an arm of a cooperative robot is counted by the pair
                LOOP_TIMES.append(time.perf_counter())
            obs = original(self, *args, **kwargs)
            cams = getattr(self, "cameras", {}) or {}
            robot_cams.update(cams)
            for cam in cams:
                if cam in obs:
                    latest[cam] = obs[cam]
            return obs

        cls.get_observation = get_observation

    hook(omx_follower.OmxFollower)
    hook(omx_bimanual.BiOmxFollower)  # a cooperative robot holds the cameras itself

    def writer() -> None:
        last_stats = 0.0
        while True:
            time.sleep(0.1)
            if time.time() - last_stats > 0.5 and robot_cams:  # fps and brightness for the panel
                last_stats = time.time()
                try:
                    tmp = f"{prefix}stats.tmp"
                    with open(tmp, "w") as f:
                        json.dump(omx_camstats.stats(robot_cams, dict(latest)), f)
                    os.replace(tmp, f"{prefix}stats.json")
                except Exception:
                    pass
            for cam, frame in list(latest.items()):
                try:
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        tmp = f"{prefix}{cam}.tmp"
                        with open(tmp, "wb") as f:
                            f.write(jpg.tobytes())
                        os.replace(tmp, f"{prefix}{cam}.jpg")
                except Exception:
                    pass

    omx_follower.OmxFollower.get_observation = get_observation
    threading.Thread(target=writer, daemon=True).start()


_install_live_view()


def _feed_live_view(robot, stop: threading.Event) -> None:
    """Camera frames only (no motor bus) about 10 times a second, until the reset period ends."""
    while not stop.is_set():
        for name, cam in list(getattr(robot, "cameras", {}).items()):
            try:
                LIVE_LATEST[name] = cam.async_read(timeout_ms=200)
            except Exception:
                pass
        stop.wait(0.1)

def _share_dataset_with_team() -> None:
    """Group-read the recorded dataset (videos are saved as 600 and skip the folder's default ACL)."""
    import os
    import subprocess
    from pathlib import Path

    repo_id = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dataset.repo_id=")), None)
    root = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dataset.root=")), None)
    if not repo_id and not root:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import omx_config

    home = os.environ.get("HF_LEROBOT_HOME") or str(omx_config.path("data_home"))
    path = Path(root) if root else Path(home) / repo_id
    group = omx_config.CFG.get("team_group")
    if path.exists() and group:
        omx_config.share_with_team(path)
        print(f"[omx] shared {path} with group {group} (read/write)", flush=True)


if __name__ == "__main__":
    try:
        lerobot_record.main()
    finally:
        _share_dataset_with_team()
