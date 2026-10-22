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

# Policy runs: limit PyTorch's CPU thread pool (the panel passes OMX_TORCH_THREADS with OMP_NUM_THREADS).
# The default of one thread per core makes the 30 Hz loop drop to ~21 Hz; see EVAL_THREADS in omx_web.py.
if os.environ.get("OMX_TORCH_THREADS"):
    import torch  # noqa: E402

    torch.set_num_threads(int(os.environ["OMX_TORCH_THREADS"]))
    print(f"[omx] torch CPU threads: {torch.get_num_threads()}", flush=True)

# Follower starts with a slow, low-power lift -> close gripper -> approach (no full-power jump)
omx_safe_start.install()

# Stopping during the reset period makes lerobot-record call save_episode() on an empty
# buffer, which raises and marks the run as failed although the finished episodes are fine.
_original_save_episode = LeRobotDataset.save_episode


def _save_episode_if_frames(self, episode_data=None, *args, **kwargs):
    buffer = episode_data if episode_data is not None else self.episode_buffer
    if buffer is not None and buffer.get("size", 0) == 0:
        print("[omx] skipped an empty episode (stopped before any frame was recorded)", flush=True)
        return
    return _original_save_episode(self, episode_data, *args, **kwargs)


LeRobotDataset.save_episode = _save_episode_if_frames

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
    if kwargs.get("policy") is None and kwargs.get("teleop") is None and kwargs.get("dataset") is None:
        events = kwargs["events"]
        print(f"[omx] reset period: {kwargs['control_time_s']}s (robot holds its pose, put the object back)", flush=True)
        end = time.perf_counter() + kwargs["control_time_s"]
        while time.perf_counter() < end:
            if events["exit_early"]:
                events["exit_early"] = False
                break
            time.sleep(0.05)
        return None
    _finish_safe_start(kwargs)
    return _original_record_loop(*args, **kwargs)


def _finish_safe_start(kwargs) -> None:
    """With a leader arm, run the safe start before the first recorded episode.

    Otherwise its first seconds would hold the slow approach, while the recorded actions are the
    leader's targets. The warm-up runs the normal teleop loop without a dataset until the follower
    has caught up with the leader.
    """
    robot, teleop = kwargs.get("robot"), kwargs.get("teleop")
    st = getattr(robot, "_omx_ss", None)
    if kwargs.get("dataset") is None or teleop is None or not st or st["phase"] == "normal":
        return
    events = kwargs["events"]
    print("[omx] safe start before recording: the follower catches up with the leader first", flush=True)

    def watch() -> None:
        while st["phase"] != "normal":
            time.sleep(0.05)
        events["exit_early"] = True

    threading.Thread(target=watch, daemon=True).start()
    warm = dict(kwargs, dataset=None, control_time_s=omx_safe_start.settings()["timeout_s"] + 30)
    _original_record_loop(**warm)
    events["exit_early"] = False


lerobot_record.record_loop = _record_loop


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
    latest: dict = {}
    prefix = f"/dev/shm/omx_live_{os.getuid()}_"

    original = omx_follower.OmxFollower.get_observation

    def get_observation(self, *args, **kwargs):
        obs = original(self, *args, **kwargs)
        for cam in getattr(self, "cameras", {}):
            if cam in obs:
                latest[cam] = obs[cam]
        return obs

    def writer() -> None:
        while True:
            time.sleep(0.1)
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
