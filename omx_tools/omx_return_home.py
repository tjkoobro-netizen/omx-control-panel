#!/usr/bin/env python3
"""Bring the follower slowly back to the eval's start pose after the eval process had to be killed.

  python omx_return_home.py <home_pose.json> [repo_id]

The panel runs this right after a SIGKILL (즉시 중단 when the eval did not answer in time, or a
second press). The killed run leaves the torque on with its last goal, so this connects to the bus
only (no robot.connect(): that rewrites the operating modes, which needs the torque off and would
drop the arm), first sets every goal to the present position at capped output so nothing jumps,
then runs the same safe stop as a normal eval end (omx_safe_start.soft_land with OMX_STOP_POSE=home):
home at capped output, gripper (OMX_HOME_GRIPPER=open), fade out, torque off. The multi-turn range
check and home_max_gap apply: if blocked, the arm is not moved and only the output is faded.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402
import omx_safe_start as ss  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else ""
    data = json.loads(path.read_text())
    if repo and data.get("repo_id") and data["repo_id"] != repo:
        ss._say(f"처음 자세 파일이 다른 평가({data['repo_id']}) 것이라 쓰지 않습니다. 제자리에서 출력만 줄입니다", "stop")
        data = {}
    # a cooperative robot's run keeps one entry per arm (port + start pose): bring each back
    arms = data.get("arms") or {}
    if arms:
        import threading

        threads = [threading.Thread(target=return_arm, args=(a.get("port"), a.get("joints"), side)) for side, a in arms.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return 0
    return_arm(omx_config.CFG["follower_port"], data.get("joints"))
    return 0


def return_arm(port: str, home: dict | None, side: str | None = None) -> None:
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

    robot = OmxFollower(OmxFollowerConfig(port=port, id="omx_follower_arm"))
    robot.bus.connect()  # open the port and ping; torque and goals stay as the killed run left them
    try:
        s = ss.settings()
        present = robot.bus.sync_read("Present_Position")
        ss.set_pwm(robot, s["pwm_ratio"])
        robot.bus.sync_write("Goal_Position", dict(present))  # hold here first: no jump to an old goal
        robot.bus.enable_torque()  # no-op when still on; holds the arm if a motor had dropped it
        time.sleep(0.2)
        robot._omx_home = home
        os.environ["OMX_STOP_POSE"] = "home"
        if home:
            ss._say(f"{'왼팔 ' if side == 'left' else '오른팔 ' if side == 'right' else ''}평가를 강제로 끝냈습니다. "
                    "다시 연결해 처음 자세로 천천히 돌아갑니다", "stop")
        ss.soft_land(robot)
    finally:
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
