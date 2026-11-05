#!/usr/bin/env python3
"""Show the robot sets by the team's names, in this account's panel. Boards and wiring do not change.

The udev rules name the boards /dev/omx_follower_N by their serial. The team calls them differently:
  team OMX 1 = /dev/omx_*_2, team OMX 2 = /dev/omx_*_3, team OMX 3 = /dev/omx_*_1
Writes "robot_sets" in the config and renames the sets in robot.json, so each arm stays the same physical arm.
Running it again changes nothing. Restart the panel afterwards: omx-panel-v3 --restart
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

CONF_DIR = Path.home() / ".config/omx/v3"
TEAM = {"OMX 1": 2, "OMX 2": 3, "OMX 3": 1}  # team name -> board number in the udev rules
RENAME = {f"OMX {board}": name for name, board in TEAM.items()}


def main() -> None:
    conf, robot = CONF_DIR / "config.json", CONF_DIR / "robot.json"
    cfg = json.loads(conf.read_text())
    if cfg.get("robot_sets"):
        print("이미 이름이 설정되어 있어 바꾸지 않습니다:", ", ".join(f"{n} = {s['follower']}" for n, s in cfg["robot_sets"].items()))
        return
    stamp = time.strftime("%Y%m%d%H%M%S")
    shutil.copy2(conf, conf.with_name(f"config.json.bak.{stamp}"))
    cfg["robot_sets"] = {n: {"follower": f"/dev/omx_follower_{b}", "leader": f"/dev/omx_leader_{b}"} for n, b in TEAM.items()}
    if robot.exists():
        setup = json.loads(robot.read_text())
        shutil.copy2(robot, robot.with_name(f"robot.json.bak.{stamp}"))
        if setup.get("single", {}).get("set") in RENAME:
            setup["single"]["set"] = RENAME[setup["single"]["set"]]
        for side in ("left", "right"):
            if setup.get("bimanual", {}).get(side) in RENAME:
                setup["bimanual"][side] = RENAME[setup["bimanual"][side]]
        if setup.get("wrist_offset_deg"):
            setup["wrist_offset_deg"] = {RENAME.get(n, n): v for n, v in setup["wrist_offset_deg"].items()}
        robot.write_text(json.dumps(setup, ensure_ascii=False, indent=2))
    conf.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    for name, board in TEAM.items():
        print(f"{name} = /dev/omx_follower_{board}, /dev/omx_leader_{board}")
    print("적용하려면 제어판을 다시 시작하세요: omx-panel-v3 --restart")


if __name__ == "__main__":
    main()
