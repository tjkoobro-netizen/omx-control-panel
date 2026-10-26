#!/usr/bin/env python3
"""Two OMX arms as one robot (cooperative / bimanual), for lerobot-record and lerobot-teleoperate.

  --robot.type=bi_omx_follower --robot.left_port=/dev/omx_follower_2 --robot.right_port=/dev/omx_follower_3
      --robot.cameras="{front: {...}, wrist_left: {...}, wrist_right: {...}}"
  --teleop.type=bi_omx_leader --teleop.left_port=/dev/omx_leader_2 --teleop.right_port=/dev/omx_leader_3

Joints are prefixed left_ / right_ (12 in all: left_shoulder_pan.pos ... right_gripper.pos), so one dataset
and one policy cover both arms at the same instant. The cameras belong to the pair (one front camera per
control PC, one wrist camera per arm) and keep plain names (front, wrist_left, wrist_right).
Each arm is a normal OmxFollower / OmxLeader (factory calibration, shared calibration files), so the
panel's safe start, multi-turn fixes, safe stop and 비상 정지 (omx_safe_start.py patches those classes)
apply to each arm; `arms` lists them for the pair-level moves (reset return, home).
Importing this module registers the two types with LeRobot's config parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
from lerobot.teleoperators.teleoperator import Teleoperator

SIDES = ("left", "right")


def _prefix(side: str, d: dict) -> dict:
    return {f"{side}_{k}": v for k, v in d.items()}


def _split(d: dict) -> dict:
    return {s: {k.removeprefix(f"{s}_"): v for k, v in d.items() if k.startswith(f"{s}_")} for s in SIDES}


# ---------------------------------------------------------------- follower pair
@RobotConfig.register_subclass("bi_omx_follower")
@dataclass
class BiOmxFollowerConfig(RobotConfig):
    left_port: str = "/dev/omx_follower_1"
    right_port: str = "/dev/omx_follower_2"
    arm_id: str = "omx_follower_arm"  # calibration file shared by every OMX arm (factory calibration)
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class BiOmxFollower(Robot):
    config_class = BiOmxFollowerConfig
    name = "bi_omx_follower"

    def __init__(self, config: BiOmxFollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm, self.right_arm = (
            OmxFollower(OmxFollowerConfig(id=config.arm_id, calibration_dir=config.calibration_dir, port=port,
                                          disable_torque_on_disconnect=config.disable_torque_on_disconnect,
                                          max_relative_target=config.max_relative_target))
            for port in (config.left_port, config.right_port))
        self.left_arm._omx_side, self.right_arm._omx_side = "left", "right"
        self.arms = [self.left_arm, self.right_arm]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {**_prefix("left", self.left_arm._motors_ft), **_prefix("right", self.right_arm._motors_ft)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return all(a.is_connected for a in self.arms) and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        for arm in self.arms:  # each arm: safe start / standard home move (omx_safe_start)
            arm.connect(calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return all(a.is_calibrated for a in self.arms)

    def calibrate(self) -> None:
        for arm in self.arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self.arms:
            arm.configure()

    def get_observation(self) -> dict:
        obs = {**_prefix("left", self.left_arm.get_observation()), **_prefix("right", self.right_arm.get_observation())}
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest()
        return obs

    def send_action(self, action: dict) -> dict:
        parts = _split(action)
        return {**_prefix("left", self.left_arm.send_action(parts["left"])),
                **_prefix("right", self.right_arm.send_action(parts["right"]))}

    def disconnect(self) -> None:
        for arm in self.arms:  # safe stop of each arm (home, gripper, fade out, torque off)
            try:
                arm.disconnect()
            except Exception as exc:  # the other arm must still land
                print(f"[omx] safe stop: {arm._omx_side} 팔 정지 중 오류 ({exc})", flush=True)
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()


# ---------------------------------------------------------------- leader pair
@TeleoperatorConfig.register_subclass("bi_omx_leader")
@dataclass
class BiOmxLeaderConfig(TeleoperatorConfig):
    left_port: str = "/dev/omx_leader_1"
    right_port: str = "/dev/omx_leader_2"
    arm_id: str = "omx_leader_arm"
    gripper_open_pos: float = 60.0


class BiOmxLeader(Teleoperator):
    config_class = BiOmxLeaderConfig
    name = "bi_omx_leader"

    def __init__(self, config: BiOmxLeaderConfig):
        super().__init__(config)
        self.config = config
        self.left_arm, self.right_arm = (
            OmxLeader(OmxLeaderConfig(id=config.arm_id, calibration_dir=config.calibration_dir, port=port,
                                      gripper_open_pos=config.gripper_open_pos))
            for port in (config.left_port, config.right_port))
        self.arms = [self.left_arm, self.right_arm]

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**_prefix("left", self.left_arm.action_features), **_prefix("right", self.right_arm.action_features)}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return all(a.is_connected for a in self.arms)

    def connect(self, calibrate: bool = True) -> None:
        for arm in self.arms:
            arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return all(a.is_calibrated for a in self.arms)

    def calibrate(self) -> None:
        for arm in self.arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self.arms:
            arm.configure()

    def get_action(self) -> dict[str, float]:
        return {**_prefix("left", self.left_arm.get_action()), **_prefix("right", self.right_arm.get_action())}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        for arm in self.arms:
            if arm.is_connected:
                arm.disconnect()
