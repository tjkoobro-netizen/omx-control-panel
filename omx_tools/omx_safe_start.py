#!/usr/bin/env python3
"""Safe start and safe stop for the OMX follower: no full-power jump, no sudden drop.

LeRobot sends the follower straight to the leader pose (or the first policy action) at full speed
and torque. If the poses differ - e.g. the follower lies on the table with the gripper open - the
arm drags along the table and can break the wrist link. install() patches OmxFollower so that the
first targets go through a short, low-power sequence:

  1. lift     arm joints (not the gripper) move slowly to the saved start pose (arm raised)
  2. gripper  the gripper closes slowly
  3. approach every joint moves at a limited speed toward the live target (leader or policy);
              as soon as the user starts moving the leader, full output comes back and the
              follower catches up quickly
  4. normal   once within tolerance, full speed and output come back

During 1-3 the motors' output is capped with Goal_PWM (pwm_ratio, default 30%).

Multi-turn guard: the arm motors run in extended (multi-turn) position mode, so a joint turned past
the encoder seam while powered reads one revolution off (e.g. leader shoulder_pan 4611 instead of
515 -> +125 instead of -75). Sending that makes the follower spin ~360 deg into its cables (seen as
an overload on ID 11). Leader readings outside -100..100 are wrapped back by one revolution (200);
a follower rotation joint read out of range is rebooted (no gravity load); a remaining gap larger
than ~160 deg refuses to start. The start pose is read from the leader arm
(arm held up by hand) with save_start_pose() and shared by everyone using the robot.
Safe stop (when teleop/record/eval ends normally, e.g. the panel's Stop): before LeRobot turns the
torque off, the follower moves slowly to the saved rest pose (if any) with capped output, then the
output is ramped down to zero over a few seconds so the arm settles under gravity instead of
dropping. A force-quit (SIGKILL) skips this and leaves the torque on.

Return home (policy evaluation, enabled by the panel with environment variables): the follower's
pose when the run starts (read just before the first move, after the multi-turn fixes) is kept as
"home". With OMX_STOP_POSE=home the safe stop goes back to that pose instead of the rest pose, and
with OMX_RESET_HOME=1 omx_record.py also returns home at the start of every reset period between
episodes. The arm moves first at capped output; with OMX_HOME_GRIPPER=open the gripper then opens
slowly back to its starting value (only after the arm has arrived, so a held object is not
dropped on the way). If a joint reads out of range (multi-turn misread) or is further than
home_max_gap from home, the arm is not moved.
With OMX_HOME_FILE set, the home pose is also written to that file, so omx_return_home.py can bring
the arm back after the run had to be killed (see the panel's 즉시 중단).

Progress is printed as "[omx] safe start: ..." / "[omx] safe stop: ..." lines, which the panel shows.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_config  # noqa: E402

import threading  # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# 비상 정지 (omx_record.py, SIGUSR2): FREEZE stops any slow move and makes disconnect wait, holding the
# arm (torque on), until RELEASE ([홈 위치로 이동] in the panel); then the usual return and safe stop.
FREEZE, RELEASE = threading.Event(), threading.Event()
PWM_MAX = 885  # X-series Goal_PWM full scale

DEFAULTS = {
    "enabled": True,
    "lift_speed": 12.0,      # units/s (joint range is -100..100, gripper 0..100) while lifting
    "approach_speed": 15.0,  # units/s while moving to the leader / first policy target
    "gripper_speed": 10.0,   # units/s
    "gripper_close": 47.0,   # follower gripper value treated as closed
    "pwm_ratio": 0.3,        # output cap during the start sequence (0..1)
    "catch_up_speed": 60.0,  # units/s once the user moves the leader during the approach
    "leader_move": 3.0,      # units of leader motion that count as "the user is driving now"
    "max_gap": 90.0,         # units (~160 deg): refuse to start when leader/follower differ more
    "home_max_gap": 150.0,   # units (~270 deg): do not return home from further away (normal eval motion
                             # reaches ~80 units from the start pose; multi-turn misreads are caught by
                             # the -100..100 range check)
    "tolerance": 2.0,        # units: "reached" threshold
    "timeout_s": 25.0,       # give up the slow approach after this long (then normal control)
    "stop_speed": 12.0,      # units/s while moving to the rest pose on stop
    "stop_ramp_s": 3.0,      # seconds to fade the output to zero before torque off
}


def settings() -> dict:
    s = dict(DEFAULTS)
    s.update(omx_config.CFG.get("safe_start") or {})
    return s


def pose_file(kind: str = "start") -> Path:
    """kind: 'start' (arm raised, used on start) or 'rest' (arm put down, used on stop)."""
    return omx_config.lock_dir() / f"omx_{kind}_pose.json"


def load_pose(kind: str) -> dict | None:
    try:
        return json.loads(pose_file(kind).read_text())
    except (OSError, ValueError):
        return None


def load_start_pose() -> dict | None:
    return load_pose("start")


def save_start_pose(kind: str = "start") -> dict:
    """Read the leader arm (read-only, no torque change) and store it as the follower start/rest pose."""
    from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig

    leader = OmxLeader(OmxLeaderConfig(port=omx_config.CFG["leader_port"], id="omx_leader_arm"))
    leader.bus.connect()
    try:
        pos = leader.bus.sync_read("Present_Position")
    finally:
        leader.bus.disconnect(disable_torque=False)
    data = {"joints": {m: round(float(pos[m]), 2) for m in ARM}, "saved_by": os.environ.get("USER", ""),
            "saved_at": time.strftime("%Y-%m-%d %H:%M")}
    pose_file(kind).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        os.chmod(pose_file(kind), 0o664)
    except PermissionError:
        pass
    return data


REV = 200.0  # one motor revolution in normalized units (4096 ticks over -100..100)
ROTATION_JOINTS = ("shoulder_pan", "wrist_roll")  # no gravity load: safe to reboot


def wrap(value: float) -> float:
    """Bring a multi-turn reading back into the calibrated single-turn range -100..100."""
    while value > 100.0:
        value -= REV
    while value < -100.0:
        value += REV
    return value


def _say(msg: str, what: str = "start") -> None:
    print(f"[omx] safe {what}: {msg}", flush=True)


def _step(current: float, target: float, max_delta: float) -> float:
    d = target - current
    return target if abs(d) <= max_delta else current + max_delta * (1 if d > 0 else -1)


def set_pwm(robot, ratio: float) -> None:
    value = int(PWM_MAX * ratio)
    for motor in robot.bus.motors:
        try:
            robot.bus.write("Goal_PWM", motor, value, normalize=False)
        except Exception:
            pass


def go_home(robot, what: str = "stop", events: dict | None = None) -> bool:
    """Move the follower slowly back to the pose it had when the run started (see module doc).

    Returns True when it got there. The output stays capped (pwm_ratio); the caller restores it.
    `events` (lerobot-record's dict) lets the panel's 다음으로/끝내기 cut the move short.
    what="start": the move to the standard home pose before a run (OMX_HOME_POSE); the gripper is
    then set to the home value (open or closed, as the training data started). On reset/stop the
    gripper only opens (OMX_HOME_GRIPPER=open), never closes, so a held object is not squeezed.
    """
    s = settings()
    home = getattr(robot, "_omx_home", None)
    if not home:
        _say("처음 자세를 아직 읽지 않아 제자리에서 멈춥니다", what)
        return False
    present = {m: float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
    bad = [m for m in ARM if not -100.0 <= present.get(m, 0.0) <= 100.0 or not -100.0 <= home.get(m, 0.0) <= 100.0]
    if bad:
        _say(f"{', '.join(bad)} 위치가 범위를 벗어나 {'홈' if what == 'start' else '처음'} 자세로 가지 않습니다", what)
        return False
    gaps = {m: abs(present[m] - home[m]) for m in ARM if m in home}
    worst = max(gaps, key=gaps.get) if gaps else None
    if worst and gaps[worst] > s["home_max_gap"]:
        _say(f"{worst}가 {'홈' if what == 'start' else '처음'} 자세와 너무 달라({gaps[worst] * 1.8:.0f}°) 움직이지 않습니다", what)
        return False
    set_pwm(robot, s["pwm_ratio"])
    _say(f"{'홈 자세로' if what == 'start' else '평가 시작 때 처음 자세로'} 천천히 {'이동합니다' if what == 'start' else '돌아갑니다'} "
         f"(출력 {int(s['pwm_ratio'] * 100)}%)", what)

    def move(goal: dict, speed: float) -> bool:
        cmd = {m: present[m] for m in goal}
        t = t0 = time.monotonic()
        while time.monotonic() - t0 < s["timeout_s"]:
            if events and (events.get("exit_early") or events.get("stop_recording")):
                return False
            if FREEZE.is_set():  # 비상 정지 during the move: stop where it is
                return False
            time.sleep(1 / 30)
            now = time.monotonic()
            dt, t = min(now - t, 0.2), now
            for m in goal:
                cmd[m] = _step(cmd[m], goal[m], speed * dt)
            robot.bus.sync_write("Goal_Position", dict(cmd))
            if all(abs(cmd[m] - goal[m]) <= 1e-6 for m in goal):
                time.sleep(0.5)  # let the servos settle on the last goal
                return True
        return False

    arrived = move({m: home[m] for m in ARM if m in home}, s["stop_speed"])
    if arrived and "gripper" in home and (what == "start" or os.environ.get("OMX_HOME_GRIPPER") == "open"):
        present["gripper"] = float(robot.bus.sync_read("Present_Position")["gripper"])
        closing = home["gripper"] < present["gripper"] - 1.0
        if closing and what != "start":
            _say("처음 자세에 도착. 그리퍼는 닫지 않습니다(쥔 물건 보호)", what)
        else:
            _say(f"{'홈' if what == 'start' else '처음'} 자세에 도착. 그리퍼를 {'닫힌' if closing else '처음'} 상태로 천천히 맞춥니다", what)
            move({"gripper": home["gripper"]}, s["gripper_speed"])
    elif arrived:
        _say("처음 자세에 도착했습니다", what)
    return arrived


def soft_land(robot) -> None:
    """Home (eval) or the rest pose (if saved) at capped output, then fade the output out and
    turn the torque off. Also used by omx_return_home.py on a bare bus connection."""
    s = settings()
    rest = (load_pose("rest") or {}).get("joints")
    set_pwm(robot, s["pwm_ratio"])
    if os.environ.get("OMX_STOP_POSE") == "home":
        go_home(robot, "stop")
        while FREEZE.is_set():  # 비상 정지 while returning: hold, then go on after [홈 위치로 이동]
            _say("비상 정지 · 제자리에서 기다립니다. [홈 위치로 이동]를 누르면 이어서 돌아갑니다", "freeze")
            RELEASE.wait()
            RELEASE.clear()
            FREEZE.clear()
            go_home(robot, "stop")
    elif rest:
        _say("끝 자세로 천천히 이동합니다", "stop")
        cmd = {m: float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
        t, t0 = time.monotonic(), time.monotonic()
        while time.monotonic() - t0 < s["timeout_s"]:
            time.sleep(1 / 30)
            now = time.monotonic()
            dt, t = min(now - t, 0.2), now
            for m in ARM:
                if m in rest:
                    cmd[m] = _step(cmd[m], rest[m], s["stop_speed"] * dt)
            robot.bus.sync_write("Goal_Position", {m: cmd[m] for m in ARM if m in rest})
            if all(abs(cmd[m] - rest[m]) <= 1e-6 for m in ARM if m in rest):
                time.sleep(0.5)  # let the servos settle on the last goal
                break
    _say(f"모터 출력을 {s['stop_ramp_s']:.0f}초에 걸쳐 줄입니다", "stop")
    steps = 15
    for k in range(1, steps + 1):
        set_pwm(robot, s["pwm_ratio"] * (1 - k / steps))
        time.sleep(s["stop_ramp_s"] / steps)
    robot.bus.disable_torque()
    set_pwm(robot, 1.0)  # leave the default output for the next program (RAM value; torque is off)
    _say("완료. 토크를 끕니다", "stop")


def arm_joints(joints: dict, side: str | None) -> dict:
    """The joints of one arm: a cooperative robot's pose has left_/right_ names (a model trained on both arms);
    a pose without them (one arm, or a task's home) is used for each arm as it is."""
    if side and any(k.startswith(f"{side}_") for k in joints):
        return {k.removeprefix(f"{side}_"): v for k, v in joints.items() if k.startswith(f"{side}_")}
    return {k: v for k, v in joints.items() if not k.startswith(("left_", "right_"))}


def arms_of(robot) -> list:
    """The arms of a robot: the two of a cooperative robot (omx_bimanual), else the robot itself."""
    return list(getattr(robot, "arms", None) or [robot])


def go_home_all(robot, what: str = "reset", events: dict | None = None) -> bool:
    """go_home for every arm at the same time (a cooperative robot's arms return together)."""
    import threading

    arms = arms_of(robot)
    if len(arms) == 1:
        return go_home(arms[0], what, events)
    done = {}
    threads = [threading.Thread(target=lambda a=a: done.__setitem__(id(a), go_home(a, what, events))) for a in arms]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return all(done.values())


def set_pwm_all(robot, ratio: float) -> None:
    for arm in arms_of(robot):
        set_pwm(arm, ratio)


def _go_standard_home(robot) -> bool:
    """OMX_HOME_POSE=<json>: move to the standard home pose right after connecting (see omx_home.py).

    Runs before lerobot-record starts the first episode, so the move is not in the recorded data.
    Same safeguards as the return home: range check (multi-turn misreads), home_max_gap, capped
    output, gripper only after the arm has arrived. If it cannot go, the run starts as before
    (from where the arm is) and says so.
    """
    path = os.environ.get("OMX_HOME_POSE")
    if not path:
        return False
    try:
        home = arm_joints({k: float(v) for k, v in json.loads(Path(path).read_text())["joints"].items()},
                          getattr(robot, "_omx_side", None))
    except (OSError, ValueError, KeyError) as exc:
        _say(f"홈 자세 파일을 읽지 못해 지금 자세에서 시작합니다 ({exc})")
        return False
    robot._omx_home = home
    try:
        arrived = go_home(robot, "start")
    except Exception as exc:
        _say(f"홈 자세 이동 실패({exc}). 지금 자세에서 시작합니다")
        arrived = False
    if not arrived:
        robot._omx_home = None
        _say("홈 자세에 가지 못해 지금 자세를 처음 자세로 씁니다")
        return False
    _say("홈 자세 준비 완료")
    return True


def _save_home_file(home: dict, robot=None) -> None:
    """The run's start pose for omx_return_home.py. A cooperative robot keeps one entry per arm
    ("arms": {"left": {"port", "joints"}, "right": ...}) in the same file."""
    path = os.environ.get("OMX_HOME_FILE")
    if not path:
        return
    joints = {m: round(v, 3) for m, v in home.items()}
    side = getattr(robot, "_omx_side", None) if robot is not None else None
    try:
        data = {"repo_id": os.environ.get("OMX_REPO_ID", ""), "pid": os.getpid(), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if side:
            try:
                old = json.loads(Path(path).read_text())
                if old.get("pid") == os.getpid():
                    data["arms"] = old.get("arms") or {}
            except (OSError, ValueError):
                pass
            data.setdefault("arms", {})[side] = {"port": robot.config.port, "joints": joints}
        else:
            data["joints"] = joints
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except OSError as exc:
        _say(f"처음 자세 파일을 쓰지 못했습니다 ({exc})")


def install() -> None:
    """Patch OmxFollower.connect/send_action. Safe to call once per process."""
    from lerobot.robots.omx_follower import omx_follower

    cls = omx_follower.OmxFollower
    if getattr(cls, "_omx_safe_start", False):
        return
    cls._omx_safe_start = True
    s = settings()
    if not s["enabled"]:
        return
    original_connect, original_send, original_disconnect = cls.connect, cls.send_action, cls.disconnect

    # Leader: wrap multi-turn readings (stateless, also fixes the recorded actions)
    from lerobot.teleoperators.omx_leader import omx_leader
    lcls = omx_leader.OmxLeader
    if not getattr(lcls, "_omx_wrap", False):
        lcls._omx_wrap = True
        original_get_action = lcls.get_action
        warned: set = set()

        def get_action(self):
            action = original_get_action(self)
            for key, val in action.items():
                motor = key.removesuffix(".pos")
                if motor in ARM and not -100.0 <= val <= 100.0:
                    fixed = wrap(float(val))
                    if motor not in warned:
                        warned.add(motor)
                        _say(f"리더 {motor} 값이 한 바퀴 넘게 읽혀 보정합니다 ({val:.1f} → {fixed:.1f})")
                    action[key] = fixed
            return action

        lcls.get_action = get_action

    def connect(self, *args, **kwargs):
        original_connect(self, *args, **kwargs)
        set_pwm(self, s["pwm_ratio"])
        if _go_standard_home(self):
            # at the home pose (before any recording): it is the start pose of this run; then the
            # usual slow approach to the leader / first policy target
            self._omx_ss = {"phase": "approach", "pose": None, "cmd": None, "t": None, "t0": None, "near": None,
                            "home_fixed": True}
            return
        pose = load_start_pose()
        self._omx_ss = {"phase": "lift" if pose else "gripper", "pose": (pose or {}).get("joints"),
                        "cmd": None, "t": None, "t0": None, "near": None}
        if pose:
            _say(f"출력 {int(s['pwm_ratio'] * 100)}%로 제한. 시작 자세로 팔을 천천히 듭니다")
        else:
            _say(f"출력 {int(s['pwm_ratio'] * 100)}%로 제한. 저장된 시작 자세가 없어 들어올리기는 건너뜁니다")

    def send_action(self, action):
        st = getattr(self, "_omx_ss", None)
        if not st or st["phase"] == "normal":
            return original_send(self, action)
        now = time.monotonic()
        if st["cmd"] is None:  # start from where the follower actually is
            present = self.bus.sync_read("Present_Position")
            bad = [m for m in ARM if not -100.0 <= present[m] <= 100.0]
            for m in bad:
                if m in ROTATION_JOINTS:  # reboot resets the turn count; this joint carries no weight
                    _say(f"팔로워 {m} 값이 한 바퀴 넘게 읽혀({present[m]:.1f}) 모터를 재부팅합니다")
                    mid = self.bus.motors[m].id
                    self.bus.packet_handler.reboot(self.bus.port_handler, mid)
                    time.sleep(0.6)
                    self.bus.write("Goal_PWM", m, int(PWM_MAX * s["pwm_ratio"]), normalize=False)
                    self.bus.enable_torque(m)
                else:
                    raise RuntimeError(f"[omx] safe start: 팔로워 {m} 위치가 범위를 벗어났습니다({present[m]:.1f}). "
                                       "팔을 받친 채 제어판의 '모터 초기화' 후 다시 시작하세요.")
            if bad:
                present = self.bus.sync_read("Present_Position")
            st["cmd"] = {m: float(v) for m, v in present.items()}
            if not st.get("home_fixed"):
                self._omx_home = dict(st["cmd"])  # where the run started (used by go_home)
            _save_home_file(self._omx_home, self)
            st["t"] = st["t0"] = now
            target0 = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}
            gaps = {m: abs(target0[m] - st["cmd"][m]) for m in ARM if m in target0}
            worst = max(gaps, key=gaps.get) if gaps else None
            if worst and gaps[worst] > s["max_gap"]:
                raise RuntimeError(f"[omx] safe start: 리더와 팔로워의 {worst} 차이가 너무 큽니다"
                                   f"({gaps[worst]:.0f}, 약 {gaps[worst] * 1.8:.0f}°). 리더를 팔로워와 비슷한 자세로 맞춘 뒤 다시 시작하세요.")
        dt = min(max(now - st["t"], 0.0), 0.2)
        st["t"] = now
        cmd, tol = st["cmd"], s["tolerance"]
        target = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}

        if st["phase"] == "lift":
            for m in ARM:
                if m in st["pose"]:
                    cmd[m] = _step(cmd[m], st["pose"][m], s["lift_speed"] * dt)
            if all(abs(cmd[m] - st["pose"][m]) <= 1e-6 for m in ARM if m in st["pose"]):
                st["phase"] = "gripper"
                _say("들어올림 완료. 그리퍼를 천천히 닫습니다")
        elif st["phase"] == "gripper":
            cmd["gripper"] = _step(cmd["gripper"], s["gripper_close"], s["gripper_speed"] * dt)
            if abs(cmd["gripper"] - s["gripper_close"]) <= 1e-6:
                st["phase"] = "approach"
                st["base"] = dict(target)  # leader pose when the approach starts
                _say("리더 자세로 천천히 이동합니다 (리더를 크게 움직이지 마세요)")
        elif st["phase"] == "approach":
            base = st.get("base") or target
            if not st.get("driving") and any(abs(target[m] - base.get(m, target[m])) > s["leader_move"] for m in ARM if m in target):
                st["driving"] = True  # the user took the leader: full output, quick catch-up
                set_pwm(self, 1.0)
                _say("리더를 움직이기 시작해 원래 출력으로 빠르게 따라갑니다")
            for m, goal in target.items():
                if st.get("driving"):
                    speed = s["catch_up_speed"]
                else:
                    speed = s["gripper_speed"] if m == "gripper" else s["approach_speed"]
                cmd[m] = _step(cmd[m], goal, speed * dt)
            far = max((abs(cmd[m] - g) for m, g in target.items()), default=0.0)
            if far <= tol:
                st["near"] = st["near"] or now
            else:
                st["near"] = None
            if (st["near"] and now - st["near"] > 0.3) or now - st["t0"] > s["timeout_s"]:
                st["phase"] = "normal"
                set_pwm(self, 1.0)
                _say("완료. 정상 속도와 출력으로 따라갑니다" if far <= tol else "시간 초과. 정상 제어로 전환합니다")
                return original_send(self, action)

        sent = original_send(self, {f"{m}.pos": v for m, v in cmd.items()})
        return sent

    import threading
    land_lock = threading.Lock()

    def disconnect(self, *args, **kwargs):
        # 즉시 중단 may land the arm from another thread while lerobot-record is still saving;
        # whoever comes second just finds the robot already disconnected.
        with land_lock:
            if not self.bus.is_connected:
                return None
            if FREEZE.is_set():
                # 비상 정지: hold here (torque on, last goal) until [홈 위치로 이동]
                _say("제자리에서 기다립니다. [홈 위치로 이동]를 누르면 천천히 돌아갑니다", "freeze")
                RELEASE.wait()
                RELEASE.clear()
                FREEZE.clear()
                _say("초기 위치로 돌아갑니다", "freeze")
            try:
                if getattr(self.config, "disable_torque_on_disconnect", True):
                    soft_land(self)
            except Exception as exc:  # never block the normal shutdown
                _say(f"건너뜀 ({exc})", "stop")
            return original_disconnect(self, *args, **kwargs)

    cls.connect, cls.send_action, cls.disconnect = connect, send_action, disconnect


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        kind = sys.argv[2] if len(sys.argv) > 2 else "start"
        if kind not in ("start", "rest"):
            raise SystemExit("usage: omx_safe_start.py save [start|rest]")
        print(json.dumps(save_start_pose(kind), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"settings": settings(), "start_pose": load_pose("start"), "rest_pose": load_pose("rest"),
                          "folder": str(omx_config.lock_dir())}, ensure_ascii=False, indent=2))
