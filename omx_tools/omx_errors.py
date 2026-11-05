#!/usr/bin/env python3
"""Why a teleop, recording or evaluation stopped abnormally: err_code_N, from the end of its output.

The number of a code never changes once released (logs and notes keep it), so a new kind gets a new number.
"""

from __future__ import annotations

import re

# (number, pattern, title, why, what to do). The first pattern that matches wins, so the narrow ones come first.
CODES = [
    (3, r"시작하지 않습니다", "시작 전 점검에서 거절", "시작하기 전 모터를 읽어 보니 값이 범위를 벗어났거나 응답이 없어, 토크를 켜지 않고 시작을 거절했습니다. 로봇은 움직이지 않았습니다.",
     "화면의 안내에 나온 팔과 모터를 확인합니다. 멈춘 보드면 USB를 다시 꽂고, 한 바퀴 넘은 모터면 모터 초기화 후 다시 시작합니다."),
    (2, r"Could not connect|could not open port|Failed to open port|Port is in use|No such file or directory: '/dev/|Input/output error|DeviceNotConnected|device disconnected", "보드 연결 끊김",
     "모터 보드의 포트를 열지 못했거나 작업 중 연결이 끊겼습니다. USB 케이블이나 허브가 빠졌거나, 다른 프로그램이 포트를 쓰고 있을 때 납니다.",
     "보드 USB와 전원을 확인하고, 시뮬레이터의 실물 연결을 끕니다. 로봇 구성 화면에서 팔이 인식되는지 본 뒤 다시 시작합니다."),
    (6, r"Overload|Overheat|hardware error|Hardware error", "모터 과부하·과열", "모터가 무리한 힘을 받거나 과열되어 스스로 멈췄습니다. 팔이 어딘가에 걸렸을 때 납니다.",
     "팔이 걸린 곳이 없는지 보고 잠시 식힌 뒤, 모터 초기화 후 다시 시작합니다."),
    (1, r"Incorrect status packet|There is no status packet|Failed to sync (read|write)|\[TxRxResult\]|\[RxPacketError\]", "모터 통신 오류",
     "모터 값을 읽거나 쓰는 통신이 여러 번 잇달아 실패했습니다. 케이블 접촉, 허브, 전원 문제나 다른 프로그램의 포트 사용이 원인일 수 있습니다.",
     "오류 내용에서 모터 번호를 봅니다(11~16 팔로워). 해당 팔의 케이블과 전원을 확인하고 다시 시작합니다. 자주 나면 케이블이나 허브 칸을 바꿉니다."),
    (4, r"차이가 너무 큽니다", "리더와 팔로워 자세 차이가 큼", "시작 때 리더와 팔로워, 또는 모델의 첫 목표와 팔로워의 차이가 안전 범위를 넘어 멈췄습니다.",
     "리더 자세를 팔로워와 비슷하게 맞춘 뒤 다시 시작합니다. 평가라면 처음 자세가 맞는지 확인합니다."),
    (5, r"위치가 범위를 벗어났습니다|범위를 벗어", "모터 값이 범위를 벗어남", "모터 값이 정상 범위 밖으로 읽혔습니다. 모터가 한 바퀴 넘게 돌았을 때 납니다.", "팔을 받친 채 모터 초기화 후 다시 시작합니다."),
    (8, r"out of memory|OutOfMemory", "메모리 부족", "GPU나 PC 메모리가 모자랐습니다. 학습 등 다른 작업과 겹칠 때 납니다.", "돌고 있는 학습과 다른 작업을 확인하고, 끝난 뒤 다시 시작합니다."),
    (7, r"(?i)camera.*(time|fail|not|error|read|open)|(time|fail|read|open).*camera|VideoCapture|/dev/video|/dev/v4l", "카메라 오류",
     "카메라를 열지 못했거나 영상이 끊겼습니다. USB 위치가 바뀌었거나, 같은 허브에 카메라가 몰려 대역폭이 모자랄 때 납니다.",
     "로봇 구성 화면에서 카메라 위치를 확인하고 미리보기로 네 대가 모두 나오는지 봅니다. 다른 프로그램이 카메라를 쓰고 있지 않은지도 확인합니다."),
    (10, r"No space left|Permission denied|Read-only file system", "저장 실패", "저장 공간이 모자라거나 저장 폴더에 쓸 권한이 없습니다.", "디스크 공간과 폴더 권한을 확인합니다."),
    (9, r"pretrained_model|safetensors|[Pp]olicy|[Cc]heckpoint|추론", "모델 오류", "모델을 불러오거나 실행하다 실패했습니다. 모델 파일이나 입력(카메라 이름, 관절 수)이 맞지 않을 때 납니다.",
     "고른 모델이 지금 로봇 구성과 같은 카메라, 팔로 학습됐는지 확인합니다."),
]
FORCED = (11, "강제 종료", "프로그램이 신호로 강제 종료됐습니다. 즉시 중단이 응답하지 않았거나, 메모리 부족으로 시스템이 끝냈을 때 납니다.", "팔 자세를 확인합니다. 토크가 남아 있을 수 있으니 팔을 받친 뒤 다시 시작합니다.")
UNKNOWN = (99, "분류되지 않은 오류", "위 목록에 없는 오류로 끝났습니다.", "오류 내용과 로그를 제어판 관리자에게 전달해 주세요.")


def _item(number: int, title: str, why: str, what: str, detail: str = "") -> dict:
    return {"code": number, "name": f"err_code_{number}", "title": title, "why": why, "what": what, "detail": detail}


def all_codes() -> list[dict]:
    items = [_item(n, t, w, h) for n, _, t, w, h in CODES] + [_item(*FORCED), _item(*UNKNOWN)]
    return sorted(items, key=lambda i: i["code"])


def classify(lines, returncode: int | None) -> dict | None:
    """err_code of an abnormal end from the job's output lines, None for a normal end."""
    if not returncode:
        return None
    lines = [str(x) for x in lines if not str(x).startswith("$ ")]  # the command line names cameras and the policy
    last = max((i for i, x in enumerate(lines) if x.startswith("Traceback")), default=None)
    # a traceback's message lines are not indented; its indented lines are file paths and source code
    tail = [x for x in lines[last:] if not x[:1].isspace() and not x.startswith("Traceback")] if last is not None else lines[-30:]
    refused = next((x for x in lines if "시작하지 않습니다" in x), None)
    detail = (refused or (tail[-1] if tail else ""))[:300]
    if refused:
        tail = [refused]
    text = "\n".join(tail)
    for number, pattern, title, why, what in CODES:
        if re.search(pattern, text):
            return _item(number, title, why, what, detail)
    return _item(*FORCED, detail) if returncode < 0 else _item(*UNKNOWN, detail)
