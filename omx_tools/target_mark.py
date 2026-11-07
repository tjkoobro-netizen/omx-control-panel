"""
목표 마커 표시 (시뮬 데이터 생성과 실제 로봇 실행이 같이 쓰는 코드)

정면 카메라 영상에서 명령받은 ArUco 마커(4x4)를 찾아 상자 둘레에 자홍색 원을 그립니다.
그리퍼가 마커를 가려 못 찾는 프레임에는 마지막으로 찾은 위치에 계속 그립니다.
자홍색 (255, 0, 255) 은 RGB/BGR 순서가 같아서 어느 쪽 영상에 그려도 같은 색입니다.

사용:
    mark = TargetMarker(15)
    mark.reset()                 # 에피소드 시작마다
    img = mark(img)              # 매 프레임 (HxWx3 uint8, 표시가 그려진 사본을 돌려줌)
"""

import cv2
import numpy as np

COLOR = (255, 0, 255)
RADIUS = 48       # 640x480 정면 영상 기준 (상자 한 변 약 70px)
THICKNESS = 6


class TargetMarker:
    def __init__(self, marker_id: int, radius: int = RADIUS, thickness: int = THICKNESS):
        self.id = int(marker_id)
        self.radius, self.thickness = radius, thickness
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self.last = None

    def reset(self) -> None:
        self.last = None

    def find(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) == self.id:
                    return c[0].mean(0)
        return None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        p = self.find(img)
        if p is not None:
            self.last = p
        out = np.ascontiguousarray(img).copy()
        if self.last is not None:
            cv2.circle(out, (int(round(self.last[0])), int(round(self.last[1]))), self.radius, COLOR,
                       self.thickness, cv2.LINE_AA)
        return out
