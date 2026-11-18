"""넘겨받기 판정 (11/17 TK 설계, 구현 전 sim 검증용) — numpy + OpenCV 만. 평가 프로세스(il3, MuJoCo 없음)에서 import 해서 쓴다.

조건 (모두 만족한 첫 순간 = placed):
 1) 이번에 나른 상자: 판정 시작(reset) 뒤 ARM_AFTER_S 안에 상자가 이미 칸에 안정이면 이 회차는 넘겨받기 안 함(blocked)
 2) 상자가 칸에 있음: 위 카메라(원본 프레임, 목표 원 그리기 전)에서 상자 윗면 마커가 2B 영역(TOP_ROI)·선반 높이 크기(SIDE_PX)로
    STABLE_S 동안 STABLE_PX 안에서 안 움직임. 판정 순간에도 SEEN_S 안에 보였고 안정된 자리에서 MOVED_PX 안 (손에 들려 나가면 취소)
 3) 그리퍼가 벗어남(관절 FK): 두 손가락 앞끝(손가락 geom 모서리 최대 y) < min(선반 2B 앞 − CLEAR_M, 상자 앞면 − CLEAR_M). 상자 앞면 = 마커 중심 y − 마커 각도로 구한 상자 반폭(긴 변·짧은 변 두 방향 중 큰 쪽)
    (11/17 sim 검증: 손끝 = 손가락 몸체 원점 평균으로 보면 패드가 아직 상자에 닿은 채 54/153 회 걸림 → 손가락 앞끝으로 바꿈)
사용: J = Judge(fk_json_path); J.reset(t0); 매 프레임(10 Hz 권장) r = J.update(frame_bgr, joints_dict, t) → r["placed"]
예외를 던지지 않는다(오류는 r["error"])."""
from __future__ import annotations

import json
import math
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VERSION = "1117e"   # 1117d + 기울기 확인 / 1117e: 시작 1 s 안에 마커가 보이기만 해도 blocked, 손이 한 번은 선반 쪽으로 들어와야 함(HANDS_IN_Y)   # 1117a 손끝=몸체 원점(폐기) / 1117b 손가락 앞끝 / 1117c + 안정 뒤 움직이면 취소·판정 순간 보임 (sim 0/131 오판)
BOX_IDS = (16, 17, 20)
SIDE_PX = (38.0, 47.0)
STABLE_S, STABLE_PX, SEEN_S = 0.5, 3.0, 0.3
MOVED_PX = 6.0     # 안정된 뒤 마커가 이만큼 넘게 움직이면 안정 취소 (11/17 sim: 손에 든 채 0.5s 멈췄다가 다시 들고 나가는 경우 8/142 오판)
ARM_AFTER_S = 1.0
HANDS_IN_Y = 0.20   # 손가락 앞끝이 이보다 앞(선반 쪽)으로 한 번은 들어와야 판정 (11/17 실물 18:54: 상자가 처음부터 칸에 있고 팔이 안 움직인 회차를 1117d 가 잡음)
CLEAR_M = 0.03
BOX_L, BOX_W = 0.108, 0.068   # 눕힌 상자 윗면 (긴 변, 짧은 변). 마커가 상자에 90° 단위 어느 방향으로 붙었는지 모르므로 앞면은 두 방향 중 큰 쪽
ROI_UP_PX = 40
MARKER_M = 0.048          # 상자 윗면 큰 마커 한 변 (real_runner mark_big_m)
# 11/17 TK: 기운 상자(윗면이 위를 안 봄) 걸러내기 — 마커 네 꼭짓점을 상자 윗면 높이 평면으로 풀어 4.8cm 정사각형인지
TILT_CHECK = True
TILT_EDGE = (0.88, 1.12)  # 네 변 길이 / 4.8cm 허용 범위
TILT_RATIO = 1.08         # 가장 긴 변 / 가장 짧은 변, 두 대각선 비 상한


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _axis_rot(a, t):
    a = np.asarray(a, float) / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


class FK:
    """MuJoCo mj_kinematics 와 같은 손끝(손가락 몸체 link6·link7 원점 평균). 패널 단위 12값 → 세계 m"""

    def __init__(self, tab):
        self.t = tab
        self.grid = np.asarray(tab["panel_grid"], float)
        self.qmap = [np.asarray(v, float) for v in tab["q_of_panel"]]
        for side in ("left", "right"):
            for ch in tab["chains"][side]:
                for b in ch:
                    b["_R"] = _quat_mat(b["quat"])
                    b["_p"] = np.asarray(b["pos"], float)

    def qpos(self, joints):
        q = np.asarray(self.t["qpos0"], float).copy()
        for k, n in enumerate(self.t["names"]):
            q[self.t["qpos_adr"][k]] = float(np.interp(float(joints[n]), self.grid, self.qmap[k]))
        return q

    def bodies(self, joints, side, q=None):
        """link6·link7 몸체의 세계 (R, p)"""
        q = self.qpos(joints) if q is None else q
        out = {}
        for ch in self.t["chains"][side]:
            Rw, pw = np.eye(3), np.zeros(3)
            for b in ch:
                pw = pw + Rw @ b["_p"]
                Rw = Rw @ b["_R"]
                for j in b["joints"]:
                    if j["type"] == 3:
                        jp = np.asarray(j["pos"], float)
                        Rj = _axis_rot(j["axis"], q[j["qadr"]])
                        pw = pw + Rw @ (jp - Rj @ jp)
                        Rw = Rw @ Rj
                    elif j["type"] == 2:
                        pw = pw + Rw @ (np.asarray(j["axis"], float) * q[j["qadr"]])
            out[ch[-1]["body"]] = (Rw, pw)
        return out

    def finger_front_y(self, joints, side, q=None):
        """손가락 geom(상자 모양) 모서리 중 가장 앞(+y, 선반 쪽) 값. 손끝(몸체 원점 평균)은 손가락 패드보다 3~4cm 뒤라 이것으로 벗어남을 본다"""
        B = self.bodies(joints, side, q)
        ys = []
        for g in self.t["finger_geoms"][side]:
            Rb, pb = B[g["body"]]
            Rg, pg = Rb @ _quat_mat(g["quat"]), pb + Rb @ np.asarray(g["pos"], float)
            hs = np.asarray(g["size"], float)
            for a in (-1, 1):
                for b in (-1, 1):
                    for c in (-1, 1):
                        ys.append(float((pg + Rg @ (hs * np.array([a, b, c])))[1]))
        return max(ys)

    def tip(self, joints, side, q=None):
        q = self.qpos(joints) if q is None else q
        pts = []
        for ch in self.t["chains"][side]:
            Rw, pw = np.eye(3), np.zeros(3)
            for b in ch:
                pw = pw + Rw @ b["_p"]
                Rw = Rw @ b["_R"]
                for j in b["joints"]:
                    if j["type"] == 3:    # hinge
                        jp = np.asarray(j["pos"], float)
                        Rj = _axis_rot(j["axis"], q[j["qadr"]])
                        pw = pw + Rw @ (jp - Rj @ jp)
                        Rw = Rw @ Rj
                    elif j["type"] == 2:  # slide
                        pw = pw + Rw @ (np.asarray(j["axis"], float) * q[j["qadr"]])
            pts.append(pw)
        return np.mean(pts, 0)


class TopCam:
    def __init__(self, c):
        self.R, self.p, self.f, self.k1 = np.asarray(c["R"], float), np.asarray(c["p"], float), float(c["f"]), float(c["k1"])

    def world(self, u, v, z):
        xd, yd = (u - 320) / self.f, (v - 240) / self.f
        xu, yu = xd, yd
        for _ in range(8):
            g = 1 + self.k1 * (xu * xu + yu * yu)
            xu, yu = xd / g, yd / g
        dw = self.R @ np.array([xu, -yu, -1.0])
        return self.p + (z - self.p[2]) / dw[2] * dw


class Judge:
    def __init__(self, fk_json=None, box_ids=BOX_IDS):
        tab = json.loads(pathlib.Path(fk_json or HERE / "handover_fk.json").read_text())
        self.fk, self.cam = FK(tab), TopCam(tab["topcam"])
        self.shelf = tab["shelf"]
        self.box_ids = set(box_ids)
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        self.reset(0.0)

    def reset(self, t0=0.0):
        self.t0, self.blocked, self.placed_at = float(t0), False, None
        self.stable_since, self.prev_c, self.stable_ok, self.last_seen, self.box = None, None, False, -1e9, None
        self.anchor = None
        self.hands_in = False

    def _marker(self, gray):
        v0, v1, u0, u1 = self.shelf["top_roi"]
        r0 = max(0, v0 - ROI_UP_PX)
        cs, ids, _ = self.det.detectMarkers(gray[r0:v1, u0:u1])
        best = None
        for k, q in ([] if ids is None else zip(ids.ravel(), cs)):
            if int(k) not in self.box_ids:
                continue
            q = q.reshape(4, 2) + [u0, r0]
            side = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))
            c = q.mean(0)
            if SIDE_PX[0] <= side <= SIDE_PX[1] and c[1] >= v0:
                z = self.shelf["top"] + 0.05
                W = np.array([self.cam.world(u, v, z)[:2] for u, v in q])
                ed = [float(np.linalg.norm(W[i] - W[(i + 1) % 4])) for i in range(4)]
                dg = [float(np.linalg.norm(W[0] - W[2])), float(np.linalg.norm(W[1] - W[3]))]
                shape = {"edge_rel": [round(e / MARKER_M, 3) for e in ed], "side_ratio": round(max(ed) / max(min(ed), 1e-6), 3), "diag_ratio": round(max(dg) / max(min(dg), 1e-6), 3)}
                flat = (TILT_EDGE[0] <= min(shape["edge_rel"]) and max(shape["edge_rel"]) <= TILT_EDGE[1]
                        and shape["side_ratio"] <= TILT_RATIO and shape["diag_ratio"] <= TILT_RATIO)
                best = {"id": int(k), "uv": [round(float(c[0]), 1), round(float(c[1]), 1)], "side_px": round(side, 1), "corners": q.tolist(), "shape": shape, "flat": bool(flat)}
                if TILT_CHECK and not flat:   # 기운 상자: 마커를 안 본 것으로 (안정·판정 안 함)
                    best = dict(best, rejected="tilted")
        return best

    def update(self, frame_bgr, joints, t):
        out = {"placed": False, "t": float(t), "version": VERSION}
        try:
            if self.placed_at is not None:
                return dict(out, placed=True, placed_at=self.placed_at, box=self.box)
            gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mk = self._marker(gray)
            out["marker_seen"] = mk and {k_: v_ for k_, v_ in mk.items() if k_ != "corners"}
            if mk is not None and t - self.t0 <= ARM_AFTER_S:   # 시작부터 칸에 상자(기울기 판정과 무관) → 이번에 나른 상자 아님
                self.blocked = True
            if mk is not None and mk.get("rejected"):
                mk = None
            if mk is not None:
                c = np.array(mk["uv"])
                if self.anchor is not None and np.linalg.norm(c - self.anchor) > MOVED_PX:   # 안정된 상자가 움직임 → 처음부터
                    self.stable_ok, self.anchor, self.box, self.stable_since = False, None, None, None
                if self.prev_c is not None and np.linalg.norm(c - self.prev_c) <= STABLE_PX and self.stable_since is not None:
                    pass
                else:
                    self.stable_since = t
                self.prev_c, self.last_seen = c, t
                if t - self.stable_since >= STABLE_S - 1e-6:
                    if t - self.t0 <= ARM_AFTER_S and not self.stable_ok:
                        self.blocked = True
                    self.stable_ok = True
                    if self.anchor is None:
                        self.anchor = c.copy()
                    z = self.shelf["top"] + 0.05
                    w = self.cam.world(c[0], c[1], z)
                    P = np.array([self.cam.world(u, v, z) for u, v in mk["corners"]])
                    e = (P[1] - P[0]) + (P[2] - P[3])
                    yaw = math.atan2(e[1], e[0])
                    sa, ca = abs(math.sin(yaw)), abs(math.cos(yaw))
                    half = max(BOX_L / 2 * sa + BOX_W / 2 * ca, BOX_L / 2 * ca + BOX_W / 2 * sa)
                    self.box = {k_: v_ for k_, v_ in mk.items() if k_ != "corners"}
                    self.box.update(y=round(float(w[1]), 4), x=round(float(w[0]), 4), yaw_deg=round(math.degrees(yaw), 1), front_y=round(float(w[1]) - half, 4))
            else:
                self.stable_since, self.prev_c = None, None
            q = self.fk.qpos(joints)
            ty = [float(self.fk.finger_front_y(joints, s, q)) for s in ("left", "right")]   # 손가락 앞끝 y
            if max(ty) > HANDS_IN_Y:
                self.hands_in = True
            out.update(marker=mk and {k_: v_ for k_, v_ in mk.items() if k_ != "corners"}, tip_y=[round(v, 4) for v in ty], stable=self.stable_ok, blocked=self.blocked)
            out["hands_in"] = self.hands_in
            if self.blocked or not self.hands_in or not self.stable_ok or t - self.last_seen > SEEN_S:
                return out
            y_out = self.shelf["front_y_2b"] - CLEAR_M
            if self.box is not None:
                y_out = min(y_out, self.box["front_y"] - CLEAR_M)
            out["y_out"] = round(y_out, 4)
            if max(ty) < y_out:
                self.placed_at = float(t)
                out.update(placed=True, placed_at=self.placed_at, box=self.box)
            return out
        except Exception as exc:  # noqa: BLE001 — 평가 루프를 멈추지 않는다
            return dict(out, error=f"{type(exc).__name__}: {exc}"[:200])
