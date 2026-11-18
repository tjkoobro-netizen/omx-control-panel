"""패널 홈 이동 전 '수직 들기' 계산 (numpy만, MuJoCo 없이 — handover_fk.json 의 FK 표 사용). 11/18 TK·Admin 사양용 시안.

lift_path(joints, side, dz=0.06, step=0.01) → 패널 단위 관절 dict 목록(손끝을 지금 기울기 그대로 step 마다 위로) 또는 None
 - 손끝 = 손가락 몸체(link6·link7) 원점 평균 (real_runner·sim 과 같은 정의), 기울기 = 관절 2+3+4 (라디안) 합
 - 팔당 관절 1~4 만 바꾼다(손목 롤·그리퍼 그대로). 감쇠 최소제곱 IK, 야코비안은 수치 미분
 - 단계마다 손끝 오차 ≤ 1 mm·기울기 오차 ≤ 0.5° 이고 관절 한계 안이어야 함. 아니면 None (부르는 쪽이 들지 않고 바로 홈)"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from handover_judge import FK

HERE = pathlib.Path(__file__).resolve().parent
VERSION = "1118a"
ARMJ = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
# 선반(2×3, lv30) 기하 — unload_check.py 와 같은 값. 위가 막힘 = 손가락이 1층 칸 안(2층 판 아래)
SHELF_X = (-0.3637, 0.3608)
LEVEL1_FRONT_Y = 0.25          # 1층 칸 앞 모서리 (2층 B 는 연장판으로 0.215)
LEVEL2_UNDER_Z = 0.300         # 2층 판 아랫면 (윗면 0.305, 두께 약 0.5cm)
BACK_CLEAR = 0.03              # 뒤로 뺄 때 손가락이 칸 앞 모서리보다 이만큼 앞까지
MAX_BACK = 0.15
BACK_LIFT = 0.02               # 뒤로 뺀 뒤 들기 높이 (1층 칸 합성 자세 200: 동기만 176 충돌 → 뒤로+2cm 41)
STEP_LIMIT = 15.0              # 한 웨이포인트 관절 변화가 이보다 크면 비정상 → 들기 중단(부르는 쪽)


class Lifter:
    def __init__(self, fk_json=None):
        self.tab = json.loads(pathlib.Path(fk_json or HERE / "handover_fk.json").read_text())
        self.fk = FK(self.tab)
        self.lim = self.tab.get("joint_limits", {})

    def _names(self, side):
        return [f"{side}_{j}" for j in ARMJ]

    def pitch(self, joints, side):
        q = self.fk.qpos(joints)
        idx = [self.tab["names"].index(n) for n in self._names(side)[1:]]
        return float(sum(q[self.tab["qpos_adr"][k]] for k in idx))

    def _err(self, joints, side, target, pitch0):
        tip = self.fk.tip(joints, side)
        return np.r_[(tip - target) * 1000.0, (self.pitch(joints, side) - pitch0) * 180 / math.pi * 2.0]   # mm, 0.5 mm/° 가중

    def solve(self, joints, side, target, pitch0, iters=60):
        j = dict(joints); names = self._names(side)
        for _ in range(iters):
            e = self._err(j, side, target, pitch0)
            if np.abs(e[:3]).max() <= 0.5 and abs(e[3]) <= 0.5:
                break
            Jm = np.zeros((4, 4))
            for c, n in enumerate(names):
                jj = dict(j); jj[n] += 0.05
                Jm[:, c] = (self._err(jj, side, target, pitch0) - e) / 0.05
            dq = -np.linalg.solve(Jm.T @ Jm + 1e-3 * np.eye(4), Jm.T @ e)
            dq = np.clip(dq, -3.0, 3.0)
            for c, n in enumerate(names):
                lo, hi = self.lim.get(n, (-100.0, 100.0))
                j[n] = float(np.clip(j[n] + dq[c], lo, hi))
        e = self._err(j, side, target, pitch0)
        ok = np.linalg.norm(e[:3]) <= 1.0 and abs(e[3]) / 2.0 <= 0.5
        return (j if ok else None), e

    def finger_box(self, joints, side):
        """손가락 geom 모서리들 (N×3, 세계)"""
        B = self.fk.bodies(joints, side)
        from handover_judge import _quat_mat
        pts = []
        for g in self.tab["finger_geoms"][side]:
            Rb, pb = B[g["body"]]
            Rg, pg = Rb @ _quat_mat(g["quat"]), pb + Rb @ np.asarray(g["pos"], float)
            hs = np.asarray(g["size"], float)
            for a in (-1, 1):
                for b in (-1, 1):
                    for c in (-1, 1):
                        pts.append(pg + Rg @ (hs * np.array([a, b, c])))
        return np.array(pts)

    def blocked_above(self, joints, side, dz=0.06):
        """손가락이 1층 칸 안(2층 판 아래)에 있고 dz 들면 판 아랫면에 닿을 만큼 → True"""
        P = self.finger_box(joints, side)
        inside = (P[:, 0] > SHELF_X[0]) & (P[:, 0] < SHELF_X[1]) & (P[:, 1] > LEVEL1_FRONT_Y - 0.01) & (P[:, 2] < LEVEL2_UNDER_Z)
        return bool(inside.any() and (P[inside, 2].max() + dz > LEVEL2_UNDER_Z - 0.01))

    def line_path(self, joints, side, delta, step=0.01):
        """손끝을 지금 기울기 그대로 delta(m, 3) 만큼 step 간격으로 곧게. 못 풀면 None"""
        tip0 = self.fk.tip(joints, side); p0 = self.pitch(joints, side)
        cur, path = dict(joints), []
        n = max(1, int(math.ceil(float(np.linalg.norm(delta)) / step)))
        for k in range(1, n + 1):
            nxt, _ = self.solve(cur, side, tip0 + np.asarray(delta, float) * k / n, p0)
            if nxt is None:
                return None
            path.append(nxt); cur = nxt
        return path

    def plan(self, joints, side, dz=0.06):
        """(모드, 관절 dict 목록). 모드: 'lift' / 'back_lift'(위가 막혀 먼저 −y 로 뺀 뒤 들기) / 'none'(못 풂 → 바로 동기 보간)"""
        try:
            if self.blocked_above(joints, side, dz):
                P = self.finger_box(joints, side)
                back = float(min(MAX_BACK, max(0.0, P[:, 1].max() - (LEVEL1_FRONT_Y - BACK_CLEAR))))
                p1 = self.line_path(joints, side, (0.0, -back, 0.0))
                if p1 is None:
                    return "none", []
                p2 = self.line_path(p1[-1], side, (0.0, 0.0, min(dz, BACK_LIFT)))   # 칸 밖으로 뺀 뒤엔 조금만 (11/18 sim: 2cm 가 6cm 보다 덜 닿음)
                return ("back_lift", p1 + p2) if p2 is not None else ("back", p1)
            p = self.line_path(joints, side, (0.0, 0.0, dz))
            return ("lift", p) if p is not None else ("none", [])
        except Exception:   # noqa: BLE001 — 계산 실패는 들지 않음
            return "none", []

    def lift_path(self, joints, side, dz=0.06, step=0.01):
        tip0 = self.fk.tip(joints, side); p0 = self.pitch(joints, side)
        cur, path = dict(joints), []
        n = max(1, int(math.ceil(dz / step)))
        for k in range(1, n + 1):
            nxt, _ = self.solve(cur, side, tip0 + np.array([0, 0, dz * k / n]), p0)
            if nxt is None:
                return None
            path.append(nxt); cur = nxt
        return path
