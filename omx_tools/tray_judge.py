"""출고(하차) 평가 중 '쟁반에 놓음' 판정 (Guide, 11/24 TK 요청 · Admin 패널 연결)

handover_judge.py(입고 놓음)와 같은 방식: 패널 평가 루프 옆 스레드가 약 6 Hz 로 update() 를 부르고, 'placed' 가 처음 True 가 되면
패널이 평가를 마친다. numpy·OpenCV 만 (MuJoCo 없음). update() 는 예외를 던지지 않는다(오류는 'error' 키).

판정 = 아래가 모두 PLACE_HOLD_S 동안 이어짐
  1. 쟁반(front): 명령 상자 윗면 마커가 쟁반 사각형 안에서 MARK_STILL_S 동안 MARK_STILL_PX 안에 멈춤.
     마커를 못 읽으면 대신 '직전 빈 쟁반' 대비 바뀐 비율 tray_changed ≥ TRAY_CHANGED_ON (pinky_read 와 같은 계산·기준 0.25)
  2. 칸(front_2): 명령 칸이 빔 (pinky_read slot_empty 와 같은 계산: 빈 선반 기준 사진 shelf_ref 대비 바뀐 비율 < 0.06, 칸 안 상자 마커 없음)
  3. 양손 끝(FK)이 쟁반 구역에서 TIP_MARGIN_M(3cm) 넘게 떨어짐 (tips_near_tray)
  4. 두 그리퍼 모두 쥔 상태 아님 (명령 − 실제 > −GRIP_HOLD_G)
  + 손이 한 번은 쟁반 구역에 들어왔어야 함 / 시작 때 이미 쟁반에 무언가 있으면(tray_busy_at_start) 판정하지 않음

'직전 빈 쟁반': 이번 평가 첫 위 카메라 프레임(팔은 홈). 작업 직전 기준(tray_check --tray-ref JSON)과 비교해 비었음을 확인하고
(바뀐 비율 < START_EMPTY_MAX) 쟁반 사각형 위치도 그 기준에서 옮겨 온다. 기준 파일이 없거나 못 쓰면 쟁반 사각형은 tray_empty_ref 기본값.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

VERSION = "1126a"   # 1126a (패널 3.30.1): 마커 멈춤 판정이 프레임 간격과 상관없게(아래 MARK_MIN_N·MARK_GAP_S) / 1124c: 한팔 2A(front_3)·2C(front_4) 측면 카메라 쟁반 + 팔별 손끝 구역, 2B 는 1124b 와 같음 / 1124b: 1124a 초안 + hold_s 인자·ok_now, 첫 프레임 쟁반 테두리 찾기 대체 경로. 11/24 두 팔 출고 72회 재생: 잘못 뜬 판정 0
HERE = pathlib.Path(__file__).resolve().parent
ALIGN_DIR = HERE.parent if (HERE.parent / "shelf_ref.json").exists() else pathlib.Path("~/il_ws/src/omx_sim_align").expanduser()

# ---- 쟁반 구역 (세계 m, handover_fk 좌표; 쟁반 중심 sim params_shelf23 pinky.pos, 크기 0.155 × 0.125)
# 손끝 = 손가락 몸체 원점 평균(FK.tip). 양손이 상자 양옆을 잡으므로 x 는 쟁반보다 넓게: 1112·1117·1121 출고 놓는 순간
# 왼손 x −0.118~−0.073, 오른손 −0.005~0.076(한 번 0.199), y −0.089~0.016, z 0.157~0.257. 홈 자세 손끝 |x| 0.19~0.215, z 0.18.
TRAY_CX = -0.007
TRAY_ZONE_HX = 0.115          # |x − TRAY_CX| < 이 값
TRAY_ZONE_Y = (-0.10, 0.08)
TRAY_ZONE_ZMAX = 0.28
TIP_MARGIN_M = 0.03
# 11/24 TK: 한팔 출고(2A 왼팔 · 2C 오른팔) — Pinky 는 로봇 바깥 양옆, 쟁반은 측면 카메라로 본다.
# 1118 한팔 출고 36회 놓는 순간 손끝: 왼손 x −0.437~−0.181(중앙 −0.413), 오른손 0.308~0.465(중앙 0.445), y −0.008~0.044, z 0.16~0.32.
# 홈 자세 손끝 |x| 0.196~0.216 → 여유 3cm 를 더해도 구역 밖
TRAY_ZONES = {   # 이름: (x 중심, x 반폭, (y 범위), z 상한)
    "2B": (TRAY_CX, TRAY_ZONE_HX, TRAY_ZONE_Y, TRAY_ZONE_ZMAX),
    "left": (-0.43, 0.085, (-0.10, 0.10), 0.30),
    "right": (0.44, 0.085, (-0.10, 0.10), 0.30),
}
SLOT_ARM = {"2A": "left", "2B": None, "2C": "right"}
SLOT_TRAY_CAM = {"2A": "front_3", "2B": "front", "2C": "front_4"}   # 11/24 1118 녹화: front_3 = 왼팔 쟁반, front_4 = 오른팔 쟁반(놓는 시각 일치)
# 측면 카메라 빈 쟁반 테두리 기본 자리(1118 녹화 첫 화면 테두리 맞춤 0.85~0.90). 평가 첫 프레임에서 다시 맞춘다
SIDE_TRAY_QUAD = {"front_3": [[347, 85], [260, 91], [268, 188], [355, 182]],
                  "front_4": [[161, 184], [80, 184], [80, 280], [161, 280]]}

GRIP_HOLD_G = 4.0             # 명령 − 실제 < −이 값 = 쥐고 있음 (DropWatch 와 같음)

# ---- 위 카메라 쟁반 (unload_check.tray_changed 와 같은 값)
TRAY_DIFF = 0.15
TRAY_OPEN_PX = 5
TRAY_INNER = 0.82
TRAY_RIM = (0.90, 1.0)
TRAY_CHANGED_ON = 0.25
START_EMPTY_MAX = 0.15        # 첫 프레임이 작업 직전 기준과 이만큼 넘게 다르면 '시작 때 쟁반 위에 무언가'
MARK_IN_SCALE = 1.08          # 상자 마커 중심이 쟁반 사각형(이 배율) 안
MARK_STILL_S = 0.5
MARK_STILL_PX = 3.0
# 1126a: 멈춤 = 지금 위치에서 MARK_STILL_PX 안인 검출이 MARK_MIN_N 번 이상, 빈틈 MARK_GAP_S 이하로 이어져 처음~지금이
# MARK_STILL_S − 0.1 이상. 1124c 는 '최근 0.5초 안 기록의 폭 ≥ 0.4초'라 0.4~0.5초 전 기록이 꼭 있어야 했고, 판독 간격이
# 0.17~0.19초(다시 확인 루프 약 0.173초)나 0.3초 이상이면 상자가 쟁반 위에 가만히 있어도 멈춤이 되지 않았다(11/26 2A 오판)
MARK_MIN_N = 3
MARK_GAP_S = 0.5             # 판독 간격 0.45초(부하가 클 때)까지 이어진 것으로 봄
MARK_KEEP_S = 1.0             # 보관하는 마커 기록 길이
BOX_TOP_IDS = tuple(range(15, 23))

# ---- 전방 카메라 칸 (unload_check.slot_empty 와 같은 값)
SLOT_DIFF = 0.12
SLOT_OPEN_PX = 5
SLOT_CHANGED = 0.06
BOX_MARKER_IDS = (15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26)
SLOT_SHIFT_LIM = 14           # 첫 front_2 프레임 ↔ 빈 선반 기준 이동 탐색(px)

PLACE_HOLD_S = 0.5            # 네 조건이 이만큼 이어지면 placed
STALE_S = 1.0                 # 카메라 판독이 이보다 오래되면 그 조건은 모름(False)


def tips_near_tray(joints, side, fk, margin=TIP_MARGIN_M, zone=None):
    """그 팔 손끝(FK.tip)이 쟁반 구역(± margin) 안인가. 홈 복귀 중 재진입 감시에도 씀. joints: 12관절 dict(handover_fk 이름).
    zone: None·'2B' = 두 팔 가운데 쟁반, 'left'·'right' = 한팔 작업 그 팔 바깥 쟁반 (Judge.zone)"""
    cx, hx, (y0, y1), zmax = TRAY_ZONES[zone or "2B"]
    x, y, z = (float(v) for v in fk.tip(joints, side))
    return bool(abs(x - cx) < hx + margin and y0 - margin < y < y1 + margin and z < zmax + margin)


# ---------------------------------------------------------------- 작은 도구 (unload_check 에서 옮김)
def _read_json(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _quad_scale(q, s):
    q = np.asarray(q, float)
    return (q - q.mean(0)) * s + q.mean(0)


def _quad_mask(shape, q):
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(q)).astype(np.int32)], 255)
    return m > 0


def _edges(gray, sel=None):
    import cv2
    e = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    if sel is not None:
        e = e * sel
    return e


def _reg_tray(ref_g, quad_ref, cur_g, lim=40, rlim=10.0):
    """지금 → 기준 2D 이동·회전 (unload_check._reg_tray 와 같음) → (M 2x3 | None, 정보)"""
    import cv2
    q = np.asarray(quad_ref, float)
    c = tuple(map(float, q.mean(0)))
    ring = _quad_mask(ref_g.shape, _quad_scale(q, 1.15)) & ~_quad_mask(ref_g.shape, _quad_scale(q, 0.85))
    dt_ = cv2.distanceTransform((255 - _edges(ref_g, ring)).astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.nonzero(_edges(cur_g, _quad_mask(cur_g.shape, _quad_scale(q, 1.6))))
    if len(u) < 30:
        return None, {"score": 0.0}
    if len(u) > 2500:
        k = np.linspace(0, len(u) - 1, 2500).astype(int)
        u, v = u[k], v[k]
    P = np.stack([u, v], 1).astype(np.float32)
    H, W = ref_g.shape

    def mat(dx, dy, r):
        M = cv2.getRotationMatrix2D(c, float(r), 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return M

    def sc(dx, dy, r, sig):
        M = mat(dx, dy, r)
        p = P @ M[:, :2].T + M[:, 2]
        pi = np.clip(p.round().astype(int), 0, [W - 1, H - 1])
        d = dt_[pi[:, 1], pi[:, 0]]
        return float(np.mean(np.exp(-np.minimum(d * d / (2 * sig * sig), 50.0))))
    best = (-1.0, 0.0, 0.0, 0.0)
    for r in np.arange(-rlim, rlim + 0.01, 2.0):
        for dx in range(-lim, lim + 1, 4):
            for dy in range(-lim, lim + 1, 4):
                s = sc(dx, dy, r, 4.0)
                if s > best[0]:
                    best = (s, float(dx), float(dy), float(r))
    best = (sc(*best[1:], 1.5), *best[1:])
    for st in (2.0, 1.0, 0.5, 0.25):
        _, bx, by, br = best
        for r in (br - st, br, br + st):
            for dx in np.arange(bx - 2 * st, bx + 2 * st + 1e-6, st):
                for dy in np.arange(by - 2 * st, by + 2 * st + 1e-6, st):
                    s = sc(dx, dy, r, 1.5)
                    if s > best[0]:
                        best = (s, float(dx), float(dy), float(r))
    s, dx, dy, r = best
    at_lim = abs(dx) >= lim - 1 or abs(dy) >= lim - 1 or abs(r) >= rlim - 0.5
    return mat(dx, dy, r), {"dx": round(dx, 1), "dy": round(dy, 1), "rot_deg": round(r, 2), "score": round(s, 3), "at_limit": bool(at_lim)}


TRAY_FIT_MIN = 0.6


def _fit_tray(gray, quad_model):
    """쟁반 바깥 테두리 사각형 찾기 (unload_check.fit_tray 와 같음, 탐색 격자만 조금 성기게) → (꼭짓점, 점수, [dx, dy, 회전°, 배율])"""
    import cv2
    q0 = np.asarray(quad_model, float)
    c0 = q0.mean(0)
    e = _edges(gray, _quad_mask(gray.shape, _quad_scale(q0, 1.7)))
    dt_ = cv2.distanceTransform((255 - e).astype(np.uint8), cv2.DIST_L2, 3)
    H, W = gray.shape
    P0 = np.concatenate([q0[i] + (q0[(i + 1) % 4] - q0[i]) * np.linspace(0, 1, 60, endpoint=False)[:, None] for i in range(4)]) - c0

    def rot(deg):
        t = math.radians(deg)
        return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])

    def sc(x, sig):
        q = (P0 * x[3]) @ rot(x[2]).T + c0 + x[:2]
        qi = np.clip(q.round().astype(int), 0, [W - 1, H - 1])
        d = dt_[qi[:, 1], qi[:, 0]]
        return float(np.mean(np.exp(-np.minimum(d * d / (2 * sig * sig), 50.0))))
    best, x = -1.0, None
    for s in np.arange(0.86, 1.061, 0.04):
        for r in np.arange(-10, 10.01, 2.0):
            for dx in range(-48, 49, 4):
                for dy in range(-32, 33, 4):
                    v = sc((dx, dy, r, s), 4.0)
                    if v > best:
                        best, x = v, np.array([dx, dy, r, s], float)
    b = sc(x, 1.5)
    for st in (2.0, 1.0, 0.5, 0.25):
        imp = True
        while imp:
            imp = False
            for i, d in enumerate((st, st, st / 2, st / 50)):
                for sg in (1, -1):
                    y = x.copy()
                    y[i] += sg * d
                    v = sc(y, 1.5)
                    if v > b + 1e-6:
                        x, b, imp = y, v, True
    q = ((q0 - c0) * x[3]) @ rot(x[2]).T + c0 + x[:2]
    return q, round(b, 3), np.round(x, 2).tolist()


def _changed(cur, ref, inner, rim=None, diff=TRAY_DIFF, open_px=TRAY_OPEN_PX):
    """기준 대비 밝기 비율이 diff 넘게 바뀐 화소 비율(inner 안). 밝기는 rim(없으면 inner) 중앙값으로 맞춤"""
    import cv2
    a = cv2.GaussianBlur(cur, (5, 5), 0).astype(np.float32) + 4
    b = cv2.GaussianBlur(ref, (5, 5), 0).astype(np.float32) + 4
    r = a / b
    k = float(np.median(r[rim if rim is not None else inner]))
    ch = ((np.abs(r / k - 1) > diff) & inner).astype(np.uint8)
    ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    return float(ch[inner].mean()), k


class Judge:
    def __init__(self, fk_json=None, box_id=None, slot="2B", tray_ref_file=None, shelf_ref_file=None, hold_s=PLACE_HOLD_S,
                 arm="auto", tray_cam="auto"):
        """slot '2B'(두 팔, 쟁반 = 위 카메라 front) · '2A'(왼팔, 쟁반 = front_3) · '2C'(오른팔, 쟁반 = front_4).
        update() 의 첫 인자는 tray_cam 카메라 회색 프레임이다(2B 는 front). 한팔 쟁반은 작업 직전 기준(tray_ref_file)이 없어 첫 프레임과 비교"""
        import cv2
        if slot not in SLOT_ARM:
            raise ValueError(f"slot {slot} 미지원 (2A·2B·2C)")
        self.arm = SLOT_ARM[slot] if arm == "auto" else arm
        self.sides = ("left", "right") if self.arm is None else (self.arm,)
        self.zone = "2B" if self.arm is None else self.arm
        self.tray_cam = SLOT_TRAY_CAM[slot] if tray_cam == "auto" else tray_cam
        self.hold_s = float(hold_s)   # 네 조건이 이만큼 이어지면 placed (TK 0.5 / 1.0 검토 중)

        from handover_judge import FK
        self.error0 = None
        tab = _read_json(fk_json or HERE / "handover_fk.json")
        self.fk = FK(tab) if tab else None
        self.names = tab["names"] if tab else []
        self.box_id = None if box_id is None else int(box_id)
        self.slot = slot
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        p = cv2.aruco.DetectorParameters()            # real_runner tray_det: 과노출 흰 윗면 마커용 완화 검출기 (번호·오류정정은 같음)
        p.adaptiveThreshConstant = 3.0
        p.perspectiveRemoveIgnoredMarginPerCell = 0.3
        self.det2 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)
        # 쟁반 기준 (작업 직전)
        self.tray_ref_file = tray_ref_file
        self.mref = None   # (회색 사진(기준 화면), quad_ref)
        ent = _read_json(tray_ref_file) if tray_ref_file and self.tray_cam == "front" else None   # tray_check 기준은 위 카메라 사진
        if ent and ent.get("kind") == "tray_pre_mission" and ent.get("usable") and ent.get("image"):
            g = cv2.imread(str(pathlib.Path(tray_ref_file).parent / ent["image"]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                self.mref = (g, np.asarray(ent["quad_ref"], float))
        self.mref_state = "ok" if self.mref is not None else ("unusable" if ent else "missing")
        dflt = (_read_json(ALIGN_DIR / "tray_empty_ref.json") or {}).get("default") or {}
        self.quad_default = np.asarray(dflt["quad_ref"], float) if dflt.get("quad_ref") else None
        if self.tray_cam in SIDE_TRAY_QUAD:
            self.quad_default = np.asarray(SIDE_TRAY_QUAD[self.tray_cam], float)
        # 빈 선반 기준 (front_2)
        sref = _read_json(shelf_ref_file or ALIGN_DIR / "shelf_ref.json")
        self.sref = None
        if sref and slot in (sref.get("slots") or {}):
            g = cv2.imread(str(pathlib.Path(shelf_ref_file or ALIGN_DIR / "shelf_ref.json").parent / sref["image"]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                self.sref = (g, np.asarray(sref["slots"][slot], np.int32), sref.get("saved_at"))
        self.reset(time.time())

    # ------------------------------------------------------------ 평가 시작
    def reset(self, t0):
        self.t0 = float(t0)
        self.tray = None          # dict(ref, quad, inner, rim, Minv(지금→기준 화면) ...) 첫 위 프레임에서
        self.tray_info = {}
        self.slot_reg = None      # dict(shift, poly, mask, ref)
        self.slot_info = {}
        self.hands_in = False
        self.blocked = None
        self.mark_hist = []       # (t, uv) 쟁반 안 마커
        self.last = {"tray": None, "slot": None}   # (t, 값 dict)
        self.since = None
        self.hit = None

    # ------------------------------------------------------------ 위 카메라
    def _tray_init(self, g):
        import cv2
        info = {"tray_reference": None}
        if self.mref is not None:
            rg, q = self.mref
            M, reg = _reg_tray(rg, q, g)
            info["tray_align"] = reg
            if M is not None and not reg["at_limit"]:
                gw = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                inner = _quad_mask(g.shape, _quad_scale(q, TRAY_INNER))
                rim = _quad_mask(g.shape, _quad_scale(q, TRAY_RIM[1])) & ~_quad_mask(g.shape, _quad_scale(q, TRAY_RIM[0]))
                c0, _ = _changed(gw, rg, inner, rim)
                info["start_vs_mission_ref"] = round(c0, 3)
                Mi = cv2.invertAffineTransform(M)
                quad = q @ Mi[:, :2].T + Mi[:, 2]            # 기준 화면 쟁반 → 지금 화면
                info["tray_reference"] = "first_frame(checked_vs_pre_mission)"
                if c0 >= START_EMPTY_MAX:
                    self.blocked = "tray_busy_at_start"
            else:
                quad = self.quad_default
                info["tray_reference"] = "first_frame(pre_mission_not_aligned)"
        else:
            quad = self.quad_default
            info["tray_reference"] = f"first_frame(pre_mission_{self.mref_state})"
            if quad is not None:   # 작업 직전 기준이 없으면 첫 프레임에서 쟁반 테두리를 다시 찾는다 (Pinky 서는 자리가 ±25px 다름)
                q2, score, x = _fit_tray(g, quad)
                info.update(tray_fit_score=score, tray_fit=x)
                if score >= TRAY_FIT_MIN:
                    quad = q2
                else:
                    info["tray_fit"] = "low_score_default_quad"
        if quad is None:
            return {"error": "no_tray_quad", **info}
        quad = np.asarray(quad, float)
        self.tray = {"ref": g.copy(), "quad": quad, "inner": _quad_mask(g.shape, _quad_scale(quad, TRAY_INNER)),
                     "rim": _quad_mask(g.shape, _quad_scale(quad, TRAY_RIM[1])) & ~_quad_mask(g.shape, _quad_scale(quad, TRAY_RIM[0])),
                     "mark_box": _quad_scale(quad, MARK_IN_SCALE).astype(np.float32)}
        x0, y0 = np.floor(_quad_scale(quad, 1.5).min(0)).astype(int)
        x1, y1 = np.ceil(_quad_scale(quad, 1.5).max(0)).astype(int)
        self.tray["crop"] = (max(x0, 0), max(y0, 0), min(x1, g.shape[1]), min(y1, g.shape[0]))
        info["quad_px"] = np.round(quad, 1).tolist()
        return info

    def _box_marker(self, g):
        """명령 상자(없으면 아무 상자 윗면) 마커 중심이 쟁반 사각형 안 → (uv, id) | None"""
        import cv2
        x0, y0, x1, y1 = self.tray["crop"]
        sub = g[y0:y1, x0:x1]
        want = (self.box_id,) if self.box_id is not None else BOX_TOP_IDS
        for det in (self.det, self.det2):
            cs, ids, _ = det.detectMarkers(sub)
            for k, q in ([] if ids is None else zip(ids.ravel(), cs)):
                if int(k) not in want:
                    continue
                c = q.reshape(4, 2).mean(0) + (x0, y0)
                if cv2.pointPolygonTest(self.tray["mark_box"], (float(c[0]), float(c[1])), False) >= 0:
                    return [round(float(c[0]), 1), round(float(c[1]), 1)], int(k)
        return None

    def _top(self, g, t):
        if self.tray is None:
            self.tray_info = self._tray_init(g)
            if self.tray is None:
                return
        ch, k = _changed(g, self.tray["ref"], self.tray["inner"], self.tray["rim"])
        m = self._box_marker(g)
        if m is not None:
            self.mark_hist.append((t, m[0]))
        self.mark_hist = [(tt, uv) for tt, uv in self.mark_hist if t - tt <= MARK_KEEP_S + 1e-6]
        still = False
        if m is not None:
            run = []   # 지금부터 거슬러 올라가며: 지금 위치에서 MARK_STILL_PX 안, 빈틈 MARK_GAP_S 이하로 이어진 검출
            for tt, uv in reversed(self.mark_hist):
                if run and run[-1][0] - tt > MARK_GAP_S + 1e-6:
                    break
                if max(abs(uv[0] - m[0][0]), abs(uv[1] - m[0][1])) > MARK_STILL_PX:
                    break
                run.append((tt, uv))
            still = len(run) >= MARK_MIN_N and t - run[-1][0] >= MARK_STILL_S - 0.1 - 1e-6
        self.last["tray"] = (t, {"tray_changed": round(ch, 3), "tray_light_k": round(k, 2), "tray_marker": bool(still),
                                 "marker_seen": m is not None, "marker_uv": m[0] if m else None, "marker_id": m[1] if m else None})

    # ------------------------------------------------------------ 전방 카메라 칸
    def _slot_init(self, g):
        import cv2
        rg, poly, at = self.sref
        m = np.zeros(g.shape, np.uint8)
        cv2.fillPoly(m, [poly], 255)
        band = cv2.dilate(m, np.ones((41, 41), np.uint8)) & ~cv2.dilate(m, np.ones((9, 9), np.uint8))   # 칸 둘레(안은 상자가 있음)
        er = _edges(rg, band > 0)
        dt_ = cv2.distanceTransform((255 - er).astype(np.uint8), cv2.DIST_L2, 3)
        big = cv2.dilate(m, np.ones((61, 61), np.uint8)) > 0
        v, u = np.nonzero(_edges(g, big & ~(cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0)))
        best = (-1.0, 0, 0)
        if len(u) >= 30:
            H, W = g.shape
            for dx in range(-SLOT_SHIFT_LIM, SLOT_SHIFT_LIM + 1):
                for dy in range(-SLOT_SHIFT_LIM, SLOT_SHIFT_LIM + 1):
                    uu, vv = np.clip(u - dx, 0, W - 1), np.clip(v - dy, 0, H - 1)   # 지금 (u,v) = 기준 (u+dx, v+dy) 로 볼 때 기준 화면 위치
                    d = dt_[vv, uu]
                    s = float(np.mean(np.exp(-d * d / 8.0)))
                    if s > best[0]:
                        best = (s, dx, dy)
        s, dx, dy = best
        at_lim = abs(dx) >= SLOT_SHIFT_LIM or abs(dy) >= SLOT_SHIFT_LIM
        if s < 0 or at_lim:
            dx = dy = 0
        polyc = poly + np.array([dx, dy], np.int32)
        mc = np.zeros(g.shape, np.uint8)
        cv2.fillPoly(mc, [polyc], 255)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        refc = cv2.warpAffine(rg, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)   # 기준 사진을 지금 화면으로
        self.slot_reg = {"poly": polyc, "mask": mc > 0, "ref": refc}
        return {"slot_shift_px": [int(dx), int(dy)], "slot_shift_score": round(s, 3), "slot_shift_at_limit": bool(at_lim),
                "shelf_ref_at": at, "slot_poly_px": polyc.tolist()}

    def _front2(self, g, t):
        import cv2
        if self.sref is None:
            self.last["slot"] = (t, {"slot_empty": None, "slot_read": "reference_missing"})
            return
        if self.slot_reg is None:
            self.slot_info = self._slot_init(g)
        sel = self.slot_reg["mask"]
        cs, ids, _ = self.det.detectMarkers(g)
        inside = []
        for k, q in ([] if ids is None else zip(ids.ravel(), cs)):
            if int(k) in BOX_MARKER_IDS:
                c = q.reshape(4, 2).mean(0)
                if cv2.pointPolygonTest(self.slot_reg["poly"].astype(np.float32), (float(c[0]), float(c[1])), False) >= 0:
                    inside.append(int(k))
        ch, _ = _changed(g, self.slot_reg["ref"], sel, None, SLOT_DIFF, SLOT_OPEN_PX)
        empty = not inside and ch < SLOT_CHANGED
        self.last["slot"] = (t, {"slot_empty": bool(empty), "slot_changed": round(ch, 3), "markers_in_slot": inside,
                                 "slot_read": "marker_in_slot" if inside else ("empty" if empty else "changed")})

    # ------------------------------------------------------------ 한 번
    def update(self, front_gray, front2_gray, joints, grips, t):
        out = {"placed": False, "version": VERSION, "error": None}
        try:
            t = float(t)
            if self.fk is None:
                raise RuntimeError("fk_json 없음")
            jd = {n: 0.0 for n in self.names}
            for k, v in (joints or {}).items():   # 한 팔 평가는 'shoulder_pan.pos' 처럼 팔 접두어가 없다 → 이 판정기의 팔로
                k = str(k).removesuffix(".pos")
                if self.arm is not None and not k.startswith(("left_", "right_")):
                    k = f"{self.arm}_{k}"
                jd[k] = float(v)
            near0 = {s: tips_near_tray(jd, s, self.fk, 0.0, self.zone) for s in self.sides}
            if any(near0.values()) and t - self.t0 > 1.0:
                self.hands_in = True
            tips_clear = not any(tips_near_tray(jd, s, self.fk, TIP_MARGIN_M, self.zone) for s in self.sides)
            g = {s: (float(c) - float(m)) for s, (c, m) in (grips or {}).items()}
            if self.arm is not None and self.arm not in g and len(g) == 1:   # 한 팔: 이름이 달라도 하나뿐이면 그 팔
                g = {self.arm: next(iter(g.values()))}
            released = all(s in g and g[s] > -GRIP_HOLD_G for s in self.sides)
            if front_gray is not None:
                self._top(front_gray, t)
            if front2_gray is not None:
                self._front2(front2_gray, t)
            tr = self.last["tray"][1] if self.last["tray"] and t - self.last["tray"][0] <= STALE_S else None
            sl = self.last["slot"][1] if self.last["slot"] and t - self.last["slot"][0] <= STALE_S else None
            on_pinky = None
            if tr is not None:
                on_pinky = bool(tr["tray_marker"] or (not tr["marker_seen"] and tr["tray_changed"] >= TRAY_CHANGED_ON))
            slot_empty = None if sl is None else sl["slot_empty"]
            ok = bool(on_pinky and slot_empty and tips_clear and released and self.hands_in and not self.blocked)
            if ok:
                self.since = t if self.since is None else self.since
            else:
                self.since = None
            placed = ok and t - self.since >= self.hold_s - 1e-6
            out.update(placed=bool(placed), on_pinky=on_pinky, tray_marker=bool(tr and tr["tray_marker"]),
                       tray_changed=None if tr is None else tr["tray_changed"], slot_empty=slot_empty, tips_clear=bool(tips_clear),
                       grippers_released=bool(released), hands_in=self.hands_in, blocked=self.blocked,
                       tips_near=near0, grip_gap={k: round(v, 1) for k, v in g.items()},
                       marker_uv=None if tr is None else tr["marker_uv"], slot_changed=None if sl is None else sl.get("slot_changed"),
                       slot_read=None if sl is None else sl.get("slot_read"), since_s=None if self.since is None else round(t - self.since, 2), ok_now=ok)
            if placed and self.hit is None:
                self.hit = dict(out, t=t, tray_info=self.tray_info, slot_info=self.slot_info)
            out["placed"] = self.hit is not None
        except Exception as exc:  # noqa: BLE001 — 판정기는 예외를 던지지 않는다
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out
