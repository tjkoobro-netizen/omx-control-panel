#!/usr/bin/env bash
# OMX Control Panel v2.1.0 - GPU 노트북 설치 (로봇과 카메라가 연결된 노트북에서, 제어판을 쓸 각 계정으로 실행)
#
#   bash install_gpu.sh            질문에 답하며 설치
#   bash install_gpu.sh --yes      감지한 값/기본값으로 자동 설치
#
# 하는 일
#   1. 환경 점검 (Ubuntu, GPU 드라이버, SSH 서버, dialout/video 그룹)
#   2. LeRobot v0.4.4 가상환경 확인 (없으면 설치 여부를 묻습니다)
#   3. 제어판 파일 복사 (~/il_ws/src/omx_tools) 와 omx-panel 명령 등록
#   4. 로봇 포트 자동 감지 (모터 ID로 팔로워/리더 구분, 로봇은 움직이지 않음)
#   5. 카메라 선택 (front, wrist)
#   6. 데이터/모델 폴더 결정 (팀 공유 폴더 또는 개인 폴더)
#   7. 설정 저장 (~/.config/omx/config.json) 후 제어판 서버 시작
# sudo가 필요한 일은 직접 하지 않고, 실행할 명령을 안내합니다.
set -uo pipefail

VERSION=2.1.0
PKG="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TARGET="$HOME/il_ws/src/omx_tools"
CONF_DIR="$HOME/.config/omx"
CONF="$CONF_DIR/config.json"
YES=0
[ "${1:-}" = "--yes" ] && YES=1

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
err()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
ask() {  # ask "질문" "기본값" -> REPLY
  local q="$1" d="${2:-}"
  if [ $YES -eq 1 ]; then REPLY="$d"; return; fi
  read -r -p "  $q${d:+ [$d]}: " REPLY
  REPLY="${REPLY:-$d}"
}
yesno() {  # yesno "질문" y|n -> 0 if yes
  local d="$2"
  if [ $YES -eq 1 ]; then [ "$d" = y ]; return; fi
  read -r -p "  $1 [$( [ "$d" = y ] && echo Y/n || echo y/N )]: " REPLY
  REPLY="${REPLY:-$d}"
  [[ "$REPLY" =~ ^[Yy] ]]
}
SUDO_TODO=()

echo "OMX Control Panel v$VERSION - GPU 노트북 설치 ($(hostname), 계정 $USER)"
prev() {  # prev key -> value from the existing config, else the package defaults (empty if none)
  python3 - "$1" "$CONF" "$PKG/omx_tools" <<'PYEOF' 2>/dev/null
import json, os, sys
key, conf, tools = sys.argv[1:]
sys.path.insert(0, tools)
try:
    import omx_config
    c = dict(omx_config.DEFAULTS)
except Exception:
    c = {}
if os.path.isfile(conf):
    c.update(json.load(open(conf)))
v = c
for k in key.split("."):
    v = v.get(k, "") if isinstance(v, dict) else ""
print(v if isinstance(v, str) else "")
PYEOF
}
[ -f "$CONF" ] && echo "  기존 설정($CONF)이 있어 그 값을 기본값으로 씁니다."

# ---------------------------------------------------------------- 1. 환경 점검
bold "1/7 환경 점검"
ok "OS: $( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"   # subshell: os-release also defines VERSION
if nvidia-smi -L >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else warn "nvidia-smi가 동작하지 않습니다. 학습은 CPU로만 됩니다 (매뉴얼 '문제 해결 - GPU 드라이버' 참고)."; fi
if systemctl is-active --quiet ssh 2>/dev/null || systemctl is-active --quiet sshd 2>/dev/null; then ok "SSH 서버 실행 중"
else warn "SSH 서버가 꺼져 있습니다. 노트북에서 원격 접속하려면 필요합니다."; SUDO_TODO+=("sudo apt install -y openssh-server && sudo systemctl enable --now ssh"); fi
for g in dialout video; do
  if id -nG | tr ' ' '\n' | grep -qx "$g"; then ok "그룹 $g"
  else warn "그룹 $g 에 속해 있지 않습니다 (로봇 포트/카메라 접근에 필요)."; SUDO_TODO+=("sudo usermod -aG $g $USER   # 실행 후 다시 로그인"); fi
done
command -v v4l2-ctl >/dev/null || { warn "v4l2-ctl이 없습니다 (카메라 확인용)."; SUDO_TODO+=("sudo apt install -y v4l-utils"); }
command -v setfacl >/dev/null || { warn "setfacl이 없습니다 (팀 공유 권한용)."; SUDO_TODO+=("sudo apt install -y acl"); }
command -v tailscale >/dev/null && tailscale status >/dev/null 2>&1 && ok "Tailscale 연결됨: $(tailscale ip -4 2>/dev/null | head -1)" \
  || warn "Tailscale이 없거나 로그인 전입니다. 원격 접속하려면 매뉴얼 1단계를 먼저 하세요."

# ---------------------------------------------------------------- 2. LeRobot 환경
bold "2/7 LeRobot 가상환경"
ask "LeRobot 가상환경 위치" "$(prev venv || true)"
[ -z "$REPLY" ] && REPLY="~/venv/il"
VENV_IN="$REPLY"; VENV="${VENV_IN/#\~/$HOME}"
check_lerobot() { "$VENV/bin/python" -c "import lerobot, dynamixel_sdk, cv2; print(lerobot.__version__)" 2>/dev/null; }
LR_VER="$(check_lerobot || true)"
if [ -n "$LR_VER" ]; then
  ok "LeRobot $LR_VER ($VENV)"
  [ "$LR_VER" = "0.4.4" ] || warn "제어판은 LeRobot 0.4.4 기준으로 만들고 시험했습니다. 다른 버전은 일부 기능이 다르게 동작할 수 있습니다."
else
  warn "$VENV 에 LeRobot이 없습니다."
  if yesno "지금 LeRobot v0.4.4를 설치할까요? (torch 포함 약 4GB, 20~40분)" n; then
    command -v python3.10 >/dev/null || { err "python3.10이 없습니다. 매뉴얼 '3-1 LeRobot 설치'의 sudo 명령을 먼저 실행하세요."; exit 1; }
    mkdir -p "$(dirname "$VENV")" ~/il_ws/src
    python3.10 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip
    [ -d ~/il_ws/src/lerobot ] || git clone -q --branch v0.4.4 https://github.com/huggingface/lerobot.git ~/il_ws/src/lerobot
    (cd ~/il_ws/src/lerobot && "$VENV/bin/pip" install -e ".[dynamixel,smolvla]") || { err "LeRobot 설치 실패"; exit 1; }
    LR_VER="$(check_lerobot || true)"; [ -n "$LR_VER" ] && ok "LeRobot $LR_VER 설치 완료" || { err "설치 확인 실패"; exit 1; }
  else
    err "LeRobot 없이는 제어판을 쓸 수 없습니다. 설치 후 다시 실행하세요."; exit 1
  fi
fi
PY="$VENV/bin/python"

# SmolVLA (optional): extra packages (transformers etc., torch unchanged) and the pretrained weights.
# Training runs offline, so lerobot/smolvla_base and its SmolVLM2 backbone are downloaded here once.
if "$PY" -c "import transformers, num2words" 2>/dev/null; then
  ok "SmolVLA 패키지 있음 (transformers $("$PY" -c 'import transformers; print(transformers.__version__)'))"
elif yesno "SmolVLA 학습·평가용 패키지를 설치할까요? (transformers 등, torch는 그대로)" n; then
  (cd ~/il_ws/src/lerobot && "$VENV/bin/pip" install -q -e ".[smolvla]") && ok "SmolVLA 패키지 설치" || warn "SmolVLA 패키지 설치 실패 (ACT·Diffusion은 그대로 쓸 수 있음)"
fi
if "$PY" -c "import transformers" 2>/dev/null && ! "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/smolvla_base', local_files_only=True)
snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', local_files_only=True, ignore_patterns=['onnx/*'])" 2>/dev/null; then
  if yesno "SmolVLA 사전학습 가중치(약 3GB)를 지금 받을까요? (SmolVLA 학습 전에 한 번 필요)" n; then
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/smolvla_base', ignore_patterns=['*.gif', '*.ipynb'])
snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', ignore_patterns=['onnx/*'])" && ok "SmolVLA 가중치 받음" || warn "다운로드 실패 (네트워크 확인 후 이 스크립트를 다시 실행)"
  fi
fi

# ---------------------------------------------------------------- 3. 파일 복사
bold "3/7 제어판 파일 설치 → $TARGET"
mkdir -p "$TARGET" ~/.local/bin ~/il_ws/omx_logs "$CONF_DIR"
if [ "$(readlink -f "$PKG/omx_tools")" != "$(readlink -f "$TARGET")" ]; then
  cp -r "$PKG/omx_tools/." "$TARGET/"
fi
chmod +x "$TARGET"/*.sh "$TARGET"/*.py 2>/dev/null
ln -sf "$TARGET/omx-panel-server.sh" ~/.local/bin/omx-panel
ok "파일 $(ls "$TARGET" | wc -l)개, 명령 omx-panel 등록 (~/.local/bin, 다음 로그인부터 PATH에 포함)"

# ---------------------------------------------------------------- 4. 로봇 포트
bold "4/7 로봇 포트 감지 (모터에 신호만 보내고 움직이지 않습니다)"
DETECT=$("$PY" - <<'PYEOF'
import glob, json, os
from dynamixel_sdk import PacketHandler, PortHandler
cands = []
for p in ["/dev/omx_follower", "/dev/omx_leader"] + sorted(glob.glob("/dev/serial/by-id/*")) + sorted(glob.glob("/dev/ttyACM*")):
    if os.path.exists(p) and os.path.realpath(p) not in [os.path.realpath(c) for c in cands]:
        cands.append(p)
found = {}
for p in cands:
    h, k = PortHandler(p), PacketHandler(2.0)
    try:
        if not h.openPort():
            continue
        h.setBaudRate(1_000_000)
        ids = [i for i in range(0, 20) if k.ping(h, i)[1] == 0]
    except Exception:
        continue
    finally:
        try: h.closePort()
        except Exception: pass
    role = "follower" if set(range(11, 17)) <= set(ids) else "leader" if set(range(1, 7)) <= set(ids) else None
    if role and role not in found:
        found[role] = {"port": p, "ids": ids}
print(json.dumps(found))
PYEOF
)
FOLLOWER=$(echo "$DETECT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('follower',{}).get('port',''))" 2>/dev/null)
LEADER=$(echo "$DETECT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('leader',{}).get('port',''))" 2>/dev/null)
[ -n "$FOLLOWER" ] && ok "팔로워: $FOLLOWER (모터 ID 11~16)" || warn "팔로워를 찾지 못했습니다 (USB·12V 전원 확인, 다른 프로그램이 쓰는 중이면 끄고 다시)."
[ -n "$LEADER" ] && ok "리더:   $LEADER (모터 ID 1~6)" || warn "리더를 찾지 못했습니다."
P_F="$(prev follower_port)"; P_L="$(prev leader_port)"
# keep a previously configured port name (e.g. a udev symlink) when it points to the detected device
[ -n "$P_F" ] && [ -n "$FOLLOWER" ] && [ "$(readlink -f "$P_F")" = "$(readlink -f "$FOLLOWER")" ] && FOLLOWER="$P_F"
[ -n "$P_L" ] && [ -n "$LEADER" ] && [ "$(readlink -f "$P_L")" = "$(readlink -f "$LEADER")" ] && LEADER="$P_L"
ask "팔로워 포트" "${FOLLOWER:-${P_F:-/dev/omx_follower}}"; FOLLOWER="$REPLY"
ask "리더 포트" "${LEADER:-${P_L:-/dev/omx_leader}}"; LEADER="$REPLY"

# ---------------------------------------------------------------- 5. 카메라
bold "5/7 카메라 선택"
mapfile -t CAMS < <(ls /dev/v4l/by-id/*-video-index0 2>/dev/null)
INTERNAL_RE='Integrated|True_Vision|Chicony|LG_Camera|FaceTime|Webcam|IR_Camera'
if [ ${#CAMS[@]} -eq 0 ]; then
  warn "연결된 카메라가 없습니다. 카메라 없이 설정하고, 나중에 ~/.config/omx/config.json 의 cameras를 고치세요."
else
  i=1; for c in "${CAMS[@]}"; do
    tag=""; [[ "$(basename "$c")" =~ $INTERNAL_RE ]] && tag="  (노트북 내장 카메라로 보임)"
    echo "    $i) $(basename "$c" | sed 's/^usb-//; s/-video-index0$//')$tag"; i=$((i+1)); done
fi
EXT=(); for c in "${CAMS[@]}"; do [[ "$(basename "$c")" =~ $INTERNAL_RE ]] || EXT+=("$c"); done
idx_of() { local t="$1" j=1; for c in "${CAMS[@]}"; do [ "$c" = "$t" ] && { echo $j; return; }; j=$((j+1)); done; echo ""; }
DEF_FRONT=$( [ ${#EXT[@]} -ge 1 ] && idx_of "${EXT[0]}" ); DEF_WRIST=$( [ ${#EXT[@]} -ge 2 ] && idx_of "${EXT[1]}" )
# cameras chosen last time win, when they are still plugged in
P_FRONT="$(prev cameras.front)"; P_WRIST="$(prev cameras.wrist)"
[ -n "$P_FRONT" ] && [ -e "$P_FRONT" ] && DEF_FRONT=$(idx_of "$P_FRONT")
[ -n "$P_WRIST" ] && [ -e "$P_WRIST" ] && DEF_WRIST=$(idx_of "$P_WRIST")
[ ${#EXT[@]} -ge 2 ] && [ -z "$P_FRONT" ] && echo "  카메라 앞뒤는 자동으로 알 수 없습니다. 목록을 보고 번호를 꼭 확인하세요."
pick() { local n="$1"; [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#CAMS[@]} ] && echo "${CAMS[$((n-1))]}"; }
if [ ${#CAMS[@]} -gt 0 ]; then
  ask "front(작업대 전체가 보이는 고정 카메라) 번호, 없으면 0" "${DEF_FRONT:-0}"; FRONT="$(pick "$REPLY")"
  ask "wrist(팔로워 그리퍼에 달린 카메라) 번호, 없으면 0" "${DEF_WRIST:-0}"; WRIST="$(pick "$REPLY")"
fi
# A camera saved last time but unplugged right now is kept (with a warning) instead of being dropped
saved() { [ -f "$CONF" ] && python3 -c "import json,sys; print(json.load(open('$CONF')).get('cameras',{}).get(sys.argv[1],''))" "$1" 2>/dev/null; }
S_FRONT="$(saved front)"; S_WRIST="$(saved wrist)"
if [ -z "${FRONT:-}" ] && [ -n "$S_FRONT" ] && [ ! -e "$S_FRONT" ]; then FRONT="$S_FRONT"; warn "front 카메라가 지금 연결되어 있지 않지만 이전 설정을 유지합니다: $(basename "$S_FRONT")"; fi
if [ -z "${WRIST:-}" ] && [ -n "$S_WRIST" ] && [ ! -e "$S_WRIST" ]; then WRIST="$S_WRIST"; warn "wrist 카메라가 지금 연결되어 있지 않지만 이전 설정을 유지합니다: $(basename "$S_WRIST")"; fi
[ -n "${FRONT:-}" ] && ok "front: $(basename "$FRONT")"; [ -n "${WRIST:-}" ] && ok "wrist: $(basename "$WRIST")"
[ -z "${FRONT:-}${WRIST:-}" ] && warn "카메라 없이 설정합니다 (녹화·평가에는 카메라가 필요합니다)."

# ---------------------------------------------------------------- 6. 폴더
bold "6/7 데이터·모델 폴더"
P_DATA="$(prev data_home)"
if [ -n "$P_DATA" ] && [ "$P_DATA" != "/srv/il/lerobot_data" ] && [ -d "$P_DATA" ]; then
  ok "기존 설정의 폴더를 그대로 씁니다"
  DATA="$P_DATA"; CKPT="$(prev checkpoints)"; SHARED="$(prev shared_root)"; GROUP="$(prev team_group)"
elif [ -d /srv/il ] && [ -w /srv/il ]; then
  GROUP=$(stat -c %G /srv/il)
  ok "팀 공유 폴더 /srv/il 사용 (그룹 $GROUP)"
  SHARED=/srv/il; DATA=/srv/il/lerobot_data; CKPT=/srv/il/checkpoints
else
  echo "    1) 개인 폴더 ~/omx_data  (혼자 쓸 때. sudo 불필요)"
  echo "    2) 팀 공유 폴더 /srv/il  (여러 계정이 같은 데이터·모델을 쓸 때. 관리자가 한 번 만들어야 함)"
  ask "선택" "1"
  if [ "$REPLY" = "2" ]; then
    GROUP=ilteam; SHARED=/srv/il; DATA=/srv/il/lerobot_data; CKPT=/srv/il/checkpoints
    SUDO_TODO+=("sudo groupadd -f ilteam && sudo usermod -aG ilteam $USER   # 팀원 계정마다
sudo mkdir -p /srv/il/lerobot_data /srv/il/checkpoints && sudo chgrp -R ilteam /srv/il && sudo chmod -R 2775 /srv/il && sudo setfacl -R -d -m g:ilteam:rwX /srv/il
# 위 명령 후 다시 로그인하고 이 설치를 한 번 더 실행하세요")
    warn "/srv/il 이 아직 없어 개인 폴더로 임시 설정합니다. 관리자가 만든 뒤 다시 실행하면 공유 폴더로 바뀝니다."
    SHARED="$HOME/omx_data"; DATA="$HOME/omx_data/lerobot_data"; CKPT="$HOME/omx_data/checkpoints"; GROUP=""
  else
    GROUP=""; SHARED="$HOME/omx_data"; DATA="$HOME/omx_data/lerobot_data"; CKPT="$HOME/omx_data/checkpoints"
  fi
fi
mkdir -p "$DATA" "$CKPT" 2>/dev/null
ok "데이터: $DATA"; ok "모델:   $CKPT"
# Existing personal calibration (lecture default ~/.cache/huggingface/lerobot) is reused in the new home
if [ ! -d "$DATA/calibration" ] && [ -d ~/.cache/huggingface/lerobot/calibration ]; then
  cp -r ~/.cache/huggingface/lerobot/calibration "$DATA/" && ok "기존 캘리브레이션을 $DATA/calibration 으로 복사"
fi
if ! grep -q "HF_LEROBOT_HOME=" ~/.bashrc 2>/dev/null; then
  if yesno "터미널에서 LeRobot을 직접 쓸 때도 이 데이터 폴더를 쓰도록 ~/.bashrc 에 추가할까요?" y; then
    printf '\n# OMX panel: LeRobot home (datasets, calibration)\nexport HF_LEROBOT_HOME=%s\n' "$DATA" >> ~/.bashrc && ok "~/.bashrc 에 HF_LEROBOT_HOME 추가"
  fi
fi

# ---------------------------------------------------------------- 7. 설정 저장, 서버 시작
bold "7/7 설정 저장, 서버 시작"
[ -f "$CONF" ] && cp "$CONF" "$CONF.bak.$(date +%Y%m%d%H%M%S)" && ok "기존 설정 백업"
FRONT="${FRONT:-}" WRIST="${WRIST:-}" FOLLOWER="$FOLLOWER" LEADER="$LEADER" DATA="$DATA" CKPT="$CKPT" SHARED="$SHARED" \
GROUP="${GROUP:-}" VENV_IN="$VENV_IN" "$PY" - "$CONF" <<'PYEOF'
import json, os, socket, sys
cams = {}
if os.environ["FRONT"]: cams["front"] = os.environ["FRONT"]
if os.environ["WRIST"]: cams["wrist"] = os.environ["WRIST"]
cfg = {
    "gpu_name": socket.gethostname(),
    "data_home": os.environ["DATA"], "checkpoints": os.environ["CKPT"], "shared_root": os.environ["SHARED"],
    "team_group": os.environ["GROUP"],
    "follower_port": os.environ["FOLLOWER"], "leader_port": os.environ["LEADER"],
    "cameras": cams, "fixed_fps_camera": "wrist" if "wrist" in cams else "",
    "venv": os.environ["VENV_IN"],
}
# keep the settings this installer does not manage (home_pose, freeze_return_after_s, safe_start, ...)
try:
    old = json.load(open(sys.argv[1]))
    cfg = {**{k: v for k, v in old.items() if k not in cfg}, **cfg}
except (OSError, ValueError):
    pass
open(sys.argv[1], "w").write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PYEOF
ok "설정: $CONF"
# Recording, teleop and evaluation run as children of the panel server (their output goes through it),
# so a restart would cut them off. Training runs on its own (omx_train.py) and is not affected.
if pgrep -u "$USER" -f "omx_record.py|omx_teleop.py|omx_return_home.py|lerobot-(record|teleoperate)" >/dev/null; then
  warn "녹화·텔레옵·평가가 실행 중이라 서버를 재시작하지 않았습니다. 끝난 뒤 'omx-panel --restart'를 실행하세요."
else
  "$TARGET/omx-panel-server.sh" --restart | sed 's/^/  /'
fi
PORT=$("$PY" -c "import sys; sys.path.insert(0, '$TARGET'); import omx_config; print(omx_config.port())")

bold "설치 완료 (v$VERSION)"
echo "  이 계정의 제어판 포트: $PORT   (설정 확인: $PY $TARGET/omx_config.py)"
echo "  이제 내 노트북에서 install_laptop.sh 를 실행하세요 (매뉴얼 4단계)."
if [ ${#SUDO_TODO[@]} -gt 0 ]; then
  bold "관리자(sudo) 권한으로 해야 할 일"
  for t in "${SUDO_TODO[@]}"; do echo "  $t"; done
fi
