#!/usr/bin/env bash
# OMX Control Panel v2.1.0 - 제어 PC 설치 (팀원 각자의 우분투 PC, 로봇팔·카메라를 연결하는 PC)
#
#   bash install_control.sh
#
# 제어 PC: 리더·팔로워·카메라를 이 PC에 연결해 텔레옵·녹화·평가를 하고,
#          학습과 AI 추론은 GPU 노트북이 맡습니다(Tailscale + SSH로 연결).
# 하는 일
#   1. 필요한 시스템 패키지 확인 (없으면 설치, sudo)
#   2. Tailscale 설치·로그인 확인
#   3. GPU 노트북 고르기, SSH 키 등록 (GPU 노트북 내 계정 비밀번호를 한 번 물음)
#   4. GPU 노트북의 내 계정에 제어판 v2 확인 (학습·추론 서버). 없거나 옛 버전이면 원격 설치
#   5. LeRobot 환경 (~/venv/il, LeRobot 0.4.4 같은 버전) 확인·설치
#   6. 로봇 장치 이름(OMX 1·2·3)과 USB 권한 (sudo)
#   7. 제어판 설치, 제어 PC 설정
#   8. 실행기·바탕화면 아이콘 "OMX Control Panel"
# 다시 실행해도 됩니다: 이미 된 단계는 확인만 하고 넘어갑니다. 설정은 백업 후 유지합니다.
set -uo pipefail

VERSION=2.1.0
LEROBOT_COMMIT=8fff0fde7c79f23a93d845d1a50e985de01f8b8a  # the team's LeRobot 0.4.4 (same as the GPU laptop)
PKG="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ALIAS=omx-gpu
APP_DIR="$HOME/.local/share/omx"
CONF_DIR="$HOME/.config/omx"
CONF="$CONF_DIR/config.json"
TARGET="$HOME/il_ws/src/omx_tools"
VENV_IN="~/venv/il"; VENV="$HOME/venv/il"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
err()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
ask()  { local q="$1" d="${2:-}"; read -r -p "  $q${d:+ [$d]}: " REPLY; REPLY="${REPLY:-$d}"; }
yesno() { read -r -p "  $1 [$( [ "$2" = y ] && echo Y/n || echo y/N )]: " REPLY; REPLY="${REPLY:-$2}"; [[ "$REPLY" =~ ^[Yy] ]]; }
YES=0; [ "${1:-}" = "--yes" ] && YES=1

echo "OMX Control Panel v$VERSION - 제어 PC 설치 ($(hostname), 계정 $USER)"
[ -f "$PKG/omx_tools/omx_web.py" ] || { err "패키지 폴더에서 실행하세요 (omx_tools 폴더가 옆에 있어야 합니다)."; exit 1; }
grep -qi ubuntu /etc/os-release 2>/dev/null || warn "우분투가 아닙니다. 우분투 22.04/24.04 기준으로 만든 설치입니다."

# ---------------------------------------------------------------- 1. 시스템 패키지
bold "1/8 시스템 패키지"
NEED=()
for c in git curl rsync v4l2-ctl; do command -v "$c" >/dev/null || NEED+=("$c"); done
command -v python3.10 >/dev/null || [ -x "$VENV/bin/python" ] || NEED+=(python3.10)
if [ ${#NEED[@]} -gt 0 ]; then
  warn "없는 프로그램: ${NEED[*]}"
  PKGS="git curl rsync v4l-utils acl"
  if [[ " ${NEED[*]} " == *" python3.10 "* ]]; then
    PKGS="$PKGS python3.10 python3.10-venv python3.10-dev"
    if ! apt-cache show python3.10 >/dev/null 2>&1; then
      echo "  우분투 24.04에는 python3.10이 기본으로 없어 deadsnakes 저장소를 추가합니다."
      sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update || { err "저장소 추가 실패"; exit 1; }
    fi
  fi
  if yesno "지금 설치할까요? (sudo apt install $PKGS)" y; then
    sudo apt install -y $PKGS || { err "설치 실패"; exit 1; }
  else
    err "필요한 프로그램 없이는 진행할 수 없습니다."; exit 1
  fi
fi
ok "git, curl, rsync, v4l-utils, python3.10"

# ---------------------------------------------------------------- 2. Tailscale
bold "2/8 Tailscale (GPU 노트북과 연결하는 사설 네트워크)"
if ! command -v tailscale >/dev/null; then
  warn "Tailscale이 설치되어 있지 않습니다."
  if yesno "지금 설치할까요? (공식 설치 스크립트, sudo 비밀번호 필요)" y; then
    curl -fsSL https://tailscale.com/install.sh | sh || { err "설치 실패. 매뉴얼 2단계를 보고 직접 설치하세요."; exit 1; }
  else
    err "Tailscale 없이는 GPU 노트북(학습·추론)에 연결할 수 없습니다."; exit 1
  fi
fi
ok "Tailscale $(tailscale version 2>/dev/null | head -1)"
if ! tailscale status >/dev/null 2>&1; then
  warn "아직 로그인하지 않았습니다. 아래에 나오는 주소를 브라우저로 열어 팀 Tailscale 계정으로 로그인하세요."
  sudo tailscale up || { err "tailscale up 실패"; exit 1; }
fi
ok "내 Tailscale 주소: $(tailscale ip -4 2>/dev/null | head -1)"

# ---------------------------------------------------------------- 3. GPU 노트북, SSH 키
bold "3/8 GPU 노트북 연결"
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$ALIAS" true 2>/dev/null && ! yesno "이미 '$ALIAS'로 GPU 노트북에 접속됩니다. 다른 GPU 노트북으로 바꿀까요?" n; then
  GPU_USER=$(ssh -o BatchMode=yes "$ALIAS" whoami); ok "ssh $ALIAS ($GPU_USER) 그대로 사용"
else
  mapfile -t PEERS < <(tailscale status 2>/dev/null | awk 'NR>1 && $1 ~ /^100\./ {st = ($5 ~ /^offline/) ? "꺼짐" : "켜짐"; print $1" "$2" "$4" "st}')
  [ ${#PEERS[@]} -eq 0 ] && { err "같은 tailnet에 다른 기기가 없습니다. GPU 노트북이 같은 Tailscale 계정에 로그인했는지 확인하세요."; exit 1; }
  i=1; DEF=""
  for p in "${PEERS[@]}"; do
    set -- $p; echo "    $i) $2  ($1, $3, $4)"
    [ -z "$DEF" ] && [ "$4" = "켜짐" ] && [[ "$2" =~ [Gg][Pp][Uu]|[Tt]eam|16 ]] && DEF=$i
    i=$((i+1))
  done
  ask "GPU 노트북 번호 (목록에 없으면 100.x.x.x 주소 직접 입력)" "${DEF:-1}"
  if [[ "$REPLY" =~ ^[0-9]+$ ]] && [ "$REPLY" -le ${#PEERS[@]} ]; then set -- ${PEERS[$((REPLY-1))]}; GPU_IP=$1; GPU_NAME=$2
  else GPU_IP="$REPLY"; GPU_NAME="$REPLY"; fi
  tailscale ping -c 1 --timeout 5s "$GPU_IP" >/dev/null 2>&1 && ok "$GPU_NAME ($GPU_IP) 응답함" \
    || warn "$GPU_NAME ($GPU_IP) 이 응답하지 않습니다. GPU 노트북이 켜져 있는지 확인하세요."
  ask "GPU 노트북에서 쓰는 내 계정 이름 (관리자에게 받은 계정)" "$USER"; GPU_USER="$REPLY"
  KEY="$HOME/.ssh/id_ed25519"
  if [ ! -f "$KEY" ]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    ssh-keygen -t ed25519 -N "" -C "$USER@$(hostname) omx" -f "$KEY" >/dev/null && ok "새 SSH 키: $KEY"
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -i "$KEY" "$GPU_USER@$GPU_IP" true 2>/dev/null; then
    echo "  GPU 노트북 $GPU_USER 계정의 비밀번호를 한 번 입력하면 키가 등록되고, 다음부터는 묻지 않습니다."
    ssh-copy-id -o StrictHostKeyChecking=accept-new -i "$KEY.pub" "$GPU_USER@$GPU_IP" || { err "키 등록 실패 (계정·비밀번호 확인)"; exit 1; }
  fi
  touch ~/.ssh/config && chmod 600 ~/.ssh/config
  python3 - "$ALIAS" "$GPU_IP" "$GPU_USER" "$KEY" <<'PYEOF'
import re, sys, os
alias, ip, user, key = sys.argv[1:]
path = os.path.expanduser("~/.ssh/config")
text = open(path).read()
block = f"Host {alias}\n    HostName {ip}\n    User {user}\n    IdentityFile {key}\n    ServerAliveInterval 30\n"
pattern = re.compile(rf"^Host {re.escape(alias)}\n(?:[ \t]+.*\n?)*", re.M)
text = pattern.sub(block, text) if pattern.search(text) else (text.rstrip("\n") + "\n\n" + block if text.strip() else block)
open(path, "w").write(text)
PYEOF
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$ALIAS" true 2>/dev/null && ok "ssh $ALIAS 접속 성공" || { err "ssh $ALIAS 접속 실패"; exit 1; }
fi

# ---------------------------------------------------------------- 4. GPU 노트북 쪽 (학습·추론 서버)
bold "4/8 GPU 노트북의 내 계정 (학습·추론 서버)"
if ssh -o BatchMode=yes "$ALIAS" 'test -f ~/il_ws/src/omx_tools/omx_infer.py && test -f ~/.config/omx/config.json'; then
  ok "GPU 노트북에 제어판 v2 이상이 설치되어 있습니다 (추론 서버 포함)"
else
  warn "GPU 노트북의 $GPU_USER 계정에 제어판 v2가 없습니다. 학습과 AI 추론에 필요합니다."
  if [ $YES = 1 ] || yesno "지금 GPU 노트북에 설치할까요? (원격 실행, 몇 가지를 물어봅니다)" y; then
    ssh "$ALIAS" 'rm -rf ~/omx_panel_install && mkdir -p ~/omx_panel_install'
    scp -q -r "$PKG/omx_tools" "$PKG/install_gpu.sh" "$ALIAS:omx_panel_install/" || { err "파일 복사 실패"; exit 1; }
    ssh -t "$ALIAS" 'bash ~/omx_panel_install/install_gpu.sh' || warn "GPU 노트북 설치가 끝까지 되지 않았습니다. 관리자에게 알려 주세요."
  else
    warn "GPU 노트북 설치 없이는 학습·평가(추론)를 쓸 수 없습니다. 텔레옵·녹화만 됩니다."
  fi
fi
GPU_PORT=$(ssh -o BatchMode=yes "$ALIAS" '~/venv/il/bin/python -c "import sys; sys.path.insert(0, \"$HOME/il_ws/src/omx_tools\"); import omx_config; print(omx_config.port())"' 2>/dev/null || echo 8765)
[[ "$GPU_PORT" =~ ^[0-9]+$ ]] || GPU_PORT=8765
ok "GPU 노트북의 내 제어판 포트: $GPU_PORT"

# ---------------------------------------------------------------- 5. LeRobot
bold "5/8 LeRobot 환경 ($VENV_IN)"
check_lerobot() { "$VENV/bin/python" -c "import lerobot, dynamixel_sdk, cv2; print(lerobot.__version__)" 2>/dev/null; }
LR_VER="$(check_lerobot || true)"
if [ -n "$LR_VER" ]; then
  ok "LeRobot $LR_VER"
  HEAD=$(cd ~/il_ws/src/lerobot 2>/dev/null && git rev-parse HEAD 2>/dev/null || true)
  [ "$HEAD" = "$LEROBOT_COMMIT" ] || warn "팀 기준 LeRobot(커밋 ${LEROBOT_COMMIT:0:8})과 다를 수 있습니다. 문제가 생기면 관리자에게 알려 주세요."
else
  warn "LeRobot이 없습니다. 지금 설치합니다 (torch 포함 약 4GB, 20~40분)."
  if [ $YES = 1 ] || yesno "설치할까요?" y; then
    mkdir -p "$(dirname "$VENV")" ~/il_ws/src
    [ -x "$VENV/bin/python" ] || { python3.10 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip; } || { err "가상환경 만들기 실패"; exit 1; }
    if [ ! -d ~/il_ws/src/lerobot ]; then
      git clone -q https://github.com/huggingface/lerobot.git ~/il_ws/src/lerobot || { err "LeRobot 내려받기 실패"; exit 1; }
    fi
    (cd ~/il_ws/src/lerobot && git fetch -q origin && git checkout -q "$LEROBOT_COMMIT") || warn "팀 기준 커밋으로 맞추지 못했습니다 (그대로 진행)"
    (cd ~/il_ws/src/lerobot && "$VENV/bin/pip" install -e ".[dynamixel]") || { err "LeRobot 설치 실패"; exit 1; }
    LR_VER="$(check_lerobot || true)"; [ -n "$LR_VER" ] && ok "LeRobot $LR_VER 설치 완료" || { err "설치 확인 실패"; exit 1; }
  else
    err "LeRobot 없이는 제어판을 쓸 수 없습니다."; exit 1
  fi
fi
PY="$VENV/bin/python"

# ---------------------------------------------------------------- 6. 로봇 장치, USB 권한
bold "6/8 로봇 장치 이름(OMX 1·2·3)과 USB 권한"
RULES=/etc/udev/rules.d/99-omx.rules
if cmp -s "$PKG/udev/99-omx.rules" "$RULES"; then
  ok "로봇 장치 규칙이 최신입니다"
else
  [ -f "$RULES" ] && { mkdir -p ~/il_ws/omx_trash; cp "$RULES" ~/il_ws/omx_trash/99-omx.rules.before.$(date +%Y%m%d%H%M%S); }
  echo "  OMX 1·2·3 로봇팔을 어느 USB 포트에 꽂아도 같은 이름(/dev/omx_follower_N)으로 잡히게 합니다 (sudo)."
  sudo cp "$PKG/udev/99-omx.rules" "$RULES" && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty \
    && ok "로봇 장치 규칙 설치 (/dev/omx_follower_1~3, /dev/omx_leader_1~3)" || warn "규칙 설치 실패 (관리자에게 알려 주세요)"
fi
if id -nG "$USER" | grep -qw dialout; then ok "USB 직렬 포트 권한(dialout) 있음"
else
  sudo usermod -aG dialout "$USER" && warn "USB 직렬 포트 권한(dialout)을 추가했습니다. 로그아웃 후 다시 로그인해야 적용됩니다." \
    || warn "dialout 권한 추가 실패"
fi

# ---------------------------------------------------------------- 7. 제어판, 설정
bold "7/8 제어판 설치 → $TARGET"
mkdir -p "$TARGET" ~/il_ws/omx_logs "$CONF_DIR" ~/omx_data/lerobot_data ~/omx_data/checkpoints
if [ "$(readlink -f "$PKG/omx_tools")" != "$(readlink -f "$TARGET")" ]; then
  rsync -a --exclude __pycache__ "$PKG/omx_tools/" "$TARGET/" || { err "파일 복사 실패"; exit 1; }
fi
chmod +x "$TARGET"/*.sh
ok "제어판 파일 v$VERSION"
[ -f "$CONF" ] && cp "$CONF" "$CONF.bak.$(date +%Y%m%d%H%M%S)" && ok "기존 설정 백업"
GPU_USER="${GPU_USER:-$(ssh -o BatchMode=yes "$ALIAS" whoami)}"
"$PY" - "$CONF" "$GPU_USER" "$GPU_PORT" "$(hostname)" "$VENV_IN" <<'PYEOF'
import json, os, sys
path, gpu_user, gpu_port, host, venv = sys.argv[1:]
home = os.path.expanduser("~")
cfg = {}
try:
    cfg = json.load(open(path))
except (OSError, ValueError):
    pass
# keep what the person set (robot sets, home pose, ...); fill the control-PC basics
cfg.update(role="control", gpu_host="omx-gpu", gpu_user=gpu_user, gpu_panel_port=int(gpu_port),
           gpu_data_home=cfg.get("gpu_data_home", "/srv/il/lerobot_data"), gpu_name=host, venv=venv,
           data_home=f"{home}/omx_data/lerobot_data", checkpoints=f"{home}/omx_data/checkpoints",
           shared_root=f"{home}/omx_data", team_group="")
cfg.setdefault("home_pose", False)
json.dump(cfg, open(path, "w"), ensure_ascii=False, indent=2)
PYEOF
ok "설정: $CONF (제어 PC, GPU 계정 $GPU_USER, 포트 $GPU_PORT)"

# ---------------------------------------------------------------- 8. 실행기
bold "8/8 실행기·바탕화면 아이콘"
mkdir -p "$APP_DIR" ~/.local/bin ~/.local/share/applications
cp "$TARGET/omx-control.sh" "$APP_DIR/" && chmod +x "$APP_DIR/omx-control.sh"
printf 'OMX_GPU_HOST=%s\n' "$ALIAS" > "$CONF_DIR/laptop.env"
ln -sf "$APP_DIR/omx-control.sh" ~/.local/bin/omx-panel
cat > ~/.local/share/applications/omx-control.desktop <<EOF
[Desktop Entry]
Type=Application
Name=OMX Control Panel
Comment=OMX 로봇 데이터 수집·학습·평가 (제어 PC)
Exec=$APP_DIR/omx-control.sh
Icon=applications-science
Terminal=false
Categories=Utility;
EOF
DESK=$(xdg-user-dir DESKTOP 2>/dev/null || echo ~/Desktop)
if [ -d "$DESK" ]; then
  cp ~/.local/share/applications/omx-control.desktop "$DESK/" && chmod +x "$DESK/omx-control.desktop"
  gio set "$DESK/omx-control.desktop" metadata::trusted true 2>/dev/null
  ok "바탕화면 아이콘 'OMX Control Panel'"
fi
ok "명령 omx-panel (새 터미널부터)"
# a running panel of an older version must not keep the old code
bash "$TARGET/omx-panel-server.sh" --restart >/dev/null 2>&1 && ok "제어판 서버 시작" || warn "제어판 서버를 켜지 못했습니다 (~/il_ws/omx_logs/web.log)"

bold "설치 완료 (v$VERSION)"
echo "  1) 바탕화면 'OMX Control Panel'을 더블클릭하세요 (처음엔 우클릭 → '실행 허용'이 필요할 수 있음)."
echo "  2) ① 탭 '로봇 구성'에서 단일/협동 로봇, 로봇 세트, 카메라를 고르고 [저장하고 점검]."
echo "  3) 모든 항목이 초록이면 텔레옵부터 시작하세요."
id -nG "$USER" | grep -qw dialout || echo "  ※ USB 권한 적용을 위해 한 번 로그아웃 후 다시 로그인하세요."
