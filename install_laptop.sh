#!/usr/bin/env bash
# OMX 제어판 v1.6.1 - 개인 PC(노트북) 설치. 내가 들고 다니는 PC에서 실행합니다.
#
#   bash install_laptop.sh
#
# 하는 일
#   1. Tailscale 설치·로그인 확인 (없으면 설치 명령 실행, sudo 비밀번호를 물음)
#   2. 같은 tailnet의 GPU 노트북 고르기, 연결 시험
#   3. SSH 키 만들기·GPU 노트북에 등록 (GPU 노트북 계정 비밀번호를 한 번 물음)
#   4. SSH 별칭 omx-gpu 등록 (~/.ssh/config)
#   5. (선택) GPU 노트북 쪽 설치를 여기서 원격으로 실행
#   6. 실행기·바탕화면 아이콘·omx-panel 명령 설치
#   7. (선택) Rerun 뷰어 설치
set -uo pipefail

VERSION=1.6.1
PKG="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ALIAS=omx-gpu
APP_DIR="$HOME/.local/share/omx"
CONF_DIR="$HOME/.config/omx"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
err()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
ask()  { local q="$1" d="${2:-}"; read -r -p "  $q${d:+ [$d]}: " REPLY; REPLY="${REPLY:-$d}"; }
yesno() { read -r -p "  $1 [$( [ "$2" = y ] && echo Y/n || echo y/N )]: " REPLY; REPLY="${REPLY:-$2}"; [[ "$REPLY" =~ ^[Yy] ]]; }

echo "OMX 제어판 v$VERSION - 개인 PC 설치 ($(hostname), 계정 $USER)"

# ---------------------------------------------------------------- 1. Tailscale
bold "1/7 Tailscale"
if ! command -v tailscale >/dev/null; then
  warn "Tailscale이 설치되어 있지 않습니다."
  if yesno "지금 설치할까요? (공식 설치 스크립트, sudo 비밀번호 필요)" y; then
    curl -fsSL https://tailscale.com/install.sh | sh || { err "설치 실패. 매뉴얼 1-3을 보고 직접 설치하세요."; exit 1; }
  else
    err "Tailscale 없이는 학원 밖에서 GPU 노트북에 접속할 수 없습니다."; exit 1
  fi
fi
ok "Tailscale $(tailscale version 2>/dev/null | head -1)"
if ! tailscale status >/dev/null 2>&1; then
  warn "아직 로그인하지 않았습니다. 아래 명령이 보여주는 주소를 브라우저로 열어, GPU 노트북과 같은 tailnet 계정으로 로그인하세요."
  sudo tailscale up || { err "tailscale up 실패"; exit 1; }
fi
ok "내 Tailscale 주소: $(tailscale ip -4 2>/dev/null | head -1)"

# ---------------------------------------------------------------- 2. GPU 노트북 고르기
bold "2/7 GPU 노트북 고르기"
# "IP NAME OS STATE" for every other device in the tailnet (first line of `tailscale status` is this PC)
mapfile -t PEERS < <(tailscale status 2>/dev/null | awk 'NR>1 && $1 ~ /^100\./ {st = ($5 ~ /^offline/) ? "꺼짐" : "켜짐"; print $1" "$2" "$4" "st}')
if [ ${#PEERS[@]} -eq 0 ]; then
  err "같은 tailnet에 다른 기기가 없습니다. GPU 노트북도 Tailscale에 로그인했는지, 같은 tailnet인지 확인하세요 (매뉴얼 1-4)."
  exit 1
fi
i=1; DEF=""; DEF2=""
for p in "${PEERS[@]}"; do
  set -- $p; echo "    $i) $2  ($1, $3, $4)"
  # default: a name that looks like a GPU/team machine, then any online device
  [ -z "$DEF" ] && [ "$4" = "켜짐" ] && [[ "$2" =~ [Gg][Pp][Uu]|[Tt]eam|[Ss]erver|[Ll]ab ]] && DEF=$i
  [ -z "$DEF2" ] && [ "$4" = "켜짐" ] && DEF2=$i
  i=$((i+1))
done
DEF="${DEF:-$DEF2}"
ask "GPU 노트북 번호 (목록에 없으면 100.x.x.x 주소 직접 입력)" "${DEF:-1}"
if [[ "$REPLY" =~ ^[0-9]+$ ]] && [ "$REPLY" -le ${#PEERS[@]} ]; then set -- ${PEERS[$((REPLY-1))]}; GPU_IP=$1; GPU_NAME=$2
else GPU_IP="$REPLY"; GPU_NAME="$REPLY"; fi
if tailscale ping -c 1 --timeout 5s "$GPU_IP" >/dev/null 2>&1; then ok "$GPU_NAME ($GPU_IP) 응답함"
else warn "$GPU_NAME ($GPU_IP) 이 응답하지 않습니다. GPU 노트북이 켜져 있고 Tailscale이 연결돼 있는지 확인하세요."; fi
ask "GPU 노트북에서 쓰는 내 계정 이름" "$USER"; GPU_USER="$REPLY"

# ---------------------------------------------------------------- 3. SSH 키
bold "3/7 SSH 키"
KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$KEY" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  ssh-keygen -t ed25519 -N "" -C "$USER@$(hostname) omx" -f "$KEY" >/dev/null && ok "새 키 생성: $KEY"
else ok "기존 키 사용: $KEY"; fi
if ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -i "$KEY" "$GPU_USER@$GPU_IP" true 2>/dev/null; then
  ok "이미 비밀번호 없이 접속됩니다"
else
  echo "  GPU 노트북 $GPU_USER 계정의 비밀번호를 한 번 입력하면 키가 등록됩니다."
  ssh-copy-id -o StrictHostKeyChecking=accept-new -i "$KEY.pub" "$GPU_USER@$GPU_IP" || { err "키 등록 실패 (GPU 노트북 SSH 서버·계정·비밀번호 확인, 매뉴얼 2-2)"; exit 1; }
fi

# ---------------------------------------------------------------- 4. SSH 별칭
bold "4/7 SSH 별칭 '$ALIAS'"
touch ~/.ssh/config && chmod 600 ~/.ssh/config
python3 - "$ALIAS" "$GPU_IP" "$GPU_USER" "$KEY" <<'PYEOF'
import re, sys, os
alias, ip, user, key = sys.argv[1:]
path = os.path.expanduser("~/.ssh/config")
text = open(path).read()
block = f"Host {alias}\n    HostName {ip}\n    User {user}\n    IdentityFile {key}\n    ServerAliveInterval 30\n"
# replace an existing block for this alias (until the next Host line)
pattern = re.compile(rf"^Host {re.escape(alias)}\n(?:[ \t]+.*\n?)*", re.M)
text = pattern.sub(block, text) if pattern.search(text) else (text.rstrip("\n") + "\n\n" + block if text.strip() else block)
open(path, "w").write(text)
PYEOF
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$ALIAS" 'echo ok' >/dev/null 2>&1; then ok "ssh $ALIAS 접속 성공"
else err "ssh $ALIAS 접속 실패"; exit 1; fi

# ---------------------------------------------------------------- 5. GPU 쪽 설치 (선택)
bold "5/7 GPU 노트북 쪽 설치 확인"
if ssh -o BatchMode=yes "$ALIAS" 'test -x ~/il_ws/src/omx_tools/omx-panel-server.sh && test -f ~/.config/omx/config.json'; then
  ok "GPU 노트북에 제어판이 이미 설치되어 있습니다"
  if yesno "이 패키지(v$VERSION)로 GPU 노트북 쪽도 업데이트할까요?" n; then DO_GPU=1; else DO_GPU=0; fi
else
  warn "GPU 노트북의 $GPU_USER 계정에 제어판이 아직 없습니다."
  if yesno "지금 여기서 GPU 노트북 설치(install_gpu.sh)를 원격으로 실행할까요?" y; then DO_GPU=1; else DO_GPU=0; fi
fi
if [ "${DO_GPU:-0}" = 1 ]; then
  ssh "$ALIAS" 'rm -rf ~/omx_panel_install && mkdir -p ~/omx_panel_install'
  scp -q -r "$PKG/omx_tools" "$PKG/install_gpu.sh" "$ALIAS:omx_panel_install/" || { err "파일 복사 실패"; exit 1; }
  ssh -t "$ALIAS" 'bash ~/omx_panel_install/install_gpu.sh' || warn "GPU 노트북 설치가 끝까지 되지 않았습니다. 위 메시지를 확인하세요."
fi

# ---------------------------------------------------------------- 6. 실행기
bold "6/7 실행기·아이콘"
mkdir -p "$APP_DIR" "$CONF_DIR" ~/.local/bin ~/.local/share/applications
cp "$PKG/omx_tools/omx-control.sh" "$APP_DIR/" && chmod +x "$APP_DIR/omx-control.sh"
printf 'OMX_GPU_HOST=%s\n' "$ALIAS" > "$CONF_DIR/laptop.env"
ln -sf "$APP_DIR/omx-control.sh" ~/.local/bin/omx-panel
cat > ~/.local/share/applications/omx-control.desktop <<EOF
[Desktop Entry]
Type=Application
Name=OMX 제어판
Comment=GPU 노트북($GPU_NAME)의 OMX 로봇 수집·학습·평가
Exec=$APP_DIR/omx-control.sh
Icon=applications-science
Terminal=false
Categories=Utility;
EOF
DESK=$(xdg-user-dir DESKTOP 2>/dev/null || echo ~/Desktop)
if [ -d "$DESK" ]; then
  cp ~/.local/share/applications/omx-control.desktop "$DESK/" && chmod +x "$DESK/omx-control.desktop"
  gio set "$DESK/omx-control.desktop" metadata::trusted true 2>/dev/null
  ok "바탕화면 아이콘 'OMX 제어판'"
fi
ok "명령 omx-panel (새 터미널부터)"

# ---------------------------------------------------------------- 7. Rerun (선택)
bold "7/7 Rerun 뷰어 (선택)"
if [ -x ~/venv/rerun/bin/rerun ] || [ -x ~/venv/il/bin/rerun ]; then ok "Rerun 뷰어가 이미 있습니다"
elif yesno "녹화·평가 중 관절 그래프를 내 PC에서 보는 Rerun 뷰어(0.26.2)를 설치할까요?" n; then
  python3 -m venv ~/venv/rerun && ~/venv/rerun/bin/pip install -q rerun-sdk==0.26.2 && ok "~/venv/rerun/bin/rerun"
fi

bold "설치 완료 (v$VERSION)"
echo "  바탕화면의 'OMX 제어판'을 더블클릭하거나 터미널에서 omx-panel 을 실행하세요."
echo "  처음엔 아이콘을 우클릭해 '실행 허용(Allow Launching)'을 눌러야 할 수 있습니다."
