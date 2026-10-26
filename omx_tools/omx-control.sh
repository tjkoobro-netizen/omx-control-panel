#!/usr/bin/env bash
# Open the OMX control panel from a laptop:
#   1. start your panel server on the GPU laptop if it is not running
#   2. open an SSH tunnel localhost:PORT -> GPU laptop 127.0.0.1:PORT (over Tailscale)
#   3. open the panel in the browser
# Each GPU-laptop account has its own port, so data is saved under your name.
# The SSH alias of the GPU laptop comes from ~/.config/omx/laptop.env (written by install_laptop.sh).
set -uo pipefail

[ -f ~/.config/omx/laptop.env ] && . ~/.config/omx/laptop.env
HOST="${OMX_GPU_HOST:-omx-gpu}"

open_url() {  # a new window, so the panel is not hidden as a tab behind other windows
  if command -v google-chrome >/dev/null; then setsid google-chrome --new-window "$1" >/dev/null 2>&1 &
  elif command -v firefox >/dev/null; then setsid firefox --new-window "$1" >/dev/null 2>&1 &
  else setsid xdg-open "$1" >/dev/null 2>&1 &
  fi
}

# v2 control PC (the robot is on this PC; training and inference on the GPU laptop): run the panel here
if grep -q '"role": *"control"' ~/.config/omx/config.json 2>/dev/null; then
  notify-send -t 4000 "OMX Control Panel" "이 PC의 제어판을 켜는 중..." 2>/dev/null
  # an SSH tunnel left by the old way of opening (laptop -> GPU panel) would hold the local port
  P0=$(~/venv/il/bin/python -c "import sys; sys.path.insert(0, '$HOME/il_ws/src/omx_tools'); import omx_config; print(omx_config.port())" 2>/dev/null)
  for pid in $(ss -ltnpH 2>/dev/null | grep "127.0.0.1:${P0} " | grep '"ssh"' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do kill "$pid"; done
  PORT=$(bash ~/il_ws/src/omx_tools/omx-panel-server.sh --port-only) || {
    notify-send -u critical "OMX Control Panel" "제어판 서버를 켜지 못했습니다. ~/il_ws/omx_logs/web.log 를 확인하세요." 2>/dev/null; exit 1; }
  open_url "http://127.0.0.1:${PORT}"
  exit 0
fi
# Path of the tools on the GPU laptop, relative to your home there
TOOLS="${OMX_TOOLS:-il_ws/src/omx_tools}"

fail() {
  notify-send -u critical "OMX Control Panel" "$1" 2>/dev/null
  echo "[실패] $1" >&2
  exit 1
}

notify-send -t 4000 "OMX Control Panel" "GPU 노트북($HOST)에 연결하는 중..." 2>/dev/null

if command -v tailscale >/dev/null && ! tailscale status >/dev/null 2>&1; then
  fail "Tailscale이 연결되어 있지 않습니다. 터미널에서 'sudo tailscale up' 후 다시 시도하세요."
fi

# Starts the server when needed and prints the port of this account's server
PORT=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "bash $TOOLS/omx-panel-server.sh --port-only") \
  || fail "GPU 노트북($HOST)에서 제어판 서버를 켜지 못했습니다. 'ssh $HOST' 접속과 GPU 노트북의 ~/il_ws/omx_logs/web.log 를 확인하세요."
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "서버 포트를 알 수 없습니다: $PORT"
URL="http://127.0.0.1:${PORT}"

if ! ss -ltn | grep -q "127.0.0.1:${PORT} "; then
  ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "${PORT}:127.0.0.1:${PORT}" "$HOST" || fail "SSH 터널을 열지 못했습니다 (포트 ${PORT})."
fi

# Open in a new window so the panel is not hidden as a tab behind other windows
if command -v google-chrome >/dev/null; then
  setsid google-chrome --new-window "$URL" >/dev/null 2>&1 &
elif command -v firefox >/dev/null; then
  setsid firefox --new-window "$URL" >/dev/null 2>&1 &
else
  setsid xdg-open "$URL" >/dev/null 2>&1 &
fi
