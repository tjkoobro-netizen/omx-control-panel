#!/usr/bin/env bash
# Run on the GPU laptop: start this account's OMX panel server if it is not running.
#   omx-panel               start if needed and show how to open it
#   omx-panel --port-only   same, but print only the port (used by omx-control.sh on the laptop)
#   omx-panel --restart     stop this account's server and start it again (after an update)
# Settings (port, Python environment, folders) come from omx_config.py / ~/.config/omx/config.json.
set -uo pipefail

TOOLS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG_DIR=~/il_ws/omx_logs
mkdir -p "$LOG_DIR"

# Python of the LeRobot environment, as configured (falls back to ~/venv/il)
VENV=$(python3 -c "import sys; sys.path.insert(0, '$TOOLS'); import omx_config; print(omx_config.venv_bin())" 2>/dev/null || echo ~/venv/il/bin)
PY="$VENV/python"
[ -x "$PY" ] || { echo "[실패] LeRobot 가상환경($VENV)이 없습니다. 설치 매뉴얼 3단계를 먼저 하세요." >&2; exit 1; }
PORT=$("$PY" -c "import sys; sys.path.insert(0, '$TOOLS'); import omx_config; print(omx_config.port())")
listening() { ss -ltn | grep -q "127.0.0.1:${PORT} "; }

if [ "${1:-}" = "--restart" ]; then
  # the inference server for control PCs (v2) is started by the panel: restart it with the new code too
  pkill -TERM -u "$(id -u)" -f "omx_infer.py serve" || true
fi
if [ "${1:-}" = "--restart" ] && listening; then
  pkill -TERM -u "$(id -u)" -f "python omx_web.py" || true
  for _ in $(seq 1 20); do listening || break; sleep 0.5; done
  if listening; then  # still serving (e.g. held by an open video stream): an update must not keep the old code
    pkill -KILL -u "$(id -u)" -f "python omx_web.py" || true
    for _ in $(seq 1 10); do listening || break; sleep 0.5; done
  fi
fi

if ! listening; then
  cd "$TOOLS" || exit 1
  OMX_PORT=$PORT setsid nohup "$PY" omx_web.py >> "$LOG_DIR/web.log" 2>&1 < /dev/null &
  for _ in $(seq 1 40); do listening && break; sleep 0.5; done
  listening || { echo "[실패] 서버를 켜지 못했습니다. $LOG_DIR/web.log 를 확인하세요." >&2; tail -5 "$LOG_DIR/web.log" >&2; exit 1; }
  started=1
fi

if [ "${1:-}" = "--port-only" ]; then
  echo "$PORT"
  exit 0
fi
[ -n "${started:-}" ] && echo "제어판 서버를 켰습니다 (포트 ${PORT})." || echo "제어판 서버가 이미 실행 중입니다 (포트 ${PORT})."
echo
echo "열기:"
echo "  내 노트북            : 바탕화면의 'OMX Control Panel' 아이콘 또는 터미널에서  omx-panel"
echo "  VS Code Remote-SSH  : 아래 패널의 [포트(Ports)] 탭 -> ${PORT} -> 브라우저에서 열기(지구본 아이콘)"
