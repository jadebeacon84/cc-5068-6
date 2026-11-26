#!/usr/bin/env bash
# GUI launcher for stripped environments (no system python3-tk).
set -euo pipefail
cd "$(dirname "$0")"

TK_PY="$PWD/.extradeps/python3.11"
export PYTHONPATH="$PWD/.pydeps:$TK_PY:$TK_PY/lib-dynload:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/.extradeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo "未检测到 DISPLAY/WAYLAND_DISPLAY：图形窗口需要桌面或 Xvfb。" >&2
fi

exec python3 -m vad_analyzer.gui "$@"
