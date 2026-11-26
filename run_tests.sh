#!/usr/bin/env bash
# Runs the unittest suite with the workspace-local numpy/Tk fallbacks.
set -euo pipefail
cd "$(dirname "$0")"

TK_PY="$PWD/.extradeps/python3.11"
export PYTHONPATH="$PWD:$PWD/.pydeps:$TK_PY:$TK_PY/lib-dynload${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/.extradeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "${1:-}" = "--gui" ]; then
  export VAD_TEST_GUI=1
  shift
fi

exec python3 -m unittest discover -s tests -p "test_*.py" -v "$@"
