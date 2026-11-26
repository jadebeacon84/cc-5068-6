"""Local runtime bootstrap.

Makes the workspace-local dependency directory ``.pydeps`` importable so the
project runs on a stripped interpreter where packages were installed with
``pip --target``. Also locates the bundled Tk extras used on machines without
the distribution's ``python3-tk`` package.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
PYDEPS = WORKSPACE / ".pydeps"
EXTRA_TK = WORKSPACE / ".extradeps"


def bootstrap_python_paths() -> None:
    p = str(PYDEPS)
    if PYDEPS.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def ensure_gui_env() -> None:
    """Make a local ``python3-tk`` fallback importable; re-exec if needed.

    On stripped containers ``import tkinter`` can be missing even though the
    shared libraries are present. We add the bundled version-specific package
    directory and, when the dynamic loader path must change, re-exec the same
    GUI module with LD_LIBRARY_PATH/PYTHONPATH populated.
    """
    bootstrap_python_paths()
    try:
        import tkinter  # noqa: F401
        return
    except Exception:
        pass

    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    py_dir = EXTRA_TK / ver
    extra_paths = [py_dir, py_dir / "lib-dynload"]
    if not py_dir.is_dir():
        raise RuntimeError(
            "当前 Python 缺少 tkinter，且未找到本地兜底目录 "
            f"{py_dir}。请在 Debian/Ubuntu 上执行 "
            "`sudo apt install python3-tk`，或使用 `bash run_gui.sh`。")

    for p in reversed(extra_paths):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    lib_dir = EXTRA_TK / "lib"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [x for x in ld.split(os.pathsep) if x]
    need_reexec = lib_dir.is_dir() and str(lib_dir) not in ld_parts
    if need_reexec:
        os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (
            os.pathsep + ld if ld else "")

    pp_parts = [str(WORKSPACE), str(PYDEPS)] + [str(p) for p in extra_paths]
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        pp_parts + ([existing_pp] if existing_pp else []))
    os.environ.setdefault("VAD_GUI_REEXEC", "0")

    try:
        import tkinter  # noqa: F401
        if need_reexec and os.environ.get("VAD_GUI_REEXEC") != "1":
            raise ImportError("force re-exec to apply LD_LIBRARY_PATH")
    except Exception:
        if os.environ.get("VAD_GUI_REEXEC") == "1":
            raise RuntimeError(
                "已尝试用本地 .extradeps 重启 GUI 仍无法导入 tkinter；"
                "请安装 python3-tk 或检查 Tcl/Tk 运行库。")
        os.environ["VAD_GUI_REEXEC"] = "1"
        os.execv(sys.executable,
                 [sys.executable, "-m", "vad_analyzer.gui"] + sys.argv[1:])


bootstrap_python_paths()
