from __future__ import annotations

import sys
from pathlib import Path

from depth_anything_mcp.settings import Settings


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepend_sys_path(directory: Path) -> None:
    resolved = str(directory.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def enable_native_imports(settings: Settings, *, kind: str) -> None:
    if kind == "image":
        if not settings.native_dav2_ready():
            raise FileNotFoundError(
                "Native Depth Anything V2 code is missing. "
                "Run `depth-anything-setup` or set DEPTH_ANYTHING_V2_HOME."
            )
        prepend_sys_path(settings.dav2_home)
        return

    if kind == "video":
        if not settings.native_vda_ready():
            raise FileNotFoundError(
                "Video Depth Anything code is missing. "
                "Run `depth-anything-setup` or set VIDEO_DEPTH_ANYTHING_HOME."
            )
        prepend_sys_path(settings.vda_home)
        return

    raise ValueError(f"Unknown backend kind: {kind}")
