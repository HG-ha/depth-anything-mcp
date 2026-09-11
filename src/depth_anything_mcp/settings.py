from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from depth_anything_mcp.mirrors import apply_download_mirrors, mirrors_enabled


def detect_device(preferred: str = "auto") -> str:
    """Keep this light: do not import onnxruntime/torch at settings load time."""
    choice = (preferred or "auto").strip().lower()
    return choice or "auto"


def default_user_home() -> Path:
    return Path.home() / ".depth-anything-mcp"


def project_root() -> Path:
    env = os.environ.get("DEPTH_ANYTHING_HOME")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "depth_anything_mcp").exists():
            return parent
    # uvx / pip installs have no source checkout; keep weights in a stable user dir.
    return default_user_home()


@dataclass(frozen=True)
class Settings:
    home: Path
    device: str
    default_encoder: str
    output_dir: Path
    checkpoint_dir: Path
    dav2_home: Path
    vda_home: Path
    hf_endpoint: str | None
    mirror: str

    @property
    def third_party_dir(self) -> Path:
        return self.home / "third_party"

    def native_dav2_ready(self) -> bool:
        return (self.dav2_home / "depth_anything_v2" / "dpt.py").is_file()

    def native_vda_ready(self) -> bool:
        return (self.vda_home / "video_depth_anything" / "video_depth.py").is_file()


def load_settings() -> Settings:
    mirror_state = apply_download_mirrors()
    home = project_root()
    output_dir = Path(os.environ.get("DEPTH_ANYTHING_OUTPUT_DIR", home / "outputs")).expanduser().resolve()
    checkpoint_dir = Path(
        os.environ.get("DEPTH_ANYTHING_CHECKPOINT_DIR", home / "checkpoints")
    ).expanduser().resolve()
    dav2_home = Path(
        os.environ.get("DEPTH_ANYTHING_V2_HOME", home / "third_party" / "Depth-Anything-V2")
    ).expanduser().resolve()
    vda_home = Path(
        os.environ.get("VIDEO_DEPTH_ANYTHING_HOME", home / "third_party" / "Video-Depth-Anything")
    ).expanduser().resolve()
    return Settings(
        home=home,
        device=detect_device(os.environ.get("DEPTH_ANYTHING_DEVICE", "auto")),
        default_encoder=os.environ.get("DEPTH_ANYTHING_DEFAULT_ENCODER", "vits").strip().lower(),
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        dav2_home=dav2_home,
        vda_home=vda_home,
        hf_endpoint=str(mirror_state["hf_endpoint"] or "") or None,
        mirror="cn" if mirrors_enabled() else "off",
    )
