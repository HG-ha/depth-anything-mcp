from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from depth_anything_mcp.catalog import DAV2_REPO, VDA_REPO
from depth_anything_mcp.checkpoints import download_onnx_weights
from depth_anything_mcp.paths import ensure_dir
from depth_anything_mcp.settings import load_settings


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def clone_repo(url: str, dest: Path) -> dict[str, str]:
    if dest.exists() and any(dest.iterdir()):
        return {"path": str(dest), "status": "already_present"}
    ensure_dir(dest.parent)
    _run(["git", "clone", "--depth", "1", url, str(dest)])
    return {"path": str(dest), "status": "cloned"}


def setup_backends(
    *,
    download_onnx: bool = True,
    encoder: str = "vits",
    variant: str = "dynamic",
    metric_scene: str | None = None,
    clone_dav2: bool = False,
    clone_vda: bool = False,
) -> dict:
    settings = load_settings()
    report: dict = {"home": str(settings.home), "steps": []}

    if download_onnx:
        path = download_onnx_weights(encoder, variant=variant, metric_scene=metric_scene, settings=settings)
        report["steps"].append({"name": "download_onnx", "path": str(path), "variant": variant})
    if clone_dav2:
        report["steps"].append({"name": "clone_depth_anything_v2", **clone_repo(DAV2_REPO, settings.dav2_home)})
    if clone_vda:
        report["steps"].append({"name": "clone_video_depth_anything", **clone_repo(VDA_REPO, settings.vda_home)})

    report["onnx_dir"] = str(settings.checkpoint_dir)
    report["dav2_ready"] = settings.native_dav2_ready()
    report["vda_ready"] = settings.native_vda_ready()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download packable Depth Anything V2 ONNX weights.")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--variant", default="dynamic", choices=["dynamic", "quantized"])
    parser.add_argument("--metric-scene", default="", choices=["", "indoor", "outdoor"])
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--clone-dav2", action="store_true")
    parser.add_argument("--clone-vda", action="store_true")
    args = parser.parse_args()

    report = setup_backends(
        download_onnx=not args.skip_onnx,
        encoder=args.encoder,
        variant=args.variant,
        metric_scene=args.metric_scene or None,
        clone_dav2=args.clone_dav2,
        clone_vda=args.clone_vda,
    )
    print("Depth Anything setup finished.")
    for step in report["steps"]:
        print(f"- {step}")
    print(f"ONNX dir: {report['onnx_dir']}")
