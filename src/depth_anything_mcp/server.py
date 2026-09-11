from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from depth_anything_mcp import __version__
from depth_anything_mcp.catalog import ONNX_FABIO, ONNX_QUANTIZED
from depth_anything_mcp.checkpoints import describe_local_weights, download_onnx_weights
from depth_anything_mcp.infer_image import estimate_images
from depth_anything_mcp.infer_video import estimate_video
from depth_anything_mcp.registry import cached_models, current_settings
from depth_anything_mcp.setup_models import setup_backends

mcp = FastMCP("depth-anything")


def _dump(payload: dict[str, Any]) -> str | list[Any]:
    preview = payload.pop("preview_png", None)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if preview:
        return [text, Image(data=preview, format="png")]
    return text


def _fail(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False, indent=2)


@mcp.tool()
def estimate_image_depth(
    image_path: str,
    encoder: str = "vits",
    backend: str = "onnx",
    variant: str = "dynamic",
    input_size: int = 518,
    metric_scene: str = "",
    output_dir: str = "",
    pred_only: bool = False,
    grayscale: bool = False,
    save_raw: bool = False,
    include_preview: bool = True,
    device: str = "",
) -> str | list[Any]:
    """Estimate monocular depth with Depth Anything V2 ONNX.

    默认使用可打包的 ONNX Runtime 模型（单文件 dynamic，约 95MB；quantized 约 38MB）。
    不需要 PyTorch。

    Args:
        image_path: Local image path, image directory, txt list, or http(s) URL.
        encoder: vits (default, Apache-2.0, ~95MB), vitb, or vitl. Base/Large are CC-BY-NC-4.0.
        backend: onnx (default), transformers, or native.
        variant: dynamic (single-file, recommended) or quantized (smaller two-file community export).
        input_size: Short-side size before rounding to a multiple of 14. Official default is 518.
        metric_scene: Empty for relative depth. indoor (20m) or outdoor (80m) uses metric ONNX.
        output_dir: Optional output directory. Defaults to <project>/outputs/images.
        pred_only: If true, save only the depth visualization. Otherwise save input | depth.
        grayscale: Save a grayscale depth map instead of Spectral_r color.
        save_raw: Also save the raw HxW depth array as .npy.
        include_preview: Attach a downsized depth preview image in the MCP response.
        device: auto (default), cpu, dml, cuda, or cuda:N. Empty uses DEPTH_ANYTHING_DEVICE.
    """
    try:
        result = estimate_images(
            image_path,
            encoder=encoder,
            backend=backend,
            variant=variant,
            input_size=input_size,
            metric_scene=metric_scene or None,
            output_dir=output_dir or None,
            pred_only=pred_only,
            grayscale=grayscale,
            save_raw=save_raw,
            include_preview=include_preview,
            device=device or None,
        )
        return _dump(result)
    except Exception as exc:
        return _fail(str(exc))


@mcp.tool()
def estimate_video_depth(
    video_path: str,
    encoder: str = "vits",
    backend: str = "onnx",
    variant: str = "dynamic",
    metric: bool = False,
    metric_scene: str = "",
    streaming: bool = False,
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    fp32: bool = False,
    grayscale: bool = False,
    save_npz: bool = False,
    save_source: bool = True,
    output_dir: str = "",
    include_preview: bool = True,
    device: str = "",
) -> str | list[Any]:
    """Estimate video depth with the packable Depth Anything V2 ONNX model.

    官方 Video Depth Anything 没有可打包的正式 ONNX。默认按帧跑 DA-V2 ONNX。
    若要时序一致的官方 VDA，把 backend 设为 native 并先克隆官方仓库。

    Args:
        video_path: Local video file path (.mp4/.mov/.avi/.mkv/.webm).
        encoder: vits (default), vitb, or vitl.
        backend: onnx (default, packable) or native (official Video Depth Anything).
        variant: dynamic or quantized. Only used by the ONNX backend.
        metric: For ONNX, load outdoor metric weights unless metric_scene is set.
        metric_scene: indoor or outdoor metric ONNX. Empty keeps relative depth.
        streaming: Native VDA experimental streaming mode only.
        input_size: Model input size. Official default is 518.
        max_res: Downscale so the longer side does not exceed this value.
        max_len: Max frames to read. -1 means no limit. Use 32 for a smoke test.
        target_fps: Resample fps. -1 keeps the original fps.
        fp32: Native backend only.
        grayscale: Save grayscale depth video.
        save_npz: Also save compressed raw depths as .npz.
        save_source: Also export the (possibly resized) RGB source video.
        output_dir: Optional output directory. Defaults to <project>/outputs/videos.
        include_preview: Attach a mid-clip depth preview image.
        device: auto, cpu, dml, cuda, or cuda:N. Empty uses DEPTH_ANYTHING_DEVICE.
    """
    try:
        result = estimate_video(
            video_path,
            encoder=encoder,
            backend=backend,
            variant=variant,
            metric=metric,
            metric_scene=metric_scene or None,
            streaming=streaming,
            input_size=input_size,
            max_res=max_res,
            max_len=max_len,
            target_fps=target_fps,
            fp32=fp32,
            grayscale=grayscale,
            save_npz=save_npz,
            save_source=save_source,
            output_dir=output_dir or None,
            include_preview=include_preview,
            device=device or None,
        )
        return _dump(result)
    except Exception as exc:
        return _fail(str(exc))


@mcp.tool()
def list_depth_models() -> str:
    """List packable ONNX models first, plus optional native/HF backends."""
    settings = current_settings()
    payload = {
        "ok": True,
        "server_version": __version__,
        "device": settings.device,
        "default_encoder": settings.default_encoder,
        "default_backend": "onnx",
        "onnx_dynamic": ONNX_FABIO,
        "onnx_quantized": ONNX_QUANTIZED,
        "local_weights": describe_local_weights(settings),
        "notes": [
            "Default weights: fabio-sim Depth-Anything-ONNX v2.0.0 single-file dynamic ONNX.",
            "vits dynamic is ~94.5MB and Apache-2.0. quantized vits is ~38.6MB (two files).",
            "Official Video Depth Anything has no packable ONNX; video defaults to per-frame DA-V2 ONNX.",
            "Base/Large weights are CC-BY-NC-4.0.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def download_depth_checkpoints(
    target: str = "onnx",
    encoder: str = "vits",
    variant: str = "dynamic",
    metric_scene: str = "",
) -> str:
    """Download extra ONNX weights. Default vits dynamic is already bundled.

    Args:
        target: onnx (default). Other legacy targets are no longer required for packaging.
        encoder: vits, vitb, or vitl.
        variant: dynamic (single .onnx) or quantized (community graph + data).
        metric_scene: Empty, indoor, or outdoor. indoor/outdoor only work with variant=dynamic.
    """
    try:
        if target.strip().lower() not in {"onnx", "image-onnx", "video"}:
            raise ValueError("Packaging path only needs target='onnx'.")
        path = download_onnx_weights(encoder, variant=variant, metric_scene=metric_scene or None)
        payload = {
            "ok": True,
            "target": "onnx",
            "encoder": encoder,
            "variant": variant,
            "metric_scene": metric_scene or None,
            "path": str(path),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return _fail(str(exc))


@mcp.tool()
def setup_depth_backends(
    download_onnx: bool = True,
    encoder: str = "vits",
    variant: str = "dynamic",
    metric_scene: str = "",
    clone_dav2: bool = False,
    clone_vda: bool = False,
) -> str:
    """Download the default packable ONNX model into checkpoints/.

    默认只拉 vits dynamic ONNX，不克隆官方 PyTorch 仓库。
    """
    try:
        report = setup_backends(
            download_onnx=download_onnx,
            encoder=encoder,
            variant=variant,
            metric_scene=metric_scene or None,
            clone_dav2=clone_dav2,
            clone_vda=clone_vda,
        )
        report["ok"] = True
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as exc:
        return _fail(str(exc))


@mcp.tool()
def get_depth_runtime_status() -> str:
    """Show ONNX Runtime providers, paths, and cached sessions."""
    from depth_anything_mcp.runtime import ensure_onnxruntime, has_nvidia, is_windows

    settings = current_settings()
    runtime = ensure_onnxruntime(settings.device)
    spec = runtime.get("spec")
    payload = {
        "ok": True,
        "version": __version__,
        "device_setting": settings.device,
        "resolved_device": spec.label if spec else "cpu",
        "nvidia": has_nvidia(),
        "windows": is_windows(),
        "home": str(settings.home),
        "output_dir": str(settings.output_dir),
        "checkpoint_dir": str(settings.checkpoint_dir),
        "hf_endpoint": settings.hf_endpoint,
        "cached_models": cached_models(),
        "ort_providers": runtime.get("providers") or [],
    }
    if runtime.get("error"):
        payload["runtime_error"] = runtime["error"]
    if runtime.get("fallback"):
        payload["fallback"] = runtime["fallback"]
    if runtime.get("requested"):
        payload["requested_device"] = runtime["requested"]
    try:
        import onnxruntime as ort

        payload["onnxruntime"] = ort.__version__
    except Exception as exc:
        payload["onnxruntime_error"] = str(exc)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.prompt()
def depth_estimation_help() -> str:
    """How to call the Depth Anything MCP tools."""
    return (
        "Default backend is ONNX Runtime. No PyTorch needed.\n"
        "Images: estimate_image_depth. Videos: estimate_video_depth (per-frame DA-V2 ONNX).\n"
        "Default vits dynamic ONNX (~95MB) is bundled. Extra models download on demand.\n"
        "Device auto: Windows uses DirectML; Linux NVIDIA uses CUDA 12 ONNX Runtime.\n"
        "For a smaller package use variant=quantized (~38MB, two files).\n"
        "Official Video Depth Anything has no packable ONNX release."
    )


def main() -> None:
    from depth_anything_mcp.runtime import ensure_onnxruntime

    ensure_onnxruntime(current_settings().device)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
