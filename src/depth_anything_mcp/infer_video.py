from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from depth_anything_mcp.catalog import ENCODER_CONFIGS, normalize_encoder, normalize_metric_scene, video_weight_spec
from depth_anything_mcp.checkpoints import download_video_weights
from depth_anything_mcp.infer_onnx import get_onnx_bundle, infer_with_session
from depth_anything_mcp.media import read_video_rgb, resolve_video_path, write_rgb_video
from depth_anything_mcp.paths import enable_native_imports, ensure_dir
from depth_anything_mcp.registry import cache_get, cache_put, current_settings, inference_lock
from depth_anything_mcp.visualize import colorize_depth, depth_stats, encode_preview_png


def _load_offline_model(encoder: str, metric: bool, device: str):
    import torch

    settings = current_settings()
    enable_native_imports(settings, kind="video")
    from video_depth_anything.video_depth import VideoDepthAnything

    weights = download_video_weights(encoder, metric, settings)
    model = VideoDepthAnything(**ENCODER_CONFIGS[encoder], metric=metric)
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return {"model": model, "weights": str(weights), "streaming": False}


def _load_streaming_model(encoder: str, metric: bool, device: str):
    import torch

    settings = current_settings()
    enable_native_imports(settings, kind="video")
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    weights = download_video_weights(encoder, metric, settings)
    model = VideoDepthAnything(**ENCODER_CONFIGS[encoder])
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return {"model": model, "weights": str(weights), "streaming": True}


def _infer_streaming(model, frames: np.ndarray, input_size: int, device: str, fp32: bool) -> np.ndarray:
    depths = [model.infer_video_depth_one(frame, input_size=input_size, device=device, fp32=fp32) for frame in frames]
    return np.stack(depths, axis=0)


def resolve_video_backend(backend: str) -> str:
    choice = (backend or "onnx").strip().lower()
    if choice in {"auto", "onnx", "ort"}:
        return "onnx"
    if choice in {"native", "vda"}:
        return "native"
    raise ValueError("backend must be onnx or native.")


def estimate_video(
    video_path: str,
    *,
    encoder: str | None = None,
    backend: str = "onnx",
    variant: str = "dynamic",
    metric: bool = False,
    metric_scene: str | None = None,
    streaming: bool = False,
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    fp32: bool = False,
    grayscale: bool = False,
    save_npz: bool = False,
    save_source: bool = True,
    output_dir: str | None = None,
    include_preview: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    settings = current_settings()
    encoder = normalize_encoder(encoder, kind="video", default=settings.default_encoder)
    backend_name = resolve_video_backend(backend)
    scene = normalize_metric_scene(metric_scene)
    if metric and scene is None and backend_name == "onnx":
        scene = "outdoor"

    path = resolve_video_path(video_path)
    out_dir = ensure_dir(Path(output_dir).expanduser().resolve() if output_dir else settings.output_dir / "videos")
    device = device or settings.device
    if backend_name == "native" and str(device).startswith("mps"):
        device = "cpu"

    with inference_lock():
        if backend_name == "onnx":
            bundle = get_onnx_bundle(encoder, variant=variant, metric_scene=scene, device=device)
            frames, fps, meta = read_video_rgb(path, max_len=max_len, target_fps=target_fps, max_res=max_res)
            depths = np.stack(
                [infer_with_session(bundle["session"], frame, input_size) for frame in frames],
                axis=0,
            )
            model_name = "Depth Anything V2 ONNX (per-frame)"
            weights = bundle["weights"]
            device = bundle.get("device") or device
            license_name = bundle["license"]
            note = (
                "Official Video Depth Anything has no packable ONNX release. "
                "This path runs Depth Anything V2 ONNX frame by frame."
            )
        else:
            cache_key = ("video", "stream" if streaming else "offline", encoder, "metric" if metric else "relative", device)
            bundle = cache_get(cache_key)
            if bundle is None:
                bundle = (
                    _load_streaming_model(encoder, metric, device)
                    if streaming
                    else _load_offline_model(encoder, metric, device)
                )
                cache_put(cache_key, bundle)
            frames, fps, meta = read_video_rgb(path, max_len=max_len, target_fps=target_fps, max_res=max_res)
            if streaming:
                depths = _infer_streaming(bundle["model"], frames, input_size, device, fp32)
            else:
                depths, fps = bundle["model"].infer_video_depth(
                    frames,
                    fps,
                    input_size=input_size,
                    device=device,
                    fp32=fp32,
                )
            model_name = "Video Depth Anything"
            weights = bundle["weights"]
            license_name = video_weight_spec(encoder, metric)["license"]
            note = "Native temporal Video Depth Anything."

        vmin = float(np.nanmin(depths))
        vmax = float(np.nanmax(depths))
        vis_frames = np.stack(
            [colorize_depth(frame, grayscale=grayscale, vmin=vmin, vmax=vmax) for frame in depths],
            axis=0,
        )

        stem = path.stem
        vis_path = out_dir / f"{stem}_vis.mp4"
        write_rgb_video(vis_path, vis_frames, fps)

        src_path = None
        if save_source:
            src_path = out_dir / f"{stem}_src.mp4"
            write_rgb_video(src_path, frames, fps)

        npz_path = None
        if save_npz:
            npz_path = out_dir / f"{stem}_depths.npz"
            np.savez_compressed(npz_path, depths=depths)

        preview = encode_preview_png(vis_frames[len(vis_frames) // 2]) if include_preview else None

    return {
        "ok": True,
        "task": "video_depth",
        "model": model_name,
        "encoder": encoder,
        "backend": backend_name,
        "variant": variant if backend_name == "onnx" else None,
        "metric": metric,
        "metric_scene": scene,
        "streaming": streaming if backend_name == "native" else False,
        "device": device,
        "weights": weights,
        "license": license_name,
        "note": note,
        "fps": float(fps),
        "video_meta": meta,
        "depth_stats": depth_stats(depths),
        "visualization": str(vis_path),
        "source_video": str(src_path) if src_path else None,
        "raw_npz": str(npz_path) if npz_path else None,
        "output_dir": str(out_dir),
        "preview_png": preview,
    }
