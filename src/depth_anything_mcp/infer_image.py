from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depth_anything_mcp.catalog import IMAGE_RELATIVE_HF, image_hf_id, normalize_encoder, normalize_metric_scene
from depth_anything_mcp.checkpoints import download_image_native_weights, native_image_checkpoint
from depth_anything_mcp.infer_onnx import get_onnx_bundle, infer_with_session
from depth_anything_mcp.media import bgr_to_pil, collect_image_sources, load_image_bgr, stem_of
from depth_anything_mcp.paths import enable_native_imports, ensure_dir
from depth_anything_mcp.registry import cache_get, cache_put, current_settings, inference_lock
from depth_anything_mcp.visualize import colorize_depth, depth_stats, encode_preview_png, save_image_bgr, side_by_side


def _load_transformers(model_id: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()
    return {"processor": processor, "model": model, "torch": torch, "model_id": model_id}


def _infer_transformers(bundle: dict[str, Any], image_rgb, device: str) -> np.ndarray:
    torch = bundle["torch"]
    inputs = bundle["processor"](images=image_rgb, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        predicted = bundle["model"](**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(image_rgb.height, image_rgb.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    return prediction.detach().float().cpu().numpy()


def _load_native(encoder: str, device: str):
    import torch

    from depth_anything_mcp.catalog import ENCODER_CONFIGS

    settings = current_settings()
    enable_native_imports(settings, kind="image")
    from depth_anything_v2.dpt import DepthAnythingV2

    weights = download_image_native_weights(encoder, settings)
    model = DepthAnythingV2(**ENCODER_CONFIGS[encoder])
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return {"model": model, "weights": str(weights)}


def resolve_image_backend(backend: str) -> str:
    choice = (backend or "onnx").strip().lower()
    if choice in {"auto", "onnx", "ort"}:
        return "onnx"
    if choice in {"transformers", "hf", "native"}:
        return "transformers" if choice != "native" else "native"
    raise ValueError("backend must be onnx, transformers, or native.")


def estimate_images(
    image_path: str,
    *,
    encoder: str | None = None,
    backend: str = "onnx",
    variant: str = "dynamic",
    input_size: int = 518,
    metric_scene: str | None = None,
    output_dir: str | None = None,
    pred_only: bool = False,
    grayscale: bool = False,
    save_raw: bool = False,
    include_preview: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    settings = current_settings()
    encoder = normalize_encoder(encoder, kind="image", default=settings.default_encoder)
    scene = normalize_metric_scene(metric_scene)
    backend_name = resolve_image_backend(backend)
    sources = collect_image_sources(image_path)
    out_dir = ensure_dir(Path(output_dir).expanduser().resolve() if output_dir else settings.output_dir / "images")
    device = device or settings.device

    with inference_lock():
        if backend_name == "onnx":
            bundle = get_onnx_bundle(encoder, variant=variant, metric_scene=scene, device=device)
        else:
            cache_key = ("image", backend_name, encoder, scene or "relative", device)
            bundle = cache_get(cache_key)
            if bundle is None:
                if backend_name == "transformers":
                    bundle = _load_transformers(image_hf_id(encoder, scene), device)
                else:
                    if scene:
                        raise ValueError("Native backend does not load metric heads here. Use backend=onnx.")
                    bundle = _load_native(encoder, device)
                cache_put(cache_key, bundle)

        results: list[dict[str, Any]] = []
        preview_bytes = None
        model_id = bundle.get("weights") or bundle.get("model_id")
        for index, source in enumerate(sources):
            bgr = load_image_bgr(source)
            if backend_name == "onnx":
                depth = infer_with_session(bundle["session"], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), input_size)
                model_id = bundle["weights"]
            elif backend_name == "transformers":
                depth = _infer_transformers(bundle, bgr_to_pil(bgr), device)
                model_id = bundle["model_id"]
            else:
                depth = bundle["model"].infer_image(bgr, input_size)
                model_id = native_image_checkpoint(settings, encoder).name

            vis_rgb = colorize_depth(depth, grayscale=grayscale)
            stem = stem_of(source)
            vis_bgr = vis_rgb[:, :, ::-1]
            vis_path = out_dir / f"{stem}_depth.png"
            if pred_only:
                save_image_bgr(vis_path, vis_bgr)
            else:
                save_image_bgr(vis_path, side_by_side(bgr, vis_rgb))

            raw_path = None
            if save_raw:
                raw_path = out_dir / f"{stem}_depth.npy"
                np.save(raw_path, depth)

            results.append(
                {
                    "source": source,
                    "visualization": str(vis_path),
                    "raw_depth": str(raw_path) if raw_path else None,
                    "stats": depth_stats(depth),
                }
            )
            if include_preview and index == 0:
                preview_bytes = encode_preview_png(vis_rgb)

    spec = IMAGE_RELATIVE_HF[encoder]
    return {
        "ok": True,
        "task": "image_depth",
        "model": "Depth Anything V2",
        "encoder": encoder,
        "backend": backend_name,
        "variant": variant if backend_name == "onnx" else None,
        "device": bundle.get("device") or device,
        "metric_scene": scene,
        "model_id": model_id,
        "license": bundle.get("license") or spec["license"],
        "source": bundle.get("source"),
        "relative_depth": scene is None,
        "count": len(results),
        "output_dir": str(out_dir),
        "results": results,
        "preview_png": preview_bytes,
    }
