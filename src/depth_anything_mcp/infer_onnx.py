from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from depth_anything_mcp.catalog import normalize_encoder, normalize_metric_scene, onnx_spec
from depth_anything_mcp.checkpoints import download_onnx_weights
from depth_anything_mcp.registry import cache_get, cache_put, current_settings
from depth_anything_mcp.runtime import DeviceSpec, ensure_onnxruntime, ort_providers_for, resolve_device_spec

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def round_to_14(value: float) -> int:
    return max(14, int(round(value / 14.0)) * 14)


def infer_hw(
    orig_h: int,
    orig_w: int,
    input_size: int = 518,
    static_hw: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if static_hw is not None:
        return static_hw
    scale = float(input_size) / float(min(orig_h, orig_w))
    return round_to_14(orig_h * scale), round_to_14(orig_w * scale)


def static_input_hw(shape: list[Any]) -> tuple[int, int] | None:
    if len(shape) < 2:
        return None
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


def preprocess_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    image = rgb.astype(np.float32) / 255.0
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def preprocess_bgr(bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    return preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), height, width)


def postprocess_depth(raw: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    depth = np.squeeze(np.asarray(raw))
    if depth.ndim != 2:
        raise ValueError(f"Unexpected ONNX depth shape: {getattr(raw, 'shape', None)}")
    return cv2.resize(depth.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def resolve_ort_device(preferred: str | None) -> DeviceSpec:
    return resolve_device_spec(preferred)


def load_onnx_session(model_path, spec: DeviceSpec):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=ort_providers_for(spec))


def infer_with_session(session, rgb: np.ndarray, input_size: int = 518) -> np.ndarray:
    spec = session.get_inputs()[0]
    orig_h, orig_w = rgb.shape[:2]
    height, width = infer_hw(orig_h, orig_w, input_size, static_input_hw(spec.shape))
    tensor = preprocess_rgb(rgb, height, width)
    if "float16" in str(spec.type):
        tensor = tensor.astype(np.float16)
    raw = session.run(None, {spec.name: tensor})[0]
    return postprocess_depth(raw, orig_h, orig_w)


def get_onnx_bundle(
    encoder: str,
    *,
    variant: str = "dynamic",
    metric_scene: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    settings = current_settings()
    encoder = normalize_encoder(encoder, kind="image", default=settings.default_encoder)
    scene = normalize_metric_scene(metric_scene)
    runtime = ensure_onnxruntime(device or settings.device)
    spec = runtime.get("spec") or resolve_ort_device(device or settings.device)
    cache_key = ("onnx", encoder, variant, scene or "relative", spec.label)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    path = download_onnx_weights(encoder, variant=variant, metric_scene=scene, settings=settings)
    session = load_onnx_session(path, spec)
    model = onnx_spec(encoder, variant, scene)
    bundle = {
        "session": session,
        "weights": str(path),
        "input_name": session.get_inputs()[0].name,
        "providers": session.get_providers(),
        "device": spec.label,
        "runtime": runtime,
        "license": model["license"],
        "source": model["source"],
        "packaging": model.get("packaging"),
        "encoder": encoder,
        "variant": variant,
        "metric_scene": scene,
    }
    return cache_put(cache_key, bundle)
