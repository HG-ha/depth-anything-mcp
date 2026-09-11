from __future__ import annotations

from typing import Any, Literal

Encoder = Literal["vits", "vitb", "vitl", "vitg"]
ImageMetricScene = Literal["indoor", "outdoor"]

ENCODER_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

IMAGE_ENCODERS = ("vits", "vitb", "vitl")
VIDEO_ENCODERS = ("vits", "vitb", "vitl")

ONNX_RELEASE = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0"


def _fabio(filename: str, size: str, license_name: str) -> dict[str, Any]:
    return {
        "source": "fabio-sim/Depth-Anything-ONNX@v2.0.0",
        "filename": filename,
        "url": f"{ONNX_RELEASE}/{filename}",
        "size": size,
        "license": license_name,
        "dynamic": True,
        "packaging": "single-file",
        "input": "image [B,3,H,W], H/W % 14 == 0",
        "output": "depth [B,H,W]",
    }


# Single-file dynamic ONNX. Small is the default packable weight (~95 MB).
ONNX_FABIO: dict[str, dict[str, dict[str, Any]]] = {
    "relative": {
        "vits": _fabio("depth_anything_v2_vits_dynamic.onnx", "94.5 MB", "Apache-2.0"),
        "vitb": _fabio("depth_anything_v2_vitb_dynamic.onnx", "370.9 MB", "CC-BY-NC-4.0"),
        "vitl": _fabio("depth_anything_v2_vitl_dynamic.onnx", "1275 MB", "CC-BY-NC-4.0"),
    },
    "indoor": {
        "vits": _fabio("depth_anything_v2_vits_indoor_dynamic.onnx", "94.5 MB", "Apache-2.0"),
        "vitb": _fabio("depth_anything_v2_vitb_indoor_dynamic.onnx", "370.9 MB", "CC-BY-NC-4.0"),
        "vitl": _fabio("depth_anything_v2_vitl_indoor_dynamic.onnx", "1275 MB", "CC-BY-NC-4.0"),
    },
    "outdoor": {
        "vits": _fabio("depth_anything_v2_vits_outdoor_dynamic.onnx", "94.5 MB", "Apache-2.0"),
        "vitb": _fabio("depth_anything_v2_vitb_outdoor_dynamic.onnx", "370.9 MB", "CC-BY-NC-4.0"),
        "vitl": _fabio("depth_anything_v2_vitl_outdoor_dynamic.onnx", "1275 MB", "CC-BY-NC-4.0"),
    },
}

# Smaller two-file community exports (graph + external data). Relative depth only.
ONNX_QUANTIZED: dict[str, dict[str, Any]] = {
    "vits": {
        "source": "onnx-community/depth-anything-v2-small-ONNX",
        "repo": "onnx-community/depth-anything-v2-small-ONNX",
        "files": ["onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"],
        "size": "38.6 MB",
        "license": "Apache-2.0",
        "packaging": "onnx+data",
    },
    "vitb": {
        "source": "onnx-community/depth-anything-v2-base-ONNX",
        "repo": "onnx-community/depth-anything-v2-base-ONNX",
        "files": ["onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"],
        "size": "varies",
        "license": "CC-BY-NC-4.0",
        "packaging": "onnx+data",
    },
    "vitl": {
        "source": "onnx-community/depth-anything-v2-large-ONNX",
        "repo": "onnx-community/depth-anything-v2-large-ONNX",
        "files": ["onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"],
        "size": "varies",
        "license": "CC-BY-NC-4.0",
        "packaging": "onnx+data",
    },
}

IMAGE_RELATIVE_HF: dict[str, dict[str, str]] = {
    "vits": {
        "hf_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "native_repo": "depth-anything/Depth-Anything-V2-Small",
        "native_file": "depth_anything_v2_vits.pth",
        "params": "24.8M",
        "license": "Apache-2.0",
    },
    "vitb": {
        "hf_id": "depth-anything/Depth-Anything-V2-Base-hf",
        "native_repo": "depth-anything/Depth-Anything-V2-Base",
        "native_file": "depth_anything_v2_vitb.pth",
        "params": "97.5M",
        "license": "CC-BY-NC-4.0",
    },
    "vitl": {
        "hf_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "native_repo": "depth-anything/Depth-Anything-V2-Large",
        "native_file": "depth_anything_v2_vitl.pth",
        "params": "335.3M",
        "license": "CC-BY-NC-4.0",
    },
}

IMAGE_METRIC_HF: dict[str, dict[str, str]] = {
    "indoor": {
        "vits": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "max_depth": "20m",
        "dataset": "Hypersim",
    },
    "outdoor": {
        "vits": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "max_depth": "80m",
        "dataset": "Virtual KITTI 2",
    },
}

VIDEO_WEIGHTS: dict[str, dict[str, dict[str, str]]] = {
    "relative": {
        "vits": {
            "repo": "depth-anything/Video-Depth-Anything-Small",
            "filename": "video_depth_anything_vits.pth",
            "params": "28.4M",
            "license": "Apache-2.0",
        },
        "vitb": {
            "repo": "depth-anything/Video-Depth-Anything-Base",
            "filename": "video_depth_anything_vitb.pth",
            "params": "113.1M",
            "license": "CC-BY-NC-4.0",
        },
        "vitl": {
            "repo": "depth-anything/Video-Depth-Anything-Large",
            "filename": "video_depth_anything_vitl.pth",
            "params": "381.8M",
            "license": "CC-BY-NC-4.0",
        },
    },
    "metric": {
        "vits": {
            "repo": "depth-anything/Metric-Video-Depth-Anything-Small",
            "filename": "metric_video_depth_anything_vits.pth",
            "params": "28.4M",
            "license": "Apache-2.0",
        },
        "vitb": {
            "repo": "depth-anything/Metric-Video-Depth-Anything-Base",
            "filename": "metric_video_depth_anything_vitb.pth",
            "params": "113.1M",
            "license": "CC-BY-NC-4.0",
        },
        "vitl": {
            "repo": "depth-anything/Metric-Video-Depth-Anything-Large",
            "filename": "metric_video_depth_anything_vitl.pth",
            "params": "381.8M",
            "license": "CC-BY-NC-4.0",
        },
    },
}

DAV2_REPO = "https://github.com/DepthAnything/Depth-Anything-V2.git"
VDA_REPO = "https://github.com/DepthAnything/Video-Depth-Anything.git"


def normalize_encoder(encoder: str | None, *, kind: str, default: str = "vits") -> str:
    value = (encoder or default).strip().lower()
    allowed = IMAGE_ENCODERS if kind == "image" else VIDEO_ENCODERS
    if value == "vitg":
        raise ValueError("Depth-Anything-V2-Giant weights are not released yet.")
    if value not in allowed:
        raise ValueError(f"Unsupported {kind} encoder '{encoder}'. Choose from: {', '.join(allowed)}")
    return value


def image_hf_id(encoder: str, metric_scene: str | None = None) -> str:
    if metric_scene:
        scene = metric_scene.strip().lower()
        if scene not in IMAGE_METRIC_HF:
            raise ValueError("metric_scene must be 'indoor' or 'outdoor'.")
        return IMAGE_METRIC_HF[scene][encoder]
    return IMAGE_RELATIVE_HF[encoder]["hf_id"]


def video_weight_spec(encoder: str, metric: bool) -> dict[str, str]:
    family = "metric" if metric else "relative"
    return VIDEO_WEIGHTS[family][encoder]


def normalize_metric_scene(metric_scene: str | None) -> str | None:
    if not metric_scene:
        return None
    scene = metric_scene.strip().lower()
    if scene not in {"indoor", "outdoor"}:
        raise ValueError("metric_scene must be 'indoor' or 'outdoor'.")
    return scene


def onnx_spec(encoder: str, variant: str = "dynamic", metric_scene: str | None = None) -> dict[str, Any]:
    choice = (variant or "dynamic").strip().lower()
    if choice in {"quantized", "int8", "community"}:
        if metric_scene:
            raise ValueError("Quantized onnx-community models are relative depth only.")
        return ONNX_QUANTIZED[encoder]
    if choice not in {"dynamic", "fabio", "onnx"}:
        raise ValueError("variant must be 'dynamic' (single-file) or 'quantized' (smaller two-file).")
    family = normalize_metric_scene(metric_scene) or "relative"
    return ONNX_FABIO[family][encoder]
