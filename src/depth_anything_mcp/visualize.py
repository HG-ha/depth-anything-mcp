from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def depth_stats(depth: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Depth map contains no finite values.")
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "height": int(depth.shape[-2]) if depth.ndim >= 2 else int(depth.shape[0]),
        "width": int(depth.shape[-1]) if depth.ndim >= 2 else 1,
    }


def normalize_depth(depth: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    data = np.asarray(depth, dtype=np.float32)
    lo = float(np.nanmin(data) if vmin is None else vmin)
    hi = float(np.nanmax(data) if vmax is None else vmax)
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def colorize_depth(depth: np.ndarray, *, grayscale: bool = False, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Return an RGB uint8 visualization. Uses OpenCV MAGMA (no matplotlib)."""
    norm = normalize_depth(depth, vmin=vmin, vmax=vmax)
    gray = (norm * 255.0).astype(np.uint8)
    if grayscale:
        return np.repeat(gray[..., None], 3, axis=-1)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def side_by_side(bgr: np.ndarray, depth_rgb: np.ndarray, gap: int = 50) -> np.ndarray:
    left = bgr
    right = depth_rgb[:, :, ::-1]
    if left.shape[0] != right.shape[0]:
        raise ValueError("Image and depth visualization heights do not match.")
    spacer = np.ones((left.shape[0], gap, 3), dtype=np.uint8) * 255
    return np.concatenate([left, spacer, right], axis=1)


def save_image_bgr(path: Path, bgr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")
    return path


def encode_preview_png(rgb: np.ndarray, max_edge: int = 768) -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.fromarray(rgb)
    longest = max(image.size)
    if longest > max_edge:
        scale = max_edge / longest
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
