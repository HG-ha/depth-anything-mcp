from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def stem_of(source: str) -> str:
    if is_url(source):
        name = Path(urlparse(source).path).stem
        return name or "remote"
    return Path(source).stem or "input"


def collect_image_sources(source: str) -> list[str]:
    text = source.strip().strip('"')
    if is_url(text):
        return [text]

    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image input not found: {path}")

    if path.is_file() and path.suffix.lower() == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line and not line.startswith("#")]

    if path.is_file():
        return [str(path.resolve())]

    files = [
        str(item.resolve())
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No images found under: {path}")
    return files


def load_image_bgr(source: str) -> np.ndarray:
    if is_url(source):
        with urlopen(source, timeout=60) as response:
            data = response.read()
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as handle:
            handle.write(data)
            tmp = Path(handle.name)
        try:
            image = cv2.imdecode(np.fromfile(tmp, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            tmp.unlink(missing_ok=True)
        if image is None:
            raise ValueError(f"Failed to decode image URL: {source}")
        return image

    path = Path(source).expanduser().resolve()
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        # Fallback for some TIFF / odd encodings.
        rgb = np.array(Image.open(path).convert("RGB"))
        return rgb[:, :, ::-1]
    return image


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def resolve_video_path(source: str) -> Path:
    path = Path(source.strip().strip('"')).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video suffix '{path.suffix}'. Use: {', '.join(sorted(VIDEO_SUFFIXES))}")
    return path


def read_video_rgb(
    path: Path,
    *,
    max_len: int = -1,
    target_fps: int = -1,
    max_res: int = 1280,
) -> tuple[np.ndarray, float, dict[str, int | float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 1e-3:
        src_fps = 30.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_fps = src_fps if target_fps is None or target_fps <= 0 else float(target_fps)
    stride = max(int(round(src_fps / out_fps)), 1) if target_fps and target_fps > 0 else 1

    scale = 1.0
    width, height = src_width, src_height
    if max_res > 0 and max(src_height, src_width) > max_res:
        scale = max_res / max(src_height, src_width)
        width = max(2, int(round(src_width * scale)))
        height = max(2, int(round(src_height * scale)))

    frames: list[np.ndarray] = []
    index = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride != 0:
            index += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
        kept += 1
        index += 1
        if max_len is not None and max_len > 0 and kept >= max_len:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")

    meta = {
        "source_fps": src_fps,
        "output_fps": out_fps,
        "source_width": src_width,
        "source_height": src_height,
        "frame_width": frames[0].shape[1],
        "frame_height": frames[0].shape[0],
        "source_frame_count": total,
        "used_frames": len(frames),
        "stride": stride,
    }
    return np.stack(frames, axis=0), out_fps, meta


def write_rgb_video(path: Path, frames: np.ndarray, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB video frames with shape [N, H, W, 3].")

    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1.0)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path
