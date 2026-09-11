from __future__ import annotations

import urllib.request
from pathlib import Path

from depth_anything_mcp.catalog import (
    IMAGE_RELATIVE_HF,
    ONNX_FABIO,
    ONNX_QUANTIZED,
    image_hf_id,
    normalize_encoder,
    onnx_spec,
    video_weight_spec,
)
from depth_anything_mcp.paths import ensure_dir
from depth_anything_mcp.settings import Settings, load_settings

BUNDLED_MODELS_DIR = Path(__file__).resolve().parent / "models"


def bundled_onnx_path(filename: str) -> Path:
    return BUNDLED_MODELS_DIR / filename


def find_bundled_onnx(spec: dict) -> Path | None:
    filename = spec.get("filename")
    if filename:
        path = bundled_onnx_path(filename)
        if checkpoint_exists(path, min_bytes=1_000_000):
            return path
    for extra in spec.get("files", []):
        path = bundled_onnx_path(Path(extra).name)
        if path.suffix == ".onnx" and checkpoint_exists(path):
            return path
    return None


def native_image_checkpoint(settings: Settings, encoder: str) -> Path:
    spec = IMAGE_RELATIVE_HF[encoder]
    return settings.checkpoint_dir / spec["native_file"]


def native_video_checkpoint(settings: Settings, encoder: str, metric: bool) -> Path:
    spec = video_weight_spec(encoder, metric)
    return settings.checkpoint_dir / spec["filename"]


def checkpoint_exists(path: Path, *, min_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    ensure_dir(dest_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(downloaded).resolve()


def download_url(url: str, dest: Path, *, min_bytes: int = 1_000_000) -> Path:
    if checkpoint_exists(dest, min_bytes=min_bytes):
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < min_bytes:
            raise RuntimeError(f"Downloaded file looks too small: {tmp} from {url}")
        tmp.replace(dest)
    finally:
        if tmp.exists() and not dest.exists():
            tmp.unlink(missing_ok=True)
    return dest


def download_onnx_quantized(encoder: str, settings: Settings) -> Path:
    spec = ONNX_QUANTIZED[encoder]
    dest_dir = settings.checkpoint_dir / spec["repo"].split("/")[-1]
    paths = [download_hf_file(spec["repo"], filename, dest_dir) for filename in spec["files"]]
    onnx_path = next(path for path in paths if path.suffix == ".onnx" and not path.name.endswith(".onnx_data"))
    return onnx_path


def download_onnx_weights(
    encoder: str = "vits",
    *,
    variant: str = "dynamic",
    metric_scene: str | None = None,
    settings: Settings | None = None,
) -> Path:
    settings = settings or load_settings()
    encoder = normalize_encoder(encoder, kind="image", default=settings.default_encoder)
    spec = onnx_spec(encoder, variant, metric_scene)
    bundled = find_bundled_onnx(spec)
    if bundled is not None:
        return bundled
    if spec.get("packaging") == "onnx+data":
        return download_onnx_quantized(encoder, settings)

    dest = settings.checkpoint_dir / spec["filename"]
    try:
        return download_url(spec["url"], dest)
    except Exception:
        if variant in {"dynamic", "fabio", "onnx"} and not metric_scene and encoder == "vits":
            return download_onnx_quantized(encoder, settings)
        raise


def download_image_native_weights(encoder: str = "vits", settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    encoder = normalize_encoder(encoder, kind="image", default=settings.default_encoder)
    spec = IMAGE_RELATIVE_HF[encoder]
    path = native_image_checkpoint(settings, encoder)
    if checkpoint_exists(path, min_bytes=1_000_000):
        return path
    return download_hf_file(spec["native_repo"], spec["native_file"], settings.checkpoint_dir)


def download_video_weights(encoder: str = "vits", metric: bool = False, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    encoder = normalize_encoder(encoder, kind="video", default=settings.default_encoder)
    spec = video_weight_spec(encoder, metric)
    path = native_video_checkpoint(settings, encoder, metric)
    if checkpoint_exists(path, min_bytes=1_000_000):
        return path
    return download_hf_file(spec["repo"], spec["filename"], settings.checkpoint_dir)


def warmup_transformers_model(encoder: str, metric_scene: str | None = None) -> str:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = image_hf_id(encoder, metric_scene)
    AutoImageProcessor.from_pretrained(model_id)
    AutoModelForDepthEstimation.from_pretrained(model_id)
    return model_id


def describe_local_weights(settings: Settings) -> dict[str, dict]:
    onnx: dict[str, dict] = {}
    for family, items in ONNX_FABIO.items():
        onnx[family] = {}
        for encoder, spec in items.items():
            path = settings.checkpoint_dir / spec["filename"]
            onnx[family][encoder] = {
                "path": str(path),
                "downloaded": checkpoint_exists(path, min_bytes=1_000_000),
                "size": spec["size"],
                "license": spec["license"],
            }

    quantized = {}
    for encoder, spec in ONNX_QUANTIZED.items():
        dest_dir = settings.checkpoint_dir / spec["repo"].split("/")[-1]
        graph = dest_dir / "onnx" / "model_quantized.onnx"
        quantized[encoder] = {
            "path": str(graph),
            "downloaded": checkpoint_exists(graph),
            "size": spec["size"],
            "license": spec["license"],
        }

    bundled = {}
    if BUNDLED_MODELS_DIR.is_dir():
        for path in sorted(BUNDLED_MODELS_DIR.glob("*.onnx")):
            bundled[path.name] = {
                "path": str(path),
                "downloaded": checkpoint_exists(path, min_bytes=1_000_000),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }

    return {
        "bundled": bundled,
        "onnx_dynamic": onnx,
        "onnx_quantized": quantized,
        "image_native_pth": {
            encoder: {
                "path": str(native_image_checkpoint(settings, encoder)),
                "downloaded": checkpoint_exists(native_image_checkpoint(settings, encoder), min_bytes=1_000_000),
            }
            for encoder in IMAGE_RELATIVE_HF
        },
    }
