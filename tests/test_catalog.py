from depth_anything_mcp.catalog import image_hf_id, normalize_encoder, onnx_spec, video_weight_spec
from depth_anything_mcp.checkpoints import download_onnx_weights, find_bundled_onnx
from depth_anything_mcp.repo import UVX_FROM


def test_uvx_from_points_at_public_repo() -> None:
    assert UVX_FROM == "git+https://github.com/HG-ha/depth-anything-mcp"


def test_normalize_encoder() -> None:
    assert normalize_encoder("ViTL", kind="image") == "vitl"
    assert normalize_encoder(None, kind="video", default="vits") == "vits"


def test_default_vits_onnx_is_bundled() -> None:
    spec = onnx_spec("vits")
    bundled = find_bundled_onnx(spec)
    assert bundled is not None
    assert bundled.name == "depth_anything_v2_vits_dynamic.onnx"
    assert download_onnx_weights("vits") == bundled


def test_onnx_default_is_single_file() -> None:
    spec = onnx_spec("vits")
    assert spec["filename"] == "depth_anything_v2_vits_dynamic.onnx"
    assert spec["packaging"] == "single-file"
    assert spec["url"].endswith("depth_anything_v2_vits_dynamic.onnx")
    indoor = onnx_spec("vits", metric_scene="indoor")
    assert "indoor" in indoor["filename"]


def test_onnx_quantized() -> None:
    spec = onnx_spec("vits", "quantized")
    assert spec["packaging"] == "onnx+data"
    assert spec["files"][0].endswith("model_quantized.onnx")


def test_image_hf_ids() -> None:
    assert image_hf_id("vits").endswith("Small-hf")
    assert "Metric-Indoor" in image_hf_id("vits", "indoor")


def test_video_weight_names() -> None:
    rel = video_weight_spec("vits", False)
    met = video_weight_spec("vitl", True)
    assert rel["filename"] == "video_depth_anything_vits.pth"
    assert met["filename"] == "metric_video_depth_anything_vitl.pth"
