import numpy as np

from depth_anything_mcp.visualize import colorize_depth, depth_stats, encode_preview_png, normalize_depth


def test_normalize_and_colorize() -> None:
    depth = np.linspace(0, 10, 64, dtype=np.float32).reshape(8, 8)
    stats = depth_stats(depth)
    assert stats["min"] == 0.0
    assert stats["width"] == 8
    gray = colorize_depth(depth)
    assert gray.shape == (8, 8, 3)
    assert gray[..., 0].max() >= gray[..., 0].min()
    rgb = colorize_depth(depth, grayscale=False)
    assert rgb.shape == (8, 8, 3)
    assert not np.array_equal(gray, rgb)
    assert np.allclose(normalize_depth(np.ones((4, 4))), 0)


def test_preview_png_roundtrip() -> None:
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[:, :] = (30, 80, 200)
    data = encode_preview_png(rgb, max_edge=16)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
