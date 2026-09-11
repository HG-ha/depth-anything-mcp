import numpy as np

from depth_anything_mcp.infer_onnx import infer_hw, postprocess_depth, preprocess_rgb, round_to_14, static_input_hw


def test_round_and_infer_hw() -> None:
    assert round_to_14(518) == 518
    assert infer_hw(100, 200, input_size=518) == (518, 1036)
    assert infer_hw(100, 200, static_hw=(518, 518)) == (518, 518)


def test_static_input_hw() -> None:
    assert static_input_hw([1, 3, 518, 518]) == (518, 518)
    assert static_input_hw([1, 3, "height", "width"]) is None


def test_preprocess_shape() -> None:
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    tensor = preprocess_rgb(rgb, 28, 56)
    assert tensor.shape == (1, 3, 28, 56)
    assert tensor.dtype == np.float32
    restored = postprocess_depth(np.ones((1, 28, 56), dtype=np.float32), 40, 80)
    assert restored.shape == (40, 80)
