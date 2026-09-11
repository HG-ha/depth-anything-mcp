from depth_anything_mcp.server import list_depth_models, mcp


def test_tools_registered() -> None:
    tools = getattr(mcp, "_tool_manager").list_tools()
    names = {tool.name for tool in tools}
    assert {
        "estimate_image_depth",
        "estimate_video_depth",
        "list_depth_models",
        "download_depth_checkpoints",
        "setup_depth_backends",
        "get_depth_runtime_status",
    } <= names


def test_list_models_json() -> None:
    payload = list_depth_models()
    assert "depth_anything_v2_vits_dynamic.onnx" in payload
    assert "fabio-sim/Depth-Anything-ONNX" in payload
