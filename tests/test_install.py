import json
from pathlib import Path

from depth_anything_mcp.install import cursor_deeplink, mcp_server_entry, merge_mcp_file


def test_merge_mcp_keeps_other_servers(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "echo"}}}), encoding="utf-8")
    entry = mcp_server_entry(Path("python"), tmp_path)
    merge_mcp_file(path, entry)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    assert data["mcpServers"]["depth-anything"]["args"] == ["-m", "depth_anything_mcp"]
    assert data["mcpServers"]["depth-anything"]["env"]["DEPTH_ANYTHING_MIRROR"] == "cn"
    assert data["mcpServers"]["depth-anything"]["env"]["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_deeplink_roundtrip() -> None:
    entry = {"command": "python", "args": ["-m", "depth_anything_mcp"]}
    link = cursor_deeplink(entry)
    assert link.startswith("cursor://anysphere.cursor-deeplink/mcp/install?name=depth-anything&config=")
