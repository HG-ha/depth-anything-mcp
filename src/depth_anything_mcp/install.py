from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from depth_anything_mcp.repo import GITHUB_REPO
from depth_anything_mcp.setup_models import setup_backends


SERVER_NAME = "depth-anything"


def default_home() -> Path:
    return Path.home() / ".depth-anything-mcp"


def cursor_mcp_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def mcp_server_entry(python_exe: Path, home: Path) -> dict:
    return {
        "command": str(python_exe.resolve()),
        "args": ["-m", "depth_anything_mcp"],
        "env": {
            "DEPTH_ANYTHING_HOME": str(home),
            "DEPTH_ANYTHING_CHECKPOINT_DIR": str(home / "checkpoints"),
            "DEPTH_ANYTHING_OUTPUT_DIR": str(home / "outputs"),
            "DEPTH_ANYTHING_DEFAULT_ENCODER": "vits",
            "DEPTH_ANYTHING_DEVICE": "auto",
        },
    }


def merge_mcp_file(path: Path, entry: dict, name: str = SERVER_NAME) -> Path:
    data: dict = {}
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                data = loaded
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[name] = entry
    data["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def cursor_deeplink(entry: dict, name: str = SERVER_NAME) -> str:
    payload = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"cursor://anysphere.cursor-deeplink/mcp/install?name={name}&config={encoded}"


def install(
    *,
    home: Path | None = None,
    python_exe: Path | None = None,
    variant: str = "dynamic",
    download_model: bool = True,
    write_mcp: bool = True,
) -> dict:
    home = (home or default_home()).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    python_exe = Path(python_exe or sys.executable).resolve()
    os.environ["DEPTH_ANYTHING_HOME"] = str(home)
    os.environ["DEPTH_ANYTHING_CHECKPOINT_DIR"] = str(home / "checkpoints")
    os.environ["DEPTH_ANYTHING_OUTPUT_DIR"] = str(home / "outputs")

    report: dict = {
        "ok": True,
        "home": str(home),
        "python": str(python_exe),
        "repo": GITHUB_REPO,
    }

    if download_model:
        model = setup_backends(download_onnx=True, encoder="vits", variant=variant)
        report["model"] = model

    entry = mcp_server_entry(python_exe, home)
    report["mcp_entry"] = entry
    report["cursor_deeplink"] = cursor_deeplink(entry)

    if write_mcp:
        report["cursor_mcp_json"] = str(merge_mcp_file(cursor_mcp_path(), entry))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Depth Anything MCP and the default ONNX model.")
    parser.add_argument("--home", default="", help="Data directory. Default: ~/.depth-anything-mcp")
    parser.add_argument("--python", default="", help="Python executable Cursor should launch.")
    parser.add_argument("--variant", default="dynamic", choices=["dynamic", "quantized"])
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-mcp", action="store_true")
    args = parser.parse_args()

    report = install(
        home=Path(args.home) if args.home else None,
        python_exe=Path(args.python) if args.python else None,
        variant=args.variant,
        download_model=not args.skip_model,
        write_mcp=not args.skip_mcp,
    )
    print("Depth Anything MCP is ready.")
    print(f"Home:     {report['home']}")
    print(f"Python:   {report['python']}")
    if report.get("cursor_mcp_json"):
        print(f"Cursor:   {report['cursor_mcp_json']}")
    print("Reload MCP servers in Cursor, or click:")
    print(report["cursor_deeplink"])
    if report.get("model"):
        print(f"ONNX dir: {report['model'].get('onnx_dir')}")


if __name__ == "__main__":
    main()
