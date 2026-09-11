# Depth Anything MCP

把 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 做成 Cursor / Claude 可用的 MCP 工具。默认用 **单文件 ONNX**（约 95MB），不依赖 PyTorch。

## 用 uvx 一键接入（推荐）

本机先装 [uv](https://docs.astral.sh/uv/)（Windows：`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`）。

把下面整段加进 Cursor 的 `~/.cursor/mcp.json`，或项目里的 `.cursor/mcp.json`，然后 Reload MCP：

```json
{
  "mcpServers": {
    "depth-anything": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/HG-ha/depth-anything-mcp",
        "depth-anything-mcp"
      ],
      "env": {
        "DEPTH_ANYTHING_HOME": "~/.depth-anything-mcp",
        "DEPTH_ANYTHING_DEFAULT_ENCODER": "vits",
        "DEPTH_ANYTHING_DEVICE": "auto"
      }
    }
  }
}
```

`uvx` 会拉仓库并启动 MCP。默认权重 `depth_anything_v2_vits_dynamic.onnx`（约 95MB，Apache-2.0）已放在包内 `src/depth_anything_mcp/models/`，不用再单独下。

只要量化版或米制模型时才会额外下载：

```powershell
uvx --from git+https://github.com/HG-ha/depth-anything-mcp depth-anything-setup --variant quantized
```

## 其它安装方式

PowerShell：

```powershell
irm https://raw.githubusercontent.com/HG-ha/depth-anything-mcp/main/install.ps1 | iex
```

macOS / Linux：

```bash
curl -fsSL https://raw.githubusercontent.com/HG-ha/depth-anything-mcp/main/install.sh | bash
```

或 pip：

```powershell
pip install git+https://github.com/HG-ha/depth-anything-mcp.git
python -m depth_anything_mcp.install
```

本地仓库：`.\install.ps1` / `./install.sh`。

## MCP 工具

| 工具 | 作用 |
|---|---|
| `estimate_image_depth` | 图片深度（默认 ONNX） |
| `estimate_video_depth` | 视频深度（默认逐帧 ONNX） |
| `list_depth_models` | 模型清单 |
| `download_depth_checkpoints` | 预下载 ONNX |
| `setup_depth_backends` | 下载默认权重 |
| `get_depth_runtime_status` | Runtime 状态 |

- 图片：`image_path` 指向本地文件或 URL
- 更小包：`variant=quantized`
- 室内米制：`metric_scene=indoor`
- 视频冒烟：`max_len=32`

官方 Video Depth Anything 没有可打包的正式 ONNX，视频默认按帧跑 DA-V2。

## 模型

| 文件 | 大小 | 来源 |
|---|---|---|
| `depth_anything_v2_vits_dynamic.onnx`（已随仓库） | 94.5 MB | [fabio-sim/Depth-Anything-ONNX v2.0.0](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0) |
| `model_quantized.onnx` + data | 38.6 MB | [onnx-community](https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX) |

Small 为 Apache-2.0。Base / Large 为 CC-BY-NC-4.0。

## GPU

`DEPTH_ANYTHING_DEVICE=auto`（默认）会在第一次启动时自动换 Runtime，用户不用自己对 CUDA 版本：

- **Windows**：装 `onnxruntime-directml`，NVIDIA / AMD / Intel 都能走 GPU，不看 CUDA
- **Linux + NVIDIA**：装 CUDA 12 档的 `onnxruntime-gpu[cuda,cudnn]`（ORT `>=1.21,<1.27`，避免 1.27+ 默认 CUDA 13）
- 失败或没有独显：回退 CPU

也可以在工具参数或环境变量里写 `cpu` / `dml` / `cuda` / `cuda:1`。强制 CPU：

```json
"DEPTH_ANYTHING_DEVICE": "cpu"
```

第一次走 GPU 时会换掉 CPU 版 `onnxruntime` 并拉对应 wheel，可能要一两分钟。失败会回退 CPU，日志在 stderr。

## 开发

```powershell
pip install -e ".[dev]"
python -m pytest
```

## 许可

本仓库代码 Apache-2.0。模型权重以官方卡片为准。
