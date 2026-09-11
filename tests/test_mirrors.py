from depth_anything_mcp.mirrors import (
    DEFAULT_GITHUB_PROXY,
    download_url_candidates,
    mirrors_enabled,
    pip_index_args,
    proxied_github_url,
)
from depth_anything_mcp.repo import UVX_FROM, UVX_FROM_CN


def test_uvx_from_cn_uses_github_proxy() -> None:
    assert UVX_FROM == "git+https://github.com/HG-ha/depth-anything-mcp"
    assert UVX_FROM_CN.startswith("git+https://ghfast.top/https://github.com/HG-ha/depth-anything-mcp")


def test_proxied_github_url() -> None:
    url = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/a.onnx"
    assert proxied_github_url(url, DEFAULT_GITHUB_PROXY).startswith("https://ghfast.top/https://github.com/")
    assert proxied_github_url("https://hf-mirror.com/x") == "https://hf-mirror.com/x"


def test_candidates_prefer_proxy_when_cn(monkeypatch) -> None:
    monkeypatch.setenv("DEPTH_ANYTHING_MIRROR", "cn")
    monkeypatch.delenv("DEPTH_ANYTHING_GITHUB_PROXY", raising=False)
    url = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/a.onnx"
    candidates = download_url_candidates(url)
    assert candidates[0].startswith("https://ghfast.top/")
    assert url in candidates


def test_official_mirror_keeps_github(monkeypatch) -> None:
    monkeypatch.setenv("DEPTH_ANYTHING_MIRROR", "off")
    url = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/a.onnx"
    assert download_url_candidates(url) == [url]
    assert mirrors_enabled() is False


def test_pip_index_args_cn(monkeypatch) -> None:
    monkeypatch.setenv("DEPTH_ANYTHING_MIRROR", "cn")
    args = pip_index_args()
    assert args[:2] == ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
