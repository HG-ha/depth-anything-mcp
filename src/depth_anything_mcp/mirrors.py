from __future__ import annotations

import os

HF_MIRROR = "https://hf-mirror.com"
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_TRUSTED_HOST = "pypi.tuna.tsinghua.edu.cn"
DEFAULT_GITHUB_PROXY = "https://ghfast.top/"
GITHUB_PROXIES = (
    "https://ghfast.top/",
    "https://gh-proxy.com/",
)


def mirrors_enabled() -> bool:
    raw = os.environ.get("DEPTH_ANYTHING_MIRROR", "cn").strip().lower()
    return raw not in {"0", "off", "false", "no", "none", "official"}


def github_proxies() -> list[str]:
    custom = (os.environ.get("DEPTH_ANYTHING_GITHUB_PROXY") or "").strip()
    proxies: list[str] = []
    if custom:
        proxies.append(custom if custom.endswith("/") else custom + "/")
    if DEFAULT_GITHUB_PROXY not in proxies:
        proxies.append(DEFAULT_GITHUB_PROXY)
    for item in GITHUB_PROXIES:
        if item not in proxies:
            proxies.append(item)
    return proxies


def pypi_index() -> str:
    return (os.environ.get("DEPTH_ANYTHING_PYPI_INDEX") or PYPI_MIRROR).strip()


def hf_endpoint() -> str:
    return (
        os.environ.get("HF_ENDPOINT")
        or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
        or HF_MIRROR
    ).rstrip("/")


def is_github_url(url: str) -> bool:
    text = url.lower()
    return "github.com/" in text or "githubusercontent.com/" in text


def _already_proxied(url: str) -> bool:
    lowered = url.lower()
    return any(
        token in lowered
        for token in ("ghfast.top", "gh-proxy.com", "ghproxy", "gitclone.com", "kkgithub.com")
    )


def proxied_github_url(url: str, proxy: str | None = None) -> str:
    if not is_github_url(url) or _already_proxied(url):
        return url
    prefix = (proxy or github_proxies()[0]).rstrip("/")
    return f"{prefix}/{url}"


def download_url_candidates(url: str) -> list[str]:
    if not mirrors_enabled() or not is_github_url(url) or _already_proxied(url):
        return [url]
    seen: list[str] = []
    for proxy in github_proxies():
        candidate = proxied_github_url(url, proxy)
        if candidate not in seen:
            seen.append(candidate)
    if url not in seen:
        seen.append(url)
    return seen


def apply_download_mirrors() -> dict[str, str | bool]:
    """Set HF / UV index env vars for China. Safe to call more than once."""
    report = {
        "enabled": mirrors_enabled(),
        "hf_endpoint": os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT") or "",
        "pypi_index": "",
    }
    if not mirrors_enabled():
        return report
    if not report["hf_endpoint"]:
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        report["hf_endpoint"] = HF_MIRROR
    if not os.environ.get("UV_DEFAULT_INDEX"):
        os.environ["UV_DEFAULT_INDEX"] = pypi_index()
    report["pypi_index"] = pypi_index()
    return report


def pip_index_args() -> list[str]:
    if not mirrors_enabled():
        return []
    index = pypi_index()
    host = index.split("://", 1)[-1].split("/", 1)[0]
    return ["-i", index, "--trusted-host", host]


def uv_index_args() -> list[str]:
    if not mirrors_enabled():
        return []
    return ["--default-index", pypi_index()]
