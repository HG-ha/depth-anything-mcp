"""Canonical public repository URL."""

from depth_anything_mcp.mirrors import DEFAULT_GITHUB_PROXY, proxied_github_url

GITHUB_REPO = "https://github.com/HG-ha/depth-anything-mcp"
GITHUB_REPO_GIT = f"{GITHUB_REPO}.git"
UVX_FROM = f"git+{GITHUB_REPO}"
UVX_FROM_CN = f"git+{proxied_github_url(GITHUB_REPO, DEFAULT_GITHUB_PROXY)}"
