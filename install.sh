#!/usr/bin/env bash
# One-click installer: venv + package + ONNX model + Cursor MCP config.
# Usage:
#   ./install.sh
#   curl -fsSL https://raw.githubusercontent.com/HG-ha/depth-anything-mcp/main/install.sh | bash

set -euo pipefail

REPO="${DEPTH_ANYTHING_REPO:-https://github.com/HG-ha/depth-anything-mcp.git}"
VARIANT="${1:-dynamic}"
HOME_DIR="${HOME}/.depth-anything-mcp"
VENV_DIR="${HOME_DIR}/venv"
SRC_DIR="${HOME_DIR}/src"
SCRIPT_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

find_python() {
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  echo "Python 3.10+ is required." >&2
  exit 1
}

PYTHON="$(find_python)"
mkdir -p "$HOME_DIR"

if [[ -n "${SCRIPT_DIR}" && -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
  echo "Installing from local checkout: ${SCRIPT_DIR}"
  INSTALL_SPEC="${SCRIPT_DIR}"
else
  command -v git >/dev/null 2>&1 || { echo "git is required to clone ${REPO}" >&2; exit 1; }
  if [[ -f "${SRC_DIR}/pyproject.toml" ]]; then
    echo "Updating ${SRC_DIR}"
    git -C "${SRC_DIR}" pull --ff-only
  else
    echo "Cloning ${REPO}"
    rm -rf "${SRC_DIR}"
    git clone --depth 1 "${REPO}" "${SRC_DIR}"
  fi
  INSTALL_SPEC="${SRC_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating venv ${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -U pip
"${VENV_DIR}/bin/python" -m pip install -e "${INSTALL_SPEC}"
"${VENV_DIR}/bin/python" -m depth_anything_mcp.install --home "${HOME_DIR}" --python "${VENV_DIR}/bin/python" --variant "${VARIANT}"

echo
echo "Done. Reload MCP in Cursor, then try estimate_image_depth."
