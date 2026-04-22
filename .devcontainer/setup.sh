#!/usr/bin/env bash
set -euo pipefail

profile="${1:-cpu}"

case "$profile" in
  cpu)
    sync_args=(--frozen)
    ;;
  gpu)
    sync_args=(--frozen --group torch-gpu --no-group torch-cpu)
    ;;
  *)
    echo "Usage: /setup.sh [cpu|gpu]" >&2
    exit 1
    ;;
esac

cd /workspace

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed in the devcontainer image." >&2
  exit 1
fi

install -m 0644 /workspace/.devcontainer/.bashrc "${HOME}/.bashrc"
install -m 0644 /workspace/.devcontainer/.zshrc "${HOME}/.zshrc"

python_version="$(python3 --version | awk '{print $2}')"
if [[ "$python_version" != 3.10.* ]]; then
  echo "Python 3.10 is required, found ${python_version}." >&2
  exit 1
fi

export UV_NO_MANAGED_PYTHON=1
uv sync "${sync_args[@]}"

echo "TPU-MLIR devcontainer is ready."
echo "profile: ${profile}"
echo "python: $(/workspace/.venv/bin/python --version 2>&1)"
echo "uv: $(uv --version)"
echo "venv: /workspace/.venv"
