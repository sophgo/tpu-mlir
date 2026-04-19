# Auto-activate the project environment when it already exists.
if [ -f /workspace/.venv/bin/activate ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  . /workspace/.venv/bin/activate
fi

# Auto-source TPU-MLIR environment variables once per shell.
if [ -f /workspace/envsetup.sh ] && [ -z "${TPU_MLIR_ENVSETUP_SOURCED:-}" ]; then
  export TPU_MLIR_ENVSETUP_SOURCED=1
  # shellcheck disable=SC1091
  . /workspace/envsetup.sh >/dev/null 2>&1 || true
fi

if command -v uv >/dev/null 2>&1; then
  eval "$(uv generate-shell-completion bash)" 2>/dev/null || true
fi
