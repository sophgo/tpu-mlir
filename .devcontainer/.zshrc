# Auto-activate the project environment when it already exists.
if [ -f /workspace/.venv/bin/activate ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  source /workspace/.venv/bin/activate
fi

# zsh does not understand bash-style PS1 escapes, so define its prompt
# explicitly instead of inheriting the prompt from a parent bash shell.
setopt PROMPT_SUBST
PROMPT='${VIRTUAL_ENV:+(%F{green}${VIRTUAL_ENV:t}%f) }%n@%m:%~%# '
RPROMPT=''

# Auto-source TPU-MLIR environment variables once per shell.
if [ -f /workspace/envsetup.sh ] && [ -z "${TPU_MLIR_ENVSETUP_SOURCED:-}" ]; then
  export TPU_MLIR_ENVSETUP_SOURCED=1
  source /workspace/envsetup.sh >/dev/null 2>&1 || true
fi

if command -v uv >/dev/null 2>&1; then
  eval "$(uv generate-shell-completion zsh)" 2>/dev/null || true
fi
