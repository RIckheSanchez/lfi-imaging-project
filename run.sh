#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${LFI_PYTHON:-}" ]]; then
    interpreter="$LFI_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    interpreter="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    interpreter="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/bin/python" ]]; then
    interpreter="$HOME/miniconda3/bin/python"
else
    interpreter="python3"
fi
exec "$interpreter" -B -m part4_ui.app "$@"
