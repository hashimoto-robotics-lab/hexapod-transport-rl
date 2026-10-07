#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${HEXAPOD_TRANSPORT_PYTHON:-}" ]]; then
    interpreter="$HEXAPOD_TRANSPORT_PYTHON"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
    interpreter="$project_dir/.venv/bin/python"
else
    echo 'Python environment not found. Run uv sync or set HEXAPOD_TRANSPORT_PYTHON.' >&2
    exit 1
fi
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$interpreter" -m hexapod_transport_rl.cli "$@"
