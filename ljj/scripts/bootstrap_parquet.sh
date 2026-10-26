#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ljj/vllm-pr-59863/.venv/bin/python3}"
VERSION="21.0.0"
mkdir -p "$HERE/_deps" "$HERE/cache/pip-tmp"
if PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
  "import pyarrow; assert pyarrow.__version__ == '$VERSION'" 2>/dev/null; then
  exit 0
fi
TMPDIR="$HERE/cache/pip-tmp" "$PYTHON" -m pip install \
  --no-cache-dir --no-deps --target "$HERE/_deps" "pyarrow==$VERSION"
PYTHONPATH="$HERE/_deps${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
  "import pyarrow; assert pyarrow.__version__ == '$VERSION'"
