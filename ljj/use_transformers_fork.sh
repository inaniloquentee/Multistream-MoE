#!/usr/bin/env bash
# Source this file before invoking the Qwen scripts; dependencies stay in the venv.
FORK_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_ROOT="$FORK_HERE/_upstream/transformers"
FORK_REVISION="c4c4605f961877852b89b97478536bb3a26b6fd1"
if [[ ! -f "$FORK_ROOT/src/transformers/__init__.py" ]] ||
   [[ "$(git -C "$FORK_ROOT" rev-parse HEAD 2>/dev/null)" != "$FORK_REVISION" ]]; then
  echo "Run bash $FORK_HERE/bootstrap_transformers.sh first" >&2
  return 1 2>/dev/null || exit 1
fi
export TRANSFORMERS_FORK_ROOT="$FORK_ROOT"
export PYTHONPATH="$FORK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
