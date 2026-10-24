#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK="https://github.com/inaniloquentee/transformers.git"
BRANCH="ljj/qwen3-0.6b-validation"
REVISION="c4c4605f961877852b89b97478536bb3a26b6fd1"
CHECKOUT="$HERE/_upstream/transformers"

if [[ ! -d "$CHECKOUT/.git" ]]; then
  mkdir -p "$HERE/_upstream"
  git clone --filter=blob:none --depth=1 --branch "$BRANCH" "$FORK" "$CHECKOUT"
fi

ORIGIN="$(git -C "$CHECKOUT" remote get-url origin)"
if [[ "$ORIGIN" != "$FORK" ]]; then
  echo "Unexpected Transformers remote: $ORIGIN (expected $FORK)" >&2
  exit 1
fi
ACTUAL="$(git -C "$CHECKOUT" rev-parse HEAD)"
if [[ "$ACTUAL" != "$REVISION" ]]; then
  echo "Unexpected Transformers fork revision: $ACTUAL (expected $REVISION)" >&2
  exit 1
fi
printf 'Transformers fork: %s at %s\n' "$FORK" "$ACTUAL"
