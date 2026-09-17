#!/usr/bin/env bash

# Regenerate scripts/whisper.cpp.patch from the working-tree changes in
# src/whisper.rn/vendor/whisper.cpp.
#
# whisper.rn vendors whisper.cpp with its own patches already applied (see
# src/whisper.rn/vendor/README.md). This patch carries only the whisper.node
# specific changes on top of that tree, so it must be regenerated against a
# clean whisper.rn checkout: edit the vendored files in place, then run this.
#
# Usage: ./scripts/regenerate-patch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_RN_DIR="$PROJECT_ROOT/src/whisper.rn"
PATCH_FILE="$SCRIPT_DIR/whisper.cpp.patch"

echo "Regenerating whisper.cpp.patch..."

if [ ! -d "$WHISPER_RN_DIR/vendor/whisper.cpp" ]; then
  echo "Error: whisper.rn submodule not found at $WHISPER_RN_DIR"
  exit 1
fi

cd "$WHISPER_RN_DIR"

if git diff --quiet -- vendor/whisper.cpp; then
  echo "No changes found in src/whisper.rn/vendor/whisper.cpp"
  exit 0
fi

# Paths are relative to the whisper.node root so CMake can `git apply` from there
git diff --src-prefix=a/src/whisper.rn/ --dst-prefix=b/src/whisper.rn/ -- vendor/whisper.cpp > "$PATCH_FILE"

echo "Patch regenerated successfully at $PATCH_FILE"
