#!/usr/bin/env bash

# Regenerate whisper.cpp.patch from current changes in the whisper.cpp submodule
# Usage: ./scripts/regenerate-patch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_DIR="$PROJECT_ROOT/whisper.cpp"
PATCH_FILE="$SCRIPT_DIR/whisper.cpp.patch"

echo "Regenerating whisper.cpp.patch..."

# Check if whisper.cpp submodule exists
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "Error: whisper.cpp submodule not found at $LLAMA_CPP_DIR"
  exit 1
fi

# Change to whisper.cpp directory
cd "$LLAMA_CPP_DIR"

# Check if there are changes
if git diff --quiet; then
  echo "No changes found in whisper.cpp submodule"
  exit 0
fi

# Generate patch with correct path prefix
git diff --src-prefix=a/whisper.cpp/ --dst-prefix=b/whisper.cpp/ > "$PATCH_FILE"

echo "Patch regenerated successfully at $PATCH_FILE"
