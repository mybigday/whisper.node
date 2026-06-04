#!/bin/bash

set -euo pipefail

THREADS=${THREADS:-BOTH}
ARGS=()
HAS_WEBGPU_ARG=OFF

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS=ON
      ;;
    --no-threads)
      THREADS=OFF
      ;;
    --all-threads)
      THREADS=BOTH
      ;;
    --webgpu|--no-webgpu)
      HAS_WEBGPU_ARG=ON
      ARGS+=("$1")
      ;;
    *)
      ARGS+=("$1")
      ;;
  esac
  shift
done

if [[ "$HAS_WEBGPU_ARG" == "OFF" ]]; then
  ARGS+=("--webgpu")
fi

case "$THREADS" in
  ON)
    node scripts/build-wasm-package.js "${ARGS[@]}" --threads
    ;;
  OFF)
    node scripts/build-wasm-package.js "${ARGS[@]}" --no-threads
    ;;
  BOTH)
    node scripts/build-wasm-package.js "${ARGS[@]}" --no-threads
    node scripts/build-wasm-package.js "${ARGS[@]}" --threads
    ;;
  *)
    echo "Invalid THREADS value: $THREADS. Expected ON, OFF, or BOTH."
    exit 1
    ;;
esac
