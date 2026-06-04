#!/bin/bash

set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build/wasm}
WEBGPU=${WEBGPU:-ON}
WEBGPU_JSPI=${WEBGPU_JSPI:-OFF}
SINGLE_FILE=${SINGLE_FILE:-OFF}
BUILD_TYPE=${BUILD_TYPE:-Release}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift
      ;;
    --webgpu)
      WEBGPU=ON
      ;;
    --no-webgpu)
      WEBGPU=OFF
      ;;
    --webgpu-jspi)
      WEBGPU_JSPI=ON
      ;;
    --no-webgpu-jspi)
      WEBGPU_JSPI=OFF
      ;;
    --single-file)
      SINGLE_FILE=ON
      ;;
    --no-single-file)
      SINGLE_FILE=OFF
      ;;
    --debug)
      BUILD_TYPE=Debug
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
  shift
done

if ! command -v emcmake >/dev/null 2>&1; then
  echo "emcmake is required. Install and activate Emscripten before building WASM."
  exit 1
fi

mkdir -p packages/node-whisper-wasm/wasm
rm -f packages/node-whisper-wasm/wasm/whisper-node.js \
  packages/node-whisper-wasm/wasm/whisper-node.wasm \
  packages/node-whisper-wasm/wasm/whisper-node.worker.js \
  packages/node-whisper-wasm/whisper-node.js \
  packages/node-whisper-wasm/whisper-node.wasm \
  packages/node-whisper-wasm/whisper-node.worker.js

emcmake cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DWHISPER_NODE_WASM=ON \
  -DWHISPER_NODE_WASM_SINGLE_FILE="$SINGLE_FILE" \
  -DGGML_WEBGPU="$WEBGPU" \
  -DGGML_WEBGPU_JSPI="$WEBGPU_JSPI"

cmake --build "$BUILD_DIR" --target whisper-node-wasm --parallel
