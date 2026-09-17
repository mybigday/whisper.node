#!/bin/bash

set -e

# General

ARCH=${ARCH:-$(uname -m)}
TARGET=${TARGET:-"default"}

HEXAGON_SDK_VERSION=${HEXAGON_SDK_VERSION:-"6.4.0.2"}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--arch) ARCH="$2"; shift ;;
    -t|--target) TARGET="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done


if [ $ARCH == "x86_64" ]; then
  # default
  if [ $TARGET == "all" ] || [ $TARGET == "default" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON --CDCLANG_USE_GOMP=ON
  fi

  # vulkan
  if [ $TARGET == "all" ] || [ $TARGET == "vulkan" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON  --CDCLANG_USE_GOMP=ON --CDGGML_VULKAN=1 --CDVARIANT=vulkan
  fi

  # cuda
  if [ $TARGET == "all" ] || [ $TARGET == "cuda" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=89 # > GeForce RTX 40 series
  fi
else
  # default
  if [ $TARGET == "all" ] || [ $TARGET == "default" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON --CDCLANG_USE_GOMP=ON --CDGGML_NATIVE=OFF \
      --CDGGML_OPENMP=0 \
      --CDGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
  fi

  # vulkan
  if [ $TARGET == "all" ] || [ $TARGET == "vulkan" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_OPENMP=0 \
      --CDGGML_NATIVE=OFF \
      --CDGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      --CDGGML_VULKAN=1 \
      --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" \
      --CDVARIANT=vulkan
  fi

  # cuda
  if [ $TARGET == "all" ] || [ $TARGET == "cuda" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_OPENMP=0 \
      --CDGGML_NATIVE=OFF \
      --CDGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      --CDGGML_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=87 # > Jetson Orin series
  fi

  # snapdragon (Hexagon NPU; whisper.rn does not enable OpenCL)
  if [ $TARGET == "all" ] || [ $TARGET == "snapdragon" ]; then
    if [ $ARCH != "arm64" ]; then
      echo "Snapdragon is only supported on arm64" >&2
      exit 1
    fi

    source externals/Hexagon_SDK/Hexagon_SDK/$HEXAGON_SDK_VERSION/setup_sdk_env.source

    ARGS=(
      -a $ARCH \
      --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_NATIVE=OFF \
      --CDGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      --CDGGML_OPENMP=0 \
      --CDGGML_HEXAGON=1 \
      --CDHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
      --CDPREBUILT_LIB_DIR=UbuntuARM_aarch64 \
      --CDVARIANT=snapdragon \
    )

    # Check if cross-compilation is needed
    if [ $(uname -m) == "x86_64" ] && [ $ARCH == "arm64" ]; then
      echo "Cross-compiling for arm64 using GCC..."
      ARGS+=( --CDCMAKE_TOOLCHAIN_FILE="$(realpath cmake/aarch64-linux-gnu.toolchain.cmake)" )
    else
      ARGS=( -C "${ARGS[@]}" )
    fi
    npx cmake-js rebuild "${ARGS[@]}"
  fi
fi
