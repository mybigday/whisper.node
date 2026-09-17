#!/bin/bash

set -e

function run_as_root() {
  if [ $UID -ne 0 ]; then
    sudo -E $@
  else
    $@
  fi
}

export DEBIAN_FRONTEND=noninteractive

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

run_as_root apt-get update

if [ $TARGET == "snapdragon" ]; then
  # Snapdragon (Hexagon NPU) is cross-compiled for arm64 with GCC; the HTP
  # skels are built by the Hexagon SDK toolchain.
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache unzip gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu

  # Download and extract Hexagon SDK
  if [ ! -d "externals/Hexagon_SDK" ]; then
    echo "Downloading Hexagon SDK..."
    mkdir -p externals
    wget -O externals/Hexagon_SDK_lnx.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/$HEXAGON_SDK_VERSION/Hexagon_SDK_lnx.zip
    echo "Extracting Hexagon SDK..."
    unzip -q externals/Hexagon_SDK_lnx.zip -d externals/Hexagon_SDK
  fi

  source externals/Hexagon_SDK/Hexagon_SDK/$HEXAGON_SDK_VERSION/setup_sdk_env.source
else
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache
fi
