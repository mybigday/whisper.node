#!/bin/bash

set -e

# macOS deployment target
#
# Keep this out of GitHub Actions workflows so local builds and CI behave the same.
# Default to 15.0 for Xcode 26+ compatibility, but allow callers to override.
: "${MACOSX_DEPLOYMENT_TARGET:=15.0}"
: "${CMAKE_OSX_DEPLOYMENT_TARGET:=${MACOSX_DEPLOYMENT_TARGET}}"
export MACOSX_DEPLOYMENT_TARGET
export CMAKE_OSX_DEPLOYMENT_TARGET

npx cmake-js rebuild -C --CDTO_PACKAGE=ON
