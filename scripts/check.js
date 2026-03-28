const path = require('path');

const validAccelerators = process.platform === 'darwin' ? [] : ['vulkan', 'cuda'];

// macOS deployment target
//
// Keep this out of GitHub Actions workflows so local builds and CI behave the same.
// Default to 15.0 for Xcode 26+ compatibility, but allow callers to override.
if (process.platform === 'darwin') {
  if (!process.env.MACOSX_DEPLOYMENT_TARGET) {
    process.env.MACOSX_DEPLOYMENT_TARGET = '15.0'
  }
  if (!process.env.CMAKE_OSX_DEPLOYMENT_TARGET) {
    process.env.CMAKE_OSX_DEPLOYMENT_TARGET = process.env.MACOSX_DEPLOYMENT_TARGET
  }
}

const accelerator = process.env.npm_config_accelerator || '';

if (process.env.npm_config_build_from_source) {
  console.log('Build from source is enabled');
} else {
  process.exit(0);
}

if (accelerator && !validAccelerators.includes(accelerator)) {
  console.error(`Invalid accelerator: ${accelerator}`);
  process.exit(1);
}

let BuildSystem;
try {
  ({ BuildSystem } = require('cmake-js'));
} catch (error) {
  console.error('cmake-js is not installed, please install it');
  process.exit(1);
}

const buildSystem = new BuildSystem({
  directory: path.resolve(__dirname, '../'),
  arch: process.arch,
  preferClang: true,
  out: path.resolve(__dirname, '../build'),
  extraCMakeArgs: [accelerator && `--CDVARIANT=${accelerator}`].filter(Boolean),
});

buildSystem.build();
