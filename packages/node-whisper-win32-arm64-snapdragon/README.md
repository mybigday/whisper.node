# @fugood/node-whisper-win32-arm64-snapdragon

Native module for whisper.node targeting win32-arm64-snapdragon.

This package contains the pre-compiled native module for the specified platform and architecture with Qualcomm Snapdragon Hexagon NPU support (ggml-hexagon). The HTP skels (`libggml-htp-*.so`) ship next to `index.node`; whisper.node points `ADSP_LIBRARY_PATH` at this directory when the `snapdragon` variant is loaded.

## Installation

This package is typically installed automatically as a dependency of `@fugood/whisper.node`.

## Platform Support

- **OS**: win32
- **Architecture**: arm64
- **Variant**: snapdragon
- **Backends**: Hexagon NPU

## Usage

This package is not meant to be used directly. It is consumed by the main `@fugood/whisper.node` package:

```js
const { initWhisper } = require('@fugood/whisper.node')
const ctx = await initWhisper({ filePath: 'ggml-base.bin', useGpu: true }, 'snapdragon')
```
