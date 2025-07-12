# whisper.node

[![CI](https://github.com/mybigday/whisper.node/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mybigday/whisper.node/actions/workflows/ci.yml)
[![NPM Version](https://img.shields.io/npm/v/%40fugood%2Fwhisper.node)](https://www.npmjs.com/package/@fugood/whisper.node)
![NPM Downloads](https://img.shields.io/npm/dw/%40fugood%2Fwhisper.node)

An another Node binding of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) to make same API with [whisper.rn](https://github.com/mybigday/whisper.rn) as much as possible.

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Automatic speech recognition with multi-platform support
- [whisper.rn](https://github.com/mybigday/whisper.rn): React Native binding of whisper.cpp

## Platform Support

- macOS
  - arm64: CPU and Metal GPU acceleration
  - x86_64: CPU only
- Windows (x86_64 and arm64)
  - CPU
  - GPU acceleration via Vulkan
  - GPU acceleration via CUDA (x86_64)
- Linux (x86_64 and arm64)
  - CPU
  - GPU acceleration via Vulkan
  - GPU acceleration via CUDA

## Installation

```sh
npm install @fugood/whisper.node
```

## Usage

### Basic Transcription

```js
import { initWhisper } from '@fugood/whisper.node'

const context = await initWhisper({
  model: 'path/to/ggml-base.en.bin',
  useGpu: true,
}, libVariant)

// transcribeFile returns { stop, promise }
const { stop: stop1, promise: promise1 } = context.transcribeFile('audio1.wav', {
  language: 'en',
  temperature: 0.0,
  // ...
})

const result1 = await promise1

// transcribeData also returns { stop, promise }
let audioBuffer // PCM 16-bit, mono, 16kHz
const { stop: stop2, promise: promise2 } = context.transcribeData(audioBuffer, {
  language: 'en',
  temperature: 0.0,
  // ...
})

const result2 = await promise2

// You can also cancel transcription if needed
// await stop1() // Cancels the first transcription
// await stop2() // Cancels the second transcription

// Always release the context when done
await context.release()
```

### Voice Activity Detection (VAD)

```js
import { initWhisperVad } from '@fugood/whisper.node'

// Context-based VAD (for multiple detections)
const vadContext = await initWhisperVad({
  model: 'path/to/ggml-vad.bin',
  useGpu: true,
  nThreads: 2
}, libVariant)

const result = await vadContext.detectSpeechFile('audio.wav')

const result2 = await vadContext.detectSpeechData(audioBuffer)
await vadContext.release()
```

**Note**: Audio data should be 16-bit PCM, mono, 16kHz format. The library expects ArrayBuffer containing raw audio data.

## Lib Variants

- [x] `default`: General usage, not support GPU except macOS (Metal)
- [x] `vulkan`: Support GPU Vulkan (Windows/Linux), but some scenario might unstable
- [x] `cuda`: Support GPU CUDA (Windows/Linux), but only for limited capability
  > Linux: (x86_64: 8.9, arm64: 8.7)
  > Windows: x86_64 - 12.0

## License

MIT

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
