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
- Web
  - WASM
  - Optional WebGPU through `ggml-webgpu` when the WASM package is built with `GGML_WEBGPU=ON`

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

### Browser WASM

The browser package keeps the same promise-based `initWhisper` and
`initWhisperVad` entry points. In browsers, `filePath` is treated as a URL and
the model is fetched into the WASM filesystem.

```js
import { initWhisper, initWhisperVad } from '@fugood/whisper.node'

const whisper = await initWhisper({
  filePath: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin',
  maxModelBytes: 1536 * 1024 * 1024,
  useGpu: false,
})

const { promise } = whisper.transcribeFile('https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', {
  language: 'en',
  temperature: 0,
})

console.log(await promise)
await whisper.release()

const vad = await initWhisperVad({
  filePath: 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin',
  useGpu: false,
})
console.log(await vad.detectSpeechFile('https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav'))
await vad.release()
```

The WASM build uses pthreads, so the page must be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`) to expose `SharedArrayBuffer`.
Oversized model downloads fail before loading into MEMFS. Firefox is capped at
256 MiB by default; other browsers default to 75% of the configured WASM maximum
memory. Pass `maxModelBytes` only when you know the target browser can allocate
the model. Whisper transcription defaults to up to 8 threads based on browser
hardware concurrency; pass `maxThreads` to override it. Browser WASM clamps
`maxThreads` to the compiled pool limit of 8. Browser pages run model loading,
transcription, benchmarks, and VAD in a dedicated module worker by default so
the UI thread can keep rendering. Use the main `whisper.node` package entrypoint
in browser code too:

```js
import { configureWasm, initWhisper } from '@fugood/whisper.node'
```

Use `configureWasm({ worker: false })` only when you explicitly need the
in-thread runtime, or pass `workerPath`, `jsPath`, and `wasmPath` when serving
the package files from custom URLs. The older `workerUrl` and
`runtimeScriptUrl` option names still work. Model
downloads are cached in browser Cache Storage by default. Pass
`cacheModel: false` to disable persistent caching, `modelCacheName` to isolate
the cache namespace, or `modelCacheKey` when the fetch URL is a proxy or signed
URL but should reuse the same cached model.

Build the browser package with:

```sh
npm run build-wasm
```

`npm run build-wasm` enables `GGML_WEBGPU=ON` by default. Use
`bash scripts/build-wasm.sh --no-webgpu` for a CPU-only WASM build. The default
build emits `wasm/whisper-node.js` and `wasm/whisper-node.wasm`; pass
`--single-file` only when you want the WASM binary embedded into
`wasm/whisper-node.js`. Modern Emscripten embeds the pthread worker bootstrap in
the main JS file, so a separate `whisper-node.worker.js` is not expected. The
browser package also ships its own module `worker.js` wrapper for non-blocking
model load and inference. A local smoke page is available after building:

```sh
node examples/wasm/server.mjs
```

In the WASM package, `useGpu: true` enables WebGPU for whisper transcription
when the browser supports `navigator.gpu`. VAD currently falls back to CPU in
the browser package because the Silero VAD graph hits unsupported WebGPU ops.

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
