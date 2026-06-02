# @fugood/node-whisper-wasm

Browser WASM package for `@fugood/whisper.node`.

The package exposes the same high-level context API as the native packages, but
model and audio file paths are fetched as URLs and copied into the Emscripten
filesystem before inference.

```js
const whisper = await WhisperNodeWasm.initWhisper({
  filePath: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin',
  useGpu: false,
})

const { promise } = whisper.transcribeFile('https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', {
  language: 'en',
})

console.log(await promise)
await whisper.release()
```

The WASM build uses pthreads, so browsers must serve the page with COOP/COEP
headers and expose `SharedArrayBuffer`. Whisper transcription defaults to up to
8 threads based on browser hardware concurrency; pass `maxThreads` to override
it. Browser pages run model loading, transcription, benchmarks, and VAD in a
dedicated worker by default so the UI thread can keep rendering. Use
`configureWasm({ worker: false })` only when you explicitly need the old
in-thread runtime, or pass `workerUrl`, `indexScriptUrl`, and `runtimeScriptUrl`
when serving the package files from custom URLs. Set `useGpu: true` only with a
package built using `GGML_WEBGPU=ON` and a browser that exposes `navigator.gpu`.
VAD currently falls back to CPU in the browser package because the Silero VAD
graph hits unsupported WebGPU ops.
