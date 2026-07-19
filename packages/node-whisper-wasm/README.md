# @fugood/node-whisper-wasm

Browser WASM implementation package for `@fugood/whisper.node`. Application
code should import `@fugood/whisper.node`; this package is pulled in by the main
package for browser builds.

The package exposes the same high-level context API as the native packages
(`initWhisper`, `initWhisperVad`, and `initParakeet`), but model and audio file
paths are fetched as URLs and copied into the Emscripten filesystem before
inference.

```js
import { initWhisper } from '@fugood/whisper.node'

const whisper = await initWhisper({
  filePath: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin',
  useGpu: false,
})

const { promise } = whisper.transcribeFile('https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav', {
  language: 'en',
})

console.log(await promise)
await whisper.release()
```

The package ships both single-thread and pthread WASM artifacts. On
cross-origin isolated pages with `SharedArrayBuffer`, the loader uses the
pthread artifact; otherwise it falls back to the single-thread artifact
automatically. Whisper transcription defaults to up to 8 threads based on
browser hardware concurrency when pthreads are available; pass `maxThreads` to
override it. Browser WASM clamps `maxThreads` to the compiled pool limit of 8,
or 1 in the single-thread fallback. Browser pages run model loading,
transcription, benchmarks, and VAD in a dedicated module worker by default so
the UI thread can keep rendering. Use the main `whisper.node` package entrypoint
in browser code:

```js
import { configureWasm, initWhisper } from '@fugood/whisper.node'
```

Use `configureWasm({ worker: false })` only when you explicitly need the
in-thread runtime, `configureWasm({ threads: false })` to force the
single-thread artifact, or pass `workerPath`, `jsPath`, and `wasmPath` when
serving the package files from custom URLs. The older `workerUrl` and
`runtimeScriptUrl` option names still work. Model downloads are cached in
browser Cache Storage by default. Pass `cacheModel: false` to disable persistent
caching, `modelCacheName` to isolate the cache namespace, or `modelCacheKey`
when the fetch URL is a proxy or signed URL but should reuse the same cached
model. Set `useGpu: true` only with a package built using `GGML_WEBGPU=ON` and a
browser that exposes `navigator.gpu`. VAD currently falls back to CPU in the
browser package because the Silero VAD graph hits unsupported WebGPU ops.

The default build emits `wasm/whisper-node.js`, `wasm/whisper-node.wasm`,
`wasm/whisper-node.threads.js`, and `wasm/whisper-node.threads.wasm`. Use
`bash scripts/build-wasm.sh --single-file` only when you want the WASM binary
embedded into each generated JS file. Modern Emscripten embeds the pthread
worker bootstrap in the main JS file, so a separate `whisper-node.worker.js` is
not expected. The package module worker that keeps UI work off the main thread
is `worker.js`. Use `npm run build-wasm-docker` to build through Docker; it
selects a native arm64 Emscripten image on Apple Silicon hosts.
