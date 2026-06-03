(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(function () {
      return require('./whisper-node.js')
    }, root)
  } else {
    root.WhisperNodeWasm = factory(function () {
      return root.createWhisperNodeModule
    }, root)
  }
})(
  typeof globalThis !== 'undefined'
    ? globalThis
    : typeof self !== 'undefined'
      ? self
      : typeof window !== 'undefined'
        ? window
        : this,
  function (loadModuleFactory, root) {
    'use strict'

    var SAMPLE_RATE = 16000
    var MIB = 1024 * 1024
    var FIREFOX_MODEL_LIMIT_BYTES = 256 * MIB
    var MODEL_MEMORY_RATIO = 0.75
    var MAX_WASM_THREADS = 8
    var DEFAULT_MODEL_CACHE_NAME = 'whisper.node.wasm.models'

    var runtimePromise = null
    var runtimeOptions = {}
    var workerProxyPromise = null
    var capturedScriptUrl = getCurrentScriptUrl()
    var modelCache = Object.create(null)
    var logEnabled = false
    var logListeners = []
    var nativeLogCallback = null

    function configureWasm(options) {
      if (runtimePromise || workerProxyPromise) {
        throw new Error('configureWasm must be called before the WASM runtime is loaded')
      }
      runtimeOptions = Object.assign({}, runtimeOptions, options || {})
    }

    function getCurrentScriptUrl() {
      if (
        root.document &&
        root.document.currentScript &&
        root.document.currentScript.src
      ) {
        return root.document.currentScript.src
      }
      return null
    }

    function isBrowserLike() {
      return typeof root.window !== 'undefined' || typeof root.importScripts === 'function'
    }

    function isMainBrowserThread() {
      return (
        typeof root.window !== 'undefined' &&
        root.window === root &&
        typeof root.Worker === 'function'
      )
    }

    function resolveUrl(value, base) {
      try {
        return new URL(value, base || (root.location && root.location.href)).href
      } catch (_) {
        return value
      }
    }

    function getIndexScriptUrl() {
      var configured = runtimeOptions.indexScriptUrl || runtimeOptions.scriptUrl
      if (configured) {
        return resolveUrl(configured)
      }
      return capturedScriptUrl
    }

    function getRuntimeScriptUrl(indexScriptUrl) {
      var configured = runtimeOptions.runtimeScriptUrl
      if (configured) {
        return resolveUrl(configured, indexScriptUrl)
      }
      return indexScriptUrl ? resolveUrl('whisper-node.js', indexScriptUrl) : null
    }

    function getWorkerScriptUrl(indexScriptUrl) {
      var configured = runtimeOptions.workerUrl
      if (configured) {
        return resolveUrl(configured, indexScriptUrl)
      }
      return indexScriptUrl ? resolveUrl('worker.js', indexScriptUrl) : null
    }

    function shouldUseWorker(options) {
      return (
        isMainBrowserThread() &&
        runtimeOptions.worker !== false &&
        (!options || options.worker !== false) &&
        !!getIndexScriptUrl()
      )
    }

    function getWorkerRuntimeOptions() {
      var blocked = {
        worker: true,
        workerUrl: true,
        indexScriptUrl: true,
        scriptUrl: true,
        runtimeScriptUrl: true,
        locateFileBaseUrl: true,
        locateFile: true,
        print: true,
        printErr: true,
        mainScriptUrlOrBlob: true,
      }
      var options = {}
      Object.keys(runtimeOptions).forEach(function (key) {
        var value = runtimeOptions[key]
        if (!blocked[key] && typeof value !== 'function' && value !== undefined) {
          options[key] = value
        }
      })
      return options
    }

    function assertThreadSupport() {
      if (isBrowserLike() && typeof root.SharedArrayBuffer === 'undefined') {
        throw new Error(
          'whisper.node WASM is built with pthreads and requires SharedArrayBuffer. Serve the page with COOP/COEP headers so the browser is cross-origin isolated.',
        )
      }
    }

    function emitLog(level, text) {
      if (!logEnabled) {
        return
      }
      if (typeof nativeLogCallback === 'function') {
        nativeLogCallback(level, text)
      }
      logListeners.slice().forEach(function (listener) {
        listener(level, text)
      })
    }

    function loadRuntime() {
      if (!runtimePromise) {
        assertThreadSupport()

        var moduleFactory = loadModuleFactory()
        if (moduleFactory && moduleFactory.default) {
          moduleFactory = moduleFactory.default
        }
        if (typeof moduleFactory !== 'function') {
          throw new Error(
            'Failed to load whisper.node WASM runtime. Make sure whisper-node.js is built and loaded before index.js.',
          )
        }

        var options = Object.assign({}, runtimeOptions)
        var userPrint = options.print
        var userPrintErr = options.printErr

        options.noInitialRun = true
        options.print = function (text) {
          emitLog('INFO', String(text))
          if (typeof userPrint === 'function') {
            userPrint(text)
          }
        }
        options.printErr = function (text) {
          emitLog('ERROR', String(text))
          if (typeof userPrintErr === 'function') {
            userPrintErr(text)
          }
        }

        runtimePromise = Promise.resolve(moduleFactory(options))
      }

      return runtimePromise
    }

    function toggleNativeLog(enable, callback) {
      logEnabled = !!enable
      nativeLogCallback = typeof callback === 'function' ? callback : null
      return Promise.resolve()
    }

    function addNativeLogListener(listener) {
      logListeners.push(listener)
      return {
        remove: function () {
          var index = logListeners.indexOf(listener)
          if (index >= 0) {
            logListeners.splice(index, 1)
          }
        },
      }
    }

    function rejectAllWorkerRequests(proxy, error) {
      Object.keys(proxy.pending).forEach(function (id) {
        proxy.pending[id].reject(error)
        delete proxy.pending[id]
      })
    }

    function createWorkerProxy() {
      var indexScriptUrl = getIndexScriptUrl()
      var runtimeScriptUrl = getRuntimeScriptUrl(indexScriptUrl)
      var workerScriptUrl = getWorkerScriptUrl(indexScriptUrl)

      if (!indexScriptUrl || !runtimeScriptUrl || !workerScriptUrl) {
        return null
      }

      var worker = new root.Worker(workerScriptUrl, {
        name: 'whisper.node.wasm',
      })
      var proxy = {
        worker: worker,
        nextRequestId: 1,
        pending: Object.create(null),
        failed: null,
      }

      worker.onmessage = function (event) {
        var message = event.data || {}
        if (message.type === 'log') {
          emitLog(message.level || 'INFO', String(message.text || ''))
          return
        }

        if (message.type === 'callback') {
          var pendingCallback = proxy.pending[message.id]
          var callback =
            pendingCallback &&
            pendingCallback.callbacks &&
            pendingCallback.callbacks[message.name]
          if (typeof callback === 'function') {
            callback(message.value)
          }
          return
        }

        if (message.type !== 'response') {
          return
        }

        var pending = proxy.pending[message.id]
        if (!pending) {
          return
        }
        delete proxy.pending[message.id]

        if (message.error) {
          var error = new Error(message.error.message || 'WASM worker failed')
          if (message.error.stack) {
            error.stack = message.error.stack
          }
          pending.reject(error)
        } else {
          pending.resolve(message.result)
        }
      }

      worker.onerror = function (event) {
        var error = new Error(
          event && event.message ? event.message : 'WASM worker failed',
        )
        proxy.failed = error
        workerProxyPromise = null
        rejectAllWorkerRequests(proxy, error)
      }

      proxy.requestOperation = function (method, args, transfer, callbacks) {
        var id = proxy.nextRequestId++
        if (proxy.failed) {
          return {
            id: id,
            promise: Promise.reject(proxy.failed),
          }
        }
        var promise = new Promise(function (resolve, reject) {
          proxy.pending[id] = {
            resolve: resolve,
            reject: reject,
            callbacks: callbacks || {},
          }
          worker.postMessage(
            {
              type: 'request',
              id: id,
              method: method,
              args: args || [],
            },
            transfer || [],
          )
        })
        return {
          id: id,
          promise: promise,
        }
      }

      proxy.request = function (method, args, transfer, callbacks) {
        return proxy.requestOperation(method, args, transfer, callbacks).promise
      }

      proxy.cancel = function (id) {
        worker.postMessage({
          type: 'cancel',
          id: id,
        })
      }

      proxy.ready = proxy.request('__init', [
        {
          indexScriptUrl: indexScriptUrl,
          runtimeScriptUrl: runtimeScriptUrl,
          runtimeOptions: getWorkerRuntimeOptions(),
          locateFileBaseUrl:
            runtimeOptions.locateFileBaseUrl || resolveUrl('.', runtimeScriptUrl),
        },
      ])

      return proxy
    }

    async function getWorkerProxy() {
      if (!shouldUseWorker()) {
        return null
      }

      if (!workerProxyPromise) {
        workerProxyPromise = (async function () {
          var proxy = createWorkerProxy()
          if (!proxy) {
            return null
          }
          await proxy.ready
          return proxy
        })().catch(function (error) {
          workerProxyPromise = null
          emitLog(
            'WARN',
            'Falling back to main-thread WASM because the worker failed to start: ' +
              (error && error.message ? error.message : String(error)),
          )
          return null
        })
      }

      return workerProxyPromise
    }

    function getFetch() {
      if (typeof root.fetch === 'function') {
        return root.fetch.bind(root)
      }
      throw new Error('fetch is required to load models or audio by URL')
    }

    function getCacheStorage() {
      return root.caches || null
    }

    function isModelCacheEnabled(options) {
      return !options || options.cacheModel !== false
    }

    function getModelCacheName(options) {
      return (
        (options && options.modelCacheName) ||
        runtimeOptions.modelCacheName ||
        DEFAULT_MODEL_CACHE_NAME
      )
    }

    function getModelCacheKey(source, options) {
      var key = (options && options.modelCacheKey) || source
      try {
        return new URL(key, root.location && root.location.href).href
      } catch (_) {
        return key
      }
    }

    function canUseModelCache(options) {
      return (
        isModelCacheEnabled(options) &&
        !!getCacheStorage() &&
        typeof root.Response === 'function'
      )
    }

    function formatBytes(bytes) {
      if (bytes >= 1024 * MIB) {
        return (bytes / (1024 * MIB)).toFixed(2) + ' GiB'
      }
      return (bytes / MIB).toFixed(2) + ' MiB'
    }

    function isFirefox() {
      return !!(
        root.navigator &&
        typeof root.navigator.userAgent === 'string' &&
        root.navigator.userAgent.indexOf('Firefox/') >= 0
      )
    }

    function getModelSizeLimit(runtime, options) {
      if (options && Number.isFinite(options.maxModelBytes) && options.maxModelBytes > 0) {
        return options.maxModelBytes
      }

      var wasmLimit =
        runtime && typeof runtime.__wasm_maximum_memory_bytes === 'function'
          ? runtime.__wasm_maximum_memory_bytes()
          : 2000 * MIB
      var limit = Math.floor(wasmLimit * MODEL_MEMORY_RATIO)

      if (isFirefox()) {
        limit = Math.min(limit, FIREFOX_MODEL_LIMIT_BYTES)
      }

      return limit
    }

    function assertModelSize(size, limit, source) {
      if (size > limit) {
        throw new Error(
          'Whisper model ' +
            source +
            ' is ' +
            formatBytes(size) +
            ', which exceeds the WASM model size limit of ' +
            formatBytes(limit) +
            '. Use a smaller or quantized model, or pass maxModelBytes only if this browser can allocate it.',
        )
      }
    }

    function hashString(value) {
      var hash = 2166136261
      for (var i = 0; i < value.length; i += 1) {
        hash ^= value.charCodeAt(i)
        hash = Math.imul(hash, 16777619)
      }
      return (hash >>> 0).toString(16)
    }

    function basenameFromUrl(source) {
      var path = source
      try {
        path = new URL(source, root.location && root.location.href).pathname
      } catch (_) {}
      var name = path.split('/').filter(Boolean).pop() || 'model.bin'
      try {
        name = decodeURIComponent(name)
      } catch (_) {}
      return name.replace(/[^a-zA-Z0-9._-]/g, '_') || 'model.bin'
    }

    function mkdirp(FS, path) {
      var current = ''
      path
        .split('/')
        .filter(Boolean)
        .forEach(function (part) {
          current += '/' + part
          try {
            FS.mkdir(current)
          } catch (_) {}
        })
    }

    function fsPathExists(FS, path) {
      try {
        return FS.analyzePath(path).exists
      } catch (_) {
        return false
      }
    }

    async function fetchArrayBuffer(source, limit) {
      var response = await getFetch()(source)
      if (!response.ok) {
        throw new Error('Failed to fetch ' + source + ': HTTP ' + response.status)
      }

      var contentLength = Number(response.headers.get('content-length') || 0)
      if (contentLength > 0 && limit) {
        assertModelSize(contentLength, limit, source)
      }

      if (limit && response.body && typeof response.body.getReader === 'function') {
        var reader = response.body.getReader()
        var chunks = []
        var total = 0

        while (true) {
          var next = await reader.read()
          if (next.done) {
            break
          }
          total += next.value.byteLength
          assertModelSize(total, limit, source)
          chunks.push(next.value)
        }

        var bytes = new Uint8Array(total)
        var offset = 0
        chunks.forEach(function (chunk) {
          bytes.set(chunk, offset)
          offset += chunk.byteLength
        })
        return bytes.buffer
      }

      var buffer = await response.arrayBuffer()
      if (limit) {
        assertModelSize(buffer.byteLength, limit, source)
      }
      return buffer
    }

    async function readCachedModel(source, limit, options) {
      if (!canUseModelCache(options)) {
        return null
      }

      var cacheName = getModelCacheName(options)
      var cacheKey = getModelCacheKey(source, options)

      try {
        var cache = await getCacheStorage().open(cacheName)
        var response = await cache.match(cacheKey)
        if (!response) {
          return null
        }

        var buffer = await response.arrayBuffer()
        if (limit) {
          assertModelSize(buffer.byteLength, limit, source)
        }
        emitLog('INFO', 'Loaded cached WASM model: ' + source)
        return {
          buffer: buffer,
          cacheHit: true,
          cacheStored: false,
          cacheName: cacheName,
          cacheKey: cacheKey,
        }
      } catch (error) {
        emitLog(
          'WARN',
          'Failed to read cached WASM model ' +
            source +
            ': ' +
            (error && error.message ? error.message : String(error)),
        )
        return null
      }
    }

    async function writeCachedModel(source, buffer, options) {
      if (!canUseModelCache(options)) {
        return {
          cacheStored: false,
          cacheName: null,
          cacheKey: null,
        }
      }

      var cacheName = getModelCacheName(options)
      var cacheKey = getModelCacheKey(source, options)

      try {
        var cache = await getCacheStorage().open(cacheName)
        await cache.put(
          cacheKey,
          new root.Response(buffer.slice(0), {
            headers: {
              'content-type': 'application/octet-stream',
            },
          }),
        )
        emitLog('INFO', 'Cached WASM model download: ' + source)
        return {
          cacheStored: true,
          cacheName: cacheName,
          cacheKey: cacheKey,
        }
      } catch (error) {
        emitLog(
          'WARN',
          'Failed to cache WASM model ' +
            source +
            ': ' +
            (error && error.message ? error.message : String(error)),
        )
        return {
          cacheStored: false,
          cacheName: cacheName,
          cacheKey: cacheKey,
        }
      }
    }

    async function fetchModelArrayBuffer(source, limit, options) {
      var cached = await readCachedModel(source, limit, options)
      if (cached) {
        return cached
      }

      var buffer = await fetchArrayBuffer(source, limit)
      var cache = await writeCachedModel(source, buffer, options)
      return {
        buffer: buffer,
        cacheHit: false,
        cacheStored: cache.cacheStored,
        cacheName: cache.cacheName,
        cacheKey: cache.cacheKey,
      }
    }

    async function ensureModel(runtime, source, kind, options) {
      if (!source) {
        throw new Error('Model path is required')
      }

      if (source[0] === '/' && fsPathExists(runtime.FS, source)) {
        return { virtualPath: source, bytes: 0 }
      }

      var cacheKey = kind + ':' + source
      if (!modelCache[cacheKey]) {
        modelCache[cacheKey] = (async function () {
          var limit = getModelSizeLimit(runtime, options)
          var loaded = await fetchModelArrayBuffer(source, limit, options)
          var bytes = new Uint8Array(loaded.buffer)
          var virtualPath =
            '/models/' + kind + '-' + hashString(source) + '-' + basenameFromUrl(source)

          mkdirp(runtime.FS, '/models')
          runtime.FS.writeFile(virtualPath, bytes)

          return {
            virtualPath: virtualPath,
            bytes: bytes.byteLength,
            cacheHit: loaded.cacheHit,
            cacheStored: loaded.cacheStored,
            cacheName: loaded.cacheName,
            cacheKey: loaded.cacheKey,
          }
        })()
      }

      return modelCache[cacheKey]
    }

    function unwrapWasmResult(result) {
      if (!result || result.ok === false) {
        throw new Error((result && result.error) || 'WASM operation failed')
      }

      var unwrapped = {}
      Object.keys(result).forEach(function (key) {
        if (key !== 'ok') {
          unwrapped[key] = normalizeWasmValue(result[key])
        }
      })
      return unwrapped
    }

    function normalizeWasmValue(value) {
      if (typeof value === 'bigint') {
        return Number(value)
      }

      if (Array.isArray(value)) {
        return value.map(normalizeWasmValue)
      }

      if (value && typeof value === 'object') {
        var normalized = {}
        Object.keys(value).forEach(function (key) {
          normalized[key] = normalizeWasmValue(value[key])
        })
        return normalized
      }

      return value
    }

    function normalizeMaxThreads(value) {
      var nThreads = Number(value)
      if (!Number.isFinite(nThreads) || nThreads <= 0) {
        return null
      }
      return Math.max(1, Math.min(MAX_WASM_THREADS, Math.floor(nThreads)))
    }

    function normalizeThreadOptions(options) {
      var normalized = Object.assign({}, options || {})
      var maxThreads = normalizeMaxThreads(normalized.maxThreads)
      if (maxThreads) {
        normalized.maxThreads = maxThreads
      } else {
        delete normalized.maxThreads
      }
      return normalized
    }

    function normalizeTranscribeOptions(options) {
      var normalized = normalizeThreadOptions(options)
      if (typeof normalized.onProgress !== 'function') {
        delete normalized.onProgress
      }
      if (typeof normalized.onNewSegments !== 'function') {
        delete normalized.onNewSegments
      }
      return normalized
    }

    function splitTranscribeOptions(options) {
      var normalized = normalizeThreadOptions(options)
      var callbacks = {}

      if (typeof normalized.onProgress === 'function') {
        callbacks.onProgress = normalized.onProgress
        normalized.onProgress = true
      } else {
        delete normalized.onProgress
      }

      if (typeof normalized.onNewSegments === 'function') {
        callbacks.onNewSegments = normalized.onNewSegments
        normalized.onNewSegments = true
      } else {
        delete normalized.onNewSegments
      }

      return {
        options: normalized,
        callbacks: callbacks,
      }
    }

    function abortedResult() {
      return {
        result: '',
        segments: [],
        isAborted: true,
      }
    }

    function defer(fn) {
      return new Promise(function (resolve, reject) {
        var run = function () {
          Promise.resolve().then(fn).then(resolve, reject)
        }
        if (typeof root.setTimeout === 'function') {
          root.setTimeout(run, 0)
        } else {
          Promise.resolve().then(run)
        }
      })
    }

    function sliceViewBuffer(view) {
      return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
    }

    function pcm16ToFloat32(buffer) {
      if (buffer.byteLength % 2 !== 0) {
        throw new Error('Audio buffer size must be even for 16-bit PCM')
      }

      var input = new Int16Array(buffer)
      var output = new Float32Array(input.length)
      for (var i = 0; i < input.length; i += 1) {
        output[i] = input[i] / 32768
      }
      return output
    }

    function toFloat32Audio(input) {
      if (input instanceof Float32Array) {
        return input
      }
      if (input instanceof Int16Array) {
        return pcm16ToFloat32(sliceViewBuffer(input))
      }
      if (input instanceof ArrayBuffer) {
        return pcm16ToFloat32(input)
      }
      if (ArrayBuffer.isView(input)) {
        return pcm16ToFloat32(sliceViewBuffer(input))
      }
      throw new TypeError('Expected ArrayBuffer or typed audio array')
    }

    function copyFloat32Audio(input) {
      var audio = toFloat32Audio(input)
      var copy = new Float32Array(audio.length)
      copy.set(audio)
      return copy
    }

    function readAscii(view, offset, length) {
      var value = ''
      for (var i = 0; i < length; i += 1) {
        value += String.fromCharCode(view.getUint8(offset + i))
      }
      return value
    }

    function readInt24(view, offset, littleEndian) {
      var b0 = view.getUint8(offset)
      var b1 = view.getUint8(offset + 1)
      var b2 = view.getUint8(offset + 2)
      var value = littleEndian ? b0 | (b1 << 8) | (b2 << 16) : b2 | (b1 << 8) | (b0 << 16)
      return value & 0x800000 ? value | 0xff000000 : value
    }

    function resampleLinear(input, sourceRate, targetRate) {
      if (sourceRate === targetRate || input.length === 0) {
        return input
      }

      var outputLength = Math.max(1, Math.round((input.length * targetRate) / sourceRate))
      var output = new Float32Array(outputLength)
      var ratio = sourceRate / targetRate

      for (var i = 0; i < outputLength; i += 1) {
        var position = i * ratio
        var left = Math.floor(position)
        var right = Math.min(left + 1, input.length - 1)
        var weight = position - left
        output[i] = input[left] * (1 - weight) + input[right] * weight
      }

      return output
    }

    function decodeWav(buffer) {
      if (buffer.byteLength < 44) {
        return null
      }

      var view = new DataView(buffer)
      if (readAscii(view, 0, 4) !== 'RIFF' || readAscii(view, 8, 4) !== 'WAVE') {
        return null
      }

      var offset = 12
      var format = null
      var dataOffset = 0
      var dataSize = 0

      while (offset + 8 <= view.byteLength) {
        var chunkId = readAscii(view, offset, 4)
        var chunkSize = view.getUint32(offset + 4, true)
        offset += 8

        if (chunkId === 'fmt ') {
          format = {
            audioFormat: view.getUint16(offset, true),
            channels: view.getUint16(offset + 2, true),
            sampleRate: view.getUint32(offset + 4, true),
            byteRate: view.getUint32(offset + 8, true),
            blockAlign: view.getUint16(offset + 12, true),
            bitsPerSample: view.getUint16(offset + 14, true),
          }
        } else if (chunkId === 'data') {
          dataOffset = offset
          dataSize = chunkSize
          break
        }

        offset += chunkSize + (chunkSize % 2)
      }

      if (!format || !dataOffset || !dataSize) {
        return null
      }

      var bytesPerSample = format.bitsPerSample / 8
      var frameCount = Math.floor(dataSize / format.blockAlign)
      var output = new Float32Array(frameCount)

      for (var frame = 0; frame < frameCount; frame += 1) {
        var sum = 0
        for (var channel = 0; channel < format.channels; channel += 1) {
          var sampleOffset =
            dataOffset + frame * format.blockAlign + channel * bytesPerSample
          var sample = 0

          if (format.audioFormat === 1) {
            if (format.bitsPerSample === 8) {
              sample = (view.getUint8(sampleOffset) - 128) / 128
            } else if (format.bitsPerSample === 16) {
              sample = view.getInt16(sampleOffset, true) / 32768
            } else if (format.bitsPerSample === 24) {
              sample = readInt24(view, sampleOffset, true) / 8388608
            } else if (format.bitsPerSample === 32) {
              sample = view.getInt32(sampleOffset, true) / 2147483648
            } else {
              throw new Error('Unsupported PCM WAV bit depth: ' + format.bitsPerSample)
            }
          } else if (format.audioFormat === 3 && format.bitsPerSample === 32) {
            sample = view.getFloat32(sampleOffset, true)
          } else {
            throw new Error('Unsupported WAV format: ' + format.audioFormat)
          }

          sum += sample
        }
        output[frame] = sum / format.channels
      }

      return resampleLinear(output, format.sampleRate, SAMPLE_RATE)
    }

    function mixAudioBuffer(audioBuffer) {
      var length = audioBuffer.length
      var output = new Float32Array(length)

      for (var channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
        var channelData = audioBuffer.getChannelData(channel)
        for (var i = 0; i < length; i += 1) {
          output[i] += channelData[i] / audioBuffer.numberOfChannels
        }
      }

      return resampleLinear(output, audioBuffer.sampleRate, SAMPLE_RATE)
    }

    async function decodeAudioBuffer(buffer) {
      var wav = decodeWav(buffer)
      if (wav) {
        return wav
      }

      var AudioContext = root.AudioContext || root.webkitAudioContext
      if (!AudioContext) {
        return pcm16ToFloat32(buffer)
      }

      var audioContext = new AudioContext()
      try {
        var decoded = await audioContext.decodeAudioData(buffer.slice(0))
        return mixAudioBuffer(decoded)
      } finally {
        if (typeof audioContext.close === 'function') {
          audioContext.close()
        }
      }
    }

    async function loadAudioUrl(source) {
      var buffer = await fetchArrayBuffer(source)
      return decodeAudioBuffer(buffer)
    }

    function validateWebGpu(runtime, useGpu) {
      if (!useGpu) {
        return
      }

      if (
        !runtime ||
        typeof runtime.__wasm_webgpu_enabled !== 'function' ||
        !runtime.__wasm_webgpu_enabled()
      ) {
        throw new Error(
          'This @fugood/node-whisper-wasm build was not compiled with GGML_WEBGPU=ON',
        )
      }

      if (!root.navigator || !root.navigator.gpu) {
        throw new Error('WebGPU was requested, but navigator.gpu is not available')
      }
    }

    function WhisperContext(options) {
      if (shouldUseWorker(options || {})) {
        return createWorkerWhisperContext(options || {})
      }
      return createWhisperContext(options)
    }

    WhisperContext.toggleNativeLog = toggleNativeLog
    WhisperContext.loadModelInfo = function (path) {
      return {
        path: path,
        type: 'whisper',
      }
    }

    async function createWorkerWhisperContext(options) {
      var proxy = await getWorkerProxy()
      if (!proxy) {
        return createWhisperContext(options)
      }

      var created = await proxy.request('initWhisper', [options || {}])
      created.meta.worker = true
      return new WorkerWhisperContextInstance(proxy, created.id, created.meta)
    }

    function WorkerWhisperContextInstance(proxy, id, meta) {
      this._proxy = proxy
      this._id = id
      this._meta = meta
      this._released = false
    }

    WorkerWhisperContextInstance.prototype._assertValid = function () {
      if (this._released) {
        throw new Error('Invalid whisper context')
      }
    }

    WorkerWhisperContextInstance.prototype.getModelInfo = function () {
      return this._meta
    }

    WorkerWhisperContextInstance.prototype._transcribeWorker = function (
      method,
      args,
      transfer,
      callbacks,
    ) {
      this._assertValid()

      var cancelled = false
      var operation = this._proxy.requestOperation(method, args, transfer, callbacks)
      var proxy = this._proxy

      return {
        _requestId: operation.id,
        stop: function () {
          cancelled = true
          proxy.cancel(operation.id)
          return Promise.resolve()
        },
        promise: operation.promise.then(function (result) {
          if (cancelled) {
            result.isAborted = true
          }
          return result
        }),
      }
    }

    WorkerWhisperContextInstance.prototype.transcribeData = function (audioData, options) {
      var audio = copyFloat32Audio(audioData)
      var split = splitTranscribeOptions(options)
      var operation = this._transcribeWorker(
        'transcribeData',
        [this._id, audio, split.options],
        [audio.buffer],
        split.callbacks,
      )
      return operation
    }

    WorkerWhisperContextInstance.prototype.transcribeFile = function (filePath, options) {
      var split = splitTranscribeOptions(options)
      var operation = this._transcribeWorker(
        'transcribeFile',
        [this._id, filePath, split.options],
        [],
        split.callbacks,
      )
      return operation
    }

    WorkerWhisperContextInstance.prototype.transcribe =
      WorkerWhisperContextInstance.prototype.transcribeFile

    WorkerWhisperContextInstance.prototype.bench = function (nThreads) {
      this._assertValid()
      return this._proxy.request('benchWhisper', [this._id, nThreads || 1])
    }

    WorkerWhisperContextInstance.prototype.release = async function () {
      if (!this._released) {
        await this._proxy.request('releaseWhisper', [this._id])
        this._released = true
      }
    }

    async function createWhisperContext(options) {
      options = options || {}
      var modelSource = options.filePath || options.modelUrl
      var runtime = await loadRuntime()
      var useGpu = options.useGpu === true

      validateWebGpu(runtime, useGpu)

      var model = await ensureModel(runtime, modelSource, 'whisper', options)
      var init = unwrapWasmResult(
        await runtime.__wasm_init_whisper(
          model.virtualPath,
          useGpu,
          options.useFlashAttn === true,
        ),
      )

      return new WhisperContextInstance(runtime, init.id, {
        filePath: modelSource,
        wasmFilePath: model.virtualPath,
        useGpu: useGpu,
        useFlashAttn: options.useFlashAttn === true,
        bytes: model.bytes,
        cacheHit: model.cacheHit,
        cacheStored: model.cacheStored,
        cacheName: model.cacheName,
        cacheKey: model.cacheKey,
        wasm: true,
      })
    }

    function WhisperContextInstance(runtime, id, meta) {
      this._runtime = runtime
      this._id = id
      this._meta = meta
      this._released = false
    }

    WhisperContextInstance.prototype._assertValid = function () {
      if (this._released) {
        throw new Error('Invalid whisper context')
      }
    }

    WhisperContextInstance.prototype.getModelInfo = function () {
      return this._meta
    }

    WhisperContextInstance.prototype._transcribeFloat32 = function (audio, options, isCancelled) {
      this._assertValid()

      if (isCancelled && isCancelled()) {
        return Promise.resolve(abortedResult())
      }

      var runtime = this._runtime
      var id = this._id
      return defer(async function () {
        if (isCancelled && isCancelled()) {
          return abortedResult()
        }

        var result = unwrapWasmResult(
          await runtime.__wasm_transcribe(
            id,
            audio,
            normalizeTranscribeOptions(options),
          ),
        )
        if (isCancelled && isCancelled()) {
          result.isAborted = true
        }
        return result
      })
    }

    WhisperContextInstance.prototype.transcribeData = function (audioData, options) {
      var cancelled = false
      var audio = toFloat32Audio(audioData)
      return {
        stop: function () {
          cancelled = true
          return Promise.resolve()
        },
        promise: this._transcribeFloat32(audio, options, function () {
          return cancelled
        }),
      }
    }

    WhisperContextInstance.prototype.transcribeFile = function (filePath, options) {
      var cancelled = false
      var self = this
      return {
        stop: function () {
          cancelled = true
          return Promise.resolve()
        },
        promise: (async function () {
          if (cancelled) {
            return abortedResult()
          }
          var audio = await loadAudioUrl(filePath)
          return self._transcribeFloat32(audio, options, function () {
            return cancelled
          })
        })(),
      }
    }

    WhisperContextInstance.prototype.transcribe =
      WhisperContextInstance.prototype.transcribeFile

    WhisperContextInstance.prototype.bench = async function (nThreads) {
      this._assertValid()
      return unwrapWasmResult(await this._runtime.__wasm_bench(this._id, nThreads || 1))
    }

    WhisperContextInstance.prototype.release = function () {
      if (!this._released) {
        this._runtime.__wasm_free_whisper(this._id)
        this._released = true
      }
      return Promise.resolve()
    }

    function WhisperVadContext(options) {
      if (shouldUseWorker(options || {})) {
        return createWorkerWhisperVadContext(options || {})
      }
      return createWhisperVadContext(options)
    }

    WhisperVadContext.toggleNativeLog = toggleNativeLog
    WhisperVadContext.loadModelInfo = function (path) {
      return {
        path: path,
        type: 'whisper_vad',
      }
    }

    async function createWorkerWhisperVadContext(options) {
      var proxy = await getWorkerProxy()
      if (!proxy) {
        return createWhisperVadContext(options)
      }

      var created = await proxy.request('initWhisperVad', [options || {}])
      created.meta.worker = true
      return new WorkerWhisperVadContextInstance(proxy, created.id, created.meta)
    }

    function WorkerWhisperVadContextInstance(proxy, id, meta) {
      this._proxy = proxy
      this._id = id
      this._meta = meta
      this._released = false
    }

    WorkerWhisperVadContextInstance.prototype._assertValid = function () {
      if (this._released) {
        throw new Error('Invalid VAD context')
      }
    }

    WorkerWhisperVadContextInstance.prototype.getModelInfo = function () {
      return this._meta
    }

    WorkerWhisperVadContextInstance.prototype.detectSpeechData = function (
      audioData,
      options,
    ) {
      this._assertValid()
      var audio = copyFloat32Audio(audioData)
      return this._proxy.request(
        'detectSpeechData',
        [this._id, audio, options || {}],
        [audio.buffer],
      )
    }

    WorkerWhisperVadContextInstance.prototype.detectSpeechFile = function (
      filePath,
      options,
    ) {
      this._assertValid()
      return this._proxy.request('detectSpeechFile', [
        this._id,
        filePath,
        options || {},
      ])
    }

    WorkerWhisperVadContextInstance.prototype.detectSpeech =
      WorkerWhisperVadContextInstance.prototype.detectSpeechFile

    WorkerWhisperVadContextInstance.prototype.release = async function () {
      if (!this._released) {
        await this._proxy.request('releaseVad', [this._id])
        this._released = true
      }
    }

    async function createWhisperVadContext(options) {
      options = options || {}
      var modelSource = options.filePath || options.modelUrl
      var runtime = await loadRuntime()
      var useGpu = false
      var nThreads = options.nThreads || 1

      if (options.useGpu === true) {
        emitLog(
          'WARN',
          'WASM VAD currently falls back to CPU because ggml-webgpu does not support the VAD graph safely yet.',
        )
      }

      var model = await ensureModel(runtime, modelSource, 'vad', options)
      var init = unwrapWasmResult(
        await runtime.__wasm_init_vad(model.virtualPath, useGpu, nThreads),
      )

      return new WhisperVadContextInstance(runtime, init.id, {
        filePath: modelSource,
        wasmFilePath: model.virtualPath,
        useGpu: useGpu,
        nThreads: init.nThreads,
        bytes: model.bytes,
        cacheHit: model.cacheHit,
        cacheStored: model.cacheStored,
        cacheName: model.cacheName,
        cacheKey: model.cacheKey,
        wasm: true,
      })
    }

    function WhisperVadContextInstance(runtime, id, meta) {
      this._runtime = runtime
      this._id = id
      this._meta = meta
      this._released = false
    }

    WhisperVadContextInstance.prototype._assertValid = function () {
      if (this._released) {
        throw new Error('Invalid VAD context')
      }
    }

    WhisperVadContextInstance.prototype.getModelInfo = function () {
      return this._meta
    }

    WhisperVadContextInstance.prototype.detectSpeechData = async function (
      audioData,
      options,
    ) {
      this._assertValid()
      var audio = toFloat32Audio(audioData)
      var result = unwrapWasmResult(
        await this._runtime.__wasm_detect_speech(this._id, audio, options || {}),
      )
      return result.segments || []
    }

    WhisperVadContextInstance.prototype.detectSpeechFile = async function (filePath, options) {
      this._assertValid()
      var audio = await loadAudioUrl(filePath)
      return this.detectSpeechData(audio, options)
    }

    WhisperVadContextInstance.prototype.detectSpeech =
      WhisperVadContextInstance.prototype.detectSpeechFile

    WhisperVadContextInstance.prototype.release = function () {
      if (!this._released) {
        this._runtime.__wasm_free_vad(this._id)
        this._released = true
      }
      return Promise.resolve()
    }

    function loadWhisperModule() {
      return Promise.resolve(api)
    }

    function initWhisper(options) {
      return Promise.resolve(new WhisperContext(options))
    }

    function initWhisperVad(options) {
      return Promise.resolve(new WhisperVadContext(options))
    }

    var api = {
      WhisperContext: WhisperContext,
      WhisperVadContext: WhisperVadContext,
      configureWasm: configureWasm,
      loadWasmModule: loadRuntime,
      loadWhisperModule: loadWhisperModule,
      initWhisper: initWhisper,
      initWhisperVad: initWhisperVad,
      toggleNativeLog: toggleNativeLog,
      addNativeLogListener: addNativeLogListener,
      DEFAULT_WASM_MODEL_SIZE_LIMIT_BYTES: 1500 * MIB,
      MAX_WASM_THREADS: MAX_WASM_THREADS,
    }

    api.default = api

    return api
  },
)
