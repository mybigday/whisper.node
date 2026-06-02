(function (root) {
  'use strict'

  var api = null
  var nextContextId = 1
  var contexts = Object.create(null)
  var activeStops = Object.create(null)
  var cancelledRequests = Object.create(null)

  function resolveUrl(value, base) {
    return new URL(value, base || root.location.href).href
  }

  function serializeError(error) {
    return {
      message: error && error.message ? error.message : String(error),
      stack: error && error.stack ? error.stack : undefined,
    }
  }

  function postResponse(id, result) {
    root.postMessage({
      type: 'response',
      id: id,
      result: result,
    })
  }

  function postError(id, error) {
    root.postMessage({
      type: 'response',
      id: id,
      error: serializeError(error),
    })
  }

  function postCallback(id, name, value) {
    root.postMessage({
      type: 'callback',
      id: id,
      name: name,
      value: value,
    })
  }

  function getContext(id, type) {
    var entry = contexts[id]
    if (!entry || entry.type !== type) {
      throw new Error('Invalid ' + type + ' context')
    }
    return entry.context
  }

  function abortedResult() {
    return {
      result: '',
      segments: [],
      isAborted: true,
    }
  }

  function inflateTranscribeOptions(requestId, options) {
    var inflated = Object.assign({}, options || {})

    if (inflated.onProgress) {
      inflated.onProgress = function (progress) {
        if (!cancelledRequests[requestId]) {
          postCallback(requestId, 'onProgress', progress)
        }
      }
    }

    if (inflated.onNewSegments) {
      inflated.onNewSegments = function (result) {
        if (!cancelledRequests[requestId]) {
          postCallback(requestId, 'onNewSegments', result)
        }
      }
    }

    return inflated
  }

  async function ensureApi(config) {
    if (api) {
      return api
    }

    config = config || {}
    var indexScriptUrl = resolveUrl(config.indexScriptUrl || 'index.js')
    var runtimeScriptUrl = resolveUrl(
      config.runtimeScriptUrl || 'whisper-node.js',
      indexScriptUrl,
    )
    var locateFileBaseUrl =
      config.locateFileBaseUrl || resolveUrl('.', runtimeScriptUrl)

    root.importScripts(runtimeScriptUrl)
    root.importScripts(indexScriptUrl)

    api = root.WhisperNodeWasm
    if (!api) {
      throw new Error('Failed to load WhisperNodeWasm in worker')
    }

    var runtimeOptions = Object.assign({}, config.runtimeOptions || {})
    runtimeOptions.mainScriptUrlOrBlob = runtimeScriptUrl
    runtimeOptions.locateFile = function (path, prefix) {
      return resolveUrl(path, locateFileBaseUrl || prefix)
    }

    api.configureWasm(runtimeOptions)
    api.addNativeLogListener(function (level, text) {
      root.postMessage({
        type: 'log',
        level: level,
        text: text,
      })
    })
    await api.toggleNativeLog(true)

    return api
  }

  async function runTranscribeOperation(requestId, operation) {
    activeStops[requestId] = operation.stop
    try {
      var result = await operation.promise
      if (cancelledRequests[requestId]) {
        result.isAborted = true
      }
      return result
    } finally {
      delete activeStops[requestId]
    }
  }

  async function dispatch(message) {
    var args = message.args || []

    switch (message.method) {
      case '__init':
        await ensureApi(args[0])
        return { ok: true }

      case 'initWhisper': {
        var whisper = await api.initWhisper(args[0] || {})
        var whisperId = nextContextId++
        contexts[whisperId] = {
          type: 'whisper',
          context: whisper,
        }
        return {
          id: whisperId,
          meta: whisper.getModelInfo(),
        }
      }

      case 'transcribeData': {
        if (cancelledRequests[message.id]) {
          return abortedResult()
        }
        var whisperDataContext = getContext(args[0], 'whisper')
        var transcribeDataOptions = inflateTranscribeOptions(message.id, args[2])
        return runTranscribeOperation(
          message.id,
          whisperDataContext.transcribeData(args[1], transcribeDataOptions),
        )
      }

      case 'transcribeFile': {
        if (cancelledRequests[message.id]) {
          return abortedResult()
        }
        var whisperFileContext = getContext(args[0], 'whisper')
        var transcribeFileOptions = inflateTranscribeOptions(message.id, args[2])
        return runTranscribeOperation(
          message.id,
          whisperFileContext.transcribeFile(args[1], transcribeFileOptions),
        )
      }

      case 'benchWhisper':
        return getContext(args[0], 'whisper').bench(args[1] || 1)

      case 'releaseWhisper':
        await getContext(args[0], 'whisper').release()
        delete contexts[args[0]]
        return { ok: true }

      case 'initWhisperVad': {
        var vad = await api.initWhisperVad(args[0] || {})
        var vadId = nextContextId++
        contexts[vadId] = {
          type: 'vad',
          context: vad,
        }
        return {
          id: vadId,
          meta: vad.getModelInfo(),
        }
      }

      case 'detectSpeechData':
        return getContext(args[0], 'vad').detectSpeechData(args[1], args[2] || {})

      case 'detectSpeechFile':
        return getContext(args[0], 'vad').detectSpeechFile(args[1], args[2] || {})

      case 'releaseVad':
        await getContext(args[0], 'vad').release()
        delete contexts[args[0]]
        return { ok: true }

      default:
        throw new Error('Unknown WASM worker method: ' + message.method)
    }
  }

  root.onmessage = function (event) {
    var message = event.data || {}

    if (message.type === 'cancel') {
      cancelledRequests[message.id] = true
      if (activeStops[message.id]) {
        Promise.resolve(activeStops[message.id]()).catch(function () {})
      }
      return
    }

    if (message.type !== 'request') {
      return
    }

    Promise.resolve()
      .then(function () {
        return dispatch(message)
      })
      .then(function (result) {
        postResponse(message.id, result)
      })
      .catch(function (error) {
        postError(message.id, error)
      })
      .finally(function () {
        delete cancelledRequests[message.id]
      })
  }
})(
  typeof self !== 'undefined'
    ? self
    : typeof globalThis !== 'undefined'
      ? globalThis
      : this,
)
