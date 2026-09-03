import fs from 'fs'
import path from 'path'
import {
  initWhisper,
  toggleNativeLog,
  addNativeLogListener,
  isNativeLogEnabled,
} from '../lib/index'

// Test configuration
const TEST_TIMEOUT = 30000 // 30 seconds timeout for model loading
const STOP_SETTLE_TIMEOUT = 60000 // how long a stopped job may take to settle on slow CI
const STOP_TEST_TIMEOUT = 180000
const SAMPLE_AUDIO_PATH = path.join(__dirname, '../whisper.cpp/samples/jfk.wav')

// Helper function to create test audio data (16-bit PCM, mono, 16kHz)
function createTestAudioBuffer(durationMs = 1000, frequency = 440) {
  const sampleRate = 16000
  const samples = Math.floor((durationMs * sampleRate) / 1000)
  const buffer = new ArrayBuffer(samples * 2) // 16-bit = 2 bytes per sample
  const view = new Int16Array(buffer)

  for (let i = 0; i < samples; i += 1) {
    // Generate sine wave
    const t = i / sampleRate
    const amplitude = 16000 // Scale to 16-bit range
    view[i] = Math.sin(2 * Math.PI * frequency * t) * amplitude
  }

  return buffer
}

// Helper function to load WAV file and convert to ArrayBuffer
function loadWavFile(filePath: string) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Audio file not found: ${filePath}`)
  }

  const buffer = fs.readFileSync(filePath)

  // Simple WAV file parsing - skip header and get audio data
  // WAV header is typically 44 bytes
  const headerSize = 44
  const audioData = buffer.slice(headerSize)

  // Convert to ArrayBuffer
  return audioData.buffer.slice(
    audioData.byteOffset,
    audioData.byteOffset + audioData.byteLength,
  )
}

// Resolves true if the promise settles (either way) within `ms`, false otherwise
function settlesWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), ms)
    const done = () => {
      clearTimeout(timer)
      resolve(true)
    }
    promise.then(done, done)
  })
}

describe('Whisper transcription', () => {
  const modelPath = path.join(
    __dirname,
    '../whisper.cpp/models/ggml-tiny.en.bin',
  )

  test(
    'should load whisper model and get system info',
    async () => {
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      expect(context).toBeDefined()

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test('toggleNativeLog should work correctly', async () => {
    const logs: Array<{ level: string; text: string }> = []

    // Add log listener
    const { remove } = addNativeLogListener((level, text) => {
      logs.push({ level, text })
    })

    try {
      // Enable native logging
      await toggleNativeLog(true)
      expect(isNativeLogEnabled()).toBe(true)

      // Load a model to trigger some logging
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      // Wait a bit for any async logging
      await new Promise((resolve) => setTimeout(resolve, 100))

      // Clean up context
      await context.release()

      // We should have received some logs (though the exact content depends on the whisper implementation)
      expect(logs.length).toBeGreaterThan(0)
      expect(logs.some((log) => log.level === 'info')).toBe(true)
    } finally {
      // Always disable logging and remove listener, even if test fails
      await toggleNativeLog(false)
      expect(isNativeLogEnabled()).toBe(false)
      remove()
    }
  })

  test(
    'should transcribe JFK audio sample',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      let lastSegment = undefined
      const { promise } = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
        nProcessors: 1,
        onNewSegments: (seg) => {
          lastSegment = seg
        },
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(typeof result.language).toBe('string')
      expect(typeof result.result).toBe('string')
      expect(result.result.length).toBeGreaterThan(0)
      expect(Array.isArray(result.segments)).toBe(true)
      expect(lastSegment).toBeDefined()

      // JFK sample should contain recognizable words
      const lowerText = result.result.toLowerCase()
      expect(lowerText).toMatch(/ask|not|what|your|country|can|do|for|you/)

      console.log('Language:', result.language)
      console.log('Transcription result:', result.result)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should transcribe with context reuse',
    async () => {
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const audioBuffer = createTestAudioBuffer(2000, 440)

      const { promise } = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(typeof result.result).toBe('string')
      expect(Array.isArray(result.segments)).toBe(true)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test('should handle transcription errors gracefully', async () => {
    const invalidModelPath = 'nonexistent/model.bin'

    await expect(initWhisper({ filePath: invalidModelPath })).rejects.toThrow()
  })

  test('should handle invalid audio data', async () => {
    const modelPath = path.join(
      __dirname,
      '../whisper.cpp/models/ggml-tiny.en.bin',
    )

    const invalidBuffer = new ArrayBuffer(0) // Empty buffer

    const context = await initWhisper({
      filePath: modelPath,
      useGpu: false,
    })

    const { promise } = context.transcribeData(invalidBuffer)
    const result = await promise

    expect(result).toEqual({
      isAborted: false,
      result: '',
      segments: [],
    })

    await context.release()
  })

  test('should handle model loading errors', async () => {
    const invalidModelPath = 'nonexistent/model.bin'

    await expect(initWhisper({ filePath: invalidModelPath })).rejects.toThrow()
  })

  test(
    'should return correct API structure with stop and promise',
    async () => {
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const audioBuffer = createTestAudioBuffer(100, 440) // Very short audio

      const result = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
      })

      // Verify the API structure matches whisper.rn standard
      expect(result).toBeDefined()
      expect(typeof result).toBe('object')
      expect(typeof result.stop).toBe('function')
      expect(typeof result.promise).toBe('object')
      expect(typeof result.promise.then).toBe('function')
      expect(result.promise.constructor.name).toBe('Promise')

      // Wait for transcription to complete
      const transcriptionResult = await result.promise

      expect(transcriptionResult).toBeDefined()
      expect(typeof transcriptionResult.result).toBe('string')
      expect(Array.isArray(transcriptionResult.segments)).toBe(true)
      expect(typeof transcriptionResult.isAborted).toBe('boolean')

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should call onProgress callback during transcription',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const progressUpdates: number[] = []

      const { promise } = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
        onProgress: (progress) => {
          console.log('Progress:', progress)
          progressUpdates.push(progress)
        },
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(progressUpdates.length).toBeGreaterThan(0)

      // Progress should be between 0 and 100
      progressUpdates.forEach((progress) => {
        expect(progress).toBeGreaterThanOrEqual(0)
        expect(progress).toBeLessThanOrEqual(100)
      })

      console.log('Total progress updates:', progressUpdates.length)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should call onNewSegments callback during transcription',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const segmentUpdates: any[] = []

      const { promise } = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
        onNewSegments: (segmentResult) => {
          console.log(
            'New segments:',
            segmentResult.nNew,
            'Total:',
            segmentResult.totalNNew,
          )
          console.log('Segment text:', segmentResult.result)
          segmentUpdates.push(segmentResult)
        },
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(segmentUpdates.length).toBeGreaterThan(0)

      // Verify segment structure
      segmentUpdates.forEach((update) => {
        expect(typeof update.nNew).toBe('number')
        expect(typeof update.totalNNew).toBe('number')
        expect(typeof update.result).toBe('string')
        expect(Array.isArray(update.segments)).toBe(true)
      })

      console.log('Total segment updates:', segmentUpdates.length)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should call both onProgress and onNewSegments callbacks',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const progressUpdates: number[] = []
      const segmentUpdates: any[] = []

      const { promise } = context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
        onProgress: (progress) => {
          progressUpdates.push(progress)
        },
        onNewSegments: (segmentResult) => {
          segmentUpdates.push(segmentResult)
        },
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(progressUpdates.length).toBeGreaterThan(0)
      expect(segmentUpdates.length).toBeGreaterThan(0)

      console.log('Progress updates:', progressUpdates.length)
      console.log('Segment updates:', segmentUpdates.length)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  // Repeat the JFK sample so the job is long enough to be interrupted mid-way
  function createLongSpeechBuffer(repeats: number) {
    const sample = new Int16Array(loadWavFile(SAMPLE_AUDIO_PATH))
    const out = new Int16Array(sample.length * repeats)
    for (let i = 0; i < repeats; i += 1) out.set(sample, i * sample.length)
    return out.buffer
  }

  test(
    'should settle the promise with isAborted after stop() on an in-flight job',
    async () => {
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      // Stop from the first onNewSegments callback: the job is guaranteed to be
      // in flight at that point regardless of how fast the machine is.
      let stopCalls = 0
      const job = context.transcribeData(createLongSpeechBuffer(6), {
        language: 'en',
        maxThreads: 2,
        onNewSegments: () => {
          if (stopCalls === 0) {
            stopCalls += 1
            job.stop()
          }
        },
      })

      expect(await settlesWithin(job.promise, STOP_SETTLE_TIMEOUT)).toBe(true)
      expect(stopCalls).toBe(1)
      const result = await job.promise
      expect(result.isAborted).toBe(true)
      expect(typeof result.result).toBe('string')
      expect(Array.isArray(result.segments)).toBe(true)

      // The context stays usable after a stopped job
      const next = await context.transcribeData(createTestAudioBuffer(500, 440), {
        language: 'en',
      }).promise
      expect(next.isAborted).toBe(false)

      await context.release()
    },
    STOP_TEST_TIMEOUT,
  )

  test(
    'should settle the promise when the context is released mid-job',
    async () => {
      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      let released: Promise<void> | null = null
      const { promise } = context.transcribeData(createLongSpeechBuffer(6), {
        language: 'en',
        maxThreads: 2,
        onNewSegments: () => {
          if (!released) released = context.release()
        },
      })

      expect(await settlesWithin(promise, STOP_SETTLE_TIMEOUT)).toBe(true)
      expect(released).not.toBeNull()
      await released
    },
    STOP_TEST_TIMEOUT,
  )
})
