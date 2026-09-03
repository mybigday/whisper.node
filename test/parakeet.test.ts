import fs from 'fs'
import path from 'path'
import { initParakeet } from '../lib/index'

// Test configuration
const TEST_TIMEOUT = 60000 // 60 seconds timeout for model loading
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

describe('Parakeet transcription', () => {
  const modelPath = path.join(
    __dirname,
    '../whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-q4_0.bin',
  )

  test(
    'should load parakeet model',
    async () => {
      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      expect(context).toBeDefined()

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should transcribe JFK audio sample',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      const { promise } = context.transcribeData(audioBuffer, {
        maxThreads: 4,
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(typeof result.result).toBe('string')
      expect(result.result.length).toBeGreaterThan(0)
      expect(Array.isArray(result.segments)).toBe(true)
      expect(result.isAborted).toBe(false)

      // JFK sample should contain recognizable words
      const lowerText = result.result.toLowerCase()
      expect(lowerText).toMatch(/ask|not|what|your|country|can|do|for|you/)

      console.log('Transcription result:', result.result)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should transcribe audio file path',
    async () => {
      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      const { promise } = context.transcribeFile(SAMPLE_AUDIO_PATH, {
        maxThreads: 4,
      })

      const result = await promise

      expect(result).toBeDefined()
      expect(typeof result.result).toBe('string')
      expect(result.result.length).toBeGreaterThan(0)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should return correct API structure with stop and promise',
    async () => {
      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      const audioBuffer = createTestAudioBuffer(100, 440) // Very short audio

      const result = context.transcribeData(audioBuffer, {
        maxThreads: 2,
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

  test('should handle invalid audio data', async () => {
    const invalidBuffer = new ArrayBuffer(0) // Empty buffer

    const context = await initParakeet({
      filePath: modelPath,
      useGpu: false,
    })

    const { promise } = context.transcribeData(invalidBuffer)
    const result = await promise

    expect(result).toEqual({
      isAborted: false,
      language: '',
      result: '',
      segments: [],
    })

    await context.release()
  }, TEST_TIMEOUT)

  test('should handle model loading errors', async () => {
    const invalidModelPath = 'nonexistent/model.bin'

    await expect(initParakeet({ filePath: invalidModelPath })).rejects.toThrow()
  })

  test(
    'should settle the promise with isAborted after stop() on an in-flight job',
    async () => {
      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      // Long CPU decode so the job is still running when we stop it. The abort
      // flag is only checked between graph nodes and the encoder runs the whole
      // clip in one graph, so keep the clip short and allow a generous settle
      // window for slow CI runners.
      const audioBuffer = createTestAudioBuffer(30000, 440)
      const { stop, promise } = context.transcribeData(audioBuffer, {
        maxThreads: 2,
      })

      expect(await settlesWithin(promise, 300)).toBe(false)

      await stop()

      expect(await settlesWithin(promise, STOP_SETTLE_TIMEOUT)).toBe(true)
      const result = await promise
      expect(result.isAborted).toBe(true)
      expect(typeof result.result).toBe('string')
      expect(Array.isArray(result.segments)).toBe(true)

      // The context stays usable after a stopped job
      const next = await context.transcribeData(createTestAudioBuffer(500, 440), {
        maxThreads: 2,
      }).promise
      expect(next.isAborted).toBe(false)

      await context.release()
    },
    STOP_TEST_TIMEOUT,
  )

  test(
    'should settle the promise when the context is released mid-job',
    async () => {
      const context = await initParakeet({
        filePath: modelPath,
        useGpu: false,
      })

      const audioBuffer = createTestAudioBuffer(30000, 440)
      const { promise } = context.transcribeData(audioBuffer, {
        maxThreads: 2,
      })

      expect(await settlesWithin(promise, 300)).toBe(false)

      await context.release()

      expect(await settlesWithin(promise, STOP_SETTLE_TIMEOUT)).toBe(true)
    },
    STOP_TEST_TIMEOUT,
  )
})
