import fs from 'fs'
import path from 'path'
import {
  initWhisper,
} from '../lib/index'

// Test configuration
const TEST_TIMEOUT = 30000 // 30 seconds timeout for model loading
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
      expect(typeof context.getSystemInfo).toBe('function')

      const systemInfo = context.getSystemInfo()
      expect(typeof systemInfo).toBe('string')
      expect(systemInfo.length).toBeGreaterThan(0)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should transcribe JFK audio sample',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisper({
        filePath: modelPath,
        useGpu: false,
      })

      const result = await context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
      })

      expect(result).toBeDefined()
      expect(typeof result.result).toBe('string')
      expect(result.result.length).toBeGreaterThan(0)
      expect(Array.isArray(result.segments)).toBe(true)

      // JFK sample should contain recognizable words
      const lowerText = result.result.toLowerCase()
      expect(lowerText).toMatch(/ask|not|what|your|country|can|do|for|you/)

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

      const result = await context.transcribeData(audioBuffer, {
        language: 'en',
        temperature: 0.0,
      })

      expect(result).toBeDefined()
      expect(typeof result.result).toBe('string')
      expect(Array.isArray(result.segments)).toBe(true)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test('should handle transcription errors gracefully', async () => {
    const invalidModelPath = 'nonexistent/model.bin'

    await expect(
      initWhisper({ filePath: invalidModelPath }),
    ).rejects.toThrow()
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

    expect(await context.transcribeData(invalidBuffer)).toEqual({
      isAborted: false,
      result: '',
      segments: [],
    })

    await context.release()
  })

  test('should handle model loading errors', async () => {
    const invalidModelPath = 'nonexistent/model.bin'

    await expect(
      initWhisper({ filePath: invalidModelPath }),
    ).rejects.toThrow()
  })
})
