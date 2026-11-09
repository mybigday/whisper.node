import fs from 'fs'
import path from 'path'
import { initWhisperVad } from '../lib/index'

// Test configuration
const TEST_TIMEOUT = 30000 // 30 seconds timeout for model loading
const SAMPLE_AUDIO_PATH = path.join(__dirname, '../src/whisper.rn/vendor/whisper.cpp/samples/jfk.wav')

// Load the PCM16 samples of a WAV file (the "data" chunk) as an ArrayBuffer
function loadWavFile(filePath: string) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Audio file not found: ${filePath}`)
  }

  const buffer = fs.readFileSync(filePath)
  let offset = 12 // RIFF header
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4)
    const size = buffer.readUInt32LE(offset + 4)
    if (id === 'data') {
      const start = buffer.byteOffset + offset + 8
      return buffer.buffer.slice(start, start + size)
    }
    offset += 8 + size + (size % 2)
  }
  throw new Error(`No data chunk in ${filePath}`)
}

describe('Voice Activity Detection (VAD)', () => {
  const vadModelPath = path.join(
    __dirname,
    'models/ggml-silero-v6.2.0.bin',
  )

  test(
    'should load VAD model',
    async () => {
      const context = await initWhisperVad({
        filePath: vadModelPath,
        useGpu: false,
        nThreads: 2,
      })

      expect(context).toBeDefined()
      expect(context.sampleRate).toBe(16000)
      expect(context.hopSamples).toBe(512)
      expect(context.arch).toBe('silero')

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should detect speech in JFK audio sample',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)

      const context = await initWhisperVad({
        filePath: vadModelPath,
        useGpu: false,
        nThreads: 2,
      })

      const result = await context.detectSpeechData(audioBuffer, {
        threshold: 0.5,
        minSpeechDurationMs: 250,
        minSilenceDurationMs: 100,
        maxSpeechDurationS: 30,
        speechPadMs: 30,
      })

      expect(result).toBeDefined()
      expect(Array.isArray(result)).toBe(true)

      // JFK sample should contain speech segments
      expect(result.length).toBeGreaterThan(0)

      // Each segment should have t0 and t1 properties
      if (result.length > 0) {
        expect(typeof result[0].t0).toBe('number')
        expect(typeof result[0].t1).toBe('number')
        expect(result[0].t1).toBeGreaterThan(result[0].t0)
      }

      console.log('VAD result:', {
        segments_count: result.length,
        segments: result.map((seg) => ({ t0: seg.t0, t1: seg.t1 })),
      })

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'should detect no speech in silence',
    async () => {
      // Create silent audio buffer
      const silentBuffer = new ArrayBuffer(32000) // 1 second of silence

      const context = await initWhisperVad({
        filePath: vadModelPath,
        useGpu: false,
        nThreads: 2,
      })

      const result = await context.detectSpeechData(silentBuffer)

      expect(result).toBeDefined()
      expect(Array.isArray(result)).toBe(true)

      // Silent audio should not contain speech segments
      expect(result).toHaveLength(0)

      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'detectSpeechFile should match detectSpeechData',
    async () => {
      const context = await initWhisperVad({ filePath: vadModelPath })
      const fromFile = await context.detectSpeechFile(SAMPLE_AUDIO_PATH)
      const fromData = await context.detectSpeechData(loadWavFile(SAMPLE_AUDIO_PATH))
      expect(fromFile.length).toBeGreaterThan(0)
      expect(fromFile).toEqual(fromData)
      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'stream should produce the same probabilities in chunks as in one push',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)
      const context = await initWhisperVad({ filePath: vadModelPath })
      const nSamples = audioBuffer.byteLength / 2

      const whole = await context.createStream()
      expect(whole.sampleRate).toBe(16000)
      expect(whole.hopSamples).toBe(512)
      const wholeProbs = [...(await whole.push(audioBuffer)), ...(await whole.flush())]
      expect(wholeProbs).toHaveLength(Math.ceil(nSamples / whole.hopSamples))

      // Batched hops (one big push) and single hops take different kernel
      // paths, so allow a small numeric difference.
      // Odd chunk sizes, pushed without awaiting: calls must run in order.
      // The chunk buffer is reused, so push() must copy it.
      const chunked = await context.createStream()
      const bytes = new Uint8Array(audioBuffer)
      const chunk = new Uint8Array(3000 * 2)
      const pending: Promise<number[]>[] = []
      for (let offset = 0; offset < bytes.length; offset += chunk.length) {
        const part = bytes.subarray(offset, offset + chunk.length)
        chunk.set(part)
        pending.push(chunked.push(chunk.subarray(0, part.length)))
      }
      pending.push(chunked.flush())
      const chunkedProbs = (await Promise.all(pending)).flat()

      expect(chunkedProbs).toHaveLength(wholeProbs.length)
      chunkedProbs.forEach((p, i) => expect(p).toBeCloseTo(wholeProbs[i], 2))
      expect(Math.max(...wholeProbs)).toBeGreaterThan(0.5)

      // reset() starts over with a fresh state
      await chunked.reset()
      const again = [...(await chunked.push(audioBuffer)), ...(await chunked.flush())]
      again.forEach((p, i) => expect(p).toBeCloseTo(wholeProbs[i], 5))

      await whole.release()
      await expect(whole.push(audioBuffer)).rejects.toThrow('VAD stream not found')
      await context.release()
      // Releasing the context frees its streams
      await expect(chunked.flush()).rejects.toThrow('VAD stream not found')
      await chunked.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'stream should report silence with low probabilities',
    async () => {
      const context = await initWhisperVad({ filePath: vadModelPath })
      const stream = await context.createStream()
      const probs = [
        ...(await stream.push(new ArrayBuffer(32000))),
        ...(await stream.flush()),
      ]
      expect(probs).toHaveLength(Math.ceil(16000 / 512))
      probs.forEach((p) => expect(p).toBeLessThan(0.5))
      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'stream should accept Float32Array samples',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)
      const pcm = new Int16Array(audioBuffer)
      const floats = Float32Array.from(pcm, (v) => v / 32768)
      const context = await initWhisperVad({ filePath: vadModelPath })
      const a = await context.createStream()
      const b = await context.createStream()
      const fromPcm = [...(await a.push(pcm)), ...(await a.flush())]
      const fromFloat = [...(await b.push(floats)), ...(await b.flush())]
      expect(fromFloat).toHaveLength(fromPcm.length)
      fromFloat.forEach((p, i) => expect(p).toBeCloseTo(fromPcm[i], 5))
      expect(() => a.push(new Float64Array(4) as any)).toThrow('Expected ArrayBuffer')
      await context.release()
    },
    TEST_TIMEOUT,
  )

  test(
    'release should let calls made before it finish',
    async () => {
      const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH)
      const context = await initWhisperVad({ filePath: vadModelPath })
      const stream = await context.createStream()
      const detecting = context.detectSpeechData(audioBuffer)
      const pushing = stream.push(audioBuffer)
      const released = context.release()
      expect((await detecting).length).toBeGreaterThan(0)
      expect((await pushing).length).toBeGreaterThan(0)
      await released
      // Calls made after release() fail
      await expect(stream.flush()).rejects.toThrow('VAD stream not found')
      await expect(context.detectSpeechData(audioBuffer)).rejects.toThrow()
      await stream.release()
    },
    TEST_TIMEOUT,
  )

  test('should handle VAD errors gracefully', async () => {
    const invalidModelPath = 'nonexistent/vad-model.bin'

    await expect(initWhisperVad({ filePath: invalidModelPath })).rejects.toThrow()
  })
})
