import fs from 'fs'
import path from 'path'
import { initWhisper } from '../lib/index'
import type { TranscribeOptions } from '../lib/index'

const TEST_TIMEOUT = 60000
const MODEL_PATH = path.join(__dirname, 'models/ggml-tiny.en.bin')
const SAMPLE_AUDIO_PATH = path.join(
  __dirname,
  '../src/whisper.rn/vendor/whisper.cpp/samples/jfk.wav',
)
const BASE_OPTIONS: TranscribeOptions = { language: 'en', temperature: 0 }

// 16-bit PCM of the sample (header skipped, as in whisper.test.ts)
function loadPcm(): ArrayBuffer {
  const data = fs.readFileSync(SAMPLE_AUDIO_PATH).subarray(44)
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
}

// whisper.rn's cpp/jsi/AudioCtx.h fitAudioCtx, for a 1500 audio ctx model
function fitAudioCtx(nSamples: number, maxAudioCtx = 1500) {
  const frames = Math.ceil(nSamples / 320)
  const padded = Math.floor((frames + 100 + 63) / 64) * 64
  return Math.min(maxAudioCtx, Math.max(256, padded))
}

// "0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11]
function parseCpuList(list: string): number[] {
  return list
    .trim()
    .split(',')
    .flatMap((part) => {
      const [lo, hi] = part.split('-').map(Number)
      return Array.from({ length: (hi ?? lo) - lo + 1 }, (_, i) => lo + i)
    })
}

// Allowed CPUs of each thread of this process
function threadCpus(): number[][] {
  const result: number[][] = []
  for (const tid of fs.readdirSync('/proc/self/task')) {
    try {
      const status = fs.readFileSync(`/proc/self/task/${tid}/status`, 'utf8')
      const match = /Cpus_allowed_list:\s*(.+)/.exec(status)
      if (match) result.push(parseCpuList(match[1]))
    } catch {
      // the thread exited
    }
  }
  return result
}

const sameCpus = (a: number[], b: number[]) =>
  a.length === b.length && a.every((cpu, i) => cpu === b[i])

describe('Whisper audioCtx', () => {
  test(
    "'auto' fits the context to the input, and matches the explicit size",
    async () => {
      const context = await initWhisper({ filePath: MODEL_PATH, useGpu: false })
      try {
        const pcm = loadPcm()
        const expected = fitAudioCtx(pcm.byteLength / 2)
        expect(expected).toBe(704)

        const auto = await context.transcribeData(pcm, { ...BASE_OPTIONS, audioCtx: 'auto' }).promise
        const fixed = await context.transcribeData(pcm, { ...BASE_OPTIONS, audioCtx: expected }).promise
        expect(auto.result.toLowerCase()).toMatch(/ask not what your country can do for you/)
        expect(auto.result).toBe(fixed.result)
        expect(auto.segments).toEqual(fixed.segments)

        // offset/duration count: 2 s of input fit into the minimum of 256
        const twoSeconds = await context.transcribeFile(SAMPLE_AUDIO_PATH, {
          ...BASE_OPTIONS,
          duration: 2000,
          audioCtx: 'auto',
        }).promise
        const twoSecondsFixed = await context.transcribeFile(SAMPLE_AUDIO_PATH, {
          ...BASE_OPTIONS,
          duration: 2000,
          audioCtx: 256,
        }).promise
        expect(twoSeconds.result).toBe(twoSecondsFixed.result)

        // 0 is the model default
        const byDefault = await context.transcribeFile(SAMPLE_AUDIO_PATH, BASE_OPTIONS).promise
        const zero = await context.transcribeFile(SAMPLE_AUDIO_PATH, { ...BASE_OPTIONS, audioCtx: 0 }).promise
        expect(zero.result).toBe(byDefault.result)
      } finally {
        await context.release()
      }
    },
    TEST_TIMEOUT,
  )

  test(
    'rejects anything but a non-negative integer or auto',
    async () => {
      const context = await initWhisper({ filePath: MODEL_PATH, useGpu: false })
      try {
        const invalid: unknown[] = [-1, 1.5, Number.NaN, Infinity, 2 ** 31, 'full', '768', true, {}]
        for (const audioCtx of invalid) {
          const options = { audioCtx } as unknown as TranscribeOptions
          await expect(context.transcribeFile(SAMPLE_AUDIO_PATH, options).promise).rejects.toThrow(
            "audioCtx must be a non-negative integer or 'auto'",
          )
          await expect(context.transcribeData(loadPcm(), options).promise).rejects.toThrow(
            "audioCtx must be a non-negative integer or 'auto'",
          )
        }
        // null / undefined keep the default
        const nullCtx = { audioCtx: null } as unknown as TranscribeOptions
        await expect(context.transcribeData(new ArrayBuffer(0), nullCtx).promise).resolves.toBeDefined()
        await expect(
          context.transcribeData(new ArrayBuffer(0), { audioCtx: undefined }).promise,
        ).resolves.toBeDefined()
      } finally {
        await context.release()
      }
    },
    TEST_TIMEOUT,
  )
})

describe('Whisper cpu_mask', () => {
  test(
    'rejects an invalid mask and nProcessors > 1',
    async () => {
      const context = await initWhisper({ filePath: MODEL_PATH, useGpu: false })
      try {
        for (const cpuMask of ['zz', '0xG', '6-x', '9999-']) {
          await expect(
            context.transcribeFile(SAMPLE_AUDIO_PATH, { cpu_mask: cpuMask }).promise,
          ).rejects.toThrow(`Invalid cpu_mask: ${cpuMask}`)
        }
        await expect(
          context.transcribeData(loadPcm(), { cpu_mask: '0x3', nProcessors: 2 }).promise,
        ).rejects.toThrow('cpu_mask is not supported with nProcessors > 1')
        // an empty mask is no mask
        await expect(
          context.transcribeData(new ArrayBuffer(0), { cpu_mask: '', nProcessors: 2 }).promise,
        ).resolves.toBeDefined()
      } finally {
        await context.release()
      }
    },
    TEST_TIMEOUT,
  )

  test(
    'gives the same transcript as without a mask',
    async () => {
      const context = await initWhisper({ filePath: MODEL_PATH, useGpu: false })
      try {
        const options = { ...BASE_OPTIONS, maxThreads: 2 }
        const plain = await context.transcribeFile(SAMPLE_AUDIO_PATH, options).promise
        const masked = await context.transcribeFile(SAMPLE_AUDIO_PATH, { ...options, cpu_mask: '0-1' }).promise
        const strict = await context.transcribeData(loadPcm(), {
          ...options,
          cpu_mask: '0x3',
          cpu_strict: true,
          audioCtx: 'auto',
        }).promise
        expect(masked.result).toBe(plain.result)
        expect(masked.segments).toEqual(plain.segments)
        expect(strict.result.toLowerCase()).toMatch(/ask not what your country can do for you/)
      } finally {
        await context.release()
      }
    },
    TEST_TIMEOUT,
  )

  const linuxTest = process.platform === 'linux' ? test : test.skip
  linuxTest(
    'runs the compute threads on the mask and leaves no thread pinned',
    async () => {
      const allowed = parseCpuList(
        /Cpus_allowed_list:\s*(.+)/.exec(fs.readFileSync('/proc/self/status', 'utf8'))![1],
      )
      // the last two CPUs this process may use (one if that is all)
      const cpus = allowed.slice(-2)
      const mask = '0x' + cpus.reduce((bits, cpu) => bits | (BigInt(1) << BigInt(cpu)), BigInt(0)).toString(16)
      const isPinned = (threadAllowed: number[]) => sameCpus(threadAllowed, cpus) && !sameCpus(cpus, allowed)

      const context = await initWhisper({ filePath: MODEL_PATH, useGpu: false })
      try {
        const sample = async (options: TranscribeOptions) => {
          const seen: number[][] = []
          const timer = setInterval(() => seen.push(...threadCpus()), 2)
          try {
            await context.transcribeFile(SAMPLE_AUDIO_PATH, { ...BASE_OPTIONS, ...options }).promise
          } finally {
            clearInterval(timer)
          }
          return seen
        }

        const during = await sample({ maxThreads: cpus.length, cpu_mask: mask })
        if (cpus.length < allowed.length) {
          expect(during.some(isPinned)).toBe(true)
        }

        // the job's thread (and an OpenMP team it owned) exits with the job
        const deadline = Date.now() + 5000
        while (threadCpus().some(isPinned) && Date.now() < deadline) {
          await new Promise((resolve) => setTimeout(resolve, 20))
        }
        expect(threadCpus().some(isPinned)).toBe(false)
        expect(threadCpus().every((threadAllowed) => sameCpus(threadAllowed, allowed))).toBe(true)

        // a later job without a mask runs unpinned
        const unmasked = await sample({ maxThreads: 2 })
        expect(unmasked.every((threadAllowed) => sameCpus(threadAllowed, allowed))).toBe(true)
      } finally {
        await context.release()
      }
    },
    TEST_TIMEOUT,
  )
})
