export interface NativeContextOptions {
  filePath: string,
  modelUrl?: string,
  /** Use flash attention. Always on when the model runs on the Hexagon NPU (snapdragon variant). */
  useFlashAttn?: boolean,
  useGpu?: boolean,
  maxModelBytes?: number,
}

export interface NativeVadContextOptions {
  /**
   * Speech model file: a vad.cpp GGUF (Silero, FireRedVAD) or whisper.cpp's
   * `ggml-silero-*.bin`
   */
  filePath: string,
  modelUrl?: string,
  /** @deprecated Ignored. VAD always runs on the CPU (vad.cpp's kernels), which is fastest for a single stream. */
  useGpu?: boolean,
  /** Number of threads to use during computation (Default: 1, best for the small per-hop graphs) */
  nThreads?: number,
  maxModelBytes?: number,
}

export interface NativeParakeetContextOptions {
  filePath: string,
  modelUrl?: string,
  useGpu?: boolean,
  maxModelBytes?: number,
}

export interface TranscribeOptions {
  language?: string
  translate?: boolean
  maxThreads?: number
  maxContext?: number
  maxLen?: number
  tokenTimestamps?: boolean
  tdrzEnable?: boolean
  wordThold?: number
  offset?: number
  duration?: number
  temperature?: number
  temperatureInc?: number
  beamSize?: number
  bestOf?: number
  prompt?: string
  nProcessors?: number
  /**
   * Progress callback, the progress is between 0 and 100
   */
  onProgress?: (progress: number) => void
  /**
   * Callback when new segments are transcribed
   */
  onNewSegments?: (result: TranscribeNewSegmentsResult) => void
}

export interface TranscribeNewSegmentsResult {
  nNew: number
  totalNNew: number
  result: string
  segments: TranscribeResult['segments']
}

export interface ParakeetTranscribeOptions {
  /** Number of threads to use during computation. */
  maxThreads?: number
  /** Override the model audio context size (0 uses the model default). */
  audioCtx?: number
}

export interface TranscribeResult {
  language?: string
  result: string
  segments: Array<{
    text: string
    t0: number
    t1: number
  }>
  /**
   * True when the job was interrupted by stop() or release(); `result` and
   * `segments` then hold what was decoded before the interruption.
   */
  isAborted: boolean
}

export interface VadOptions {
  /** Probability threshold to consider as speech (Default: 0.5) */
  threshold?: number,
  /**
   * Probability below which audio counts as silence while in speech, the lower
   * edge of the hysteresis (Default: max(threshold - 0.15, 0.01))
   */
  negThreshold?: number,
  /** Min duration for a valid speech segment in ms (Default: 250) */
  minSpeechDurationMs?: number,
  /** Min silence duration to consider speech as ended in ms (Default: 100) */
  minSilenceDurationMs?: number,
  /** Max duration of a speech segment before forcing a new segment in seconds (Default: no limit) */
  maxSpeechDurationS?: number,
  /** Padding added before and after speech segments in ms (Default: 30) */
  speechPadMs?: number,
  /**
   * @deprecated Ignored. It applied to whisper.cpp's windowed VAD; detection
   * now keeps the model state across the whole input, so nothing overlaps.
   */
  samplesOverlap?: number,
}

/** Speech segment, times in 10 ms units (centiseconds) */
export interface VadSegment {
  t0: number
  t1: number
}

/**
 * Stateful voice activity detection over live audio, created with
 * `WhisperVadContext.createStream()`.
 *
 * The model state carries over between pushes, so every sample is analyzed
 * once and each probability takes all earlier audio into account (Silero's
 * exact streaming semantics). Calls on one stream run in call order.
 */
export interface WhisperVadStream {
  /** Sample rate the stream expects (16000) */
  readonly sampleRate: number
  /** Samples covered by each probability (512 = 32 ms for Silero) */
  readonly hopSamples: number
  /**
   * Feed 16 kHz mono audio of any length: PCM16 (ArrayBuffer, Uint8Array,
   * Int16Array) or float samples (Float32Array). It is copied, so the buffer
   * may be reused right away. Resolves with the speech probabilities that
   * became available, in order: probability `i` of the stream covers samples
   * `[i * hopSamples, (i + 1) * hopSamples)`.
   */
  push(audio: ArrayBuffer | Uint8Array | Int16Array | Float32Array): Promise<number[]>
  /** End of audio: resolves with the probabilities still pending (a partial hop is zero padded). */
  flush(): Promise<number[]>
  /** Clear the model state and buffered audio to start a new audio stream. */
  reset(): Promise<void>
  /** Free the stream. Releasing the context frees its streams as well. */
  release(): Promise<void>
}

export interface BenchResult {
  config: string
  nThreads: number
  encodeMs: number
  decodeMs: number
  batchdMs: number
  promptMs: number
}

export interface WhisperContext {
  new (options: NativeContextOptions): WhisperContext
  transcribe(
    filePath: string,
    options?: TranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  transcribeFile(
    filePath: string,
    options?: TranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  transcribeData(
    audioData: ArrayBuffer,
    options?: TranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  bench(nThreads: number): Promise<BenchResult>
  release(): Promise<void>
  /**
   * Release the context synchronously. Intended for process exit handlers,
   * where release() cannot be awaited.
   */
  releaseSync(): void
  getModelInfo(): object
  
  // static methods
  /**
   * Release every live context of this type synchronously.
   */
  releaseAllSync(): void
  toggleNativeLog(
    enable: boolean,
    callback?: (level: string, text: string) => void,
  ): void
}

export interface ParakeetContext {
  new (options: NativeParakeetContextOptions): ParakeetContext
  transcribe(
    filePath: string,
    options?: ParakeetTranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  transcribeFile(
    filePath: string,
    options?: ParakeetTranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  transcribeData(
    audioData: ArrayBuffer,
    options?: ParakeetTranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  release(): Promise<void>
  /**
   * Release the context synchronously. Intended for process exit handlers,
   * where release() cannot be awaited.
   */
  releaseSync(): void
  getModelInfo(): object

  // static methods
  /**
   * Release every live context of this type synchronously.
   */
  releaseAllSync(): void
  toggleNativeLog(
    enable: boolean,
    callback?: (level: string, text: string) => void,
  ): void
}

export interface WhisperVadContext {
  new (options: NativeVadContextOptions): WhisperVadContext
  /** Sample rate the model expects (16000) */
  readonly sampleRate: number
  /** Samples between consecutive speech probabilities (512 for Silero) */
  readonly hopSamples: number
  /** Model architecture, e.g. `silero` */
  readonly arch: string
  detectSpeech(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechFile(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechData(
    audioData: ArrayBuffer,
    options?: VadOptions,
  ): Promise<VadSegment[]>
  /** Create a stateful stream for live audio (see `WhisperVadStream`). */
  createStream(): Promise<WhisperVadStream>
  /** Free the model and every stream created on it. */
  release(): Promise<void>
  /**
   * Release the context synchronously. Intended for process exit handlers,
   * where release() cannot be awaited.
   */
  releaseSync(): void
  getModelInfo(): object
  
  // static methods
  /**
   * Release every live context of this type synchronously.
   */
  releaseAllSync(): void
  toggleNativeLog(
    enable: boolean,
    callback?: (level: string, text: string) => void,
  ): void
}

export interface Module {
  WhisperContext: WhisperContext
  WhisperVadContext: WhisperVadContext
  ParakeetContext: ParakeetContext
}

export type LibVariant = 'default' | 'vulkan' | 'cuda' | 'snapdragon'

const getPlatformPackageName = (variant?: LibVariant): string => {
  const platform = process.platform
  const arch = process.arch
  const variantSuffix = variant && variant !== 'default' ? `-${variant}` : ''
  return `@fugood/node-whisper-${platform}-${arch}${variantSuffix}`
}

const loadPlatformPackage = async (
  packageName: string,
): Promise<Module | null> => {
  try {
    return (await import(packageName)) as Module
  } catch (error) {
    return null
  }
}

const isNodeRuntime = (): boolean => {
  return (
    typeof process !== 'undefined' &&
    !!process.versions &&
    !!process.versions.node
  )
}

const normalizeLoadedModule = (module: unknown): Module => {
  const maybeModule = module as Module & { default?: Module }
  return maybeModule.WhisperContext ? maybeModule : (maybeModule.default as Module)
}

const importEsm = new Function(
  'specifier',
  'return import(specifier)',
) as (specifier: string) => Promise<unknown>

const loadWasmPackage = async (): Promise<Module> => {
  const module = await importEsm('@fugood/node-whisper-wasm')
  return normalizeLoadedModule(module)
}

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  if (!isNodeRuntime()) {
    if (variant && variant !== 'default') {
      console.warn(`Browser WASM build ignores native variant "${variant}"`)
    }
    return loadWasmPackage()
  }

  const packageName = getPlatformPackageName(variant)

  // Snapdragon: ggml-hexagon loads the HTP skels (libggml-htp-*.so) from
  // ADSP_LIBRARY_PATH, which the platform package ships next to index.node.
  if (variant === 'snapdragon') {
    if (!process.env.ADSP_LIBRARY_PATH) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const path = require('path') as typeof import('path')
        process.env.ADSP_LIBRARY_PATH = path.dirname(require.resolve(packageName))
      } catch {
        /* no-op */
      }
    }
    if (!process.env.GGML_HEXAGON_NDEV) process.env.GGML_HEXAGON_NDEV = '16'
  }

  // Try to load the requested variant
  let module = await loadPlatformPackage(packageName)
  if (module) {
    return module
  }

  // Fallback to default if variant not found
  module = await loadPlatformPackage(getPlatformPackageName())
  if (module) {
    if (variant && variant !== 'default') {
      console.warn(
        `Not found package for variant "${variant}", fallback to default`,
      )
    }
    return module
  }

  // Final fallback to local build
  console.warn(`Not found package for your platform, fallback to local build`)
  try {
    // @ts-ignore
    return (await import('../build/Release/index.node')) as Module
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    throw new Error(`Failed to load whisper.node: ${errorMessage}`)
  }
}
