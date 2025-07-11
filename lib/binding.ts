export interface NativeContextOptions {
  filePath: string,
  useFlashAttn?: boolean,
  useGpu?: boolean,
}

export interface NativeVadContextOptions {
  filePath: string,
  useGpu?: boolean,
  nThreads?: number,
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
}

export interface TranscribeResult {
  result: string
  segments: Array<{
    text: string
    t0: number
    t1: number
  }>
  isAborted: boolean
}

export interface VadOptions {
  /** Probability threshold to consider as speech (Default: 0.5) */
  threshold?: number,
  /** Min duration for a valid speech segment in ms (Default: 250) */
  minSpeechDurationMs?: number,
  /** Min silence duration to consider speech as ended in ms (Default: 100) */
  minSilenceDurationMs?: number,
  /** Max duration of a speech segment before forcing a new segment in seconds (Default: 30) */
  maxSpeechDurationS?: number,
  /** Padding added before and after speech segments in ms (Default: 30) */
  speechPadMs?: number,
  /** Overlap in seconds when copying audio samples from speech segment (Default: 0.1) */
  samplesOverlap?: number,
}

export interface VadSegment {
  t0: number
  t1: number
}

export interface WhisperContext {
  new (options: NativeContextOptions): WhisperContext
  transcribe(
    filePath: string,
    options?: TranscribeOptions,
  ): Promise<TranscribeResult>
  transcribeFile(
    filePath: string,
    options?: TranscribeOptions,
  ): Promise<TranscribeResult>
  transcribeData(
    audioData: ArrayBuffer,
    options?: TranscribeOptions,
  ): Promise<TranscribeResult>
  release(): Promise<void>
  getModelInfo(): object
}

export interface WhisperVadContext {
  new (options: NativeVadContextOptions): WhisperVadContext
  detectSpeech(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechFile(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechData(
    audioData: ArrayBuffer,
    options?: VadOptions,
  ): Promise<VadSegment[]>
  release(): Promise<void>
  getModelInfo(): object
}

export interface Module {
  WhisperContext: WhisperContext
  WhisperVadContext: WhisperVadContext
}

export type LibVariant = 'default' | 'vulkan' | 'cuda'

const getPlatformPackageName = (variant?: LibVariant): string => {
  const platform = process.platform
  const arch = process.arch
  const variantSuffix = variant && variant !== 'default' ? `-${variant}` : ''
  return `@whisper-node/whisper-${platform}-${arch}${variantSuffix}`
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

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  // Try to load the requested variant
  let module = await loadPlatformPackage(getPlatformPackageName(variant))
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
