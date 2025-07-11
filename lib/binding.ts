export interface WhisperModelOptions {
  model: string
  n_threads?: number
  n_processors?: number
  use_gpu?: boolean
  language?: string
  translate?: boolean
  no_context?: boolean
  single_segment?: boolean
  print_special?: boolean
  print_progress?: boolean
  print_realtime?: boolean
  print_timestamps?: boolean
  token_timestamps?: boolean
  thold_pt?: number
  thold_ptsum?: number
  max_len?: number
  split_on_word?: boolean
  max_tokens?: number
  speed_up?: boolean
  audio_ctx?: number
  prompt_tokens?: number[]
  prompt_n_tokens?: number
  suppress_blank?: boolean
  suppress_non_speech_tokens?: boolean
  temperature?: number
  max_initial_ts?: number
  length_penalty?: number
  temperature_inc?: number
  entropy_thold?: number
  logprob_thold?: number
  no_speech_thold?: number
  greedy?: {
    best_of?: number
  }
  beam_search?: {
    beam_size?: number
    patience?: number
  }
}

export interface VadModelOptions {
  model: string
  n_threads?: number
  use_gpu?: boolean
}

export interface TranscribeOptions {
  language?: string
  translate?: boolean
  offset_ms?: number
  duration_ms?: number
  max_len?: number
  split_on_word?: boolean
  token_timestamps?: boolean
  speed_up?: boolean
  prompt?: string
  suppress_blank?: boolean
  suppress_non_speech_tokens?: boolean
  temperature?: number
  max_initial_ts?: number
  length_penalty?: number
  temperature_inc?: number
  entropy_thold?: number
  logprob_thold?: number
  no_speech_thold?: number
}

export interface TranscribeResult {
  result: string
  segments: Array<{
    start: number
    end: number
    text: string
    tokens?: number[]
    words?: Array<{
      start: number
      end: number
      word: string
      probability: number
    }>
  }>
  language: string
}

export interface VadOptions {
  offset_ms?: number
  duration_ms?: number
}

export interface VadSegment {
  t0: number
  t1: number
}

export interface WhisperContext {
  new (options: WhisperModelOptions): WhisperContext
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
  getSystemInfo(): string
  getModelInfo(): object
}

export interface WhisperVadContext {
  new (options: VadModelOptions): WhisperVadContext
  detectSpeech(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechFile(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechData(
    audioData: ArrayBuffer,
    options?: VadOptions,
  ): Promise<VadSegment[]>
  release(): Promise<void>
  getSystemInfo(): string
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
