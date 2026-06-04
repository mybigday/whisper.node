export interface NativeContextOptions {
  filePath: string
  modelUrl?: string
  useFlashAttn?: boolean
  useGpu?: boolean
  maxModelBytes?: number
  cacheModel?: boolean
  modelCacheName?: string
  modelCacheKey?: string
  worker?: boolean
}

export interface NativeVadContextOptions {
  filePath: string
  modelUrl?: string
  useGpu?: boolean
  nThreads?: number
  maxModelBytes?: number
  cacheModel?: boolean
  modelCacheName?: string
  modelCacheKey?: string
  worker?: boolean
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
  onProgress?: (progress: number) => void
  onNewSegments?: (result: TranscribeNewSegmentsResult) => void
}

export interface TranscribeNewSegmentsResult {
  nNew: number
  totalNNew: number
  result: string
  segments: TranscribeResult['segments']
}

export interface TranscribeResult {
  language?: string
  result: string
  segments: Array<{
    text: string
    t0: number
    t1: number
  }>
  isAborted: boolean
}

export interface VadOptions {
  threshold?: number
  minSpeechDurationMs?: number
  minSilenceDurationMs?: number
  maxSpeechDurationS?: number
  speechPadMs?: number
  samplesOverlap?: number
}

export interface VadSegment {
  t0: number
  t1: number
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
    audioData: ArrayBuffer | ArrayBufferView | Float32Array,
    options?: TranscribeOptions,
  ): {
    stop: () => Promise<void>
    promise: Promise<TranscribeResult>
  }
  bench(nThreads: number): Promise<BenchResult>
  release(): Promise<void>
  getModelInfo(): object
}

export interface WhisperVadContext {
  detectSpeech(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechFile(filePath: string, options?: VadOptions): Promise<VadSegment[]>
  detectSpeechData(
    audioData: ArrayBuffer | ArrayBufferView | Float32Array,
    options?: VadOptions,
  ): Promise<VadSegment[]>
  release(): Promise<void>
  getModelInfo(): object
}

export interface Module {
  WhisperContext: {
    new (options: NativeContextOptions): Promise<WhisperContext>
    toggleNativeLog(
      enable: boolean,
      callback?: (level: string, text: string) => void,
    ): void | Promise<void>
  }
  WhisperVadContext: {
    new (options: NativeVadContextOptions): Promise<WhisperVadContext>
    toggleNativeLog(
      enable: boolean,
      callback?: (level: string, text: string) => void,
    ): void | Promise<void>
  }
}

export interface WasmRuntimeOptions {
  worker?: boolean
  workerUrl?: string
  workerPath?: string
  indexScriptUrl?: string
  scriptUrl?: string
  runtimeScriptUrl?: string
  jsPath?: string
  wasmPath?: string
  threads?: boolean
  locateFileBaseUrl?: string
  locateFile?: (path: string, prefix: string) => string
  mainScriptUrlOrBlob?: string | Blob
  modelCacheName?: string
  moduleFactory?: (options?: Record<string, unknown>) => Promise<unknown> | unknown
  moduleOptions?: Record<string, unknown>
  print?: (text: string) => void
  printErr?: (text: string) => void
}

export declare const WhisperContext: Module['WhisperContext']
export declare const WhisperVadContext: Module['WhisperVadContext']
export declare const DEFAULT_WASM_MODEL_SIZE_LIMIT_BYTES: number
export declare const MAX_WASM_THREADS: number
export declare const WASM_CONFIG_PATHS: {
  index: string
  js: string
  wasm: string
  threadsJs: string
  threadsWasm: string
  worker: string
}

export declare function configureWasm(options: WasmRuntimeOptions): void
export declare function isWasmThreadsSupported(): boolean
export declare function loadWasmModule(): Promise<unknown>
export declare function loadWhisperModule(): Promise<Module>
export declare function initWhisper(
  options: NativeContextOptions,
): Promise<WhisperContext>
export declare function initWhisperVad(
  options: NativeVadContextOptions,
): Promise<WhisperVadContext>
export declare function toggleNativeLog(
  enable: boolean,
  callback?: (level: string, text: string) => void,
): Promise<void>
export declare function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void }

declare const _default: {
  WhisperContext: typeof WhisperContext
  WhisperVadContext: typeof WhisperVadContext
  configureWasm: typeof configureWasm
  loadWasmModule: typeof loadWasmModule
  loadWhisperModule: typeof loadWhisperModule
  initWhisper: typeof initWhisper
  initWhisperVad: typeof initWhisperVad
  toggleNativeLog: typeof toggleNativeLog
  addNativeLogListener: typeof addNativeLogListener
  isWasmThreadsSupported: typeof isWasmThreadsSupported
  DEFAULT_WASM_MODEL_SIZE_LIMIT_BYTES: typeof DEFAULT_WASM_MODEL_SIZE_LIMIT_BYTES
  MAX_WASM_THREADS: typeof MAX_WASM_THREADS
  WASM_CONFIG_PATHS: typeof WASM_CONFIG_PATHS
}

export default _default
