import { loadModule } from './binding';
import type {
  WhisperContext,
  WhisperVadContext,
  ParakeetContext,
  NativeContextOptions,
  NativeVadContextOptions,
  NativeParakeetContextOptions,
  TranscribeOptions,
  ParakeetTranscribeOptions,
  TranscribeResult,
  TranscribeNewSegmentsResult,
  VadOptions,
  VadSegment,
  BenchResult,
  LibVariant,
  Module
} from './binding';

// Export types
export type {
  WhisperContext,
  WhisperVadContext,
  ParakeetContext,
  NativeContextOptions,
  NativeVadContextOptions,
  NativeParakeetContextOptions,
  TranscribeOptions,
  ParakeetTranscribeOptions,
  TranscribeResult,
  TranscribeNewSegmentsResult,
  VadOptions,
  VadSegment,
  BenchResult,
  LibVariant
};

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

export const WASM_CONFIG_PATHS = {
  index: '',
  js: '',
  wasm: '',
  threadsJs: '',
  threadsWasm: '',
  worker: '',
}

export const configureWasm = (_options: WasmRuntimeOptions) => {
  throw new Error('configureWasm is only available in the browser WASM build')
}

export const isWasmThreadsSupported = () => false

// Global module cache
let moduleCache: Module | null = null;

// Log management
const logListeners: Array<(level: string, text: string) => void> = []

const logCallback = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

let logEnabled = false

const refreshNativeLogSetup = () => {
  if (moduleCache) {
    if (logEnabled) {
      moduleCache.WhisperContext.toggleNativeLog(logEnabled, logCallback)
      moduleCache.WhisperVadContext.toggleNativeLog(logEnabled, logCallback)
      // Older prebuilt packages may not export ParakeetContext
      moduleCache.ParakeetContext?.toggleNativeLog(logEnabled, logCallback)
    } else {
      moduleCache.WhisperContext.toggleNativeLog(false)
      moduleCache.WhisperVadContext.toggleNativeLog(false)
      moduleCache.ParakeetContext?.toggleNativeLog(false)
    }
  }
}

export const toggleNativeLog = async (enable: boolean) => {
  logEnabled = enable
  refreshNativeLogSetup()
}

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      const index = logListeners.indexOf(listener)
      if (index >= 0) {
        logListeners.splice(index, 1)
      }
    },
  }
}

/**
 * Load the whisper.node module with the specified variant
 * @param variant - The backend variant to use ('default', 'vulkan', 'cuda')
 * @returns Promise that resolves to the loaded module
 */
export const loadWhisperModule = async (variant?: LibVariant): Promise<Module> => {
  if (!moduleCache) {
    moduleCache = await loadModule(variant);
    refreshNativeLogSetup()
  }
  return moduleCache;
};


/**
 * Create a new WhisperContext for transcription
 * @param options - Configuration options for the whisper model
 * @param variant - Optional backend variant to use
 * @returns Promise that resolves to a WhisperContext instance
 */
export const initWhisper = async (
  options: NativeContextOptions,
  variant?: LibVariant
): Promise<WhisperContext> => {
  const module = await loadWhisperModule(variant);
  return await (new module.WhisperContext(options) as
    | WhisperContext
    | Promise<WhisperContext>);
};

/**
 * Create a new WhisperVadContext for voice activity detection
 * @param options - Configuration options for the VAD model
 * @param variant - Optional backend variant to use
 * @returns Promise that resolves to a WhisperVadContext instance
 */
export const initWhisperVad = async (
  options: NativeVadContextOptions,
  variant?: LibVariant
): Promise<WhisperVadContext> => {
  const module = await loadWhisperModule(variant);
  return await (new module.WhisperVadContext(options) as
    | WhisperVadContext
    | Promise<WhisperVadContext>);
};

/**
 * Create a new ParakeetContext for transcription with NVIDIA Parakeet models
 * @param options - Configuration options for the parakeet model
 * @param variant - Optional backend variant to use
 * @returns Promise that resolves to a ParakeetContext instance
 */
export const initParakeet = async (
  options: NativeParakeetContextOptions,
  variant?: LibVariant
): Promise<ParakeetContext> => {
  const module = await loadWhisperModule(variant);
  if (!module.ParakeetContext) {
    throw new Error(
      'ParakeetContext is not available in the loaded whisper.node binary, please update the platform package',
    );
  }
  return await (new module.ParakeetContext(options) as
    | ParakeetContext
    | Promise<ParakeetContext>);
};

export const isNativeLogEnabled = () => logEnabled

// Default export
export default {
  initWhisper,
  initWhisperVad,
  initParakeet,
  loadWhisperModule,
  toggleNativeLog,
  addNativeLogListener,
  configureWasm,
  isWasmThreadsSupported,
  isNativeLogEnabled,
  WASM_CONFIG_PATHS,
};
