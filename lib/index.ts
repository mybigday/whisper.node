import { loadModule } from './binding';
import type {
  WhisperContext,
  WhisperVadContext,
  NativeContextOptions,
  NativeVadContextOptions,
  TranscribeOptions,
  TranscribeResult,
  TranscribeNewSegmentsResult,
  VadOptions,
  VadSegment,
  LibVariant,
  Module
} from './binding';

// Export types
export type {
  WhisperContext,
  WhisperVadContext,
  NativeContextOptions,
  NativeVadContextOptions,
  TranscribeOptions,
  TranscribeResult,
  TranscribeNewSegmentsResult,
  VadOptions,
  VadSegment,
  LibVariant
};

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
    } else {
      moduleCache.WhisperContext.toggleNativeLog(false)
      moduleCache.WhisperVadContext.toggleNativeLog(false)
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
      logListeners.splice(logListeners.indexOf(listener), 1)
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
    // Don't automatically setup logging on module load
    // Users should explicitly call toggleNativeLog(true) if they want logging
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
  return new module.WhisperContext(options);
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
  return new module.WhisperVadContext(options);
};

// Default export
export default {
  initWhisper,
  initWhisperVad,
  loadWhisperModule,
  toggleNativeLog,
  addNativeLogListener
};
