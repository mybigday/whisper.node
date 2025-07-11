import { loadModule } from './binding';
import type {
  WhisperContext,
  WhisperVadContext,
  WhisperModelOptions,
  VadModelOptions,
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
  VadResult,
  LibVariant,
  Module
} from './binding';

// Export types
export type {
  WhisperContext,
  WhisperVadContext,
  WhisperModelOptions,
  VadModelOptions,
  TranscribeOptions,
  TranscribeResult,
  VadOptions,
  VadResult,
  LibVariant
};

// Global module cache
let moduleCache: Module | null = null;

/**
 * Load the whisper.node module with the specified variant
 * @param variant - The backend variant to use ('default', 'vulkan', 'cuda')
 * @returns Promise that resolves to the loaded module
 */
export const loadWhisperModule = async (variant?: LibVariant): Promise<Module> => {
  if (!moduleCache) {
    moduleCache = await loadModule(variant);
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
  options: WhisperModelOptions,
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
  options: VadModelOptions,
  variant?: LibVariant
): Promise<WhisperVadContext> => {
  const module = await loadWhisperModule(variant);
  return new module.WhisperVadContext(options);
};

// Default export
export default {
  initWhisper,
  initWhisperVad,
  loadWhisperModule
};
