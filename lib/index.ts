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

/**
 * Utility function to create AudioBuffer from Float32Array
 * @param audioData - Float32Array audio data
 * @param sampleRate - Sample rate (default: 16000)
 * @returns ArrayBuffer containing 16-bit PCM audio data
 */
export const createAudioBuffer = (audioData: Float32Array, sampleRate = 16000): ArrayBuffer => {
  const buffer = new ArrayBuffer(audioData.length * 2);
  const view = new DataView(buffer);

  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    view.setInt16(i * 2, sample * 0x7FFF, true);
  }

  return buffer;
};

/**
 * Utility function to convert ArrayBuffer to Float32Array
 * @param buffer - ArrayBuffer containing 16-bit PCM audio data
 * @returns Float32Array with normalized audio data
 */
export const audioBufferToFloat32Array = (buffer: ArrayBuffer): Float32Array => {
  const view = new DataView(buffer);
  const samples = new Float32Array(buffer.byteLength / 2);

  for (let i = 0; i < samples.length; i++) {
    samples[i] = view.getInt16(i * 2, true) / 0x7FFF;
  }

  return samples;
};

/**
 * Get system information about the loaded whisper module
 * @param variant - Optional backend variant to use
 * @returns Promise that resolves to system information string
 */
export const getSystemInfo = async (variant?: LibVariant): Promise<string> => {
  const module = await loadWhisperModule(variant);
  const context = new module.WhisperContext({ model: '' });
  try {
    return context.getSystemInfo();
  } finally {
    await context.release();
  }
};

// Default export
export default {
  initWhisper,
  initWhisperVad,
  createAudioBuffer,
  audioBufferToFloat32Array,
  getSystemInfo,
  loadWhisperModule
};
