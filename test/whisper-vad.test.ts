import fs from 'fs'
import path from 'path'
import {
  initWhisperVad,
} from '../lib/index'

// Test configuration
const TEST_TIMEOUT = 30000 // 30 seconds timeout for model loading
const SAMPLE_AUDIO_PATH = path.join(__dirname, '../whisper.cpp/samples/jfk.wav')

// Helper function to load WAV file and convert to ArrayBuffer
function loadWavFile(filePath: string) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Audio file not found: ${filePath}`)
  }

  const buffer = fs.readFileSync(filePath)

  // Simple WAV file parsing - skip header and get audio data
  // WAV header is typically 44 bytes
  const headerSize = 44
  const audioData = buffer.slice(headerSize)

  // Convert to ArrayBuffer
  return audioData.buffer.slice(
    audioData.byteOffset,
    audioData.byteOffset + audioData.byteLength,
  )
}

describe('Voice Activity Detection (VAD)', () => {
  const vadModelPath = path.join(__dirname, '../whisper.cpp/models/ggml-silero-v5.1.2.bin');

  test('should load VAD model and get system info', async () => {

    const context = await initWhisperVad({
      model: vadModelPath,
      use_gpu: false,
      n_threads: 2
    });

    expect(context).toBeDefined();
    expect(typeof context.getSystemInfo).toBe('function');

    const systemInfo = context.getSystemInfo();
    expect(typeof systemInfo).toBe('string');

    await context.release();
  }, TEST_TIMEOUT);

  test('should detect speech in JFK audio sample', async () => {
    const audioBuffer = loadWavFile(SAMPLE_AUDIO_PATH);

    const context = await initWhisperVad({
      model: vadModelPath,
      use_gpu: false,
      n_threads: 2
    });

    const result = await context.detectSpeechData(audioBuffer, {
      // VAD options
    });

    expect(result).toBeDefined();
    expect(typeof result.is_speech).toBe('boolean');
    expect(typeof result.speech_probability).toBe('number');
    expect(Array.isArray(result.speech_timestamps)).toBe(true);

    // JFK sample should contain speech
    expect(result.is_speech).toBe(true);
    expect(result.speech_probability).toBeGreaterThan(0);

    console.log('VAD result:', {
      is_speech: result.is_speech,
      speech_probability: result.speech_probability,
      segments: result.speech_timestamps.length
    });

    await context.release();
  }, TEST_TIMEOUT);

  test('should detect no speech in silence', async () => {
    // Create silent audio buffer
    const silentBuffer = new ArrayBuffer(32000); // 1 second of silence

    const context = await initWhisperVad({
      model: vadModelPath,
      use_gpu: false,
      n_threads: 2
    });

    const result = await context.detectSpeechData(silentBuffer);

    expect(result).toBeDefined();
    expect(typeof result.is_speech).toBe('boolean');
    expect(typeof result.speech_probability).toBe('number');
    expect(Array.isArray(result.speech_timestamps)).toBe(true);

    // Silent audio should not contain speech
    expect(result.is_speech).toBe(false);
    expect(result.speech_probability).toBeLessThan(0.5);

    await context.release();
  }, TEST_TIMEOUT);

  test('should handle VAD errors gracefully', async () => {
    const invalidModelPath = 'nonexistent/vad-model.bin';

    expect(initWhisperVad({ model: invalidModelPath }))
  })
})
