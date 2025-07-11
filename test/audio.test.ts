const fs = require('fs');
const path = require('path');
const { createAudioBuffer, audioBufferToFloat32Array } = require('../lib/index');

describe('Audio utilities', () => {
  test('should create audio buffer from Float32Array', () => {
    const float32Data = new Float32Array([0.5, -0.5, 1.0, -1.0, 0.0]);
    const buffer = createAudioBuffer(float32Data);

    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBe(10); // 5 samples * 2 bytes per sample

    // Verify the data is correctly converted
    const int16View = new Int16Array(buffer);
    expect(int16View[0]).toBe(16383); // 0.5 * 32767
    expect(int16View[1]).toBe(-16383); // -0.5 * 32767
    expect(int16View[2]).toBe(32767); // 1.0 * 32767 (clamped)
    expect(int16View[3]).toBe(-32767); // -1.0 * 32767 (clamped)
    expect(int16View[4]).toBe(0); // 0.0 * 32767
  });

  test('should convert ArrayBuffer to Float32Array', () => {
    // Create a test buffer with known values
    const buffer = new ArrayBuffer(10);
    const int16View = new Int16Array(buffer);
    int16View[0] = 16383; // Should convert to ~0.5
    int16View[1] = -16383; // Should convert to ~-0.5
    int16View[2] = 32767; // Should convert to ~1.0
    int16View[3] = -32767; // Should convert to ~-1.0
    int16View[4] = 0; // Should convert to 0.0

    const float32Data = audioBufferToFloat32Array(buffer);

    expect(float32Data).toBeInstanceOf(Float32Array);
    expect(float32Data.length).toBe(5);
    expect(float32Data[0]).toBeCloseTo(0.5, 3);
    expect(float32Data[1]).toBeCloseTo(-0.5, 3);
    expect(float32Data[2]).toBeCloseTo(1.0, 3);
    expect(float32Data[3]).toBeCloseTo(-1.0, 3);
    expect(float32Data[4]).toBeCloseTo(0.0, 3);
  });

  test('should handle edge cases in audio conversion', () => {
    // Test with empty array
    const emptyBuffer = createAudioBuffer(new Float32Array(0));
    expect(emptyBuffer.byteLength).toBe(0);

    // Test with values outside [-1, 1] range (should be clamped)
    const clampedBuffer = createAudioBuffer(new Float32Array([2.0, -2.0]));
    const clampedView = new Int16Array(clampedBuffer);
    expect(clampedView[0]).toBe(32767); // Clamped to 1.0
    expect(clampedView[1]).toBe(-32767); // Clamped to -1.0
  });

  test('should verify JFK audio sample exists', () => {
    const jfkPath = path.join(__dirname, '../whisper.cpp/samples/jfk.wav');
    expect(fs.existsSync(jfkPath)).toBe(true);

    const stats = fs.statSync(jfkPath);
    expect(stats.size).toBeGreaterThan(0);
    expect(stats.size).toBeLessThan(1024 * 1024); // Should be less than 1MB
  });
});
