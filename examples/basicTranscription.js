const fs = require('fs')
const path = require('path')
const { initWhisper } = require('../lib/index')

// Configuration
const MODEL_PATH = path.join(__dirname, '../whisper.cpp/models/ggml-tiny.en.bin')
const AUDIO_PATH = path.join(__dirname, '../whisper.cpp/samples/jfk.wav')

/**
 * Load WAV file and convert to ArrayBuffer (skipping WAV header)
 */
function loadWavFile(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Audio file not found: ${filePath}`)
  }

  const buffer = fs.readFileSync(filePath)

  // WAV header is typically 44 bytes
  const headerSize = 44
  const audioData = buffer.slice(headerSize)

  return audioData.buffer.slice(
    audioData.byteOffset,
    audioData.byteOffset + audioData.byteLength,
  )
}

async function main() {
  console.log('Loading whisper model...')
  console.log(`Model: ${MODEL_PATH}`)

  // Initialize whisper context
  const context = await initWhisper({
    filePath: MODEL_PATH,
    useGpu: true, // Set to false if GPU is not available
  })

  console.log('Model loaded successfully!')
  console.log()

  // Load audio file
  console.log(`Loading audio: ${AUDIO_PATH}`)
  const audioBuffer = loadWavFile(AUDIO_PATH)
  console.log(`Audio size: ${audioBuffer.byteLength} bytes`)
  console.log()

  // Transcribe with progress and segment callbacks
  console.log('Starting transcription...')
  console.log()

  const startTime = Date.now()

  const { promise } = context.transcribeData(audioBuffer, {
    language: 'en',
    temperature: 0.0,
    nProcessors: 1,
    onProgress: (progress) => {
      process.stdout.write(`\rProgress: ${progress}%`)
    },
    onNewSegments: (result) => {
      // Clear progress line and print segment
      process.stdout.write('\r' + ' '.repeat(20) + '\r')
      result.segments.forEach((segment) => {
        const t0 = (segment.t0 / 1000).toFixed(2)
        const t1 = (segment.t1 / 1000).toFixed(2)
        console.log(`[${t0}s - ${t1}s] ${segment.text}`)
      })
    },
  })

  const result = await promise

  const elapsed = Date.now() - startTime

  console.log()
  console.log('='.repeat(50))
  console.log('Transcription complete!')
  console.log('='.repeat(50))
  console.log()
  console.log('Full text:')
  console.log(result.result)
  console.log()
  console.log(`Time elapsed: ${elapsed}ms`)
  console.log(`Segments: ${result.segments.length}`)
  console.log(`Aborted: ${result.isAborted}`)

  // Release context
  await context.release()
  console.log()
  console.log('Context released.')
}

main().catch(console.error)
