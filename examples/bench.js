const path = require('path')
const { initWhisper } = require('../lib/index')

// Optional native variant, e.g. WHISPER_VARIANT=snapdragon
const VARIANT = process.env.WHISPER_VARIANT || 'default'

// Configuration
const MODEL_PATH = path.join(__dirname, '../test/models/ggml-tiny.en.bin')

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2)
  const modelPath = args[0] || MODEL_PATH
  const nThreads = parseInt(args[1], 10) || 4

  console.log('Whisper.node Benchmark')
  console.log('='.repeat(50))
  console.log()
  console.log(`Model: ${modelPath}`)
  console.log(`Threads: ${nThreads}`)
  console.log()

  // Initialize whisper context
  console.log('Loading model...')
  const context = await initWhisper({
    filePath: modelPath,
    useGpu: process.env.WHISPER_USE_GPU !== '0',
  }, VARIANT)
  console.log('Model loaded!')
  console.log()

  // Run benchmark
  console.log('Running benchmark...')
  const result = await context.bench(nThreads)
  console.log()

  // Print results
  console.log(`System: ${result.config}`)
  console.log()
  console.log('|    Encode |    Decode |     Batch |    Prompt |')
  console.log('| --------: | --------: | --------: | --------: |')
  console.log(`| ${result.encodeMs.toFixed(2).padStart(6)} ms | ${result.decodeMs.toFixed(2).padStart(6)} ms | ${result.batchdMs.toFixed(2).padStart(6)} ms | ${result.promptMs.toFixed(2).padStart(6)} ms |`)
  console.log()

  // Release context
  await context.release()
  console.log('Done.')
}

main().catch(console.error)
