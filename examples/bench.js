const path = require('path')
const { initWhisper } = require('../lib/index')

// Configuration
const MODEL_PATH = path.join(__dirname, '../whisper.cpp/models/ggml-tiny.en.bin')

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
  const context = await initWhisper(
    {
      filePath: modelPath,
      useGpu: true, // Set to false to benchmark CPU only
      // useFlashAttn: true, // Recommend for GPU
    },
    process.env.WHISPER_LIB_VARIANT, // 'default' | 'vulkan' | 'cuda' | 'snapdragon'
  )
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
