#!/usr/bin/env node

const path = require('path')
const fs = require('fs')
const http = require('http')
const https = require('https')
const { pipeline } = require('stream/promises')

const modelsDir = path.join(__dirname, '../whisper.cpp/models')

// Ensure the models directory exists
if (!fs.existsSync(modelsDir)) {
  console.error(`Models directory does not exist: ${modelsDir}`)
  process.exit(1)
}

const requiredModels = [
  {
    name: 'tiny.en',
    fileName: 'ggml-tiny.en.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin',
  },
  {
    name: 'silero-v5.1.2',
    fileName: 'ggml-silero-v5.1.2.bin',
    url: 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin',
  },
  {
    name: 'silero-v6.2.0',
    fileName: 'ggml-silero-v6.2.0.bin',
    url: 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin',
  },
]

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return 'unknown size'
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
  }
  return `${(bytes / 1024).toFixed(1)} KiB`
}

function downloadWithHttp(url, destination, redirects = 0) {
  return new Promise((resolve, reject) => {
    const request = (url.startsWith('https:') ? https : http).get(url, (response) => {
      const status = response.statusCode || 0
      const location = response.headers.location

      if (status >= 300 && status < 400 && location) {
        response.resume()
        if (redirects >= 5) {
          reject(new Error(`Too many redirects while downloading ${url}`))
          return
        }
        downloadWithHttp(new URL(location, url).href, destination, redirects + 1)
          .then(resolve, reject)
        return
      }

      if (status < 200 || status >= 300) {
        response.resume()
        reject(new Error(`HTTP ${status} while downloading ${url}`))
        return
      }

      pipeline(response, fs.createWriteStream(destination)).then(resolve, reject)
    })

    request.on('error', reject)
  })
}

async function downloadFile(url, destination) {
  const tempDestination = `${destination}.tmp-${process.pid}`
  await fs.promises.rm(tempDestination, { force: true })

  try {
    await downloadWithHttp(url, tempDestination)
    await fs.promises.rename(tempDestination, destination)
  } catch (error) {
    await fs.promises.rm(tempDestination, { force: true })
    throw error
  }
}

async function ensureModel(model) {
  const destination = path.join(modelsDir, model.fileName)
  if (fs.existsSync(destination)) {
    const stats = await fs.promises.stat(destination)
    console.log(`Model ${model.name} already exists (${formatBytes(stats.size)}). Skipping download.`)
    return
  }

  console.log(`Downloading ${model.name} from ${model.url}`)
  await downloadFile(model.url, destination)
  const stats = await fs.promises.stat(destination)
  console.log(`Saved ${model.fileName} (${formatBytes(stats.size)})`)
}

/**
 * Main function
 */
async function main() {
  try {
    for (const model of requiredModels) {
      await ensureModel(model)
    }
  } catch (error) {
    console.error('Error downloading models:', error.message)
    process.exit(1)
  }
}

// Run the main function
main().catch((error) => {
  console.error('Unexpected error:', error)
  process.exit(1)
})
