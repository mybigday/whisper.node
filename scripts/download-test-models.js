#!/usr/bin/env node

const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

const isWindows = process.platform === 'win32'
const modelsDir = path.join(__dirname, '../whisper.cpp/models')

// Ensure the models directory exists
if (!fs.existsSync(modelsDir)) {
  console.error(`Models directory does not exist: ${modelsDir}`)
  process.exit(1)
}

/**
 * Execute a command and wait for it to complete
 * @param {string} command - The command to run
 * @param {string[]} args - Command arguments
 * @param {string} cwd - Working directory
 * @returns {Promise<void>}
 */
function runCommand(command, args, cwd) {
  return new Promise((resolve, reject) => {
    console.log(`$ ${command} ${args.join(' ')}`)
    const proc = spawn(command, args, {
      cwd,
      stdio: 'inherit',
      shell: isWindows ? 'cmd.exe' : false,
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`Command failed with exit code ${code}`))
      }
    })

    proc.on('error', (err) => {
      reject(err)
    })
  })
}

/**
 * Main function
 */
async function main() {
  try {
    if (isWindows) {
      // On Windows, use .cmd scripts
      await runCommand('download-ggml-model.cmd', ['tiny.en'], modelsDir)
      await runCommand('download-vad-model.cmd', ['silero-v5.1.2'], modelsDir)
    } else {
      // On Unix-like systems, use .sh scripts
      await runCommand('./download-ggml-model.sh', ['tiny.en'], modelsDir)
      await runCommand('./download-vad-model.sh', ['silero-v5.1.2'], modelsDir)
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
