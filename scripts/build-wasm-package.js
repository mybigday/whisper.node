#!/usr/bin/env node

const fs = require('fs')
const os = require('os')
const path = require('path')
const { execFileSync } = require('child_process')

const rootDir = path.resolve(__dirname, '..')
const buildRoot = path.join(rootDir, 'build-wasm')
const packageDir = path.join(rootDir, 'packages', 'node-whisper-wasm')
const packageWasmDir = path.join(packageDir, 'wasm')
const emCacheDir = process.env.EM_CACHE
  ? path.resolve(rootDir, process.env.EM_CACHE)
  : path.join(buildRoot, 'emcache')

const boolEnv = (name, defaultValue) => {
  const value = process.env[name]
  if (value == null || value === '') return defaultValue
  return !/^(0|false|off|no)$/i.test(value)
}

const hasArg = (name) => process.argv.includes(name)

const readArgValue = (name) => {
  const index = process.argv.indexOf(name)
  if (index < 0) return null
  const value = process.argv[index + 1]
  if (!value || value.startsWith('--')) {
    throw new Error(`${name} requires a value`)
  }
  return value
}

const webgpu = hasArg('--webgpu')
  ? true
  : hasArg('--no-webgpu')
    ? false
    : boolEnv('WASM_WEBGPU', false)
const threads = hasArg('--threads')
  ? true
  : hasArg('--no-threads')
    ? false
    : boolEnv('WASM_THREADS', false)
const singleFile = hasArg('--single-file')
  ? true
  : hasArg('--no-single-file')
    ? false
    : boolEnv('WASM_SINGLE_FILE', false)
const buildType = hasArg('--debug')
  ? 'Debug'
  : process.env.WASM_BUILD_TYPE || process.env.BUILD_TYPE || 'Release'
const buildVariant = `${webgpu ? 'webgpu' : 'cpu'}${threads ? '-threads' : ''}`
const buildDir = readArgValue('--build-dir')
  ? path.resolve(rootDir, readArgValue('--build-dir'))
  : process.env.WASM_BUILD_DIR
    ? path.resolve(rootDir, process.env.WASM_BUILD_DIR)
    : path.join(buildRoot, buildVariant)
const requestedGenerator = process.env.WASM_CMAKE_GENERATOR
const requestedCcache = boolEnv('WASM_CCACHE', true)

const run = (cmd, args, options = {}) => {
  console.log(`$ ${cmd} ${args.join(' ')}`)
  execFileSync(cmd, args, {
    cwd: rootDir,
    env: {
      ...process.env,
      EM_CACHE: emCacheDir,
    },
    stdio: 'inherit',
    ...options,
  })
}

const commandExists = (cmd) => {
  try {
    execFileSync(process.platform === 'win32' ? 'where' : 'which', [cmd], {
      stdio: 'ignore',
    })
    return true
  } catch {
    return false
  }
}

const readCmakeCacheValue = (dir, key) => {
  const cachePath = path.join(dir, 'CMakeCache.txt')
  if (!fs.existsSync(cachePath)) return null

  const cache = fs.readFileSync(cachePath, 'utf8')
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = cache.match(new RegExp(`^${escapedKey}:[^=]*=(.*)$`, 'm'))
  return match?.[1] ?? null
}

const preferredGenerator = () => {
  if (requestedGenerator) return requestedGenerator
  return commandExists('ninja') ? 'Ninja' : null
}

const generatorArgs = () => {
  const generator = preferredGenerator()
  if (!generator) return []

  const existing = readCmakeCacheValue(buildDir, 'CMAKE_GENERATOR')
  if (!existing) return ['-G', generator]
  if (existing === generator) return ['-G', generator]

  if (requestedGenerator) {
    throw new Error(
      `Build directory ${buildDir} already uses CMake generator "${existing}". ` +
        `Set WASM_BUILD_DIR to a new path or remove the directory before using "${generator}".`,
    )
  }

  console.log(
    `Keeping existing CMake generator "${existing}" for ${buildDir}. ` +
      `Remove the directory to reconfigure with "${generator}".`,
  )
  return []
}

const ccacheArgs = () => {
  if (!requestedCcache || !commandExists('ccache')) return []
  return [
    '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
    '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
  ]
}

const ensureEmscripten = () => {
  if (!commandExists('emcmake')) {
    throw new Error(
      'emcmake was not found. Install and activate Emscripten SDK before running npm run build-wasm.',
    )
  }
}

const main = () => {
  ensureEmscripten()

  fs.mkdirSync(buildRoot, { recursive: true })
  fs.mkdirSync(buildDir, { recursive: true })
  fs.mkdirSync(emCacheDir, { recursive: true })
  fs.mkdirSync(packageWasmDir, { recursive: true })

  run('emcmake', [
    'cmake',
    ...generatorArgs(),
    '-S',
    rootDir,
    '-B',
    buildDir,
    `-DCMAKE_BUILD_TYPE=${buildType}`,
    '-DWHISPER_NODE_WASM=ON',
    `-DWHISPER_NODE_WASM_SINGLE_FILE=${singleFile ? 'ON' : 'OFF'}`,
    `-DWHISPER_NODE_WASM_THREADS=${threads ? 'ON' : 'OFF'}`,
    `-DGGML_WEBGPU=${webgpu ? 'ON' : 'OFF'}`,
    `-DGGML_WEBGPU_JSPI=${boolEnv('WASM_WEBGPU_JSPI', false) ? 'ON' : 'OFF'}`,
    ...ccacheArgs(),
  ])
  run('cmake', [
    '--build',
    buildDir,
    '--target',
    'whisper-node-wasm',
    '--parallel',
    String(process.env.JOBS || os.cpus().length),
  ])

  const outputName = threads ? 'whisper-node.threads' : 'whisper-node'
  const jsSrc = path.join(buildDir, `${outputName}.js`)
  const wasmSrc = path.join(buildDir, `${outputName}.wasm`)
  const jsDest = path.join(packageWasmDir, `${outputName}.js`)
  const wasmDest = path.join(packageWasmDir, `${outputName}.wasm`)

  if (!fs.existsSync(jsSrc) || (!singleFile && !fs.existsSync(wasmSrc))) {
    throw new Error(
      `WASM build did not produce ${outputName}.js` +
        (singleFile ? '' : ` and ${outputName}.wasm`),
    )
  }

  fs.copyFileSync(jsSrc, jsDest)
  console.log(`Wrote ${jsDest} (${fs.statSync(jsDest).size} bytes)`)

  if (!singleFile) {
    fs.copyFileSync(wasmSrc, wasmDest)
    console.log(`Wrote ${wasmDest} (${fs.statSync(wasmDest).size} bytes)`)
  }
}

main()
