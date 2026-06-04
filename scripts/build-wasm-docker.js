#!/usr/bin/env node

const os = require('os')
const path = require('path')
const { execFileSync } = require('child_process')

const rootDir = path.resolve(__dirname, '..')
const forwardedArgs = process.argv.slice(2)
const isArm64Host = os.arch() === 'arm64'

if (forwardedArgs.includes('--help') || forwardedArgs.includes('-h')) {
  console.log(`Usage: npm run build-wasm-docker -- [options]

Options are forwarded to scripts/build-wasm.sh:
  --webgpu / --no-webgpu
  --threads / --no-threads / --all-threads
  --single-file / --no-single-file
  --debug
  --build-dir <path>

Environment overrides:
  EMSCRIPTEN_IMAGE, EMSCRIPTEN_PLATFORM, JOBS, THREADS, WASM_*`)
  process.exit(0)
}

const image =
  process.env.EMSCRIPTEN_IMAGE ||
  (isArm64Host ? 'emscripten/emsdk:4.0.14-arm64' : 'emscripten/emsdk:4.0.13')
const platform =
  process.env.EMSCRIPTEN_PLATFORM ||
  (isArm64Host ? 'linux/arm64' : 'linux/amd64')

const readCommand = (cmd, args) =>
  execFileSync(cmd, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim()

const hostUid =
  process.env.HOST_UID ||
  (typeof process.getuid === 'function'
    ? String(process.getuid())
    : readCommand('id', ['-u']))
const hostGid =
  process.env.HOST_GID ||
  (typeof process.getgid === 'function'
    ? String(process.getgid())
    : readCommand('id', ['-g']))

const envNames = [
  'EM_CACHE',
  'JOBS',
  'THREADS',
  'WASM_BUILD_DIR',
  'WASM_BUILD_TYPE',
  'WASM_CCACHE',
  'WASM_CMAKE_GENERATOR',
  'WASM_SINGLE_FILE',
  'WASM_THREADS',
  'WASM_WEBGPU',
  'WASM_WEBGPU_JSPI',
]

const dockerArgs = [
  'run',
  '--rm',
  '--platform',
  platform,
  '-e',
  'HOME=/tmp',
  '-e',
  `HOST_UID=${hostUid}`,
  '-e',
  `HOST_GID=${hostGid}`,
]

for (const name of envNames) {
  if (process.env[name] != null) dockerArgs.push('-e', name)
}

dockerArgs.push(
  '-v',
  `${rootDir}:/src`,
  '-w',
  '/src',
  image,
  'bash',
  '-lc',
  [
    'source /emsdk/emsdk_env.sh',
    'bash scripts/build-wasm.sh "$@"',
    'chown -R "$HOST_UID:$HOST_GID" build-wasm packages/node-whisper-wasm/wasm',
  ].join(' && '),
  'build-wasm.sh',
  ...forwardedArgs,
)

console.log(`Using ${image} (${platform})`)
execFileSync('docker', dockerArgs, {
  cwd: rootDir,
  stdio: 'inherit',
})
