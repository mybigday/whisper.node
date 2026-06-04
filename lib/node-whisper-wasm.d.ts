declare module '@fugood/node-whisper-wasm' {
  import type { Module } from './binding'

  const module: Module & { default?: Module }
  export const WhisperContext: Module['WhisperContext']
  export const WhisperVadContext: Module['WhisperVadContext']
  export default module
}
