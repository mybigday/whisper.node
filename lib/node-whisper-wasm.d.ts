declare module '@fugood/node-whisper-wasm' {
  import type { Module } from './binding'

  const module: Module & { default?: Module }
  export = module
}
