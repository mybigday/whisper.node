declare module '@fugood/node-whisper-wasm' {
  import type { Module } from './binding'

  export function toggleNativeLog(
    enable: boolean,
    callback?: (level: string, text: string) => void,
  ): Promise<void>
  export function addNativeLogListener(
    listener: (level: string, text: string) => void,
  ): { remove: () => void }
  export function isNativeLogEnabled(): boolean
  export const WhisperContext: Module['WhisperContext']
  export const WhisperVadContext: Module['WhisperVadContext']
  export const ParakeetContext: Module['ParakeetContext']
  const module: Module & {
    default?: Module
    toggleNativeLog: typeof toggleNativeLog
    addNativeLogListener: typeof addNativeLogListener
    isNativeLogEnabled: typeof isNativeLogEnabled
  }
  export default module
}
