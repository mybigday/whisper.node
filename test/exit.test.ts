import path from 'path'
import { spawnSync } from 'child_process'

// Each case runs in a child process: the failure happens during process teardown,
// after a context was created without being released.
const runChild = (body: string) => {
  const script = `
    const { initWhisper, initParakeet } = require(${JSON.stringify(path.resolve(__dirname, '../lib'))})
    const models = ${JSON.stringify(path.resolve(__dirname, '../whisper.cpp/models'))}
    const samples = ${JSON.stringify(path.resolve(__dirname, '../whisper.cpp/samples'))}
    ;(async () => {
      ${body}
    })()
  `
  return spawnSync(process.execPath, ['-e', script], { encoding: 'utf8', timeout: 60000 })
}

const expectCleanTeardown = (result: ReturnType<typeof runChild>, status: number) => {
  // a child that failed before loading the model would otherwise look like a clean exit code
  expect(result.stderr).toContain('[exit-test] model loaded')
  expect(result.signal).toBeNull()
  expect(result.stderr).not.toContain('GGML_ASSERT')
  expect(result.status).toBe(status)
}

const loadWhisper = `const ctx = await initWhisper({ filePath: models + '/ggml-tiny.en.bin', useGpu: true }); console.error('[exit-test] model loaded')`

test('process.exit() with a loaded whisper context', () => {
  expectCleanTeardown(runChild(`${loadWhisper}; process.exit(0)`), 0)
})

test('uncaught exception with a loaded whisper context', () => {
  expectCleanTeardown(runChild(`${loadWhisper}; throw new Error('boom')`), 1)
})

test('process.exit() from a progress callback during transcription', () => {
  expectCleanTeardown(
    runChild(`
      ${loadWhisper}
      let exited = false
      const { promise } = ctx.transcribe(samples + '/jfk.wav', {
        onProgress: () => { if (!exited) { exited = true; process.exit(0) } },
      })
      promise.catch(() => {})
    `),
    0,
  )
})

test('releaseSync() in an exit handler', () => {
  expectCleanTeardown(runChild(`${loadWhisper}; process.on('exit', () => ctx.releaseSync()); process.exit(0)`), 0)
})

test('process.exit() with a loaded parakeet context', () => {
  expectCleanTeardown(
    runChild(`
      const ctx = await initParakeet({ filePath: models + '/ggml-parakeet-tdt-0.6b-v3-q4_0.bin', useGpu: true })
      console.error('[exit-test] model loaded')
      process.exit(0)
    `),
    0,
  )
})
