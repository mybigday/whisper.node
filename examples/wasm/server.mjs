import http from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve } from 'node:path'
import { Readable } from 'node:stream'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)))
const port = Number(process.env.PORT || 8080)

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
}

const baseHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cache-Control': 'no-store',
}

function resolvePath(url) {
  const pathname = decodeURIComponent(new URL(url, `http://localhost:${port}`).pathname)
  const staticPath =
    pathname === '/'
      ? '/examples/wasm/index.html'
      : pathname.endsWith('/')
        ? `${pathname}index.html`
        : pathname
  const filePath = normalize(join(root, staticPath))
  if (!filePath.startsWith(root)) {
    return null
  }
  return filePath
}

const server = http.createServer((request, response) => {
  const requestUrl = new URL(request.url || '/', `http://localhost:${port}`)

  if (requestUrl.pathname === '/proxy') {
    const source = requestUrl.searchParams.get('url')
    if (!source || !/^https?:\/\//.test(source)) {
      response.writeHead(400, baseHeaders)
      response.end('Missing http(s) url')
      return
    }

    fetch(source)
      .then((upstream) => {
        if (!upstream.ok || !upstream.body) {
          response.writeHead(upstream.status || 502, baseHeaders)
          response.end(`Proxy fetch failed: HTTP ${upstream.status}`)
          return
        }

        const headers = {
          ...baseHeaders,
          'Content-Type':
            upstream.headers.get('content-type') || 'application/octet-stream',
          'Access-Control-Allow-Origin': '*',
        }
        const contentLength = upstream.headers.get('content-length')
        if (contentLength) {
          headers['Content-Length'] = contentLength
        }

        response.writeHead(200, headers)
        Readable.fromWeb(upstream.body).pipe(response)
      })
      .catch((error) => {
        response.writeHead(502, baseHeaders)
        response.end(error instanceof Error ? error.message : String(error))
      })
    return
  }

  const filePath = resolvePath(request.url || '/')
  if (!filePath || !existsSync(filePath) || !statSync(filePath).isFile()) {
    response.writeHead(404, baseHeaders)
    response.end('Not found')
    return
  }

  response.writeHead(200, {
    ...baseHeaders,
    'Content-Type': contentTypes[extname(filePath)] || 'application/octet-stream',
  })
  createReadStream(filePath).pipe(response)
})

server.listen(port, () => {
  console.log(`Serving WASM smoke test at http://localhost:${port}/examples/wasm/`)
})
