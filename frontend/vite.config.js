import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import net from 'node:net'
import { spawn, spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { existsSync } from 'node:fs'

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const backendDir = path.join(projectRoot, 'backend')

function portIsOpen(port) {
  return new Promise((resolve) => {
    const socket = net.connect({ host: '127.0.0.1', port })
    socket.once('connect', () => { socket.destroy(); resolve(true) })
    socket.once('error', () => { socket.destroy(); resolve(false) })
    socket.setTimeout(500, () => { socket.destroy(); resolve(false) })
  })
}

function commandWorks(command) {
  if (!command) return false
  try {
    const result = spawnSync(command, ['--version'], { stdio: 'ignore', windowsHide: true, timeout: 3_000 })
    return result.status === 0
  } catch {
    return false
  }
}

async function waitForPort(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await portIsOpen(port)) return true
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  return false
}

// Starting the frontend is the normal user entry point, so it owns the local
// API lifecycle too. An existing API is left untouched.
function localBackend(env) {
  let backendChild
  let ollamaChild
  return {
    name: 'local-backend',
    async configureServer() {
      if (!await portIsOpen(11434)) {
        const installedOllama = path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Ollama', 'ollama.exe')
        const ollamaCommand = process.platform === 'win32' && existsSync(installedOllama) ? installedOllama : (process.platform === 'win32' ? 'ollama.exe' : 'ollama')
        ollamaChild = spawn(ollamaCommand, ['serve'], { cwd: backendDir, stdio: 'ignore', windowsHide: true })
        ollamaChild.once('error', () => {
          console.error('Ollama could not start. Install Ollama and the required local model, then restart the app.')
        })
        await waitForPort(11434, 8_000)
      }
      if (!await portIsOpen(8000)) {
        const venvPython = path.join(backendDir, '.venv', 'Scripts', 'python.exe')
        // A copied project can contain a virtual environment created by a
        // different Python installation.  Checking only whether python.exe
        // exists selects that broken environment and makes the API silently
        // fail to start.  Allow an explicit interpreter override and otherwise
        // use the venv only when it can actually execute.
        const configuredPython = env.VITE_BACKEND_PYTHON || process.env.AGENT_OTG_PYTHON
        const fallbackPython = process.platform === 'win32' ? 'python' : 'python3'
        const pythonCommand = configuredPython || (existsSync(venvPython) && commandWorks(venvPython) ? venvPython : fallbackPython)
        backendChild = spawn(pythonCommand, ['-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'], {
          cwd: backendDir,
          stdio: ['ignore', 'pipe', 'pipe'],
          windowsHide: true,
        })
        backendChild.stdout?.on('data', (chunk) => {
          const line = String(chunk).trim()
          if (line) console.log(`[backend] ${line}`)
        })
        backendChild.stderr?.on('data', (chunk) => {
          const line = String(chunk).trim()
          if (line) console.error(`[backend] ${line}`)
        })
        backendChild.once('error', () => {
          console.error('Could not start the local backend. Run: cd backend && py -m pip install -r requirements.txt')
        })
        const backendReady = await waitForPort(8000, 45_000)
        if (!backendReady) {
          console.error('Backend did not start on port 8000 within 45s. Check backend dependencies and Python setup.')
        }
      }
    },
    closeBundle() {
      if (backendChild && !backendChild.killed) backendChild.kill()
      // Ollama is a shared local runtime. Do not terminate an existing model service.
      if (ollamaChild) ollamaChild.unref()
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [tailwindcss(), react(), localBackend(env)],
    server: {
      host: '0.0.0.0',
      strictPort: true,
      port: 5173,
      proxy: {
        '/api': {
          target: env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        '/files': {
          target: env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
        '/sync': {
          target: env.VITE_BACKEND_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
