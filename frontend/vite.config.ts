import react from '@vitejs/plugin-react'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'

// Content hash of the heart model: the app requests /heart.glb?v=<hash>, so a rebuilt model always gets a new
// URL and a browser can never pair new app code with a cached old model.
const heartVersion = createHash('sha256').update(readFileSync('public/heart.glb')).digest('hex').slice(0, 12)

export default defineConfig({
  plugins: [react()],
  define: { __HEART_VERSION__: JSON.stringify(heartVersion) },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { chunkSizeWarningLimit: 1500 },
})
