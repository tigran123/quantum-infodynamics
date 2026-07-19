import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// Dev: the backend runs on 8010 (see backend/config.py); Vite proxies both
// REST and the frame-streaming WebSocket. Prod: FastAPI serves dist/.
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8010', ws: true },
    },
  },
})
