import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// Dev: the backend runs on 8010 (see backend/config.py); Vite proxies both
// REST and the frame-streaming WebSocket. Prod: FastAPI serves dist/.
//
// base: the SPA's URL prefix, taken from APP_ROOT_PATH (wignerf.env) at BUILD
// time — "/" (or unset) in dev, "/wignerf/" behind the nginx prod prefix. Must
// end with a slash. It drives asset URLs AND import.meta.env.BASE_URL, from
// which src/api.ts derives the REST base. Export APP_ROOT_PATH before
// `npm run build` for a prefixed prod build (the runtime service's
// EnvironmentFile does not reach the build).
const rootPath = process.env.APP_ROOT_PATH || '/'
const base = rootPath.endsWith('/') ? rootPath : rootPath + '/'

export default defineConfig({
  base,
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8010', ws: true },
    },
  },
})
