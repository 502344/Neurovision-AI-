import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api/* to the FastAPI backend (uvicorn backend.main:app --reload, port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
    },
  },
})
