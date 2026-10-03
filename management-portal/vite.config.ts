import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API is proxied rather than called cross-origin, so the portal needs no CORS changes on
// the backend and the browser treats every request as same-origin.
const apiTarget = process.env.VITE_API_URL ?? 'http://localhost:5210';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    strictPort: false,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/health': { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // The charting library dominates the bundle and this is an internal console served from a
    // local network, so splitting it further would add complexity for no real gain.
    chunkSizeWarningLimit: 1500,
  },
});
