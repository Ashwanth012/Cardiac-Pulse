import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev and preview servers are bound to 127.0.0.1 only (mitigates the open Vite/esbuild
// dev-server advisories, which affect only these local servers, not the production build).
const api = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } };

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy: api },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy: api },
  build: { sourcemap: false, chunkSizeWarningLimit: 900 },
});
