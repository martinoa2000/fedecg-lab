import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The Train page talks to scripts/serve.py through this proxy, so the
    // training server only ever sees same-origin requests from localhost.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
