import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [
    {
      ...basicSsl({
        name: "pocket-ai-local",
        domains: ["10.0.0.103", "localhost", "127.0.0.1"],
        ttlDays: 30,
      }),
      apply: "serve",
    },
  ],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
  },
});
