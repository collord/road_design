import { defineConfig } from "vite";

// A static site: relative paths, so it can be served from any folder (e.g. GitHub Pages).
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  // The worker imports the shared evaluator from ../roadsearch.
  server: { fs: { allow: [".."] } },
  build: { target: "es2022", outDir: "dist" },
});
