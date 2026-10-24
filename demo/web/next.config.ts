import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // let other devices on your local network load the dev server (next dev listens on 0.0.0.0)
  allowedDevOrigins: ["192.168.*.*", "10.*.*.*", "172.*.*.*", "*.local"],
  // `next build` writes a static site to out/, which the game server serves (npm run demo)
  output: "export",
  reactCompiler: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
