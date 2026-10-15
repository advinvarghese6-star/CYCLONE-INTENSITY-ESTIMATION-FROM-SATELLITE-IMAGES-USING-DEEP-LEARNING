import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Allow base64 data URIs in <img> tags & disable strict image optimization */
  images: {
    unoptimized: true,
  },
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    root: process.cwd(),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
