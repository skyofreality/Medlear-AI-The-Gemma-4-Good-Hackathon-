import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: "/:path*.mjs",
        headers: [
          { key: "Content-Type", value: "application/javascript" },
        ],
      },
      {
        source: "/:path*.js",
        headers: [
          { key: "Content-Type", value: "application/javascript" },
        ],
      },
    ];
  },
};

export default nextConfig;
