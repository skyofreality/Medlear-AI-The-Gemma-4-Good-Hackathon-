import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["192.168.100.4"],

  async rewrites() {
    const backend = process.env.API_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },

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