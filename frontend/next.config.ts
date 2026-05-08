import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the Tailscale Funnel hostname to load Next.js dev assets
  // (fonts, HMR, etc). Production builds (npm run start) ignore this.
  allowedDevOrigins: ["desktop-25ct16q.tail09f6e0.ts.net"],
};

export default nextConfig;
