/** @type {import('next').NextConfig} */
const nextConfig = {
  // standalone mode outputs a self-contained server.js + minimal node_modules.
  // Required for the multi-stage Docker build — without this, the runner stage
  // has nothing to copy and the image would need the full node_modules (~700MB).
  output: 'standalone',
};
export default nextConfig;
