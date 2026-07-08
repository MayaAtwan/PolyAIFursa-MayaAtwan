# =============================================================================
# Stage 1: Builder
# Purpose: compile TypeScript, run Tailwind, bundle assets via Next.js.
# This stage produces .next/standalone/ — a self-contained Node server.
# Everything in this stage (node_modules ~700MB, TypeScript compiler, etc.)
# is DISCARDED after the build. It never enters the final image.
# =============================================================================
FROM node:20-alpine AS builder
WORKDIR /app

# Layer caching: copy lock files before source code.
# npm ci installs ~200 packages and takes ~30 seconds.
# If only your .tsx files changed (not package-lock.json), Docker skips this.
COPY package.json package-lock.json ./
RUN npm ci

# Copy source after installing deps so the npm ci layer above stays cached
COPY . .

# NEXT_PUBLIC_* variables are baked into the JS bundle at build time.
# They are NOT available at runtime — the correct URL must be passed NOW
# via --build-arg when running docker build (handled by CI in deploy workflows).
ARG NEXT_PUBLIC_AGENT_URL=http://localhost:8000
ENV NEXT_PUBLIC_AGENT_URL=$NEXT_PUBLIC_AGENT_URL

RUN npm run build

# =============================================================================
# Stage 2: Runner
# Purpose: a clean, minimal image that only contains what's needed to run.
# Starts fresh — no leftover build tools, TypeScript compiler, or node_modules.
# Result: ~100MB image instead of ~1GB.
# =============================================================================
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy only the compiled output from the builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

# server.js is the self-contained production server that Next.js generated
CMD ["node", "server.js"]
