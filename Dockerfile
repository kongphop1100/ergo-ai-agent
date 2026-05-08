FROM python:3.11-slim

# OS deps for opencv (libgl1, glib) and ffmpeg (final.mp4 merge in local sessions; safe to include for parity)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lockfile + manifest first for build cache
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY . .

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

EXPOSE 8000

# Render injects $PORT — fall back to 8000 for local docker run
CMD ["sh", "-c", "uv run agent.py serve --host 0.0.0.0 --port ${PORT:-8000}"]
