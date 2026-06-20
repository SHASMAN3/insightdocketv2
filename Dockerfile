# ─────────────────────────────────────────────────────────────────────────────
# InsightDocket — Multi-stage Dockerfile
#
# Stage 1: builder  — install dependencies with uv into /app/.venv
# Stage 2: runtime  — copy venv + app code, run as uid 1001 (non-root)
#
# Interview note: Multi-stage builds reduce the final image size by leaving
# build tools (uv, compilers) in the builder stage. The runtime image
# contains only the venv and application code.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies required for unstructured hi-res PDF parsing
# poppler-utils: pdf2image (page rendering)
# libmagic1:     python-magic file type detection
# tesseract-ocr: OCR fallback for image-heavy PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (Docker layer cache)
COPY pyproject.toml .python-version ./

# Install all dependencies into a virtual environment
# --no-install-project: don't install the package itself yet (just deps)
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        fastapi==0.115.5 \
        uvicorn[standard]==0.32.1 \
        python-multipart==0.0.12 \
        pydantic==2.10.3 \
        pydantic-settings==2.6.1 \
        langchain==0.3.13 \
        langchain-google-genai==2.0.7 \
        langchain-community==0.3.13 \
        langsmith==0.2.3 \
        google-generativeai==0.8.3 \
        motor==3.6.0 \
        pymongo==4.10.1 \
        sqlalchemy==2.0.36 \
        aiomysql==0.2.0 \
        cryptography==43.0.3 \
        "unstructured[pdf]==0.16.11" \
        pdfminer.six==20231228 \
        pillow==11.0.0 \
        pdf2image==1.17.0 \
        sentence-transformers==3.3.1 \
        "torch==2.5.1" \
        python-dotenv==1.0.1 \
        httpx==0.28.1 \
        tenacity==9.0.0 \
        structlog==24.4.0 \
        bleach==6.2.0 \
        "python-jose[cryptography]==3.3.0" \
        "passlib[bcrypt]==1.7.4"

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user uid=1001 for security
# Interview note: Running as non-root is a container security baseline.
# uid=1001 avoids conflicts with common system users (1000 = ubuntu).
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY conftest.py ./

# Create storage and log directories with correct ownership
RUN mkdir -p /app/storage/pdfs /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Activate virtual environment by prepending to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# HEALTHCHECK: probe the /health endpoint every 30s
# --start-period: give the app 60s to start before first check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
