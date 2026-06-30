# =============================================================================
# Stage 1: Builder — install dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install production dependencies
COPY requirements/prod.txt requirements/prod.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/prod.txt

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=appuser:appgroup app/ app/
COPY --chown=appuser:appgroup migrations/ migrations/
COPY --chown=appuser:appgroup alembic.ini alembic.ini
COPY --chown=appuser:appgroup pyproject.toml pyproject.toml

# Create directories for data and logs
RUN mkdir -p data/faiss logs && chown -R appuser:appgroup data logs

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Start with gunicorn in production
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
