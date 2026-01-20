# ============================================
# SAAStelligence Engine - Production Dockerfile
# ============================================
# Multi-stage build for smaller, more secure images
#
# Build: docker build -t saastelligence:latest .
# Run:   docker run -p 8000:8000 --env-file .env saastelligence:latest
# ============================================

# ----------------------------------------------
# Stage 1: Builder
# ----------------------------------------------
FROM python:3.10.13-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# ----------------------------------------------
# Stage 2: Production
# ----------------------------------------------
FROM python:3.10.13-slim-bookworm AS production

# Labels
LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="SAAStelligence Engine - AI-powered SaaS intelligence API"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # App configuration
    APP_HOME=/app \
    APP_USER=appuser \
    APP_GROUP=appgroup \
    # Default configuration (override with --env-file or -e)
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    WORKERS=1

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_GROUP} && \
    useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

WORKDIR ${APP_HOME}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories with correct permissions
RUN mkdir -p \
    ${APP_HOME}/models \
    ${APP_HOME}/data \
    ${APP_HOME}/logs \
    && chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_GROUP} config/ ${APP_HOME}/config/
COPY --chown=${APP_USER}:${APP_GROUP} agents/ ${APP_HOME}/agents/
COPY --chown=${APP_USER}:${APP_GROUP} models/*.py ${APP_HOME}/models/
COPY --chown=${APP_USER}:${APP_GROUP} utils/ ${APP_HOME}/utils/
COPY --chown=${APP_USER}:${APP_GROUP} web/ ${APP_HOME}/web/
COPY --chown=${APP_USER}:${APP_GROUP} main.py ${APP_HOME}/

# Copy data files if they exist (optional)
COPY --chown=${APP_USER}:${APP_GROUP} data/*.csv ${APP_HOME}/data/ 2>/dev/null || true

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Default command - run the API server
CMD ["sh", "-c", "uvicorn web.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]