# ============================================
# SAAStelligence Engine - Training Dockerfile
# ============================================
# Use this for model training jobs
#
# Build: docker build -f Dockerfile.train -t saastelligence-train:latest .
# Run:   docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data saastelligence-train:latest
# ============================================

FROM python:3.10.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy training code
COPY config/ /app/config/
COPY models/ /app/models/
COPY utils/ /app/utils/
COPY data/ /app/data/

# Run training
CMD ["python", "-m", "models.train_intent_model", "--verbose"]