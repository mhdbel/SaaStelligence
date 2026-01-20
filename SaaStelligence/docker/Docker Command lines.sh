# =====================================
# BUILD
# =====================================

# Build production image
docker build -t saastelligence:latest .

# Build with no cache (force rebuild)
docker build --no-cache -t saastelligence:latest .

# Build training image
docker build -f Dockerfile.train -t saastelligence-train:latest .

# =====================================
# RUN
# =====================================

# Run with environment file
docker run -d \
  --name saastelligence \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  saastelligence:latest

# Run training container
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  saastelligence-train:latest

# =====================================
# DOCKER COMPOSE
# =====================================

# Start API service
docker-compose up -d api

# Start with training first
docker-compose --profile training up trainer
docker-compose up -d api

# Start full stack (with Redis)
docker-compose --profile full up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build api

# =====================================
# MAINTENANCE
# =====================================

# Check health
docker inspect --format='{{.State.Health.Status}}' saastelligence

# Shell into container
docker exec -it saastelligence /bin/bash

# View resource usage
docker stats saastelligence

# Clean up
docker system prune -f