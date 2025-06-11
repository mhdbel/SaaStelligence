# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p models data config agents web utils

# Copy all application files
COPY . /app/

# Make sure the model directory exists
RUN mkdir -p models

# Optional: Mount trained model from host or train it inside the container
CMD ["sh", "-c", "cd models && python train_intent_model.py && cd .. && python main.py"]