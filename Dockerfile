FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY templates/ templates/

# Download model at build time (faster startup)
RUN mkdir -p artifacts/training
RUN pip install gdown && \
    gdown "https://drive.google.com/uc?id=146vCH9kMZ7m6jVx7kGg2yKxwVBKr6vmt" -O artifacts/training/model.h5

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
