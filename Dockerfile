# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies required for OpenCV and EasyOCR (PyTorch, C++ libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create folder for output data
RUN mkdir -p data/processed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OCR_SAVE_DIR=/app/data/processed
ENV CAMERA_INDEX=0

# Default entrypoint runs the detector CLI on the sample image
CMD ["python", "src/detector.py", "--image", "data/raw/test.png"]
