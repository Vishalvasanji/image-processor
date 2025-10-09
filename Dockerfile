FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Pillow
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV CANVAS_SIZE=2048
ENV FILL_RATIO=0.68
ENV BUFFER_PCT=0.025
ENV BRIGHTNESS=1.01
ENV CONTRAST=1.03
ENV MAX_FILESIZE_MB=1.2
ENV TIMEOUT_S=20
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
