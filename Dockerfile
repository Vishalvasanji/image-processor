FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models

COPY main.py .

ENV U2NET_ONNX_PATH=/app/models/u2netp.onnx
ENV CANVAS_SIZE=2048
ENV FILL_RATIO=0.68
ENV BUFFER_PCT=0.025
ENV BRIGHTNESS=1.01
ENV CONTRAST=1.03
ENV MAX_FILESIZE_MB=1.2
ENV TIMEOUT_S=20
ENV LOG_LEVEL=INFO

EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
