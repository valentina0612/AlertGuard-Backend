FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos 8000 solo para referencia, no es obligatorio
EXPOSE 8000

# Usar $PORT para que Railway asigne dinámicamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
