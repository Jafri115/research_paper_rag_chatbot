# Use official Python runtime
FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured/pdf parsing (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Ensure required dirs
RUN mkdir -p data chroma_db

# Expose API and UI ports
EXPOSE 8000 8501

# Build vector store at image build time (optional; expects PDFs in data/)
RUN python src/ingestion.py || true
RUN python src/embeddings.py || true

# Entrypoint script to run API and UI
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
