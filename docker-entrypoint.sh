#!/usr/bin/env bash
set -e

# Start FastAPI backend (background)
echo "Starting FastAPI backend..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend (foreground)
echo "Starting Streamlit frontend..."
streamlit run src/ui.py --server.port 8501 --server.address 0.0.0.0
