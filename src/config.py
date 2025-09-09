import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "0") == "1" else "cpu"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
