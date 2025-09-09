import os
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "0") == "1" else "cpu"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
