import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embeddings import get_embedding_model
from .config import FAISS_INDEX_PATH

def build_or_load_vectorstore(documents):
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    else:
        print("No existing FAISS index found. Creating a new one...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, add_start_index=True
        )
        splits = splitter.split_documents(documents)
        print(f"Split {len(documents)} docs into {len(splits)} chunks.")
        vectorstore = FAISS.from_documents(splits, get_embedding_model())
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Vector store created and saved to {FAISS_INDEX_PATH}")
    return vectorstore
