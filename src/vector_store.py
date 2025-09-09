import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from semantic_text_splitter import TextSplitter as SemanticTextSplitter  # type: ignore
    _HAS_SEMANTIC = True
except ImportError:  # graceful fallback if package missing
    _HAS_SEMANTIC = False
from langchain_core.documents import Document
from .embeddings import get_embedding_model
from .config import FAISS_INDEX_PATH

def _chunk_documents(
    documents: List[Document],
    method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 120
):
    if method == "semantic" and _HAS_SEMANTIC:
        try:
            # Newer versions expose factory; fallback to direct init
            if hasattr(SemanticTextSplitter, "from_tiktoken_encoder"):
                splitter = SemanticTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:  # try simple init signature
                splitter = SemanticTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            semantic_chunks: List[Document] = []
            for d in documents:
                try:
                    parts = splitter.chunks(d.page_content)
                except AttributeError:
                    # Fallback: naive sentence-ish split
                    parts = d.page_content.split('. ')
                for part in parts:
                    cleaned = part.strip()
                    if not cleaned:
                        continue
                    semantic_chunks.append(
                        Document(page_content=cleaned, metadata=d.metadata)
                    )
            return semantic_chunks
        except Exception as e:
            print(f"[semantic chunking fallback] {e}; reverting to recursive splitter.")
    # fallback / default
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(documents)

def build_or_load_vectorstore(
    documents: List[Document],
    force_rebuild: bool = False,
    chunk_method: str = "recursive",  # or "semantic"
    chunk_size: int = 1000,
    chunk_overlap: int = 120
):
    if os.path.exists(FAISS_INDEX_PATH) and not force_rebuild:
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vectorstore

    print("Building FAISS index (force_rebuild=%s, method=%s)..." % (force_rebuild, chunk_method))
    splits = _chunk_documents(
        documents,
        method=chunk_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Split {len(documents)} docs into {len(splits)} chunks (method={chunk_method}).")
    vectorstore = FAISS.from_documents(splits, get_embedding_model())
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Vector store created and saved to {FAISS_INDEX_PATH}")
    return vectorstore

def build_filtered_retriever(vectorstore, primary_category: Optional[str] = None, k: int = 3):
    base = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if not primary_category:
        return base
    # Simple wrapper applying post-filtering by metadata; could be replaced by a VectorStore-specific filter if supported
    def _get_relevant_documents(query):
        docs = base.get_relevant_documents(query)
        return [d for d in docs if d.metadata.get("primary_category") == primary_category]
    base.get_relevant_documents = _get_relevant_documents  # monkey patch
    return base
