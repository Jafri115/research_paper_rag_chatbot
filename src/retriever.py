def get_retriever(vectorstore, k=3):
    """Return a retriever object from a FAISS vectorstore."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
