from fastapi import FastAPI
from .vector_store import build_or_load_vectorstore
from .retriever import get_retriever
from .rag_pipeline import build_rag_chain
from .ingestion import df_to_documents, preprocess_dataframe, load_data_subset
from .config import DATA_PATH
import os

app = FastAPI()

# Load documents and vectorstore at startup
df = load_data_subset(os.path.join(DATA_PATH, "arxiv-metadata-oai-snapshot.json"))
df = preprocess_dataframe(df)
docs = df_to_documents(df)
vectorstore = build_or_load_vectorstore(docs)
retriever = get_retriever(vectorstore)
rag_chain = build_rag_chain(retriever)

@app.get("/query")
def query_rag(q: str):
    return {"answer": rag_chain.invoke(q).content}
