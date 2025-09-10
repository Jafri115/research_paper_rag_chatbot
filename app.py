import os
import torch
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq

from src.vector_store import build_or_load_vectorstore
from src.ingestion import load_data_subset, preprocess_dataframe, df_to_documents
from src.retriever import build_advanced_retriever
from src.config import DATA_PATH, FAISS_INDEX_PATH, GROQ_API_KEY

load_dotenv(find_dotenv())

st.set_page_config(page_title="📄 Research Paper RAG Chatbot", page_icon="💬", layout="wide")
st.title("📄 Research Paper RAG Chatbot (Groq + FAISS + Rerank)")

# Sidebar controls
with st.sidebar:
    st.header("Retrieval Settings")
    base_k = st.slider("Initial fetch (base_k)", 4, 30, 16, 1)
    rerank_k = st.slider("Final docs (rerank_k)", 1, 12, 6, 1)
    dynamic = st.checkbox("Dynamic k", True)
    use_rerank = st.checkbox("Use reranking", True)
    primary_category = st.text_input("Primary category filter", "") or None
    year_min = st.number_input("Min year", value=0, step=1)
    year_max = st.number_input("Max year", value=0, step=1)
    if year_min == 0:
        year_min = None
    if year_max == 0:
        year_max = None
    rebuild = st.button("Rebuild index (semantic)")
    subset_size = st.number_input("Subset records (rebuild)", 1000, 100000, 50000, 1000)

# Build or load vectorstore
if rebuild or not os.path.exists(FAISS_INDEX_PATH):
    data_file = os.path.join(DATA_PATH, "arxiv-metadata-oai-snapshot.json")
    if not os.path.exists(data_file):
        st.error("Dataset missing. Run main pipeline first.")
        st.stop()
    with st.spinner("Building vector index..."):
        df = load_data_subset(data_file, num_records=int(subset_size))
        df = preprocess_dataframe(df)
        docs = df_to_documents(df)
        vectorstore = build_or_load_vectorstore(
            docs,
            force_rebuild=True,
            chunk_method="semantic",
            chunk_size=800,
            chunk_overlap=120
        )
else:
    vectorstore = build_or_load_vectorstore([], force_rebuild=False)

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.7,
    max_tokens=512,
    groq_api_key=GROQ_API_KEY,
)

prompt_template = """Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_chain():
    retriever = build_advanced_retriever(
        vectorstore,
        base_k=base_k,
        rerank_k=rerank_k,
        primary_category=primary_category,
        year_min=year_min,
        year_max=year_max,
        dynamic=dynamic,
        use_rerank=use_rerank,
    )
    retrieval_runnable = RunnableLambda(lambda q: format_docs(retriever.get_relevant_documents(q)))
    chain = {"context": retrieval_runnable, "question": RunnablePassthrough()} | prompt | llm
    return chain, retriever

if "messages" not in st.session_state:
    st.session_state["messages"] = []

query = st.chat_input("Ask me something...")

if query:
    rag_chain, adv_retriever = build_chain()
    docs = adv_retriever.get_relevant_documents(query)
    answer = rag_chain.invoke(query)
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    st.session_state["messages"].append({
        "query": query,
        "answer": answer_text,
        "context": docs
    })

for msg in st.session_state["messages"]:
    st.chat_message("user").write(msg["query"])
    with st.chat_message("assistant"):
        st.write(msg["answer"])
        with st.expander("Documents"):
            for i, doc in enumerate(msg["context"]):
                st.markdown(f"**Doc {i+1}**")
                st.write(doc.page_content)
                if doc.metadata:
                    st.caption(str(doc.metadata))
