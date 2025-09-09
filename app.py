import os
import torch
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

save_path = "faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

if os.path.exists(save_path):
    vectorstore = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    st.error("⚠️ FAISS index not found. Please build it first.")
    st.stop()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.7,
    max_tokens=512,
    groq_api_key=os.environ["GROQ_API_KEY"],
)

rag_prompt_template = """Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

st.set_page_config(page_title="📄 Research Paper RAG Chatbot", page_icon="💬", layout="wide")

st.title("📄 Research Paper RAG Chatbot (Groq + FAISS)")

# Session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # [{role, query, answer, context}]

# Chat input
query = st.chat_input("Ask me something...")

if query:
    # Retrieve docs
    retrieved_docs = retriever.invoke(query)

    # Get answer
    answer = rag_chain.invoke(query)
    answer_text = answer.content if hasattr(answer, "content") else str(answer)

    # Store conversation
    st.session_state["messages"].append({
        "role": "user",
        "query": query,
        "answer": answer_text,
        "context": retrieved_docs
    })

# Display conversation
for msg in st.session_state["messages"]:
    st.chat_message("user").write(msg["query"])
    with st.chat_message("assistant"):
        st.write(msg["answer"])

        # Expander for context
        with st.expander("📄 View supporting documents"):
            for i, doc in enumerate(msg["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                if doc.metadata:
                    st.caption(f"Metadata: {doc.metadata}")
