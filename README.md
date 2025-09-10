# Research Paper RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange.svg)](https://faiss.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-purple.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **conversational question-answering system** built to operate at **ArXiv scale**.  
It ingests large collections of scientific abstracts, cleans and splits them into semantically meaningful chunks, embeds them, and stores the vectors in a FAISS index.  
When a user asks a question, the system retrieves the most relevant chunks via similarity search, reranks them with a cross-encoder for higher precision, and then uses a **Groq-hosted large language model** to generate accurate, context-aware answers.

## Features

- **Text Processing**: LaTeX removal, URL cleaning, semantic chunking
- **Embeddings**: Sentence-Transformers with FAISS vector store
- **Retrieval**: Dynamic k-selection and cross-encoder reranking
- **UI**: Streamlit interface with parameter controls
- **API**: FastAPI endpoint for programmatic access

## Project Structure

```
app.py              # Streamlit UI
src/main.py         # Main pipeline
src/ingestion.py    # Data loading & preprocessing
src/retriever.py    # Advanced retriever with reranking
src/rag_pipeline.py # RAG chain
requirements.txt
```

## Quick Start

1. **Setup**:
```bash
pip install -r requirements.txt
```

2. **Environment**: Create `.env` with `GROQ_API_KEY=your_key`

3. **Build Index**:
```bash
python -m src.main
```

4. **Run UI**:
```bash
streamlit run app.py
```

## Usage

The system uses FAISS similarity search with cross-encoder reranking and dynamic k-selection based on query complexity. Configure retrieval parameters via the Streamlit sidebar.

## Configuration

Set `GROQ_API_KEY` in `.env`. GPU usage enabled via `CUDA_AVAILABLE=1`. Runtime parameters adjustable in Streamlit sidebar.

## Evaluation

Run `python scripts/evaluate_rag.py` for performance metrics (Recall@K, MRR, latency).

