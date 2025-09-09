import os
import kagglehub
from .ingestion import load_data_subset, preprocess_dataframe, df_to_documents
from .vector_store import build_or_load_vectorstore
from .retriever import get_retriever
from .rag_pipeline import build_rag_chain
from .config import DATA_PATH
import shutil 

def download_dataset():
    """Download the ArXiv dataset via KaggleHub if not already present."""
    os.makedirs(DATA_PATH, exist_ok=True)
    dataset_file = os.path.join(DATA_PATH, "arxiv-metadata-oai-snapshot.json")

    if not os.path.exists(dataset_file):
        print("Downloading ArXiv dataset via KaggleHub...")
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        extracted_file = os.path.join(path, "arxiv-metadata-oai-snapshot.json")
        shutil.copy(extracted_file, dataset_file)  # ✅ copy works across drives
        print(f"Dataset copied to {dataset_file}")
    else:
        print(f"Dataset already exists at {dataset_file}")

    return dataset_file


def run_sample_queries(rag_chain):
    """Run a few sample queries through the RAG pipeline."""
    sample_questions = [
        "What are the recent advancements in graph neural networks?",
        "Explain the applications of transformers in natural language processing.",
        "How is reinforcement learning applied in robotics?",
    ]

    for q in sample_questions:
        print("\n---")
        print(f"Question: {q}")
        answer = rag_chain.invoke(q).content
        print(f"Answer: {answer}")


def main():
    dataset_file = download_dataset()
    df = load_data_subset(dataset_file, num_records=50000)
    df = preprocess_dataframe(df)
    documents = df_to_documents(df)
    vectorstore = build_or_load_vectorstore(documents)
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    run_sample_queries(rag_chain)


if __name__ == "__main__":
    main()
