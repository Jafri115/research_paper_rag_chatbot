"""Data loading and preprocessing for ArXiv dataset."""

import os
import json
import pandas as pd
from langchain_core.documents import Document
from .config import DATA_PATH

def load_data_subset(file_path, num_records=50000):
    records = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_records:
                break
            records.append(json.loads(line))
    return pd.DataFrame(records)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['update_date'] = pd.to_datetime(df['update_date'])
    df['year'] = df['update_date'].dt.year
    df = df.dropna(subset=['abstract'])
    df = df[df['abstract'].str.strip() != '']
    return df

def df_to_documents(df: pd.DataFrame):
    documents = []
    for _, row in df.iterrows():
        page_content = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
        metadata = {
            "id": row.get('id', 'N/A'),
            "authors": row.get('authors', 'N/A'),
            "year": row.get('year', 'N/A'),
            "categories": row.get('categories', 'N/A')
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents
