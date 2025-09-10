"""Data loading, cleaning and preprocessing for ArXiv dataset."""

import os
import json
import pandas as pd
from langchain_core.documents import Document
from .config import DATA_PATH
from .text_processing import clean_text

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

def df_to_documents(
    df: pd.DataFrame,
    lowercase: bool = False,
    remove_stopwords: bool = False
):
    documents = []
    for _, row in df.iterrows():
        title_clean = clean_text(str(row['title']), lowercase=lowercase, remove_stopwords=remove_stopwords)
        abstract_clean = clean_text(str(row['abstract']), lowercase=lowercase, remove_stopwords=remove_stopwords)
        page_content = f"Title: {title_clean}\n\nAbstract: {abstract_clean}"
        categories_raw = row.get('categories', 'N/A') or 'N/A'
        primary_category = categories_raw.split()[0] if isinstance(categories_raw, str) else 'N/A'
        metadata = {
            "id": row.get('id', 'N/A'),
            "authors": row.get('authors', 'N/A'),
            "year": int(row.get('year')) if not pd.isna(row.get('year')) else None,
            "categories": categories_raw,
            "primary_category": primary_category
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents
