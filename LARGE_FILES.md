# Handling Large Files

This project generates large files that are not suitable for Git:

- `data/arxiv-metadata-oai-snapshot.json` (~522 MB) - ArXiv metadata dataset
- `faiss_index/index.faiss` (~118 MB) - FAISS vector index
- `faiss_index/index.pkl` (~57 MB) - Index metadata

## Regenerating Data

These files will be automatically generated when you run:

```bash
python -m src.main
```

The pipeline will:
1. Download the ArXiv dataset to `data/`
2. Process and chunk the documents  
3. Generate embeddings and build the FAISS index in `faiss_index/`

## First Time Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your `.env` file with `GROQ_API_KEY`
4. Run the main pipeline: `python -m src.main`
5. Launch the UI: `streamlit run app.py`

The data processing step may take some time initially but subsequent runs will be faster due to caching.
