"""Benchmark retrieval & answer quality with curated or heuristic QA.

Supports:
1. Curated QA dataset (JSONL) with fields:
     {"question": str, "answer": str, "doc_ids": [optional list of str], "has_answer": bool (optional)}
2. Heuristic fallback QA (keyword-derived) if no dataset provided.

Train/Test Split Modes:
- closed: all docs indexed (standard retrieval evaluation)
- holdout: 80/20 split; index built on train subset, QA sampled from test subset (tests generalization; recall may drop if answers not in index)

Metrics:
- Recall@K, Precision@K, F1@K (K=5,10)
- MRR@K (5,10)
- Answer token F1 (generated vs ground truth)
- Embedding similarity (cosine) between generated and gold answers
- No-answer accuracy (for questions labeled has_answer=False)
- Latency (retrieval + generation)

Usage:
    python scripts/evaluate_rag.py --qa_path data/qa_curated.jsonl --mode closed

NOTE: This is still lightweight; for publication-grade results integrate human relevance judgments & stronger metrics (ROUGE, BLEU, etc.).
"""
from __future__ import annotations
import os
import time
import json
import random
import argparse
import statistics as stats
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vector_store import build_or_load_vectorstore
from src.ingestion import load_data_subset, preprocess_dataframe, df_to_documents
from src.retriever import build_advanced_retriever
from src.rag_pipeline import build_rag_chain
from src.config import DATA_PATH
from src.embeddings import get_embedding_model

# -----------------------------
# Simple QA generation heuristic
# -----------------------------
GENERIC_FALLBACK = [
    ("What is the main contribution?", ["introduce", "propose", "present"]),
    ("What dataset is used?", ["dataset", "data"]),
    ("What method is compared against?", ["baseline", "compare", "comparison"]),
]

def sample_qa_pairs(df, n=25) -> List[Tuple[str, List[str]]]:
    pairs = []
    rows = df.sample(min(n, len(df)), random_state=42)
    for _, r in rows.iterrows():
        abstract = str(r.get("abstract", ""))
        lowered = abstract.lower()
        if "transformer" in lowered:
            pairs.append(("What does the paper say about transformers?", ["transformer", "attention"]))
        elif "graph" in lowered:
            pairs.append(("How are graphs used?", ["graph", "node", "edge"]))
        elif "reinforcement" in lowered:
            pairs.append(("What reinforcement learning aspect is discussed?", ["agent", "reward"]))
        else:
            q, kws = random.choice(GENERIC_FALLBACK)
            pairs.append((q, kws))
    return pairs

# -----------------------------
# Metrics
# -----------------------------

def recall_at_k(relevant: List[int], k: int) -> float:
    return 1.0 if any(i < k for i in relevant) else 0.0


def mrr_at_k(relevant: List[int], k: int) -> float:
    for idx in relevant:
        if idx < k:
            return 1.0 / (idx + 1)
    return 0.0

# -----------------------------
# Evaluation Loop
# -----------------------------

@dataclass
class EvalResult:
    queries: int
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    f1_at_5: float
    f1_at_10: float
    mrr_at_5: float
    mrr_at_10: float
    ans_token_f1: float
    ans_embed_sim: float
    no_answer_accuracy: Optional[float]
    avg_retrieval_ms: float
    avg_generation_ms: float


def precision_recall_f1(relevant_positions: List[int], k: int, total_retrieved: int) -> Tuple[float, float, float]:
    if total_retrieved == 0:
        return 0.0, 0.0, 0.0
    found = sum(1 for pos in relevant_positions if pos < k)
    precision = found / min(k, total_retrieved)
    recall = 1.0 if found > 0 else 0.0  # binary relevance presence
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def answer_token_f1(pred: str, gold: str) -> float:
    if not gold.strip():
        return 1.0 if not pred.strip() else 0.0
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def cosine_similarity(vec_a, vec_b):
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


def load_curated_qa(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    # If file is a JSON array
    if raw.startswith('['):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass  # fall back to line parsing
    # Fallback: treat as JSONL (ignore array brackets if present)
    qa: List[Dict] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s in ('[', ']'):  # skip empty / brackets
            continue
        # Remove trailing commas from pretty-printed arrays
        if s.endswith(','):
            s = s[:-1]
        try:
            qa.append(json.loads(s))
        except json.JSONDecodeError:
            # Attempt to recover multi-line objects (very simple join logic)
            if qa and isinstance(qa[-1], dict) and not s.startswith('{'):
                # Append extra text to previous 'answer' if present
                prev = qa[-1]
                if 'answer' in prev:
                    prev['answer'] = (prev['answer'] + '\n' + s).rstrip(',')
            else:
                raise
    return qa


def evaluate(
    num_records: int = 5000,
    qa_pairs: int = 25,
    qa_path: Optional[str] = None,
    mode: str = "closed",
    no_answer_threshold: float = 0.30,
    min_no_answer_len: int = 15,
):
    data_file = os.path.join(DATA_PATH, "arxiv-metadata-oai-snapshot.json")
    df = load_data_subset(data_file, num_records=num_records)
    df = preprocess_dataframe(df)

    # Train/test split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if mode == "closed":
        index_df = df  # all docs
    elif mode == "holdout":
        index_df = train_df
    else:
        raise ValueError("mode must be 'closed' or 'holdout'")

    index_docs = df_to_documents(index_df)
    vectorstore = build_or_load_vectorstore(index_docs, force_rebuild=True, chunk_method="semantic", chunk_size=800, chunk_overlap=120)
    retriever = build_advanced_retriever(vectorstore, base_k=16, rerank_k=6, dynamic=True, use_rerank=True)
    rag_chain = build_rag_chain(retriever)

    if qa_path and os.path.exists(qa_path):
        curated = load_curated_qa(qa_path)
        qa_items = curated[:qa_pairs] if qa_pairs else curated
    else:
        source_df = test_df if mode == "holdout" else df
        heuristic_pairs = sample_qa_pairs(source_df, n=qa_pairs)
        # Convert to curated-like dicts
        qa_items = [
            {"question": q, "answer": "", "keywords": kws, "has_answer": True} for q, kws in heuristic_pairs
        ]

    emb_model = get_embedding_model()

    recalls5 = []
    recalls10 = []
    precisions5 = []
    precisions10 = []
    f1s5 = []
    f1s10 = []
    mrrs5 = []
    mrrs10 = []
    ans_f1_scores = []
    ans_sims = []
    no_answer_correct = []
    retrieval_times = []
    gen_times = []

    for qa in qa_items:
        q = qa["question"]
        gold_answer = qa.get("answer", "")
        has_answer = qa.get("has_answer", True) and bool(gold_answer.strip())
        doc_ids = set(qa.get("doc_ids", []) or [])
        keywords = qa.get("keywords", [])

        start_r = time.perf_counter()
        retrieved_docs = retriever.get_relevant_documents(q)
        retrieval_times.append((time.perf_counter() - start_r) * 1000)

        # Determine relevance positions
        relevant_positions: List[int] = []
        for idx, d in enumerate(retrieved_docs):
            if doc_ids:
                if d.metadata.get("id") in doc_ids:
                    relevant_positions.append(idx)
            else:  # fallback keyword match
                content_l = d.page_content.lower()
                if any(k.lower() in content_l for k in keywords):
                    relevant_positions.append(idx)

        # Metrics @5 / @10
        recalls5.append(recall_at_k(relevant_positions, 5))
        recalls10.append(recall_at_k(relevant_positions, 10))
        mrrs5.append(mrr_at_k(relevant_positions, 5))
        mrrs10.append(mrr_at_k(relevant_positions, 10))
        p5, _, f5 = precision_recall_f1(relevant_positions, 5, len(retrieved_docs))
        p10, _, f10 = precision_recall_f1(relevant_positions, 10, len(retrieved_docs))
        precisions5.append(p5)
        precisions10.append(p10)
        f1s5.append(f5)
        f1s10.append(f10)

        # Generation
        start_g = time.perf_counter()
        answer_obj = rag_chain.invoke(q)
        gen_times.append((time.perf_counter() - start_g) * 1000)
        pred_answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

        # Answer metrics
        if has_answer:
            ans_f1_scores.append(answer_token_f1(pred_answer, gold_answer))
            try:
                sim = cosine_similarity(
                    emb_model.embed_query(pred_answer),
                    emb_model.embed_query(gold_answer),
                )
            except Exception:
                sim = 0.0
            ans_sims.append(sim)
        else:
            # No-answer case
            length_condition = len(pred_answer.split()) < min_no_answer_len
            try:
                gold_vec = emb_model.embed_query(gold_answer) if gold_answer else [0.0]
                pred_vec = emb_model.embed_query(pred_answer)
                sim = cosine_similarity(pred_vec, gold_vec)
            except Exception:
                sim = 0.0
            ans_sims.append(sim)
            no_answer_correct.append(length_condition or sim < no_answer_threshold)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = EvalResult(
        queries=len(qa_items),
        recall_at_5=avg(recalls5),
        recall_at_10=avg(recalls10),
        precision_at_5=avg(precisions5),
        precision_at_10=avg(precisions10),
        f1_at_5=avg(f1s5),
        f1_at_10=avg(f1s10),
        mrr_at_5=avg(mrrs5),
        mrr_at_10=avg(mrrs10),
        ans_token_f1=avg(ans_f1_scores),
        ans_embed_sim=avg(ans_sims),
        no_answer_accuracy=avg(no_answer_correct) if no_answer_correct else None,
        avg_retrieval_ms=stats.mean(retrieval_times),
        avg_generation_ms=stats.mean(gen_times),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system.")
    parser.add_argument('--qa_path', type=str, default=None, help='Path to curated QA JSONL file.')
    parser.add_argument('--num_records', type=int, default=5000, help='Number of records to load from dataset.')
    parser.add_argument('--qa_pairs', type=int, default=25, help='Number of QA pairs (heuristic or slice of curated).')
    parser.add_argument('--mode', type=str, choices=['closed', 'holdout'], default='closed', help='Indexing mode.')
    parser.add_argument('--no_answer_threshold', type=float, default=0.30, help='Embedding similarity threshold for no-answer detection.')
    parser.add_argument('--min_no_answer_len', type=int, default=15, help='Max tokens allowed for a correct no-answer prediction.')
    args = parser.parse_args()

    res = evaluate(
        num_records=args.num_records,
        qa_pairs=args.qa_pairs,
        qa_path=args.qa_path,
        mode=args.mode,
        no_answer_threshold=args.no_answer_threshold,
        min_no_answer_len=args.min_no_answer_len,
    )
    print("Queries:", res.queries)
    print(f"Recall@5:        {res.recall_at_5:.3f}")
    print(f"Recall@10:       {res.recall_at_10:.3f}")
    print(f"Precision@5:     {res.precision_at_5:.3f}")
    print(f"Precision@10:    {res.precision_at_10:.3f}")
    print(f"F1@5:            {res.f1_at_5:.3f}")
    print(f"F1@10:           {res.f1_at_10:.3f}")
    print(f"MRR@5:           {res.mrr_at_5:.3f}")
    print(f"MRR@10:          {res.mrr_at_10:.3f}")
    print(f"Answer Token F1: {res.ans_token_f1:.3f}")
    print(f"Answer Emb Sim:  {res.ans_embed_sim:.3f}")
    if res.no_answer_accuracy is not None:
        print(f"No-Answer Acc:   {res.no_answer_accuracy:.3f}")
    print(f"Avg Retrieval Latency (ms): {res.avg_retrieval_ms:.1f}")
    print(f"Avg Generation Latency (ms): {res.avg_generation_ms:.1f}")


if __name__ == "__main__":
    main()
