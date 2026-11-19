from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.analytics import classify_disclosure_change

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_or_load_index(chunks, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    texts = [item["text"] for item in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(index_dir / "embeddings.npy", vectors.astype("float32"))
    with (index_dir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def _load_index(index_dir: Path):
    emb_path = index_dir / "embeddings.npy"
    data_path = index_dir / "chunks.json"
    if not emb_path.exists() or not data_path.exists():
        return None, None
    with data_path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(emb_path)
    if len(chunks) != len(vectors):
        raise ValueError("The saved text chunks and embeddings do not match. Rebuild the index.")
    return chunks, vectors

def search_chunks(query, index_dir: Path, top_k=4, source_filter=None):
    if not query or not query.strip():
        return []
    chunks, vectors = _load_index(index_dir)
    if chunks is None:
        return []
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q
    ranked = np.argsort(scores)[::-1]
    results = []
    for idx in ranked:
        item = chunks[int(idx)]
        if source_filter and item["source"] != source_filter:
            continue
        result = dict(item)
        result["score"] = float(scores[idx])
        results.append(result)
        if len(results) >= top_k:
            break
    return results

def compare_report_chunks(query, index_dir: Path, earlier_source: str, later_source: str, top_k=6, min_pair_similarity=0.32):
    if not query or not query.strip():
        return []
    chunks, vectors = _load_index(index_dir)
    if chunks is None:
        return []

    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    query_scores = vectors @ query_vector

    def select_source(source_name):
        indices = [i for i, chunk in enumerate(chunks) if chunk.get("source") == source_name]
        indices.sort(key=lambda i: float(query_scores[i]), reverse=True)
        return indices[:top_k]

    earlier_indices = select_source(earlier_source)
    later_indices = select_source(later_source)
    if not earlier_indices and not later_indices:
        return []

    candidates = []
    for old_idx in earlier_indices:
        for new_idx in later_indices:
            similarity = float(np.dot(vectors[old_idx], vectors[new_idx]))
            candidates.append((similarity, old_idx, new_idx))
    candidates.sort(reverse=True, key=lambda row: row[0])

    used_old, used_new, comparisons = set(), set(), []
    for similarity, old_idx, new_idx in candidates:
        if similarity < min_pair_similarity:
            break
        if old_idx in used_old or new_idx in used_new:
            continue
        used_old.add(old_idx)
        used_new.add(new_idx)
        old_item, new_item = dict(chunks[old_idx]), dict(chunks[new_idx])
        label, explanation = classify_disclosure_change(old_item["text"], new_item["text"])
        comparisons.append({
            "classification": label,
            "explanation": explanation,
            "similarity": similarity,
            "earlier": old_item,
            "later": new_item,
            "match_type": "Matched passages",
        })

    for new_idx in later_indices:
        if new_idx not in used_new:
            comparisons.append({
                "classification": "New",
                "explanation": "No sufficiently similar passage was found among the earlier report's retrieved passages.",
                "similarity": None,
                "earlier": None,
                "later": dict(chunks[new_idx]),
                "match_type": "No earlier retrieved match",
            })

    for old_idx in earlier_indices:
        if old_idx not in used_old:
            comparisons.append({
                "classification": "Reduced (verify)",
                "explanation": "This earlier passage has no close match in the later report's retrieved set; it may have been reworded, relocated, or omitted.",
                "similarity": None,
                "earlier": dict(chunks[old_idx]),
                "later": None,
                "match_type": "No later retrieved match",
            })

    order = {"New": 0, "Expanded": 1, "Reduced": 2, "Reduced (verify)": 3, "Unchanged": 4}
    comparisons.sort(key=lambda item: (order.get(item["classification"], 9), -(item["similarity"] or 0)))
    return comparisons