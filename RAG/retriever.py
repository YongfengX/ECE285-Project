import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index("calculus.index")

with open("calculus_meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


def search(query, k=3):
    query_text = f"query: {query}"

    emb = model.encode(
        [query_text],
        normalize_embeddings=True
    )
    emb = np.array(emb, dtype="float32")

    scores, indices = index.search(emb, k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        item = metadata[i].copy()
        item["score"] = float(score)
        results.append(item)

    return results
