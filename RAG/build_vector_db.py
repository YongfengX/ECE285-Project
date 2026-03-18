import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ===== input files =====
INPUT_FILES = [
    "calculus_concepts_chunks.jsonl",
    "calculus_theorems_chunks.jsonl",
    "calculus_formulas_chunks.jsonl",
    "calculus_examples_chunks.jsonl"
]

INDEX_FILE = "calculus.index"
META_FILE = "calculus_meta.json"

# 推荐先用这个，稳定一点
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def load_chunks(jsonl_files):
    texts = []
    metadata = []

    for file in jsonl_files:
        path = Path(file)
        if not path.exists():
            print(f"[Warning] File not found: {file}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Bad JSON in {file} line {line_num}")
                    continue

                text = data.get("text", "").strip()
                if not text:
                    continue

                # 记录来源文件，后面展示引用更方便
                data["jsonl_source"] = path.name

                texts.append(text)
                metadata.append(data)

    return texts, metadata


def main():
    print("Loading chunk files...")
    texts, metadata = load_chunks(INPUT_FILES)

    if not texts:
        raise ValueError("No valid chunks found in input files.")

    print(f"Loaded {len(texts)} chunks from {len(INPUT_FILES)} files.")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # 对 BGE，最好给 query/document 加前缀；这里只做文档 embedding
    corpus_texts = [f"passage: {t}" for t in texts]

    print("Encoding chunks...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    # 归一化后用 Inner Product ≈ cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)

    print(f"Saving metadata to {META_FILE}")
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Total chunks indexed: {index.ntotal}")


if __name__ == "__main__":
    main()