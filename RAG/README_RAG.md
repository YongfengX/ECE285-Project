```markdown
# Efficient Mathematical Reasoning with SFT, QLoRA and RAG

This project explores an efficient pipeline for improving mathematical reasoning in large language models using **Supervised Fine-Tuning (SFT)**, **QLoRA**, and **Retrieval-Augmented Generation (RAG)**.

The system combines parameter-efficient adaptation with external knowledge retrieval to achieve strong reasoning performance under limited computational resources.

---

## Project Structure

```

.
├── chunk.py                # PDF parsing and chunk generation
├── build_vector_db.py      # Build FAISS index from chunks
├── retriever.py            # Simple retrieval interface
├── rag_chat.py             # End-to-end RAG + QLoRA chat pipeline
├── *.jsonl                 # Chunked dataset
├── *.index / *.json        # FAISS index + metadata

````

---

## Pipeline Overview

1. **Chunking**
   - Extract text from PDFs (homework, solutions, notes)
   - Split into structured chunks (problem / solution / QA pairs)

2. **Embedding & Indexing**
   - Use BGE embedding model (`bge-base-en-v1.5`)
   - Store vectors in FAISS for fast similarity search

3. **Retrieval (RAG)**
   - Retrieve top-k relevant chunks based on query similarity
   - Construct grounded prompt using retrieved context

4. **Generation (QLoRA)**
   - Load Qwen3-4B + LoRA adapter
   - Generate answers conditioned on retrieved knowledge

---

## Installation

```bash
pip install torch transformers sentence-transformers faiss-cpu peft pymupdf
````

(Optional GPU setup recommended for faster inference)

---

## Usage

### 1. Build chunks from PDFs

```bash
python chunk.py
```

### 2. Build vector database

```bash
python build_vector_db.py
```

### 3. Run RAG system

```bash
python rag_chat.py
```

Then interact:

```
You: What is a projection matrix?
Assistant: ...
```

---

## Key Features

* Efficient fine-tuning via QLoRA (low memory footprint)
* Grounded reasoning via retrieval augmentation
* Structured knowledge base from course materials
* Improved accuracy and reduced hallucination

---

## Model Details

* Base model: Qwen3-4B
* Fine-tuning: LoRA / QLoRA
* Embedding model: BAAI/bge-base-en-v1.5
* Vector DB: FAISS

---

## Notes

* Retrieval quality strongly affects final answer quality
* Early LoRA checkpoints often perform better than later ones
* Windows users may disable 4-bit quantization for stability

---

## Future Work

* Improve retrieval ranking (e.g., reranking)
* Support multi-hop reasoning
* Extend dataset beyond course materials

```
```
