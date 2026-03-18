import json
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================
# Config
# =========================
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
INDEX_FILE = "ece269.index"
META_FILE = "ece269_meta.json"

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER_PATH = "E:\\ECE 285\\ECE285-Project\\project\\checkpoint-2400"

TOP_K = 3
MAX_NEW_TOKENS = 1024
USE_4BIT = False   # Windows 上 bitsandbytes 不稳，先默认 False


# =========================
# Load retriever
# =========================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def search(query, k=TOP_K):
    instruction = "Represent this sentence for searching relevant passages: "
    query_text = f"{instruction}{query}"
    emb = embed_model.encode(
        [query_text],
        normalize_embeddings=True
    )
    emb = np.array(emb, dtype="float32")

    scores, indices = index.search(emb, k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0 or i >= len(metadata):
            continue
        item = metadata[i].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def format_source(item):
    assignment = item.get("assignment", "")
    problem_id = item.get("problem_id", "")
    sub_id = item.get("sub_id", "")
    chunk_type = item.get("chunk_type", "")

    src = assignment if assignment else "UnknownAssignment"
    if problem_id:
        src += f" Q{problem_id}"
    if sub_id and sub_id != "main":
        src += f"({sub_id})"
    if chunk_type:
        src += f" [{chunk_type}]"
    return src


def build_prompt(question, retrieved_chunks):
    context_blocks = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        source_name = format_source(item)
        block = f"[Source {idx}: {source_name}]\n{item.get('text', '').strip()}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)
 
    # Separate system commands from user questions
    messages = [
        {"role": "system", "content": (
            "You are a helpful and concise calculus assistant. "
            "Answer the user's question using ONLY the retrieved context below. "
            "Rule 1: Be direct and concise. Do not repeat the same point. "
            "Rule 2: Do NOT output any proof-ending symbols like \\(\\blacksquare\\). "
            "Rule 3: Do not output source labels like [Source 1] in the text.\n\n"
            f"Context:\n{context_text}"
        )},
        {"role": "user", "content": question}
    ]
    return messages


# =========================
# Load generator
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
if USE_4BIT:
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


def generate_answer(messages, max_new_tokens=MAX_NEW_TOKENS):
    # 1. 使用 Qwen 自带的模板格式化对话
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,  # 降低温度，减少废话
            top_p=0.3,
            repetition_penalty=1.1,  # 增加重复惩罚，防止它一直复读
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 2. 截取新生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 3. 后处理：强行清洗掉截图里那种奇怪的结尾符号
    answer = answer.strip()
    unwanted_symbols = ["\\(\\blacksquare\\)", "\\blacksquare", "The final answer is:"]
    for symbol in unwanted_symbols:
        answer = answer.replace(symbol, "").strip()
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # --- 新增：过滤掉 <think> 标签及其内部内容 ---
    import re
    # 匹配 <think> 到 </think> 之间的所有内容（包括换行符）并替换为空
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    
    return clean_answer   


def print_retrieved_chunks(chunks):
    print("\n" + "=" * 100)
    print("Retrieved Chunks")
    print("=" * 100)
    for i, item in enumerate(chunks, start=1):
        print(f"[{i}] {format_source(item)} | score={item.get('score', 0):.4f}")
        text = item.get("text", "").replace("\n", " ")
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 100)


def main():
    print("\nECE269 RAG chat started.")
    print("Type your question. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        retrieved = search(question, k=TOP_K)
        print_retrieved_chunks(retrieved)

        prompt = build_prompt(question, retrieved)
        answer = generate_answer(prompt)

        print("\n" + "=" * 100)
        print("Assistant")
        print("=" * 100)
        print(answer)
        print()

        
if __name__ == "__main__":
    main()



