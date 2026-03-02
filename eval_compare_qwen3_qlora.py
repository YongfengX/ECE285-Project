import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base Qwen3-0.6B vs QLoRA-finetuned model outputs")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter output directory")
    parser.add_argument("--dataset_name", type=str, default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset_config", type=str, default="rc.nocontext")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--max_eval_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_file", type=str, default="./outputs/eval_compare_results.json")
    return parser.parse_args()


def _extract_question(example: Dict[str, Any]) -> Optional[str]:
    if "question" in example:
        q = example["question"]
        if isinstance(q, str):
            return q.strip()
        if isinstance(q, dict):
            text = q.get("text")
            if isinstance(text, str):
                return text.strip()
    q_text = example.get("question_text")
    if isinstance(q_text, str):
        return q_text.strip()
    return None


def _load_quant_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = _load_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, args.adapter_path)
    finetuned_model.eval()

    raw_eval = load_dataset(args.dataset_name, args.dataset_config, split=args.eval_split)
    sample_n = min(args.max_eval_samples, len(raw_eval))
    raw_eval = raw_eval.select(range(sample_n))

    questions: List[str] = []
    for ex in raw_eval:
        q = _extract_question(ex)
        if q:
            questions.append(q)
    if not questions:
        raise ValueError("No valid question fields found in evaluation split.")

    results: List[Dict[str, str]] = []
    for idx, q in enumerate(questions, start=1):
        prompt = f"Question: {q}\nAnswer:"
        base_out = _generate(base_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
        ft_out = _generate(finetuned_model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
        row = {
            "id": str(idx),
            "question": q,
            "base_output": base_out,
            "finetuned_output": ft_out,
        }
        results.append(row)
        print("=" * 80)
        print(f"[{idx}] Question: {q}")
        print(f"Base: {base_out}")
        print(f"Finetuned: {ft_out}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=" * 80)
    print(f"Saved comparison results to: {args.output_file}")


if __name__ == "__main__":
    main()
