import argparse
import json
import os
import random
from typing import Dict, List

import torch

from eval_compare_with_minimax import ModelBundle, build_eval_dataset, load_base_and_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-driven OpenR1 compare: base vs finetuned, raw outputs only, no LLM judge."
    )
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400")
    parser.add_argument("--dataset_name", type=str, default="qwedsacf/competition_math")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="competition_math",
        choices=["auto", "gsm8k", "competition_math", "svamp", "commonsense_qa", "arc"],
    )
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help="Comma-separated competition_math levels to keep, e.g. 'Level 1,Level 3,Level 5'.",
    )
    parser.add_argument(
        "--samples_per_level",
        type=int,
        default=0,
        help="If > 0, sample up to this many examples per level after optional level filtering.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="reasoning",
        choices=["reasoning", "qa"],
        help="Use reasoning prompt for math/reasoning models, or qa prompt for direct answer models.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./outputs/chat_openr1_dataset_eval.json",
    )
    return parser.parse_args()


def build_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "qa":
        return f"Question: {question}\nAnswer:"
    return (
        f"Question: {question}\n"
        "Please reason step by step, then provide the final answer.\n"
        "Use this format exactly:\n"
        "Reasoning:\n"
        "<your reasoning>\n\n"
        "Answer:\n"
        "<final answer>\n"
    )


def generate_answer(
    bundle: ModelBundle,
    question: str,
    prompt_style: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_prompt(question, prompt_style)
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=1.1,
            eos_token_id=bundle.tokenizer.eos_token_id,
            pad_token_id=bundle.tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return bundle.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def write_results(output_file: str, args: argparse.Namespace, results: List[Dict]) -> None:
    payload = {
        "config": {
            "base_model_name": args.base_model_name,
            "adapter_path": args.adapter_path,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "dataset_format": args.dataset_format,
            "max_samples": args.max_samples,
            "levels": args.levels,
            "samples_per_level": args.samples_per_level,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt_style": args.prompt_style,
            "load_in_4bit": args.load_in_4bit,
            "seed": args.seed,
        },
        "sample_count": len(results),
        "samples": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print("Loading models...")
    base_bundle, finetuned_bundle = load_base_and_adapter(args)

    print("Loading evaluation dataset...")
    dataset = build_eval_dataset(args)
    total_samples = len(dataset)
    results: List[Dict] = []

    for idx, row in enumerate(dataset):
        question = str(row["question"]).strip()
        reference_reasoning = str(row.get("reference_reasoning", "")).strip()
        reference_answer = str(row.get("reference_answer", "")).strip()
        level = str(row.get("level", "")).strip()

        print(f"[{idx + 1}/{total_samples}] Generating base answer...")
        base_answer = generate_answer(
            bundle=base_bundle,
            question=question,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("[Base Raw Output]")
        print(base_answer)

        print(f"[{idx + 1}/{total_samples}] Generating finetuned answer...")
        finetuned_answer = generate_answer(
            bundle=finetuned_bundle,
            question=question,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("[Finetuned Raw Output]")
        print(finetuned_answer)

        sample = {
            "index": idx,
            "question": question,
            "reference_reasoning": reference_reasoning,
            "reference_answer": reference_answer,
            "level": level,
            "base_answer": base_answer,
            "finetuned_answer": finetuned_answer,
        }
        results.append(sample)
        write_results(args.output_file, args, results)

    print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
