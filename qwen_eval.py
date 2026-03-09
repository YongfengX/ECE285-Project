import argparse
import concurrent.futures
import json
import os
import random
from typing import Dict

import torch

from eval_compare_with_minimax import (
    ModelBundle,
    aggregate_results,
    append_log_record,
    build_eval_dataset,
    judge_sample_task,
    load_answer_cache,
    load_base_and_adapter,
    load_existing_results,
    parse_level_filter,
    resolve_answer_cache_file,
    resolve_log_file,
    write_output_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat-style evaluation: compare base vs finetuned and judge with MiniMax-compatible API."
    )
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400")
    parser.add_argument("--judge_model_name", type=str, default="qwen3.5-plus")
    parser.add_argument("--judge_api_key", type=str, required=True)
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
        "--output_file",
        type=str,
        default="./outputs/eval_compare_with_minimax_chat_style_competition_math.json",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="reasoning",
        choices=["reasoning", "qa"],
        help="Use reasoning prompt for math/reasoning models, or qa prompt for direct answer models.",
    )
    parser.add_argument("--judge_retries", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="JSONL log file for per-sample incremental results. Defaults to <output_file>.jsonl.",
    )
    parser.add_argument("--resume", action="store_true")
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    log_file = resolve_log_file(args)
    answer_cache_file = resolve_answer_cache_file(log_file)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(answer_cache_file) or ".", exist_ok=True)

    print("Loading models...")
    base_bundle, finetuned_bundle = load_base_and_adapter(args)

    print("Loading evaluation dataset...")
    dataset = build_eval_dataset(args)
    results = load_existing_results(args.output_file, log_file) if args.resume else []
    cached_answers = load_answer_cache(answer_cache_file) if args.resume else {}
    results_by_index: Dict[int, Dict] = {int(sample["index"]): sample for sample in results}
    completed_indices = set(results_by_index)
    if completed_indices:
        print(f"[resume] loaded {len(completed_indices)} completed samples from existing files.")
    if cached_answers:
        print(f"[resume] loaded {len(cached_answers)} generated-answer cache entries.")

    total_samples = len(dataset)
    pending_futures: Dict[concurrent.futures.Future, int] = {}

    def flush_completed(block: bool) -> None:
        if not pending_futures:
            return
        done, _ = concurrent.futures.wait(
            pending_futures.keys(),
            timeout=None if block else 0,
            return_when=concurrent.futures.FIRST_COMPLETED if not block else concurrent.futures.ALL_COMPLETED,
        )
        for future in done:
            idx = pending_futures.pop(future)
            sample_result = future.result()
            results_by_index[idx] = sample_result
            append_log_record(log_file, sample_result)
            write_output_snapshot(args, list(results_by_index.values()))
            judge_result = sample_result["judge"]
            print(
                f"[{idx + 1}/{total_samples}] "
                f"base={judge_result['base_scores']['total']:.2f} "
                f"finetuned={judge_result['finetuned_scores']['total']:.2f} "
                f"winner={judge_result['winner']}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        for idx, row in enumerate(dataset):
            if idx in completed_indices:
                print(f"[{idx + 1}/{total_samples}] Skipping completed sample from resume log.")
                continue

            while len(pending_futures) >= max(1, args.threads):
                flush_completed(block=True)

            if idx in cached_answers:
                sample_payload = cached_answers[idx]
                print(f"[{idx + 1}/{total_samples}] Reusing cached generated answers.")
            else:
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

                print(f"[{idx + 1}/{total_samples}] Generating finetuned answer...")
                finetuned_answer = generate_answer(
                    bundle=finetuned_bundle,
                    question=question,
                    prompt_style=args.prompt_style,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                sample_payload = {
                    "index": idx,
                    "question": question,
                    "reference_reasoning": reference_reasoning,
                    "reference_answer": reference_answer,
                    "level": level,
                    "base_answer": base_answer,
                    "finetuned_answer": finetuned_answer,
                }
                append_log_record(answer_cache_file, sample_payload)
                cached_answers[idx] = sample_payload

            print(f"[{idx + 1}/{total_samples}] Judging with {args.judge_model_name}...")
            future = executor.submit(judge_sample_task, args, sample_payload)
            pending_futures[future] = idx
            flush_completed(block=False)

        flush_completed(block=True)

    results = sorted(results_by_index.values(), key=lambda sample: int(sample["index"]))
    summary = aggregate_results(results)
    write_output_snapshot(args, results)

    print("\n=== Aggregate Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {args.output_file}")
    print(f"Incremental log saved to: {log_file}")
    print(f"Generated-answer cache saved to: {answer_cache_file}")
    if args.levels:
        print(f"Levels filter: {parse_level_filter(args.levels)}")


if __name__ == "__main__":
    main()
