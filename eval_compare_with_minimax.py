import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DIMENSION_WEIGHTS = {
    "correctness": 0.70,
    "reasoning": 0.10,
    "completeness": 0.05,
    "clarity": 0.05,
    "instruction_following": 0.10,
}


@dataclass
class ModelBundle:
    model: Any
    tokenizer: AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a fine-tuned model against Qwen3-4B using MiniMax as an LLM judge."
    )
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400/")
    parser.add_argument("--judge_model_name", type=str, default="qwen3.5-plus")
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default="sk-345e0f458bdd4291826b54bac8099ad2",
    )
    parser.add_argument("--dataset_name", type=str, default="qwedsacf/competition_math")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="competition_math",
        choices=[
            "auto",
            "gsm8k",
            "competition_math",
            "svamp",
            "commonsense_qa",
            "arc",
        ],
        help="How to normalize dataset rows into question/reference pairs.",
    )
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="./outputs/eval_compare_with_minimax_competition_math.json")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="reasoning",
        choices=["reasoning", "qa"],
        help="Use reasoning prompt for math/reasoning models, or qa prompt for direct answer models.",
    )
    parser.add_argument(
        "--judge_retries",
        type=int,
        default=3,
        help="Retry count when the judge response is not valid JSON.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent judge worker threads.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="JSONL log file for per-sample incremental results. Defaults to <output_file>.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output/log files and skip completed sample indices.",
    )
    return parser.parse_args()


def get_quant_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
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


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_and_adapter(args: argparse.Namespace) -> Tuple[ModelBundle, ModelBundle]:
    tokenizer = load_tokenizer(args.base_model_name)
    quant_config = get_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model_dtype = dtype if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    finetuned_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_base, args.adapter_path)
    finetuned_model.eval()

    return (
        ModelBundle(model=base_model, tokenizer=tokenizer),
        ModelBundle(model=finetuned_model, tokenizer=tokenizer),
    )


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


def extract_gsm8k_reference(answer: str) -> Tuple[str, str]:
    if "####" in answer:
        reasoning, final = answer.rsplit("####", 1)
        return reasoning.strip(), normalize_final_answer(final)
    return answer.strip(), normalize_final_answer(answer)


def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return normalize_final_answer(matches[-1])
    return normalize_final_answer(text)


def extract_final_answer_from_response(text: str) -> str:
    stripped = text.strip()
    answer_match = re.search(r"Answer:\s*(.+)$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        stripped = answer_match.group(1).strip()
    return extract_boxed_answer(stripped)


def normalize_final_answer(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return text


def format_multiple_choice_question(stem: str, labels: List[str], texts: List[str]) -> str:
    options = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return f"{stem}\nChoices:\n" + "\n".join(options)


def detect_dataset_format(dataset_name: str, column_names: List[str]) -> str:
    lowered = dataset_name.lower()
    cols = set(column_names)
    if lowered == "gsm8k" or {"question", "answer"}.issubset(cols):
        return "gsm8k"
    if lowered in {"hendrycks/competition_math", "qwedsacf/competition_math"} or {"problem", "solution"}.issubset(cols):
        return "competition_math"
    if lowered == "chilled/svamp" or {"Body", "Question", "Answer"}.issubset(cols):
        return "svamp"
    if lowered == "tau/commonsense_qa" or {"question", "choices", "answerKey"}.issubset(cols):
        return "commonsense_qa"
    if lowered == "allenai/ai2_arc" or {"question", "choices", "answerKey"}.issubset(cols):
        return "arc"
    raise ValueError(
        f"Could not auto-detect dataset format for {dataset_name!r} with columns {sorted(column_names)}. "
        "Pass --dataset_format explicitly."
    )


def normalize_eval_row(row: Dict[str, Any], dataset_format: str) -> Dict[str, str]:
    if dataset_format == "gsm8k":
        question = str(row.get("question", "")).strip()
        reasoning, answer = extract_gsm8k_reference(str(row.get("answer", "")).strip())
        return {
            "question": question,
            "reference_reasoning": reasoning,
            "reference_answer": answer,
        }

    if dataset_format == "competition_math":
        question = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        return {
            "question": question,
            "reference_reasoning": solution,
            "reference_answer": extract_boxed_answer(solution),
        }

    if dataset_format == "svamp":
        body = str(row.get("Body", "")).strip()
        question_text = str(row.get("Question", "")).strip()
        question = " ".join(part for part in [body, question_text] if part).strip()
        return {
            "question": question,
            "reference_reasoning": "",
            "reference_answer": normalize_final_answer(str(row.get("Answer", "")).strip()),
        }

    if dataset_format in {"commonsense_qa", "arc"}:
        question = str(row.get("question", "")).strip()
        choices = row.get("choices") or {}
        labels = [str(x).strip() for x in choices.get("label", [])]
        texts = [str(x).strip() for x in choices.get("text", [])]
        if labels and texts and len(labels) == len(texts):
            question = format_multiple_choice_question(question, labels, texts)
        answer_key = str(row.get("answerKey", "")).strip()
        answer_text = answer_key
        if answer_key and labels and texts:
            label_to_text = {label: text for label, text in zip(labels, texts)}
            answer_text = label_to_text.get(answer_key, answer_key)
        return {
            "question": question,
            "reference_reasoning": "",
            "reference_answer": answer_text,
        }

    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def build_eval_dataset(args: argparse.Namespace) -> Dataset:
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    except ValueError as exc:
        message = str(exc)
        if "Unknown split" not in message:
            raise

        dataset_dict = load_dataset(args.dataset_name, args.dataset_config)
        available_splits = list(dataset_dict.keys())
        if len(available_splits) != 1:
            raise ValueError(
                f"Requested split {args.dataset_split!r} is unavailable for {args.dataset_name!r}. "
                f"Available splits: {available_splits}."
            ) from exc

        fallback_split = available_splits[0]
        print(
            f"[info] requested split {args.dataset_split!r} is unavailable for {args.dataset_name}. "
            f"Falling back to the only available split: {fallback_split!r}."
        )
        args.dataset_split = fallback_split
        dataset = dataset_dict[fallback_split]

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset_format = args.dataset_format
    if dataset_format == "auto":
        dataset_format = detect_dataset_format(args.dataset_name, dataset.column_names)

    normalized_rows = []
    for idx, row in enumerate(dataset):
        sample = normalize_eval_row(row, dataset_format)
        if not sample["question"] or not sample["reference_answer"]:
            print(f"[skip] sample {idx}: missing normalized question or reference answer")
            continue
        normalized_rows.append(sample)

    if not normalized_rows:
        raise ValueError("No valid evaluation samples found after dataset normalization.")
    return Dataset.from_list(normalized_rows)


def make_judge_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def repair_json_escapes(text: str) -> str:
    # Judges sometimes emit LaTeX like \boxed or \frac inside JSON strings.
    # Escape backslashes that are not part of valid JSON escape sequences.
    text = re.sub(r"\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})", r"\\\\", text)
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return json.loads(repair_json_escapes(cleaned))

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(repair_json_escapes(candidate))


def weighted_total(scores: Dict[str, float]) -> float:
    return round(sum(scores[k] * w for k, w in DIMENSION_WEIGHTS.items()), 4)


def coerce_score_block(block: Dict[str, Any]) -> Dict[str, Any]:
    numeric_scores: Dict[str, float] = {}
    for key in DIMENSION_WEIGHTS:
        value = float(block.get(key, 0.0))
        value = max(0.0, min(10.0, value))
        numeric_scores[key] = value
    return {
        **numeric_scores,
        "total": weighted_total(numeric_scores),
        "strengths": str(block.get("strengths", "")).strip(),
        "weaknesses": str(block.get("weaknesses", "")).strip(),
    }


def build_judge_messages(
    question: str,
    reference_reasoning: str,
    reference_answer: str,
    answer_a: str,
    answer_b: str,
) -> List[Dict[str, str]]:
    answer_a_final = extract_final_answer_from_response(answer_a)
    answer_b_final = extract_final_answer_from_response(answer_b)
    rubric_text = (
        "You are an impartial evaluator.\n"
        "Score each model on a 0-10 scale for these dimensions:\n"
        "- correctness: mathematical correctness of the final answer, with overwhelming priority\n"
        "- reasoning: soundness and usefulness of the reasoning process\n"
        "- completeness: whether the response fully addresses the problem\n"
        "- clarity: readability, structure, and clarity\n"
        "- instruction_following: whether the response follows the requested format and task\n\n"
        "Weight the dimensions as:\n"
        "- correctness: 0.70\n"
        "- reasoning: 0.10\n"
        "- completeness: 0.05\n"
        "- clarity: 0.05\n"
        "- instruction_following: 0.10\n\n"
        "The final answer matters far more than the reasoning quality.\n"
        "If a model's final answer is wrong, score correctness very low even if the reasoning looks strong.\n"
        "If a model's final answer matches the reference, reward it strongly even when the reasoning is brief.\n"
        "Do not blindly reward longer answers.\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "model_a": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "model_b": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "winner": "A" | "B" | "tie",\n'
        '  "summary": "..."\n'
        "}"
    )
    user_text = (
        f"Question:\n{question}\n\n"
        f"Reference reasoning:\n{reference_reasoning or '[not provided]'}\n\n"
        f"Reference final answer:\n{reference_answer}\n\n"
        f"Model A extracted final answer:\n{answer_a_final}\n\n"
        f"Model A response:\n{answer_a}\n\n"
        f"Model B extracted final answer:\n{answer_b_final}\n\n"
        f"Model B response:\n{answer_b}\n"
    )
    return [
        {"role": "system", "content": rubric_text},
        {"role": "user", "content": user_text},
    ]


def judge_once(
    client: OpenAI,
    judge_model_name: str,
    messages: List[Dict[str, str]],
    judge_retries: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(judge_retries):
        try:
            completion = client.chat.completions.create(
                model=judge_model_name,
                messages=messages,
                stream=False,
            )
            content = completion.choices[0].message.content or ""
            return extract_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 + attempt)
    assert last_error is not None
    raise last_error


def judge_pair(
    client: OpenAI,
    args: argparse.Namespace,
    question: str,
    reference_reasoning: str,
    reference_answer: str,
    base_answer: str,
    finetuned_answer: str,
) -> Dict[str, Any]:
    forward = judge_once(
        client=client,
        judge_model_name=args.judge_model_name,
        messages=build_judge_messages(
            question=question,
            reference_reasoning=reference_reasoning,
            reference_answer=reference_answer,
            answer_a=base_answer,
            answer_b=finetuned_answer,
        ),
        judge_retries=args.judge_retries,
    )
    reverse = judge_once(
        client=client,
        judge_model_name=args.judge_model_name,
        messages=build_judge_messages(
            question=question,
            reference_reasoning=reference_reasoning,
            reference_answer=reference_answer,
            answer_a=finetuned_answer,
            answer_b=base_answer,
        ),
        judge_retries=args.judge_retries,
    )

    base_forward = coerce_score_block(forward["model_a"])
    finetuned_forward = coerce_score_block(forward["model_b"])
    finetuned_reverse = coerce_score_block(reverse["model_a"])
    base_reverse = coerce_score_block(reverse["model_b"])

    base_scores = {
        key: round((base_forward[key] + base_reverse[key]) / 2.0, 4)
        for key in list(DIMENSION_WEIGHTS.keys()) + ["total"]
    }
    finetuned_scores = {
        key: round((finetuned_forward[key] + finetuned_reverse[key]) / 2.0, 4)
        for key in list(DIMENSION_WEIGHTS.keys()) + ["total"]
    }

    margin = finetuned_scores["total"] - base_scores["total"]
    if margin > 0.15:
        winner = "finetuned"
    elif margin < -0.15:
        winner = "base"
    else:
        winner = "tie"

    return {
        "base_scores": base_scores,
        "finetuned_scores": finetuned_scores,
        "base_final_answer": extract_final_answer_from_response(base_answer),
        "finetuned_final_answer": extract_final_answer_from_response(finetuned_answer),
        "reference_final_answer": reference_answer,
        "winner": winner,
        "forward_summary": str(forward.get("summary", "")).strip(),
        "reverse_summary": str(reverse.get("summary", "")).strip(),
        "forward_raw": forward,
        "reverse_raw": reverse,
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(DIMENSION_WEIGHTS.keys()) + ["total"]
    base = {key: mean([sample["judge"]["base_scores"][key] for sample in samples]) for key in keys}
    finetuned = {key: mean([sample["judge"]["finetuned_scores"][key] for sample in samples]) for key in keys}

    win_counts = {"base": 0, "finetuned": 0, "tie": 0}
    for sample in samples:
        win_counts[sample["judge"]["winner"]] += 1

    deltas = {key: round(finetuned[key] - base[key], 4) for key in keys}
    return {
        "base_average": base,
        "finetuned_average": finetuned,
        "delta": deltas,
        "win_counts": win_counts,
        "sample_count": len(samples),
    }


def resolve_log_file(args: argparse.Namespace) -> str:
    if args.log_file:
        return args.log_file
    base, _ = os.path.splitext(args.output_file)
    return f"{base}.jsonl"


def resolve_answer_cache_file(log_file: str) -> str:
    base, ext = os.path.splitext(log_file)
    suffix = ext or ".jsonl"
    return f"{base}.answers{suffix}"


def load_existing_results(output_file: str, log_file: str) -> List[Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for sample in payload.get("samples", []):
                by_index[int(sample["index"])] = sample
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load existing output file {output_file}: {exc}")

    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    by_index[int(sample["index"])] = sample
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load existing log file {log_file}: {exc}")

    return [by_index[idx] for idx in sorted(by_index)]


def append_log_record(log_file: str, sample_result: Dict[str, Any]) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample_result, ensure_ascii=False) + "\n")


def load_answer_cache(answer_cache_file: str) -> Dict[int, Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(answer_cache_file):
        return by_index

    try:
        with open(answer_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                by_index[int(sample["index"])] = sample
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to load answer cache {answer_cache_file}: {exc}")
    return by_index


def write_output_snapshot(args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    summary = aggregate_results(results)
    output = {
        "config": {
            "base_model_name": args.base_model_name,
            "adapter_path": args.adapter_path,
            "judge_model_name": args.judge_model_name,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "dataset_format": args.dataset_format,
            "max_samples": args.max_samples,
            "prompt_style": args.prompt_style,
            "weights": DIMENSION_WEIGHTS,
            "threads": args.threads,
            "log_file": resolve_log_file(args),
        },
        "summary": summary,
        "samples": sorted(results, key=lambda sample: int(sample["index"])),
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def judge_sample_task(
    args: argparse.Namespace,
    sample_payload: Dict[str, Any],
) -> Dict[str, Any]:
    client = make_judge_client(args.judge_api_key)
    judge_result = judge_pair(
        client=client,
        args=args,
        question=sample_payload["question"],
        reference_reasoning=sample_payload["reference_reasoning"],
        reference_answer=sample_payload["reference_answer"],
        base_answer=sample_payload["base_answer"],
        finetuned_answer=sample_payload["finetuned_answer"],
    )
    return {
        **sample_payload,
        "judge": judge_result,
    }


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
    results_by_index: Dict[int, Dict[str, Any]] = {int(sample["index"]): sample for sample in results}
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


if __name__ == "__main__":
    main()
