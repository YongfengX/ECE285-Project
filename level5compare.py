import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS = {
    "correctness": 0.40,
    "reasoning": 0.40,
    "completeness": 0.05,
    "clarity": 0.05,
    "instruction_following": 0.10,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: Any
    tokenizer: AutoTokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone dataset-driven OpenR1 compare: base vs adapter1 vs adapter2 with LLM judge."
    )
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-2400")
    parser.add_argument("--adapter2_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400")
    parser.add_argument("--judge_model_name", type=str, default="qwen-plus")
    parser.add_argument("--judge_api_key", type=str, default="")
    # competition_math Level 5 默认参数
    parser.add_argument("--dataset_name", type=str, default="qwedsacf/competition_math")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="competition_math",
        choices=["auto", "gsm8k", "competition_math", "svamp", "commonsense_qa", "arc"],
    )
    parser.add_argument("--max_samples", type=int, default=30)       # ← 30 道
    parser.add_argument(
        "--levels",
        type=str,
        default="Level 5",                                            # ← 只做 Level 5
        help="Comma-separated competition_math levels, e.g. 'Level 1,Level 3,Level 5'.",
    )
    parser.add_argument(
        "--samples_per_level",
        type=int,
        default=0,
        help="If > 0, sample up to this many examples per level.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge_retries", type=int, default=3)
    parser.add_argument(
        "--output_file",
        type=str,
        default="./outputs/compare_result_level5.json",               # ← 文件名对应
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


def load_base_and_adapters(args: argparse.Namespace) -> Tuple[ModelBundle, ModelBundle, ModelBundle]:
    tokenizer = load_tokenizer(args.base_model_name)
    quant_config = get_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model_dtype = dtype if torch.cuda.is_available() else torch.float32

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    print("  Loading adapter1 model...")
    adapter1_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    adapter1_model = PeftModel.from_pretrained(adapter1_base, args.adapter_path)
    adapter1_model.eval()

    print("  Loading adapter2 model...")
    adapter2_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    adapter2_model = PeftModel.from_pretrained(adapter2_base, args.adapter2_path)
    adapter2_model.eval()

    return (
        ModelBundle(model=base_model, tokenizer=tokenizer),
        ModelBundle(model=adapter1_model, tokenizer=tokenizer),
        ModelBundle(model=adapter2_model, tokenizer=tokenizer),
    )


# ---------------------------------------------------------------------------
# Prompt & generation
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
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
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate raw model output without any post-processing."""
    prompt = build_prompt(question)
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
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return bundle.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Dataset loading & normalization
# ---------------------------------------------------------------------------

def extract_gsm8k_reference(answer: str) -> Tuple[str, str]:
    if "####" in answer:
        reasoning, final = answer.rsplit("####", 1)
        return reasoning.strip(), final.strip()
    return answer.strip(), answer.strip()


def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def format_multiple_choice_question(stem: str, labels: List[str], texts: List[str]) -> str:
    options = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return f"{stem}\nChoices:\n" + "\n".join(options)


def normalize_eval_row(row: Dict[str, Any], dataset_format: str) -> Dict[str, str]:
    if dataset_format == "gsm8k":
        question = str(row.get("question", "")).strip()
        reasoning, answer = extract_gsm8k_reference(str(row.get("answer", "")).strip())
        return {"question": question, "reference_reasoning": reasoning, "reference_answer": answer, "level": ""}

    if dataset_format == "competition_math":
        question = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        return {
            "question": question,
            "reference_reasoning": solution,
            "reference_answer": extract_boxed_answer(solution),
            "level": str(row.get("level", "")).strip(),
        }

    if dataset_format == "svamp":
        body = str(row.get("Body", "")).strip()
        question_text = str(row.get("Question", "")).strip()
        question = " ".join(p for p in [body, question_text] if p).strip()
        return {"question": question, "reference_reasoning": "", "reference_answer": str(row.get("Answer", "")).strip(), "level": ""}

    if dataset_format in {"commonsense_qa", "arc"}:
        question = str(row.get("question", "")).strip()
        choices = row.get("choices") or {}
        labels = [str(x).strip() for x in choices.get("label", [])]
        texts = [str(x).strip() for x in choices.get("text", [])]
        if labels and texts and len(labels) == len(texts):
            question = format_multiple_choice_question(question, labels, texts)
        answer_key = str(row.get("answerKey", "")).strip()
        label_to_text = {l: t for l, t in zip(labels, texts)}
        answer_text = label_to_text.get(answer_key, answer_key)
        return {"question": question, "reference_reasoning": "", "reference_answer": answer_text, "level": ""}

    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def detect_dataset_format(dataset_name: str, column_names: List[str]) -> str:
    cols = set(column_names)
    lowered = dataset_name.lower()
    if lowered == "gsm8k" or {"question", "answer"}.issubset(cols):
        return "gsm8k"
    if "competition_math" in lowered or {"problem", "solution"}.issubset(cols):
        return "competition_math"
    if "svamp" in lowered or {"Body", "Question", "Answer"}.issubset(cols):
        return "svamp"
    if "commonsense_qa" in lowered or ("commonsense" in lowered):
        return "commonsense_qa"
    if "arc" in lowered:
        return "arc"
    raise ValueError(
        f"Could not auto-detect dataset format for {dataset_name!r} with columns {sorted(column_names)}. "
        "Pass --dataset_format explicitly."
    )


def parse_level_filter(levels: str) -> List[str]:
    return [lv.strip() for lv in levels.split(",") if lv.strip()]


def stratified_sample_by_level(rows: List[Dict], samples_per_level: int, seed: int) -> List[Dict]:
    grouped: Dict[str, List] = defaultdict(list)
    for row in rows:
        grouped[row.get("level", "") or "[unknown]"].append(row)
    rng = random.Random(seed)
    result = []
    for level in sorted(grouped.keys()):
        bucket = grouped[level]
        if len(bucket) <= samples_per_level:
            result.extend(bucket)
        else:
            indices = list(range(len(bucket)))
            rng.shuffle(indices)
            result.extend(bucket[i] for i in sorted(indices[:samples_per_level]))
    return result


def build_eval_dataset(args: argparse.Namespace) -> Dataset:
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    except ValueError as exc:
        if "Unknown split" not in str(exc):
            raise
        dataset_dict = load_dataset(args.dataset_name, args.dataset_config)
        available = list(dataset_dict.keys())
        if len(available) != 1:
            raise ValueError(f"Requested split {args.dataset_split!r} unavailable. Available: {available}") from exc
        args.dataset_split = available[0]
        print(f"[info] falling back to split {args.dataset_split!r}")
        dataset = dataset_dict[args.dataset_split]

    fmt = args.dataset_format
    if fmt == "auto":
        fmt = detect_dataset_format(args.dataset_name, dataset.column_names)

    allowed_levels = set(parse_level_filter(args.levels))
    normalized_rows = []
    for idx, row in enumerate(dataset):
        sample = normalize_eval_row(row, fmt)
        if not sample["question"] or not sample["reference_answer"]:
            print(f"[skip] sample {idx}: missing question or reference_answer")
            continue
        if allowed_levels and sample.get("level", "") not in allowed_levels:
            continue
        normalized_rows.append(sample)

    if not normalized_rows:
        raise ValueError("No valid evaluation samples found after normalization.")

    if args.samples_per_level > 0:
        normalized_rows = stratified_sample_by_level(normalized_rows, args.samples_per_level, args.seed)
        level_counts: Dict[str, int] = defaultdict(int)
        for row in normalized_rows:
            level_counts[row.get("level", "") or "[unknown]"] += 1
        print(f"[info] stratified sample: { {k: v for k, v in sorted(level_counts.items())} }")

    # 随机打乱后取前 max_samples 道，保证样本多样性
    if args.max_samples > 0:
        rng = random.Random(args.seed)
        rng.shuffle(normalized_rows)
        normalized_rows = normalized_rows[: args.max_samples]

    return Dataset.from_list(normalized_rows)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

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
    return re.sub(r"\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})", r"\\\\", text)


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
    numeric: Dict[str, float] = {}
    for key in DIMENSION_WEIGHTS:
        value = max(0.0, min(10.0, float(block.get(key, 0.0))))
        numeric[key] = value
    return {
        **numeric,
        "total": weighted_total(numeric),
        "strengths": str(block.get("strengths", "")).strip(),
        "weaknesses": str(block.get("weaknesses", "")).strip(),
    }


def build_judge_messages(
    question: str,
    reference_reasoning: str,
    reference_answer: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
) -> List[Dict[str, str]]:
    rubric = (
        "You are an impartial evaluator.\n"
        "Score each model on a 0-10 scale for these dimensions:\n"
        "- correctness: mathematical correctness of the final answer (highest priority)\n"
        "- reasoning: soundness and usefulness of the reasoning process\n"
        "- completeness: whether the response fully addresses the problem\n"
        "- clarity: readability, structure, and clarity\n"
        "- instruction_following: whether the response follows the requested format\n\n"
        "If a model's final answer is wrong, score correctness very low even if reasoning looks strong.\n"
        "If a model's final answer matches the reference, reward it strongly even when reasoning is brief.\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "model_a": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "model_b": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "model_c": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "winner": "A" | "B" | "C" | "tie",\n'
        '  "summary": "..."\n'
        "}"
    )
    user_text = (
        f"Question:\n{question}\n\n"
        f"Reference reasoning:\n{reference_reasoning or '[not provided]'}\n\n"
        f"Reference final answer:\n{reference_answer}\n\n"
        f"Model A response:\n{answer_a}\n\n"
        f"Model B response:\n{answer_b}\n\n"
        f"Model C response:\n{answer_c}\n"
    )
    return [
        {"role": "system", "content": rubric},
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
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise last_error  # type: ignore[misc]


def _average_numeric_scores(score_blocks: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = list(DIMENSION_WEIGHTS.keys()) + ["total"]
    return {
        key: round(sum(float(block[key]) for block in score_blocks) / max(1, len(score_blocks)), 4)
        for key in keys
    }


def judge_triplet(
    client: OpenAI,
    args: argparse.Namespace,
    question: str,
    reference_reasoning: str,
    reference_answer: str,
    base_answer: str,
    adapter1_answer: str,
    adapter2_answer: str,
) -> Dict[str, Any]:
    answers_by_model = {
        "base": base_answer,
        "adapter1": adapter1_answer,
        "adapter2": adapter2_answer,
    }
    round_orders = [
        ("base", "adapter1", "adapter2"),
        ("adapter1", "adapter2", "base"),
        ("adapter2", "base", "adapter1"),
    ]

    score_blocks: Dict[str, List[Dict[str, Any]]] = {"base": [], "adapter1": [], "adapter2": []}
    round_raw: List[Dict[str, Any]] = []
    round_summaries: List[str] = []
    for model_a, model_b, model_c in round_orders:
        round_result = judge_once(
            client=client,
            judge_model_name=args.judge_model_name,
            messages=build_judge_messages(
                question,
                reference_reasoning,
                reference_answer,
                answers_by_model[model_a],
                answers_by_model[model_b],
                answers_by_model[model_c],
            ),
            judge_retries=args.judge_retries,
        )
        score_blocks[model_a].append(coerce_score_block(round_result["model_a"]))
        score_blocks[model_b].append(coerce_score_block(round_result["model_b"]))
        score_blocks[model_c].append(coerce_score_block(round_result["model_c"]))
        round_raw.append(round_result)
        round_summaries.append(str(round_result.get("summary", "")).strip())

    base_scores = _average_numeric_scores(score_blocks["base"])
    adapter1_scores = _average_numeric_scores(score_blocks["adapter1"])
    adapter2_scores = _average_numeric_scores(score_blocks["adapter2"])

    totals = {
        "base": base_scores["total"],
        "adapter1": adapter1_scores["total"],
        "adapter2": adapter2_scores["total"],
    }
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    winner = "tie" if (len(ranked) < 2 or (ranked[0][1] - ranked[1][1]) <= 0.15) else ranked[0][0]

    return {
        "base_scores": base_scores,
        "adapter1_scores": adapter1_scores,
        "adapter2_scores": adapter2_scores,
        "winner": winner,
        "round_summaries": round_summaries,
        "round_raw": round_raw,
    }


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(DIMENSION_WEIGHTS.keys()) + ["total"]
    base_avg = {k: mean([s["judge"]["base_scores"][k] for s in samples]) for k in keys}
    adapter1_avg = {k: mean([s["judge"]["adapter1_scores"][k] for s in samples]) for k in keys}
    adapter2_avg = {k: mean([s["judge"]["adapter2_scores"][k] for s in samples]) for k in keys}
    win_counts: Dict[str, int] = {"base": 0, "adapter1": 0, "adapter2": 0, "tie": 0}
    for s in samples:
        winner = s["judge"].get("winner", "tie")
        if winner not in win_counts:
            winner = "tie"
        win_counts[winner] += 1
    overall_avg = {k: round((base_avg[k] + adapter1_avg[k] + adapter2_avg[k]) / 3.0, 4) for k in keys}
    return {
        "base_average": base_avg,
        "adapter1_average": adapter1_avg,
        "adapter2_average": adapter2_avg,
        "overall_average": overall_avg,
        "delta_adapter1_vs_base": {k: round(adapter1_avg[k] - base_avg[k], 4) for k in keys},
        "delta_adapter2_vs_base": {k: round(adapter2_avg[k] - base_avg[k], 4) for k in keys},
        "win_counts": win_counts,
        "sample_count": len(samples),
    }


def write_results(output_file: str, args: argparse.Namespace, results: List[Dict]) -> None:
    summary = aggregate_results(results) if results else {}
    payload = {
        "config": {
            "base_model_name": args.base_model_name,
            "adapter_path": args.adapter_path,
            "adapter2_path": args.adapter2_path,
            "judge_model_name": args.judge_model_name,
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
            "load_in_4bit": args.load_in_4bit,
            "seed": args.seed,
            "judge_retries": args.judge_retries,
            "weights": DIMENSION_WEIGHTS,
        },
        "summary": summary,
        "sample_count": len(results),
        "samples": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print("Loading models...")
    base_bundle, adapter1_bundle, adapter2_bundle = load_base_and_adapters(args)

    print("Loading evaluation dataset...")
    dataset = build_eval_dataset(args)
    total_samples = len(dataset)
    results: List[Dict] = []
    client = make_judge_client(args.judge_api_key)

    for idx, row in enumerate(dataset):
        question = str(row["question"]).strip()
        reference_reasoning = str(row.get("reference_reasoning", "")).strip()
        reference_answer = str(row.get("reference_answer", "")).strip()
        level = str(row.get("level", "")).strip()

        print(f"\n[{idx + 1}/{total_samples}] Question: {question[:80]}...")

        print(f"[{idx + 1}/{total_samples}] Generating base answer...")
        base_answer = generate_answer(
            bundle=base_bundle,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("[Base Raw Output]")
        print(base_answer)

        print(f"[{idx + 1}/{total_samples}] Generating adapter1 answer...")
        adapter1_answer = generate_answer(
            bundle=adapter1_bundle,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("[Adapter1 Raw Output]")
        print(adapter1_answer)

        print(f"[{idx + 1}/{total_samples}] Generating adapter2 answer...")
        adapter2_answer = generate_answer(
            bundle=adapter2_bundle,
            question=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("[Adapter2 Raw Output]")
        print(adapter2_answer)

        print(f"[{idx + 1}/{total_samples}] Judging with {args.judge_model_name}...")
        judge_result = judge_triplet(
            client=client,
            args=args,
            question=question,
            reference_reasoning=reference_reasoning,
            reference_answer=reference_answer,
            base_answer=base_answer,
            adapter1_answer=adapter1_answer,
            adapter2_answer=adapter2_answer,
        )
        print(
            f"  base={judge_result['base_scores']['total']:.2f} "
            f"adapter1={judge_result['adapter1_scores']['total']:.2f} "
            f"adapter2={judge_result['adapter2_scores']['total']:.2f} "
            f"winner={judge_result['winner']}"
        )

        results.append({
            "index": idx,
            "question": question,
            "reference_reasoning": reference_reasoning,
            "reference_answer": reference_answer,
            "level": level,
            "base_raw_output": base_answer,
            "adapter1_raw_output": adapter1_answer,
            "adapter2_raw_output": adapter2_answer,
            "judge": judge_result,
        })
        write_results(args.output_file, args, results)

    print(f"\n=== Aggregate Summary ===")
    summary = aggregate_results(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
