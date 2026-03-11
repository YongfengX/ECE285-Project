import argparse
import concurrent.futures
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
        description="Compare three models (base + adapter1 + adapter2) using MiniMax as an LLM judge."
    )
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400/")
    parser.add_argument(
        "--adapter2_path",
        type=str,
        default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400/",
        help="Path to the second LoRA adapter model.",
    )
    parser.add_argument("--judge_model_name", type=str, default="qwen3.5-plus")
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="Judge API key. Defaults to DASHSCOPE_API_KEY env var.",
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


def _load_model_bundle(
    model_name: str,
    quant_config: Optional[BitsAndBytesConfig],
    model_dtype: torch.dtype,
    tokenizer: AutoTokenizer,
) -> ModelBundle:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer)


def _load_adapter_bundle(
    base_model_name: str,
    adapter_path: str,
    quant_config: Optional[BitsAndBytesConfig],
    model_dtype: torch.dtype,
    tokenizer: AutoTokenizer,
) -> ModelBundle:
    adapter_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path)
    adapter_model.eval()
    return ModelBundle(model=adapter_model, tokenizer=tokenizer)


def load_base_and_adapters(args: argparse.Namespace) -> Tuple[ModelBundle, ModelBundle, ModelBundle]:
    tokenizer = load_tokenizer(args.base_model_name)
    quant_config = get_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model_dtype = dtype if torch.cuda.is_available() else torch.float32

    base_bundle = _load_model_bundle(args.base_model_name, quant_config, model_dtype, tokenizer)
    adapter1_bundle = _load_adapter_bundle(args.base_model_name, args.adapter_path, quant_config, model_dtype, tokenizer)
    adapter2_bundle = _load_adapter_bundle(
        args.base_model_name,
        args.adapter2_path,
        quant_config,
        model_dtype,
        tokenizer,
    )
    return base_bundle, adapter1_bundle, adapter2_bundle


def load_base_and_adapter(args: argparse.Namespace) -> Tuple[ModelBundle, ModelBundle]:
    # Backward-compatible helper used by older scripts.
    base_bundle, adapter1_bundle, _ = load_base_and_adapters(args)
    return base_bundle, adapter1_bundle


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
    raw_text = bundle.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return canonicalize_response(raw_text)


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


def split_response_sections(text: str) -> Tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "", ""

    marker_patterns = [
        r"\*\*Final Answer\*\*\s*:\s*",
        r"\*\*Answer\*\*\s*:\s*",
        r"\bFinal Answer\s*:\s*",
        r"\bAnswer\s*:\s*",
    ]
    earliest_match = None
    for pattern in marker_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match and (earliest_match is None or match.start() < earliest_match.start()):
            earliest_match = match

    if earliest_match is None:
        return "", extract_boxed_answer(stripped)

    reasoning = stripped[: earliest_match.start()].strip()
    final_segment = stripped[earliest_match.end() :].strip()
    final_answer = extract_boxed_answer(final_segment)
    return reasoning, final_answer


def canonicalize_response(text: str) -> str:
    reasoning, final_answer = split_response_sections(text)
    if reasoning and final_answer:
        return f"Reasoning:\n{reasoning}\n\nAnswer:\n{final_answer}"
    if final_answer:
        return final_answer
    return text.strip()


def extract_final_answer_from_response(text: str) -> str:
    _, final_answer = split_response_sections(text)
    return final_answer


def extract_reasoning_from_response(text: str) -> str:
    reasoning, _ = split_response_sections(text)
    reasoning = re.sub(r"^\s*Reasoning\s*:\s*", "", reasoning, flags=re.IGNORECASE).strip()
    return reasoning


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
            "level": "",
        }

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
        question = " ".join(part for part in [body, question_text] if part).strip()
        return {
            "question": question,
            "reference_reasoning": "",
            "reference_answer": normalize_final_answer(str(row.get("Answer", "")).strip()),
            "level": "",
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
            "level": "",
        }

    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def parse_level_filter(levels: str) -> List[str]:
    return [level.strip() for level in levels.split(",") if level.strip()]


def stratified_sample_by_level(
    rows: List[Dict[str, str]],
    samples_per_level: int,
    seed: int,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("level", "") or "[unknown]"].append(row)

    rng = random.Random(seed)
    sampled_rows: List[Dict[str, str]] = []
    ordered_levels = sorted(grouped.keys())
    for level in ordered_levels:
        bucket = grouped[level]
        if len(bucket) <= samples_per_level:
            sampled_rows.extend(bucket)
            continue
        indices = list(range(len(bucket)))
        rng.shuffle(indices)
        sampled_rows.extend(bucket[i] for i in sorted(indices[:samples_per_level]))
    return sampled_rows


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

    dataset_format = args.dataset_format
    if dataset_format == "auto":
        dataset_format = detect_dataset_format(args.dataset_name, dataset.column_names)

    normalized_rows = []
    allowed_levels = set(parse_level_filter(args.levels))
    for idx, row in enumerate(dataset):
        sample = normalize_eval_row(row, dataset_format)
        if not sample["question"] or not sample["reference_answer"]:
            print(f"[skip] sample {idx}: missing normalized question or reference answer")
            continue
        if allowed_levels and sample.get("level", "") not in allowed_levels:
            continue
        normalized_rows.append(sample)

    if not normalized_rows:
        raise ValueError("No valid evaluation samples found after dataset normalization.")

    if args.samples_per_level > 0:
        normalized_rows = stratified_sample_by_level(normalized_rows, args.samples_per_level, args.seed)
    elif args.max_samples > 0:
        normalized_rows = normalized_rows[: min(args.max_samples, len(normalized_rows))]

    if args.max_samples > 0 and args.samples_per_level > 0:
        normalized_rows = normalized_rows[: min(args.max_samples, len(normalized_rows))]

    if args.samples_per_level > 0:
        level_counts: Dict[str, int] = defaultdict(int)
        for row in normalized_rows:
            level_counts[row.get("level", "") or "[unknown]"] += 1
        counts_text = ", ".join(f"{level}={count}" for level, count in sorted(level_counts.items()))
        print(f"[info] stratified sample by level: {counts_text}")

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
    answer_c: str,
) -> List[Dict[str, str]]:
    answer_a_final = extract_final_answer_from_response(answer_a)
    answer_b_final = extract_final_answer_from_response(answer_b)
    answer_c_final = extract_final_answer_from_response(answer_c)
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
        '  "model_c": {"correctness": 0-10, "reasoning": 0-10, "completeness": 0-10, "clarity": 0-10, "instruction_following": 0-10, "strengths": "...", "weaknesses": "..."},\n'
        '  "winner": "A" | "B" | "C" | "tie",\n'
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
        f"\nModel C extracted final answer:\n{answer_c_final}\n\n"
        f"Model C response:\n{answer_c}\n"
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
        raw = judge_once(
            client=client,
            judge_model_name=args.judge_model_name,
            messages=build_judge_messages(
                question=question,
                reference_reasoning=reference_reasoning,
                reference_answer=reference_answer,
                answer_a=answers_by_model[model_a],
                answer_b=answers_by_model[model_b],
                answer_c=answers_by_model[model_c],
            ),
            judge_retries=args.judge_retries,
        )
        score_blocks[model_a].append(coerce_score_block(raw["model_a"]))
        score_blocks[model_b].append(coerce_score_block(raw["model_b"]))
        score_blocks[model_c].append(coerce_score_block(raw["model_c"]))
        round_raw.append(raw)
        round_summaries.append(str(raw.get("summary", "")).strip())

    base_scores = _average_numeric_scores(score_blocks["base"])
    adapter1_scores = _average_numeric_scores(score_blocks["adapter1"])
    adapter2_scores = _average_numeric_scores(score_blocks["adapter2"])

    totals = {
        "base": base_scores["total"],
        "adapter1": adapter1_scores["total"],
        "adapter2": adapter2_scores["total"],
    }
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or (ranked[0][1] - ranked[1][1]) <= 0.15:
        winner = "tie"
    else:
        winner = ranked[0][0]

    return {
        "base_scores": base_scores,
        "adapter1_scores": adapter1_scores,
        "adapter2_scores": adapter2_scores,
        "base_reasoning": extract_reasoning_from_response(base_answer),
        "base_final_answer": extract_final_answer_from_response(base_answer),
        "adapter1_reasoning": extract_reasoning_from_response(adapter1_answer),
        "adapter1_final_answer": extract_final_answer_from_response(adapter1_answer),
        "adapter2_reasoning": extract_reasoning_from_response(adapter2_answer),
        "adapter2_final_answer": extract_final_answer_from_response(adapter2_answer),
        "reference_final_answer": reference_answer,
        "winner": winner,
        "round_summaries": round_summaries,
        "round_raw": round_raw,
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_results(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(DIMENSION_WEIGHTS.keys()) + ["total"]
    base = {key: mean([sample["judge"]["base_scores"][key] for sample in samples]) for key in keys}
    adapter1 = {key: mean([sample["judge"]["adapter1_scores"][key] for sample in samples]) for key in keys}
    adapter2 = {key: mean([sample["judge"]["adapter2_scores"][key] for sample in samples]) for key in keys}

    win_counts = {"base": 0, "adapter1": 0, "adapter2": 0, "tie": 0}
    for sample in samples:
        winner = sample["judge"].get("winner", "tie")
        if winner not in win_counts:
            winner = "tie"
        win_counts[winner] += 1

    overall_average = {
        key: round((base[key] + adapter1[key] + adapter2[key]) / 3.0, 4)
        for key in keys
    }
    delta_adapter1_vs_base = {key: round(adapter1[key] - base[key], 4) for key in keys}
    delta_adapter2_vs_base = {key: round(adapter2[key] - base[key], 4) for key in keys}
    return {
        "base_average": base,
        "adapter1_average": adapter1,
        "adapter2_average": adapter2,
        "overall_average": overall_average,
        "delta_adapter1_vs_base": delta_adapter1_vs_base,
        "delta_adapter2_vs_base": delta_adapter2_vs_base,
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
            "adapter2_path": args.adapter2_path,
            "judge_model_name": args.judge_model_name,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "dataset_format": args.dataset_format,
            "max_samples": args.max_samples,
            "levels": parse_level_filter(args.levels),
            "samples_per_level": args.samples_per_level,
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
    sample_payload = dict(sample_payload)
    if "adapter1_answer" not in sample_payload:
        sample_payload["adapter1_answer"] = sample_payload.get("finetuned_answer", sample_payload["base_answer"])
    sample_payload.setdefault("base_reasoning", extract_reasoning_from_response(sample_payload["base_answer"]))
    sample_payload.setdefault("base_final_answer", extract_final_answer_from_response(sample_payload["base_answer"]))
    sample_payload.setdefault("adapter1_reasoning", extract_reasoning_from_response(sample_payload["adapter1_answer"]))
    sample_payload.setdefault(
        "adapter1_final_answer",
        extract_final_answer_from_response(sample_payload["adapter1_answer"]),
    )
    sample_payload.setdefault("adapter2_answer", sample_payload.get("adapter1_answer", sample_payload["base_answer"]))
    sample_payload.setdefault("adapter2_reasoning", extract_reasoning_from_response(sample_payload["adapter2_answer"]))
    sample_payload.setdefault(
        "adapter2_final_answer",
        extract_final_answer_from_response(sample_payload["adapter2_answer"]),
    )
    client = make_judge_client(args.judge_api_key)
    judge_result = judge_triplet(
        client=client,
        args=args,
        question=sample_payload["question"],
        reference_reasoning=sample_payload["reference_reasoning"],
        reference_answer=sample_payload["reference_answer"],
        base_answer=sample_payload["base_answer"],
        adapter1_answer=sample_payload["adapter1_answer"],
        adapter2_answer=sample_payload["adapter2_answer"],
    )
    return {
        **sample_payload,
        "judge": judge_result,
    }


def main() -> None:
    args = parse_args()
    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Pass --judge_api_key or set DASHSCOPE_API_KEY.")
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    log_file = resolve_log_file(args)
    answer_cache_file = resolve_answer_cache_file(log_file)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(answer_cache_file) or ".", exist_ok=True)

    print("Loading models...")
    base_bundle, adapter1_bundle, adapter2_bundle = load_base_and_adapters(args)

    print("Loading evaluation dataset...")
    dataset = build_eval_dataset(args)
    results = load_existing_results(args.output_file, log_file) if args.resume else []
    cached_answers = load_answer_cache(answer_cache_file) if args.resume else {}
    results_by_index: Dict[int, Dict[str, Any]] = {int(sample["index"]): sample for sample in results}
    completed_indices = {
        idx
        for idx, sample in results_by_index.items()
        if isinstance(sample.get("judge"), dict) and "adapter2_scores" in sample["judge"]
    }
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
                f"adapter1={judge_result['adapter1_scores']['total']:.2f} "
                f"adapter2={judge_result['adapter2_scores']['total']:.2f} "
                f"winner={judge_result['winner']}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        for idx, row in enumerate(dataset):
            if idx in completed_indices:
                print(f"[{idx + 1}/{total_samples}] Skipping completed sample from resume log.")
                continue

            while len(pending_futures) >= max(1, args.threads):
                flush_completed(block=True)

            if idx in cached_answers and all(
                key in cached_answers[idx]
                for key in ["base_answer", "adapter1_answer", "adapter2_answer"]
            ):
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

                print(f"[{idx + 1}/{total_samples}] Generating adapter1 answer...")
                adapter1_answer = generate_answer(
                    bundle=adapter1_bundle,
                    question=question,
                    prompt_style=args.prompt_style,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                print(f"[{idx + 1}/{total_samples}] Generating adapter2 answer...")
                adapter2_answer = generate_answer(
                    bundle=adapter2_bundle,
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
                    "base_reasoning": extract_reasoning_from_response(base_answer),
                    "base_final_answer": extract_final_answer_from_response(base_answer),
                    "adapter1_answer": adapter1_answer,
                    "adapter1_reasoning": extract_reasoning_from_response(adapter1_answer),
                    "adapter1_final_answer": extract_final_answer_from_response(adapter1_answer),
                    "adapter2_answer": adapter2_answer,
                    "adapter2_reasoning": extract_reasoning_from_response(adapter2_answer),
                    "adapter2_final_answer": extract_final_answer_from_response(adapter2_answer),
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
