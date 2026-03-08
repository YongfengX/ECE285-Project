import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


DIMENSION_WEIGHTS = {
    "correctness": 0.40,
    "reasoning": 0.20,
    "completeness": 0.15,
    "clarity": 0.15,
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
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to your LoRA adapter directory")
    parser.add_argument("--judge_model_name", type=str, default="MiniMax-M2.5")
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="./outputs/eval_compare_with_minimax_gsm8k.json")
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


def normalize_final_answer(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return text


def build_eval_dataset(args: argparse.Namespace) -> Dataset:
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    return dataset


def make_judge_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY is not set.")
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


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
    rubric_text = (
        "You are an impartial evaluator.\n"
        "Score each model on a 0-10 scale for these dimensions:\n"
        "- correctness: factual and mathematical correctness of the final answer\n"
        "- reasoning: soundness and usefulness of the reasoning process\n"
        "- completeness: whether the response fully addresses the problem\n"
        "- clarity: readability, structure, and clarity\n"
        "- instruction_following: whether the response follows the requested format and task\n\n"
        "Weight the dimensions as:\n"
        "- correctness: 0.40\n"
        "- reasoning: 0.20\n"
        "- completeness: 0.15\n"
        "- clarity: 0.15\n"
        "- instruction_following: 0.10\n\n"
        "Use the reference answer for correctness, but do not blindly reward longer answers.\n"
        "Be strict about incorrect final answers.\n"
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
        f"Model A response:\n{answer_a}\n\n"
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print("Loading models...")
    base_bundle, finetuned_bundle = load_base_and_adapter(args)

    print("Loading evaluation dataset...")
    dataset = build_eval_dataset(args)
    client = make_judge_client()

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        question = str(row.get("question", "")).strip()
        if not question:
            print(f"[skip] sample {idx}: missing question field")
            continue

        raw_reference = str(row.get("answer", "")).strip()
        reference_reasoning, reference_answer = extract_gsm8k_reference(raw_reference)

        print(f"[{idx + 1}/{len(dataset)}] Generating base answer...")
        base_answer = generate_answer(
            bundle=base_bundle,
            question=question,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"[{idx + 1}/{len(dataset)}] Generating finetuned answer...")
        finetuned_answer = generate_answer(
            bundle=finetuned_bundle,
            question=question,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"[{idx + 1}/{len(dataset)}] Judging with {args.judge_model_name}...")
        judge_result = judge_pair(
            client=client,
            args=args,
            question=question,
            reference_reasoning=reference_reasoning,
            reference_answer=reference_answer,
            base_answer=base_answer,
            finetuned_answer=finetuned_answer,
        )

        sample_result = {
            "index": idx,
            "question": question,
            "reference_reasoning": reference_reasoning,
            "reference_answer": reference_answer,
            "base_answer": base_answer,
            "finetuned_answer": finetuned_answer,
            "judge": judge_result,
        }
        results.append(sample_result)

        print(
            f"[{idx + 1}/{len(dataset)}] "
            f"base={judge_result['base_scores']['total']:.2f} "
            f"finetuned={judge_result['finetuned_scores']['total']:.2f} "
            f"winner={judge_result['winner']}"
        )

    summary = aggregate_results(results)
    output = {
        "config": {
            "base_model_name": args.base_model_name,
            "adapter_path": args.adapter_path,
            "judge_model_name": args.judge_model_name,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "max_samples": args.max_samples,
            "prompt_style": args.prompt_style,
            "weights": DIMENSION_WEIGHTS,
        },
        "summary": summary,
        "samples": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Aggregate Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
