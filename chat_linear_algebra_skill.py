import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SKILL_DIR = Path(__file__).resolve().parent / "skills" / "linear-algebra-solver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat with automatic linear algebra skill injection")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional path to LoRA adapter directory")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--always_use_skill",
        action="store_true",
        help="Inject the linear algebra skill for every user query",
    )
    return parser.parse_args()


def get_quant_config(load_in_4bit: bool):
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


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text.strip()
    return parts[2].strip()


def load_skill_bundle() -> str:
    skill_text = strip_frontmatter(load_text(SKILL_DIR / "SKILL.md"))
    methods_text = load_text(SKILL_DIR / "references" / "methods.md").strip()
    checklist_text = load_text(SKILL_DIR / "references" / "checklist.md").strip()
    return "\n\n".join([skill_text, methods_text, checklist_text]).strip()


def is_linear_algebra_query(question: str) -> bool:
    text = question.lower()
    keywords = [
        "linear algebra",
        "matrix",
        "matrices",
        "vector space",
        "basis",
        "dimension",
        "rank",
        "null space",
        "column space",
        "row space",
        "rref",
        "gaussian elimination",
        "eigenvalue",
        "eigenvector",
        "diagonalizable",
        "orthogonal",
        "projection",
        "gram-schmidt",
        "svd",
        "singular value",
        "quadratic form",
        "positive definite",
        "\u77e9\u9635",
        "\u7ebf\u6027\u4ee3\u6570",
        "\u5411\u91cf\u7a7a\u95f4",
        "\u57fa",
        "\u7ef4\u6570",
        "\u79e9",
        "\u96f6\u7a7a\u95f4",
        "\u5217\u7a7a\u95f4",
        "\u884c\u7a7a\u95f4",
        "\u9ad8\u65af\u6d88\u5143",
        "\u7279\u5f81\u503c",
        "\u7279\u5f81\u5411\u91cf",
        "\u5bf9\u89d2\u5316",
        "\u6b63\u4ea4",
        "\u6295\u5f71",
        "\u5947\u5f02\u503c\u5206\u89e3",
        "\u4e8c\u6b21\u578b",
        "\u6b63\u5b9a",
    ]
    return any(keyword in text for keyword in keywords)


def build_system_prompt(use_skill: bool, skill_bundle: str) -> str:
    base_prompt = (
        "You are a careful mathematical assistant. "
        "Prefer exact reasoning, keep notation consistent, and verify critical algebraic claims before answering."
    )
    if not use_skill:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "Apply the following linear algebra skill when solving the user's question.\n\n"
        f"{skill_bundle}\n"
    )


def build_prompt(question: str, system_prompt: str, use_skill: bool) -> str:
    skill_note = ""
    if use_skill:
        skill_note = (
            "Follow the linear algebra workflow above. "
            "Classify the problem, choose the standard method, show the essential work, include a brief check, then answer.\n"
        )
    return (
        f"System:\n{system_prompt}\n\n"
        f"User:\n{question}\n\n"
        "Assistant instructions:\n"
        f"{skill_note}"
        "If the problem is mathematical, reason carefully step by step. "
        "End with a clearly labeled final answer.\n\n"
        "Assistant:\n"
    )


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    skill_bundle: str,
    always_use_skill: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, bool]:
    use_skill = always_use_skill or is_linear_algebra_query(question)
    prompt = build_prompt(question, build_system_prompt(use_skill, skill_bundle), use_skill)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), use_skill


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = get_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    skill_bundle = load_skill_bundle()
    model, tokenizer = load_model(args)

    print("Linear algebra skill chat started.")
    print("Type a question. Type 'exit' or 'quit' to stop.")
    if args.adapter_path:
        print(f"Loaded adapter: {args.adapter_path}")
    if args.always_use_skill:
        print("Skill injection: always on")
    else:
        print("Skill injection: automatic for linear algebra queries")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exit.")
            break

        answer, used_skill = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            skill_bundle=skill_bundle,
            always_use_skill=args.always_use_skill,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"\n[Skill {'ON' if used_skill else 'OFF'}]")
        print(answer)


if __name__ == "__main__":
    main()
