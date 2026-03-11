import argparse
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenR1 compare chat: base vs finetuned (reasoning style)")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, default="./outputs/qwen3-4b-qlora-openr1-math/checkpoint-5400")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--log_dir", type=str, default="output/chat_logs")
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


def build_reasoning_prompt(question: str) -> str:
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_reasoning_prompt(question)
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
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def write_session_log(log_dir: str, answered_questions: list[dict[str, str]]) -> Path | None:
    if not answered_questions:
        return None

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / f"chatbot_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    lines = [
        f"Session ended at: {datetime.now().isoformat(timespec='seconds')}",
        f"Answered questions: {len(answered_questions)}",
        "",
    ]
    for index, record in enumerate(answered_questions, start=1):
        lines.append(f"{index}. [{record['answered_at']}] {record['question']}")

    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_path


def main() -> None:
    args = parse_args()
    answered_questions: list[dict[str, str]] = []

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
    base_model.eval()

    finetuned_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_base, args.adapter_path)
    finetuned_model.eval()

    print("OpenR1 interactive compare chat started.")
    print("Type your math question. Type 'exit' or 'quit' to stop.")

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

        base_out = generate_answer(
            base_model, tokenizer, question, args.max_new_tokens, args.temperature, args.top_p
        )
        ft_out = generate_answer(
            finetuned_model, tokenizer, question, args.max_new_tokens, args.temperature, args.top_p
        )
        answered_questions.append(
            {
                "question": question,
                "answered_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        print("\n[Base]")
        print(base_out)
        print("\n[Finetuned OpenR1]")
        print(ft_out)

    log_file = write_session_log(args.log_dir, answered_questions)
    if log_file is None:
        print("No answered questions to log.")
        return

    print("\nAnswered question log:")
    for index, record in enumerate(answered_questions, start=1):
        print(f"{index}. [{record['answered_at']}] {record['question']}")
    print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
