import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive compare chat: base model vs LoRA finetuned model")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
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


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = get_quant_config(args.load_in_4bit)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    finetuned_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_base, args.adapter_path)
    finetuned_model.eval()

    print("Interactive compare chat started.")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.")

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

        print("\n[Base]")
        print(base_out)
        print("\n[Finetuned]")
        print(ft_out)


if __name__ == "__main__":
    main()
