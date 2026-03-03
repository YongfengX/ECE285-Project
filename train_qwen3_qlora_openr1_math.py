import argparse
import importlib.util
import math
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA finetune Qwen3-4B on OpenR1-Math-220k")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="oieieio/OpenR1-Math-220k")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-qlora-openr1-math")
    parser.add_argument("--max_train_samples", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def _first_nonempty_str(example: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _extract_question(example: Dict[str, Any]) -> Optional[str]:
    return _first_nonempty_str(example, ["problem", "question", "prompt", "instruction", "query", "input"])


def _extract_reasoning(example: Dict[str, Any]) -> Optional[str]:
    return _first_nonempty_str(
        example,
        ["reasoning", "solution", "cot", "rationale", "analysis", "response", "explanation", "output"],
    )


def _extract_answer(example: Dict[str, Any]) -> Optional[str]:
    answer = example.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if isinstance(answer, dict):
        val = answer.get("value")
        if isinstance(val, str) and val.strip():
            return val.strip()

    return _first_nonempty_str(example, ["final_answer", "target", "label", "gold"])


def build_reasoning_dataset(raw_dataset: Dataset) -> Dataset:
    rows: List[Dict[str, str]] = []
    for ex in raw_dataset:
        question = _extract_question(ex)
        reasoning = _extract_reasoning(ex)
        answer = _extract_answer(ex)

        if not question:
            continue
        if not reasoning and not answer:
            continue

        rows.append(
            {
                "question": question,
                "reasoning": reasoning or "",
                "answer": answer or "",
            }
        )

    if not rows:
        raise ValueError("No valid (question, reasoning/answer) samples found in dataset.")
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    eos = tokenizer.eos_token or ""

    def _map_fn(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = (
            f"Question: {example['question']}\n"
            "Please reason step by step, then provide the final answer.\n"
        )

        reasoning = example["reasoning"].strip()
        answer = example["answer"].strip()

        if reasoning and answer:
            target = f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}{eos}"
        elif reasoning:
            target = f"Reasoning:\n{reasoning}{eos}"
        else:
            target = f"Answer:\n{answer}{eos}"

        full_text = prompt + target

        tok_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
        tok_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=True)

        labels = tok_full["input_ids"].copy()
        prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tok_full["labels"] = labels
        return tok_full

    return dataset.map(_map_fn, remove_columns=dataset.column_names, desc="Tokenizing")


def get_4bit_config() -> BitsAndBytesConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_to_resume = None
    if args.resume_from_checkpoint:
        checkpoint_to_resume = args.resume_from_checkpoint
    elif args.auto_resume:
        checkpoint_to_resume = get_last_checkpoint(args.output_dir)
        if checkpoint_to_resume:
            print(f"Auto-resume enabled. Found checkpoint: {checkpoint_to_resume}")

    tb_logging_dir = args.logging_dir or os.path.join(args.output_dir, "runs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=get_4bit_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    if args.max_train_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

    train_dataset = build_reasoning_dataset(raw_dataset)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, args.max_seq_length)

    steps_per_epoch = math.ceil(len(tokenized_train) / args.per_device_train_batch_size)
    update_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    total_train_steps = max(1, math.ceil(update_steps_per_epoch * args.num_train_epochs))
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    enable_tensorboard = args.use_tensorboard
    if enable_tensorboard:
        has_tb = importlib.util.find_spec("tensorboard") is not None
        has_tbx = importlib.util.find_spec("tensorboardX") is not None
        if not (has_tb or has_tbx):
            print("TensorBoard not installed. Disable TensorBoard logging for this run.")
            print("Install with: uv pip install tensorboard")
            enable_tensorboard = False
        else:
            os.environ["TENSORBOARD_LOGGING_DIR"] = tb_logging_dir

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        optim="paged_adamw_8bit",
        report_to="tensorboard" if enable_tensorboard else "none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
        )

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Adapter/model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
