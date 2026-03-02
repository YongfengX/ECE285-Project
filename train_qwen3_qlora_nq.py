import argparse
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
    parser = argparse.ArgumentParser(description="QLoRA finetune Qwen3-0.6B on Natural Questions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_name", type=str, default="natural_questions")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-0.6b-qlora-nq")
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def _extract_question(example: Dict[str, Any]) -> Optional[str]:
    if "question" in example:
        q = example["question"]
        if isinstance(q, str):
            return q.strip()
        if isinstance(q, dict):
            if "text" in q and isinstance(q["text"], str):
                return q["text"].strip()
    if "question_text" in example and isinstance(example["question_text"], str):
        return example["question_text"].strip()
    return None


def _tokens_to_text(example: Dict[str, Any], start: int, end: int) -> Optional[str]:
    if start is None or end is None or end <= start:
        return None
    document = example.get("document")
    if not isinstance(document, dict):
        return None
    tokens = document.get("tokens")
    if not isinstance(tokens, list) or end > len(tokens):
        return None
    words: List[str] = []
    for token_info in tokens[start:end]:
        if not isinstance(token_info, dict):
            continue
        if token_info.get("is_html", False):
            continue
        tok = token_info.get("token")
        if isinstance(tok, str) and tok:
            words.append(tok)
    if not words:
        return None
    return " ".join(words).strip()


def _extract_from_annotation(example: Dict[str, Any], ann: Dict[str, Any]) -> Optional[str]:
    short_answers = ann.get("short_answers")
    if isinstance(short_answers, list) and short_answers:
        first = short_answers[0]
        if isinstance(first, dict):
            if isinstance(first.get("text"), str) and first["text"].strip():
                return first["text"].strip()
            text = _tokens_to_text(example, first.get("start_token"), first.get("end_token"))
            if text:
                return text

    long_answer = ann.get("long_answer")
    if isinstance(long_answer, dict):
        if isinstance(long_answer.get("text"), str) and long_answer["text"].strip():
            return long_answer["text"].strip()
        text = _tokens_to_text(example, long_answer.get("start_token"), long_answer.get("end_token"))
        if text:
            return text

    yes_no = ann.get("yes_no_answer")
    if isinstance(yes_no, str):
        yes_no = yes_no.upper().strip()
        if yes_no in {"YES", "NO"}:
            return yes_no
    return None


def _extract_answer(example: Dict[str, Any]) -> Optional[str]:
    if "answer" in example and isinstance(example["answer"], str) and example["answer"].strip():
        return example["answer"].strip()
    if "answers" in example:
        answers = example["answers"]
        if isinstance(answers, list) and answers:
            first = answers[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                text = first["text"].strip()
                if text:
                    return text
        if isinstance(answers, dict):
            text_list = answers.get("text")
            if isinstance(text_list, list) and text_list:
                if isinstance(text_list[0], str) and text_list[0].strip():
                    return text_list[0].strip()
    if "short_answers_text" in example and isinstance(example["short_answers_text"], list):
        if example["short_answers_text"] and isinstance(example["short_answers_text"][0], str):
            return example["short_answers_text"][0].strip()

    annotations = example.get("annotations")
    if isinstance(annotations, dict):
        return _extract_from_annotation(example, annotations)
    if isinstance(annotations, list):
        for ann in annotations:
            if isinstance(ann, dict):
                ans = _extract_from_annotation(example, ann)
                if ans:
                    return ans
    return None


def build_qa_dataset(raw_dataset: Dataset) -> Dataset:
    qa_rows = []
    for ex in raw_dataset:
        q = _extract_question(ex)
        a = _extract_answer(ex)
        if q and a:
            qa_rows.append({"question": q, "answer": a})
    if not qa_rows:
        raise ValueError("No valid (question, answer) pairs found in Natural Questions dataset.")
    return Dataset.from_list(qa_rows)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    eos = tokenizer.eos_token or ""

    def _map_fn(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = f"Question: {example['question']}\nAnswer:"
        target = f" {example['answer']}{eos}"
        full_text = prompt + target

        tok_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
        tok_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=True)

        labels = tok_full["input_ids"].copy()
        prompt_len = min(len(tok_prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tok_full["labels"] = labels
        return tok_full

    tokenized = dataset.map(
        _map_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized


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

    qa_dataset = build_qa_dataset(raw_dataset)
    tokenized_train = tokenize_dataset(qa_dataset, tokenizer, args.max_seq_length)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Adapter/model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
