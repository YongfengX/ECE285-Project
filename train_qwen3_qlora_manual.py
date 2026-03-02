import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual QLoRA training loop for Qwen3-4B on TriviaQA.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset_config", type=str, default="rc.nocontext")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-qlora-manual")
    parser.add_argument("--max_train_samples", type=int, default=20000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    return parser.parse_args()


def _extract_question(example: Dict[str, Any]) -> Optional[str]:
    if "question" in example:
        q = example["question"]
        if isinstance(q, str):
            return q.strip()
        if isinstance(q, dict):
            text = q.get("text")
            if isinstance(text, str):
                return text.strip()
    q_text = example.get("question_text")
    if isinstance(q_text, str):
        return q_text.strip()
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
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            span_text = _tokens_to_text(example, first.get("start_token"), first.get("end_token"))
            if span_text:
                return span_text

    long_answer = ann.get("long_answer")
    if isinstance(long_answer, dict):
        text = long_answer.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        span_text = _tokens_to_text(example, long_answer.get("start_token"), long_answer.get("end_token"))
        if span_text:
            return span_text

    yes_no = ann.get("yes_no_answer")
    if isinstance(yes_no, str):
        yes_no = yes_no.upper().strip()
        if yes_no in {"YES", "NO"}:
            return yes_no
    return None


def _extract_answer(example: Dict[str, Any]) -> Optional[str]:
    answer = example.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if isinstance(answer, dict):
        value = answer.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
        aliases = answer.get("aliases")
        if isinstance(aliases, list) and aliases:
            first = aliases[0]
            if isinstance(first, str) and first.strip():
                return first.strip()

    answers = example.get("answers")
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    if isinstance(answers, dict):
        text_list = answers.get("text")
        if isinstance(text_list, list) and text_list:
            first = text_list[0]
            if isinstance(first, str) and first.strip():
                return first.strip()

    annotations = example.get("annotations")
    if isinstance(annotations, dict):
        return _extract_from_annotation(example, annotations)
    if isinstance(annotations, list):
        for ann in annotations:
            if isinstance(ann, dict):
                out = _extract_from_annotation(example, ann)
                if out:
                    return out
    return None


def build_qa_dataset(raw_dataset: Dataset) -> Dataset:
    qa_rows = []
    for ex in raw_dataset:
        q = _extract_question(ex)
        a = _extract_answer(ex)
        if q and a:
            qa_rows.append({"question": q, "answer": a})
    if not qa_rows:
        raise ValueError("No valid (question, answer) pairs found in dataset.")
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

    return dataset.map(_map_fn, remove_columns=dataset.column_names, desc="Tokenizing")


def collate_fn(features: List[Dict[str, List[int]]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    batch_input = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
    batch = tokenizer.pad(batch_input, padding=True, return_tensors="pt")
    max_len = batch["input_ids"].shape[1]

    labels = torch.full((len(features), max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        seq = torch.tensor(f["labels"], dtype=torch.long)
        labels[i, : seq.shape[0]] = seq
    batch["labels"] = labels
    return batch


def get_4bit_config() -> BitsAndBytesConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def save_checkpoint(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: str,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
) -> str:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step, "epoch": epoch, "step_in_epoch": step_in_epoch}, f, indent=2)
    return ckpt_dir


def load_resume_state(resume_dir: str) -> Tuple[int, int, int]:
    state_file = os.path.join(resume_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        return 0, 0, 0
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    return int(state.get("global_step", 0)), int(state.get("epoch", 0)), int(state.get("step_in_epoch", 0))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=get_4bit_config(),
        device_map={"": 0} if torch.cuda.is_available() else None,
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

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    if args.max_train_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))
    qa_dataset = build_qa_dataset(raw_dataset)
    tokenized_train = tokenize_dataset(qa_dataset, tokenizer, args.max_seq_length)

    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = (len(train_loader) * args.num_train_epochs + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps
    )

    start_global_step = 0
    start_epoch = 0
    start_step_in_epoch = 0

    if args.resume_checkpoint:
        model = PeftModel.from_pretrained(model, args.resume_checkpoint, is_trainable=True)
        opt_path = os.path.join(args.resume_checkpoint, "optimizer.pt")
        sch_path = os.path.join(args.resume_checkpoint, "scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        if os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        start_global_step, start_epoch, start_step_in_epoch = load_resume_state(args.resume_checkpoint)
        print(f"Resumed from {args.resume_checkpoint} (global_step={start_global_step}).")

    model.train()
    global_step = start_global_step
    device = next(model.parameters()).device

    for epoch in range(start_epoch, args.num_train_epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step_in_epoch:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"epoch={epoch+1} step={global_step} loss={avg_loss:.4f} lr={lr:.6e}")
                    running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=args.output_dir,
                        global_step=global_step,
                        epoch=epoch,
                        step_in_epoch=step + 1,
                    )
                    print(f"Saved checkpoint to: {ckpt}")

        ckpt = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=args.output_dir,
            global_step=global_step,
            epoch=epoch + 1,
            step_in_epoch=0,
        )
        print(f"Epoch {epoch+1} finished. Saved checkpoint to: {ckpt}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Final adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
