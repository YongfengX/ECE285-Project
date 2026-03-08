# ECE285 Project
English | [中文](./README.zh.md)

This repository currently contains three related workflows built around `Qwen/Qwen3-4B`:

- QLoRA fine-tuning on TriviaQA-style question answering under [`nq/`](./nq)
- QLoRA fine-tuning on `oieieio/OpenR1-Math-220k` for reasoning-style math responses
- An interactive chat entrypoint that injects a local linear algebra skill prompt bundle

The previous README referenced scripts that are no longer in the repository root. This version reflects the files that actually exist today.

## Repository Layout
- [`nq/train_qwen3_qlora_nq.py`](./nq/train_qwen3_qlora_nq.py): trainer-based QLoRA training for TriviaQA-format QA
- [`nq/train_qwen3_qlora_manual.py`](./nq/train_qwen3_qlora_manual.py): manual single-GPU training loop
- [`nq/chat_compare_qwen3_qlora.py`](./nq/chat_compare_qwen3_qlora.py): interactive base-vs-adapter QA compare chat
- [`train_qwen3_qlora_openr1_math.py`](./train_qwen3_qlora_openr1_math.py): reasoning-style QLoRA training on OpenR1-Math-220k
- [`chat_openr1_compare.py`](./chat_openr1_compare.py): interactive base-vs-adapter reasoning compare chat
- [`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py): interactive chat with automatic linear algebra skill injection
- [`skills/linear-algebra-solver/`](./skills/linear-algebra-solver): local skill prompt, references, and verification helper
- [`pyproject.toml`](./pyproject.toml): package metadata and console script definitions

## Environment Setup
This project targets Python 3.10+ and uses `uv` for environment management.

```bash
uv venv -p 3.10
```

Activate the environment:

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Core dependencies are defined in [`pyproject.toml`](./pyproject.toml), including `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, and `tensorboard`.

If you want to run the MiniMax judge evaluation script, create a local `.env` file in the repository root:

```dotenv
DASHSCOPE_API_KEY=your_api_key_here
```

The script `eval_compare_with_minimax.py` will load this file automatically.

## Quick Start
Trainer-based QA fine-tuning:

```bash
python nq/train_qwen3_qlora_nq.py
```

Reasoning-style math fine-tuning:

```bash
python train_qwen3_qlora_openr1_math.py
```

Linear algebra skill chat:

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

## Training Workflows
### 1. TriviaQA-style QA with `Trainer`
Default settings:

- model: `Qwen/Qwen3-4B`
- dataset: `mandarjoshi/trivia_qa`
- config: `rc.nocontext`
- output: `./outputs/qwen3-4b-qlora-nq`

Run:

```bash
python nq/train_qwen3_qlora_nq.py
```

Resume:

```bash
python nq/train_qwen3_qlora_nq.py --auto_resume
python nq/train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-nq/checkpoint-200
```

Enable TensorBoard logging:

```bash
python nq/train_qwen3_qlora_nq.py --use_tensorboard
tensorboard --logdir ./outputs/qwen3-4b-qlora-nq/runs --port 6006
```

### 2. TriviaQA-style QA with a manual loop
This variant keeps the training loop explicit and saves optimizer/scheduler state in each checkpoint.

```bash
python nq/train_qwen3_qlora_manual.py --use_tensorboard
```

Resume from a saved checkpoint:

```bash
python nq/train_qwen3_qlora_manual.py --resume_checkpoint ./outputs/qwen3-4b-qlora-manual/checkpoint-200
```

### 3. OpenR1 reasoning fine-tuning
Default settings:

- dataset: `oieieio/OpenR1-Math-220k`
- train split: `default`
- eval split: `extended`
- output: `./outputs/qwen3-4b-qlora-openr1-math`

Run:

```bash
python train_qwen3_qlora_openr1_math.py
```

With early stopping:

```bash
python train_qwen3_qlora_openr1_math.py --use_early_stopping
```

Resume:

```bash
python train_qwen3_qlora_openr1_math.py --auto_resume
python train_qwen3_qlora_openr1_math.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-openr1-math/checkpoint-200
```

## Interactive Chat Tools
### QA compare chat
Compare the base model against a fine-tuned QA adapter:

```bash
python nq/chat_compare_qwen3_qlora.py --adapter_path ./outputs/qwen3-4b-qlora-nq --load_in_4bit
```

### OpenR1 reasoning compare chat
Compare the base model against a reasoning adapter:

```bash
python chat_openr1_compare.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

### Linear algebra skill chat
Use the local skill bundle automatically for linear algebra questions:

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

Force skill injection for every query:

```bash
python chat_linear_algebra_skill.py --load_in_4bit --always_use_skill
```

Load a LoRA adapter on top of the base model:

```bash
python chat_linear_algebra_skill.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

## Data Formatting
### TriviaQA-style QA
The QA scripts convert dataset rows into:

```text
Question: ...
Answer: ...
```

Prompt tokens are masked with `-100`, so loss is applied only to the answer portion.

### OpenR1 reasoning
The reasoning script builds prompts like:

```text
Question: ...
Please reason step by step, then provide the final answer.

Reasoning:
...

Answer:
...
```

If a dataset row has only reasoning or only a final answer, the script still keeps the valid target.

## Notes
- Current training and chat scripts assume access to Hugging Face model and dataset downloads.
- Most workflows are designed for CUDA GPUs; 4-bit loading is supported through `bitsandbytes`.
- `pyproject.toml` still defines some console scripts as top-level modules even though the QA scripts live under `nq/`. The README commands above use direct file paths because they match the repository layout.
