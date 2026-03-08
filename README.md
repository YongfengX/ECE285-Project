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

MiniMax judge evaluation:

```bash
python eval_compare_with_minimax.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --judge_api_key YOUR_API_KEY --load_in_4bit
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

## MiniMax Judge Evaluation
Use [`eval_compare_with_minimax.py`](./eval_compare_with_minimax.py) to compare the base model against a LoRA adapter and let MiniMax score both answers.

Default behavior:

- base model: `Qwen/Qwen3-4B`
- judge model: `qwen3.5-plus`
- dataset: `qwedsacf/competition_math`
- config: `default`
- split: `train`
- prompt style: `reasoning`
- judge weighting: strongly prioritizes the final answer

Minimal example:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name qwedsacf/competition_math ^
  --dataset_config default ^
  --dataset_split train ^
  --dataset_format competition_math ^
  --prompt_style reasoning ^
  --max_samples 30 ^
  --load_in_4bit ^
  --threads 4 ^
  --resume
```

Recommended QA-style example:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-nq ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name tau/commonsense_qa ^
  --dataset_config default ^
  --dataset_split validation ^
  --dataset_format commonsense_qa ^
  --prompt_style qa ^
  --max_samples 50 ^
  --load_in_4bit ^
  --output_file ./outputs/eval_compare_with_minimax_commonsense_qa.json
```

Key arguments:

- `--adapter_path`: required; path to the LoRA adapter being evaluated
- `--judge_api_key`: required; API key for the DashScope-compatible MiniMax endpoint
- `--dataset_name`, `--dataset_config`, `--dataset_split`: Hugging Face dataset selection
- `--dataset_format`: one of `auto`, `gsm8k`, `competition_math`, `svamp`, `commonsense_qa`, `arc`
- `--prompt_style`: `reasoning` for math/reasoning adapters, `qa` for direct-answer QA adapters
- `--max_samples`: cap evaluation size to control runtime and API cost
- `--levels`: comma-separated `competition_math` levels to keep, for example `Level 1,Level 3,Level 5`
- `--samples_per_level`: sample up to this many examples per level after optional level filtering
- `--threads`: number of concurrent judge workers; generation still runs sequentially on the local model
- `--resume`: resume from existing output and log files, skipping finished samples
- `--output_file`: JSON file containing per-sample judge scores and the aggregate summary
- `--log_file`: optional JSONL path for incremental per-sample judge results; defaults to `<output_file>.jsonl`

The script normalizes supported datasets into a common schema before generation and judging. `--dataset_format auto` works for the built-in formats below.
For `qwedsacf/competition_math`, the dataset only provides a `train` split; if you request a missing split, the script automatically falls back to the only available split.
For `competition_math`, the script can also stratify by `level`, which is useful for building a balanced evaluation set across `Level 1` to `Level 5`.
The judge prompt is configured to heavily weight the extracted final answer over reasoning quality.
During long runs, the script writes:

- `<output_file>`: full aggregate snapshot after each completed sample
- `<output_file>.jsonl`: append-only incremental judge log
- `<output_file>.answers.jsonl`: cached generated base/finetuned answers for resume support

### Additional Evaluation Datasets
Besides `gsm8k`, the evaluation script now supports these datasets directly:

- `qwedsacf/competition_math`
- `ChilleD/SVAMP`
- `tau/commonsense_qa`
- `allenai/ai2_arc`

Examples:

GSM8K:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name gsm8k ^
  --dataset_config main ^
  --dataset_split test ^
  --dataset_format gsm8k ^
  --prompt_style reasoning ^
  --max_samples 30 ^
  --load_in_4bit
```

Competition Math:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name qwedsacf/competition_math ^
  --dataset_config default ^
  --dataset_split train ^
  --dataset_format competition_math ^
  --prompt_style reasoning ^
  --max_samples 30 ^
  --load_in_4bit ^
  --threads 4 ^
  --resume ^
  --output_file ./outputs/eval_compare_with_minimax_competition_math.json
```

Balanced level-based sample:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name qwedsacf/competition_math ^
  --dataset_config default ^
  --dataset_split train ^
  --dataset_format competition_math ^
  --levels "Level 1,Level 2,Level 3,Level 4,Level 5" ^
  --samples_per_level 6 ^
  --max_samples 30 ^
  --prompt_style reasoning ^
  --load_in_4bit ^
  --threads 4 ^
  --resume ^
  --output_file ./outputs/eval_compare_with_minimax_competition_math_balanced.json
```

SVAMP:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name ChilleD/SVAMP ^
  --dataset_config default ^
  --dataset_split test ^
  --dataset_format svamp ^
  --prompt_style reasoning ^
  --max_samples 30 ^
  --load_in_4bit ^
  --output_file ./outputs/eval_compare_with_minimax_svamp.json
```

CommonsenseQA:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-nq ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name tau/commonsense_qa ^
  --dataset_config default ^
  --dataset_split validation ^
  --dataset_format commonsense_qa ^
  --prompt_style qa ^
  --max_samples 50 ^
  --load_in_4bit ^
  --output_file ./outputs/eval_compare_with_minimax_commonsense_qa.json
```

AI2 ARC:

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-nq ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name allenai/ai2_arc ^
  --dataset_config ARC-Challenge ^
  --dataset_split validation ^
  --dataset_format arc ^
  --prompt_style qa ^
  --max_samples 50 ^
  --load_in_4bit ^
  --output_file ./outputs/eval_compare_with_minimax_arc_challenge.json
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
- `eval_compare_with_minimax.py` requires `--judge_api_key` because judge scoring is done through the DashScope-compatible MiniMax endpoint.
- `eval_compare_with_minimax.py` now checkpoints progress continuously, so interrupted runs can resume without regenerating finished samples.
- `pyproject.toml` still defines some console scripts as top-level modules even though the QA scripts live under `nq/`. The README commands above use direct file paths because they match the repository layout.
