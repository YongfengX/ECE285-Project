# ECE285 Project
English | [中文](./README.zh.md)

This repository is a collection of local experiments around `Qwen/Qwen3-4B`, focused on QLoRA fine-tuning, adapter comparison, and prompt-based skill injection.

The documentation below reflects the files that are actually present in the project directory today.

## What Is In This Repo

- TriviaQA-style QLoRA fine-tuning under [`nq/`](./nq)
- OpenR1 math/reasoning QLoRA fine-tuning in the repository root
- Interactive comparison chats for base vs LoRA-adapted models
- Dataset-driven evaluation scripts that use an external LLM judge
- A local linear algebra skill bundle under [`skills/linear-algebra-solver`](./skills/linear-algebra-solver)

## Current Notes

- Run the scripts with `python <path-to-script>.py`.
- [`pyproject.toml`](./pyproject.toml) still contains old console-script/module mappings, so the packaged entry points do not match the current file layout.
- The repository already contains one checked-in adapter checkpoint at `output/qwen3-4b-qlora-openr1-math/checkpoint-2400`.
- Most training scripts save new outputs to `./outputs/...`, so `output/` and `outputs/` are both relevant in the current tree.

## Repository Layout

- [`nq/train_qwen3_qlora_nq.py`](./nq/train_qwen3_qlora_nq.py): QLoRA fine-tuning with `Trainer` on `mandarjoshi/trivia_qa`
- [`nq/train_qwen3_qlora_manual.py`](./nq/train_qwen3_qlora_manual.py): manual single-GPU QLoRA training loop
- [`nq/chat_compare_qwen3_qlora.py`](./nq/chat_compare_qwen3_qlora.py): interactive QA-style base vs adapter comparison
- [`train_qwen3_qlora_openr1_math.py`](./train_qwen3_qlora_openr1_math.py): reasoning-style QLoRA fine-tuning on `oieieio/OpenR1-Math-220k`
- [`chatbot.py`](./chatbot.py): interactive reasoning compare chat for base vs OpenR1 adapter
- [`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py): chat entrypoint with automatic linear algebra skill injection
- [`eval_compare_with_minimax.py`](./eval_compare_with_minimax.py): compare base model plus two adapters with MiniMax as the judge
- [`chat_openr1_dataset_eval.py`](./chat_openr1_dataset_eval.py): dataset-driven three-way comparison with an external judge
- [`level5compare.py`](./level5compare.py): specialized `competition_math` Level 5 comparison script
- [`skills/linear-algebra-solver`](./skills/linear-algebra-solver): prompt bundle, references, and a verification helper
- [`output/`](./output): checked-in experiment artifact directory

## Environment Setup

Python `3.10+` is expected.

Create a virtual environment:

```bash
uv venv -p 3.10
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the core dependencies used by the scripts:

```bash
uv pip install torch transformers datasets accelerate peft bitsandbytes sentencepiece tensorboard openai python-dotenv
```

For judge-based evaluation scripts, set the API key shown in [`.env.example`](./.env.example):

```powershell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

## Main Workflows

### 1. QA Fine-Tuning On TriviaQA

Trainer-based workflow:

```bash
python nq/train_qwen3_qlora_nq.py
```

Manual training loop:

```bash
python nq/train_qwen3_qlora_manual.py --use_tensorboard
```

Defaults used by both QA trainers:

- base model: `Qwen/Qwen3-4B`
- dataset: `mandarjoshi/trivia_qa`
- config: `rc.nocontext`
- task format: `Question: ...` followed by `Answer: ...`

### 2. OpenR1 Math / Reasoning Fine-Tuning

```bash
python train_qwen3_qlora_openr1_math.py
```

Useful variants:

```bash
python train_qwen3_qlora_openr1_math.py --use_early_stopping
python train_qwen3_qlora_openr1_math.py --auto_resume
```

Default settings:

- base model: `Qwen/Qwen3-4B`
- dataset: `oieieio/OpenR1-Math-220k`
- train split: `default`
- eval split: `extended`
- output dir: `./outputs/qwen3-4b-qlora-openr1-math`

### 3. Interactive Chats

QA adapter vs base model:

```bash
python nq/chat_compare_qwen3_qlora.py --adapter_path .\outputs\qwen3-4b-qlora-nq\checkpoint-200 --load_in_4bit
```

Reasoning adapter vs base model using the checked-in checkpoint:

```bash
python chatbot.py --adapter_path .\output\qwen3-4b-qlora-openr1-math\checkpoint-2400 --load_in_4bit
```

Linear algebra skill chat:

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

Always force the skill:

```bash
python chat_linear_algebra_skill.py --load_in_4bit --always_use_skill
```

### 4. Dataset-Driven Evaluation

MiniMax-based comparison:

```bash
python eval_compare_with_minimax.py --adapter_path .\output\qwen3-4b-qlora-openr1-math\checkpoint-2400 --judge_api_key YOUR_API_KEY --load_in_4bit
```

What this script does:

- loads the base model
- loads `adapter_path`
- loads `adapter2_path`
- normalizes supported Hugging Face datasets into a common question/reference format
- asks an external judge model to score the answers

Supported dataset formats in the script:

- `gsm8k`
- `competition_math`
- `svamp`
- `commonsense_qa`
- `arc`

There are also two local evaluation variants:

- [`chat_openr1_dataset_eval.py`](./chat_openr1_dataset_eval.py): general three-way dataset evaluation
- [`level5compare.py`](./level5compare.py): Level 5 focused evaluation for `qwedsacf/competition_math`

These two scripts are more experimental. Review their defaults before running them.

## Linear Algebra Skill Bundle

[`skills/linear-algebra-solver`](./skills/linear-algebra-solver) contains:

- `SKILL.md`: the main linear algebra reasoning workflow
- `references/methods.md`: method notes by problem type
- `references/checklist.md`: common failure checks
- `scripts/verify_linear_algebra.py`: symbolic verification helper

[`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py) loads this bundle automatically when the query looks like linear algebra, or for every query with `--always_use_skill`.

## Existing Artifact In The Repo

The repository currently includes this checkpoint:

- [`output/qwen3-4b-qlora-openr1-math/checkpoint-2400`](./output/qwen3-4b-qlora-openr1-math/checkpoint-2400)

That checkpoint includes adapter weights, tokenizer files, optimizer state, scheduler state, and trainer metadata.

## Known Inconsistencies

- `pyproject.toml` still describes the older root-level module layout.
- The project contains both `output/` and script defaults that point to `outputs/`.
- Some evaluation scripts are experiment-specific and contain stronger assumptions than the main training scripts.

If you want the packaging commands and entry points to work cleanly, update [`pyproject.toml`](./pyproject.toml) next.
