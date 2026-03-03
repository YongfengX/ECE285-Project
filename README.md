# ECE285-Project
English | [Chinese](./README.zh.md)

QLoRA fine-tuning pipeline for `Qwen/Qwen3-4B` on `mandarjoshi/trivia_qa` (`rc.nocontext`), with:
- trainer-based training
- manual single-GPU loop training
- base vs finetuned evaluation
- interactive side-by-side chat compare
- optional TensorBoard monitoring

## Project Structure
- `train_qwen3_qlora_nq.py`: Trainer-based QLoRA training
- `train_qwen3_qlora_manual.py`: manual `for epoch / for step` training loop
- `train_qwen3_qlora_openr1_math.py`: OpenR1-Math-220k reasoning-style QLoRA training
- `eval_compare_qwen3_qlora.py`: compare base and finetuned outputs
- `chat_compare_qwen3_qlora.py`: interactive terminal compare
- `chat_openr1_compare.py`: interactive OpenR1 reasoning compare chat
- `pyproject.toml`: dependency and script management

## Quick Start
```bash
uv venv
# Activate your .venv first (see UV Quickstart section)
uv pip install "-e ."
python train_qwen3_qlora_nq.py
```

Default dataset:
```bash
--dataset_name mandarjoshi/trivia_qa --dataset_config rc.nocontext
```

## UV Quickstart
### Install uv
Recommended (with pipx):
```bash
pip install pipx
pipx ensurepath
pipx install uv
```

Directly with pip:
```bash
pip install uv
```

Using brew (macOS/Linux):
```bash
brew install uv
```

### Create virtual environment
```bash
uv venv
```
or specify Python version:
```bash
uv venv -p 3.10
```

### Activate virtual environment
macOS/Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```bash
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

Exit environment:
```bash
deactivate
```

Delete virtual environment:
```bash
rm -rf .venv
```

### Install packages with uv pip
From requirements file:
```bash
uv pip install -r requirements.txt
```

Install individual packages:
```bash
uv pip install requests beautifulsoup4
```

With extras:
```bash
uv pip install "celery[redis]"
```

From Git repository:
```bash
uv pip install "git+https://github.com/example/my-lib.git#egg=my-lib"
```

Local editable install:
```bash
uv pip install "-e ."
```

### Run project
After `.venv` is activated, use:
```bash
python app.py
```
Key point: use `python` in the activated environment. Do not use `uv python`.

## Training
Trainer-based:
```bash
python train_qwen3_qlora_nq.py
```

OpenR1-Math-220k (reasoning data):
```bash
python train_qwen3_qlora_openr1_math.py
```
This training target uses `Reasoning:` + `Answer:` format, so outputs are more likely to include reasoning steps.

Manual loop (single GPU):
```bash
python train_qwen3_qlora_manual.py
```

Resume from checkpoint:
```bash
python train_qwen3_qlora_nq.py --auto_resume
python train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-nq/checkpoint-200
```

## TensorBoard
Enable logging during training:
```bash
python train_qwen3_qlora_nq.py --use_tensorboard
```

Launch TensorBoard:
```bash
tensorboard --logdir ./outputs/qwen3-4b-qlora-nq/runs --port 6006
```

Manual-loop TensorBoard logdir:
```bash
./outputs/qwen3-4b-qlora-manual/runs
```

## Evaluation (Base vs Finetuned)
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --dataset_name mandarjoshi/trivia_qa \
  --dataset_config rc.nocontext \
  --eval_split validation \
  --max_eval_samples 20 \
  --output_file ./outputs/eval_compare_results.json
```

Low-VRAM mode:
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --load_in_4bit
```

LLM-as-a-judge:
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --judge_model_name Qwen/Qwen3-4B \
  --max_eval_samples 20 \
  --output_file ./outputs/eval_compare_with_judge.json
```

## Interactive Compare Chat
```bash
python chat_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq
```

Type `exit` or `quit` to stop.

OpenR1 reasoning compare chat:
```bash
python chat_openr1_compare.py \
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math
```

## Core QLoRA Design
1. Load base model: `Qwen/Qwen3-4B`
2. Apply 4-bit quantization (`nf4`, double quant)
3. Insert LoRA adapters on attention/MLP projections
4. Build QA pairs from TriviaQA fields
5. Supervised format:
   `Question: ...`
   `Answer: ...`
6. Mask prompt labels with `-100`, optimize answer tokens only
