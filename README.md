# ECE285-Project
QLoRA fine-tuning for Hugging Face `Qwen/Qwen3-0.6B` on Google Natural Questions.

## Files
- `train_qwen3_qlora_nq.py`: training script
- `eval_compare_qwen3_qlora.py`: compare base vs finetuned outputs
- `pyproject.toml`: dependency and environment management

## Install uv

### Windows (PowerShell)
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
uv --version
```

## Environment Setup

### Option 1: `venv + pip`
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

### Option 2: `uv` (recommended)
```bash
uv sync
```
`uv sync` will create/update the virtual environment and install all dependencies from `pyproject.toml`.

Activate environment:
```bash
.venv\Scripts\activate
```

## Train
```bash
python train_qwen3_qlora_nq.py
```
or use the script entry point:
```bash
train-qwen3-qlora-nq
```

If your local Hugging Face dataset name differs, pass it explicitly:
```bash
python train_qwen3_qlora_nq.py --dataset_name google-research-datasets/natural_questions
```

Resume from checkpoint:
```bash
python train_qwen3_qlora_nq.py --auto_resume
```
or:
```bash
python train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-0.6b-qlora-nq/checkpoint-200
```

## How This QLoRA Works
The training pipeline is:

1. Load base model  
`AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", ...)`

2. Enable 4-bit quantization (the "Q" in QLoRA)  
With `BitsAndBytesConfig`:
- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- `bnb_4bit_compute_dtype=bf16/fp16`

3. Train LoRA adapters only (the "LoRA" in QLoRA)  
`prepare_model_for_kbit_training(model)` is applied first, then `LoraConfig` is injected for:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Base model weights stay frozen; only low-rank adapter params are updated.

4. Process Natural Questions  
The script extracts `question + answer` pairs across multiple possible field layouts (`annotations`, token spans, short answers).

5. Supervised fine-tuning format  
Each sample is formatted as:
```text
Question: <question>
Answer: <answer>
```
Prompt tokens (`Question` part) are masked with `-100`, so loss is computed only on answer tokens.

6. Trainer and optimizer  
Uses `Trainer + paged_adamw_8bit` with gradient accumulation and mixed precision to reduce VRAM usage.

7. Save outputs  
After training, LoRA adapter weights and tokenizer are saved to `output_dir` (default: `./outputs/qwen3-0.6b-qlora-nq`).

## Common Args
```bash
python train_qwen3_qlora_nq.py \
  --max_train_samples 20000 \
  --max_seq_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

## Evaluation: Base vs Finetuned
Compare output quality between original model and LoRA-finetuned model:

```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-0.6b-qlora-nq \
  --dataset_name natural_questions \
  --eval_split validation \
  --max_eval_samples 20 \
  --output_file ./outputs/eval_compare_results.json
```

Script entry point:
```bash
eval-compare-qwen3-qlora --adapter_path ./outputs/qwen3-0.6b-qlora-nq
```
