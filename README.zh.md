# ECE285 项目
[English](./README.md) | 中文

这个仓库目前包含三条围绕 `Qwen/Qwen3-4B` 的工作流：

- 在 [`nq/`](./nq) 下做 TriviaQA 风格问答的 QLoRA 微调
- 在 `oieieio/OpenR1-Math-220k` 上做推理型数学回答的 QLoRA 微调
- 通过本地线性代数 skill 自动注入提示词的交互式聊天

本文档只描述当前仓库里实际存在的脚本和命令。

## 仓库结构
- [`nq/train_qwen3_qlora_nq.py`](./nq/train_qwen3_qlora_nq.py)：基于 `Trainer` 的 TriviaQA 风格问答微调
- [`nq/train_qwen3_qlora_manual.py`](./nq/train_qwen3_qlora_manual.py)：手写训练循环版本
- [`nq/chat_compare_qwen3_qlora.py`](./nq/chat_compare_qwen3_qlora.py)：基础模型和 QA LoRA 的交互式对比
- [`train_qwen3_qlora_openr1_math.py`](./train_qwen3_qlora_openr1_math.py)：OpenR1-Math-220k 推理型微调
- [`chat_openr1_compare.py`](./chat_openr1_compare.py)：基础模型和推理 LoRA 的交互式对比
- [`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py)：自动注入线性代数 skill 的聊天入口
- [`eval_compare_with_minimax.py`](./eval_compare_with_minimax.py)：用 MiniMax 作为 LLM judge 评测基础模型和 LoRA
- [`skills/linear-algebra-solver/`](./skills/linear-algebra-solver)：本地 skill 提示词、参考资料和校验脚本
- [`pyproject.toml`](./pyproject.toml)：依赖和 console script 定义

## 环境准备
项目要求 Python 3.10+，推荐使用 `uv` 管理环境。

```bash
uv venv -p 3.10
```

激活环境：

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

安装项目依赖：

```bash
uv pip install -e .
```

核心依赖见 [`pyproject.toml`](./pyproject.toml)，包括 `torch`、`transformers`、`datasets`、`peft`、`accelerate`、`bitsandbytes` 和 `tensorboard`。

## 快速开始
TriviaQA 风格问答微调：

```bash
python nq/train_qwen3_qlora_nq.py
```

OpenR1 数学推理微调：

```bash
python train_qwen3_qlora_openr1_math.py
```

线性代数 skill 聊天：

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

MiniMax 评测：

```bash
python eval_compare_with_minimax.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --judge_api_key YOUR_API_KEY --load_in_4bit
```

## 训练流程
### 1. 基于 `Trainer` 的 TriviaQA 风格问答微调
默认设置：

- model: `Qwen/Qwen3-4B`
- dataset: `mandarjoshi/trivia_qa`
- config: `rc.nocontext`
- output: `./outputs/qwen3-4b-qlora-nq`

运行：

```bash
python nq/train_qwen3_qlora_nq.py
```

断点恢复：

```bash
python nq/train_qwen3_qlora_nq.py --auto_resume
python nq/train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-nq/checkpoint-200
```

开启 TensorBoard：

```bash
python nq/train_qwen3_qlora_nq.py --use_tensorboard
tensorboard --logdir ./outputs/qwen3-4b-qlora-nq/runs --port 6006
```

### 2. 手写训练循环版 TriviaQA 微调
这个版本保留了显式训练循环，并在 checkpoint 中保存 `optimizer` 和 `scheduler` 状态。

```bash
python nq/train_qwen3_qlora_manual.py --use_tensorboard
```

从 checkpoint 恢复：

```bash
python nq/train_qwen3_qlora_manual.py --resume_checkpoint ./outputs/qwen3-4b-qlora-manual/checkpoint-200
```

### 3. OpenR1 推理型微调
默认设置：

- dataset: `oieieio/OpenR1-Math-220k`
- train split: `default`
- eval split: `extended`
- output: `./outputs/qwen3-4b-qlora-openr1-math`

运行：

```bash
python train_qwen3_qlora_openr1_math.py
```

开启 early stopping：

```bash
python train_qwen3_qlora_openr1_math.py --use_early_stopping
```

断点恢复：

```bash
python train_qwen3_qlora_openr1_math.py --auto_resume
python train_qwen3_qlora_openr1_math.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-openr1-math/checkpoint-200
```

## 交互式工具
### QA 对比聊天

```bash
python nq/chat_compare_qwen3_qlora.py --adapter_path ./outputs/qwen3-4b-qlora-nq --load_in_4bit
```

### OpenR1 推理对比聊天

```bash
python chat_openr1_compare.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

### 线性代数 skill 聊天

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

强制每个问题都注入 skill：

```bash
python chat_linear_algebra_skill.py --load_in_4bit --always_use_skill
```

在基础模型上叠加 LoRA：

```bash
python chat_linear_algebra_skill.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

## MiniMax 评测
使用 [`eval_compare_with_minimax.py`](./eval_compare_with_minimax.py) 可以让基础模型和 LoRA 适配器在同一批数据上作答，再交给 MiniMax 打分。

默认参数：

- base model: `Qwen/Qwen3-4B`
- judge model: `MiniMax-M2.5`
- dataset: `gsm8k`
- config: `main`
- split: `test`
- prompt style: `reasoning`

最小示例：

```bash
python eval_compare_with_minimax.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --judge_api_key YOUR_API_KEY --load_in_4bit
```

适合 QA LoRA 的示例：

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

关键参数：

- `--adapter_path`：必填，要评测的 LoRA 路径
- `--judge_api_key`：必填，DashScope 兼容 MiniMax 接口的 API key
- `--dataset_name`、`--dataset_config`、`--dataset_split`：Hugging Face 数据集选择
- `--dataset_format`：支持 `auto`、`gsm8k`、`competition_math`、`svamp`、`commonsense_qa`、`arc`
- `--prompt_style`：推理型模型用 `reasoning`，直接问答型模型用 `qa`
- `--max_samples`：限制评测样本数，控制耗时和 API 成本
- `--output_file`：输出 JSON，包含逐样本得分和整体汇总

脚本会先把支持的数据集统一归一化，再进行生成和评审。对下面这些内置数据集，`--dataset_format auto` 就能自动识别。

### 新增可直接评测的数据集
除了默认的 `gsm8k`，现在还支持：

- `hendrycks/competition_math`
- `ChilleD/SVAMP`
- `tau/commonsense_qa`
- `allenai/ai2_arc`

示例命令如下。

GSM8K：

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

Competition Math：

```bash
python eval_compare_with_minimax.py ^
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math ^
  --judge_api_key YOUR_API_KEY ^
  --dataset_name hendrycks/competition_math ^
  --dataset_config default ^
  --dataset_split test ^
  --dataset_format competition_math ^
  --prompt_style reasoning ^
  --max_samples 30 ^
  --load_in_4bit ^
  --output_file ./outputs/eval_compare_with_minimax_competition_math.json
```

SVAMP：

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

CommonsenseQA：

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

AI2 ARC：

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

## 数据格式
### TriviaQA 风格问答
QA 脚本会把数据整理成：

```text
Question: ...
Answer: ...
```

提示词部分的 label 会被设成 `-100`，只在答案 token 上计算 loss。

### OpenR1 推理格式
推理脚本构造的文本格式是：

```text
Question: ...
Please reason step by step, then provide the final answer.

Reasoning:
...

Answer:
...
```

如果一条数据只有 reasoning 或只有 final answer，脚本仍会保留可用目标。

## 说明
- 当前训练和聊天脚本默认需要能访问 Hugging Face 模型与数据集下载。
- 大部分流程默认面向 CUDA GPU；4-bit 加载通过 `bitsandbytes` 支持。
- `eval_compare_with_minimax.py` 需要显式传入 `--judge_api_key`，因为判分通过 DashScope 兼容接口调用 MiniMax。
- `pyproject.toml` 里部分 console script 还是按顶层模块声明，但 QA 脚本实际位于 `nq/` 目录下，所以文档统一使用文件路径命令，以匹配当前仓库布局。
