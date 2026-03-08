# ECE285 Project
[English](./README.md) | 中文

这个仓库目前包含三类和 `Qwen/Qwen3-4B` 相关的工作流：

- `nq/` 目录下的 TriviaQA 风格问答 QLoRA 微调
- 基于 `oieieio/OpenR1-Math-220k` 的推理式数学 QLoRA 微调
- 一个会自动注入本地线性代数 skill 提示词的交互式聊天入口


## 仓库结构
- [`nq/train_qwen3_qlora_nq.py`](./nq/train_qwen3_qlora_nq.py)：基于 `Trainer` 的 TriviaQA 风格问答微调
- [`nq/train_qwen3_qlora_manual.py`](./nq/train_qwen3_qlora_manual.py)：手写单卡训练循环
- [`nq/chat_compare_qwen3_qlora.py`](./nq/chat_compare_qwen3_qlora.py)：基础模型和 LoRA 适配器的问答对比聊天
- [`train_qwen3_qlora_openr1_math.py`](./train_qwen3_qlora_openr1_math.py)：OpenR1-Math-220k 推理数据微调
- [`chat_openr1_compare.py`](./chat_openr1_compare.py)：基础模型和推理适配器的对比聊天
- [`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py)：自动注入线性代数 skill 的聊天入口
- [`skills/linear-algebra-solver/`](./skills/linear-algebra-solver)：本地 skill 提示词、参考资料和校验脚本
- [`pyproject.toml`](./pyproject.toml)：依赖和命令行脚本定义

## 环境准备
项目要求 Python 3.10+，推荐用 `uv` 管理环境。

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

安装依赖：

```bash
uv pip install -e .
```

核心依赖已经写在 [`pyproject.toml`](./pyproject.toml) 中，包括 `torch`、`transformers`、`datasets`、`peft`、`accelerate`、`bitsandbytes` 和 `tensorboard`。

## 快速开始
启动 TriviaQA 风格问答训练：

```bash
python nq/train_qwen3_qlora_nq.py
```

启动 OpenR1 数学推理训练：

```bash
python train_qwen3_qlora_openr1_math.py
```

启动线性代数 skill 聊天：

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

## 训练工作流
### 1. 基于 `Trainer` 的问答微调
默认配置：

- 模型：`Qwen/Qwen3-4B`
- 数据集：`mandarjoshi/trivia_qa`
- 配置：`rc.nocontext`
- 输出目录：`./outputs/qwen3-4b-qlora-nq`

运行：

```bash
python nq/train_qwen3_qlora_nq.py
```

断点续训：

```bash
python nq/train_qwen3_qlora_nq.py --auto_resume
python nq/train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-nq/checkpoint-200
```

开启 TensorBoard：

```bash
python nq/train_qwen3_qlora_nq.py --use_tensorboard
tensorboard --logdir ./outputs/qwen3-4b-qlora-nq/runs --port 6006
```

### 2. 手写训练循环版本
这个版本显式管理 `optimizer` 和 `scheduler`，并在 checkpoint 中保存它们的状态。

```bash
python nq/train_qwen3_qlora_manual.py --use_tensorboard
```

从 checkpoint 恢复：

```bash
python nq/train_qwen3_qlora_manual.py --resume_checkpoint ./outputs/qwen3-4b-qlora-manual/checkpoint-200
```

### 3. OpenR1 推理式数学微调
默认配置：

- 数据集：`oieieio/OpenR1-Math-220k`
- 训练 split：`default`
- 验证 split：`extended`
- 输出目录：`./outputs/qwen3-4b-qlora-openr1-math`

运行：

```bash
python train_qwen3_qlora_openr1_math.py
```

开启 early stopping：

```bash
python train_qwen3_qlora_openr1_math.py --use_early_stopping
```

断点续训：

```bash
python train_qwen3_qlora_openr1_math.py --auto_resume
python train_qwen3_qlora_openr1_math.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-openr1-math/checkpoint-200
```

## 交互式工具
### 问答对比聊天
对比基础模型和问答适配器：

```bash
python nq/chat_compare_qwen3_qlora.py --adapter_path ./outputs/qwen3-4b-qlora-nq --load_in_4bit
```

### OpenR1 推理对比聊天
对比基础模型和推理适配器：

```bash
python chat_openr1_compare.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

### 线性代数 skill 聊天
对线性代数问题自动注入本地 skill：

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

对所有输入都强制注入 skill：

```bash
python chat_linear_algebra_skill.py --load_in_4bit --always_use_skill
```

在基础模型之上再加载一个 LoRA 适配器：

```bash
python chat_linear_algebra_skill.py --adapter_path ./outputs/qwen3-4b-qlora-openr1-math --load_in_4bit
```

## 数据格式
### TriviaQA 风格问答
问答训练脚本会把样本整理成：

```text
Question: ...
Answer: ...
```

训练时会把 prompt 部分的 label 置为 `-100`，只对答案 token 计算 loss。

### OpenR1 推理格式
推理训练脚本会构造如下目标：

```text
Question: ...
Please reason step by step, then provide the final answer.

Reasoning:
...

Answer:
...
```

如果样本里只有 reasoning 或只有 final answer，脚本也会保留可用的部分。

## 说明
- 当前训练和聊天脚本都默认需要从 Hugging Face 下载模型或数据集。
- 绝大多数流程按 CUDA GPU 环境设计，4-bit 量化依赖 `bitsandbytes`。
- `pyproject.toml` 里部分 console script 仍然按顶层模块声明，但实际问答脚本位于 `nq/` 目录下，所以 README 里统一使用文件路径命令，和当前仓库布局保持一致。
