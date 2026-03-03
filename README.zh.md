# ECE285-Project
[English](./README.md) | 简体中文

这是一个基于 `Qwen/Qwen3-4B` 的 QLoRA 微调仓库，默认数据集为
`mandarjoshi/trivia_qa`（`rc.nocontext`）。

支持功能：
- 基于 Trainer 的 QLoRA 训练
- 传统 `for epoch / for step` 单卡训练循环
- 基座模型 vs 微调模型输出对比评测
- 交互式终端问答对比
- 可选 TensorBoard 实时监控

## 项目结构
- `train_qwen3_qlora_nq.py`: Trainer 训练脚本
- `train_qwen3_qlora_manual.py`: 手写训练循环脚本（单卡）
- `train_qwen3_qlora_openr1_math.py`: OpenR1-Math-220k 推理风格训练脚本
- `eval_compare_qwen3_qlora.py`: 批量对比评测脚本
- `chat_compare_qwen3_qlora.py`: 交互式对比问答脚本
- `chat_openr1_compare.py`: OpenR1 交互式推理对比脚本
- `pyproject.toml`: 依赖与脚本入口管理

## 快速开始
```bash
uv venv
# 先激活 .venv（见下方 UV 快速上手）
uv pip install "-e ."
python train_qwen3_qlora_nq.py
```

默认数据集参数：
```bash
--dataset_name mandarjoshi/trivia_qa --dataset_config rc.nocontext
```

## UV 快速上手
### 安装 uv
推荐（使用 pipx，隔离安装）：
```bash
pip install pipx
pipx ensurepath
pipx install uv
```

直接用 pip：
```bash
pip install uv
```

使用 brew（macOS/Linux）：
```bash
brew install uv
```

### 创建虚拟环境
```bash
uv venv
```
或指定 Python 版本：
```bash
uv venv -p 3.10
```

### 激活虚拟环境
macOS/Linux：
```bash
source .venv/bin/activate
```

Windows（Command Prompt）：
```bash
.venv\Scripts\activate.bat
```

Windows（PowerShell）：
```bash
.venv\Scripts\Activate.ps1
```

退出环境：
```bash
deactivate
```

删除环境目录：
```bash
rm -rf .venv
```

### 使用 uv pip 安装依赖
从 requirements.txt 安装：
```bash
uv pip install -r requirements.txt
```

安装单独包：
```bash
uv pip install requests beautifulsoup4
```

安装 extras：
```bash
uv pip install "celery[redis]"
```

从 Git 仓库安装：
```bash
uv pip install "git+https://github.com/example/my-lib.git#egg=my-lib"
```

本地可编辑安装：
```bash
uv pip install "-e ."
```

### 运行项目
激活 `.venv` 后，直接用：
```bash
python app.py
```
关键点：使用激活环境中的 `python`，不要使用 `uv python`（该命令不存在）。

## 模型训练
Trainer 版本：
```bash
python train_qwen3_qlora_nq.py
```

OpenR1-Math-220k（推理数据）：
```bash
python train_qwen3_qlora_openr1_math.py
```
该脚本训练目标是 `Reasoning:` + `Answer:`，因此模型更容易输出推理过程。

手写循环版本（单卡）：
```bash
python train_qwen3_qlora_manual.py
```

Checkpoint 恢复：
```bash
python train_qwen3_qlora_nq.py --auto_resume
python train_qwen3_qlora_nq.py --resume_from_checkpoint ./outputs/qwen3-4b-qlora-nq/checkpoint-200
```

## TensorBoard 实时监控
开启训练日志：
```bash
python train_qwen3_qlora_nq.py --use_tensorboard
```

启动 TensorBoard：
```bash
tensorboard --logdir ./outputs/qwen3-4b-qlora-nq/runs --port 6006
```

手写循环日志目录：
```bash
./outputs/qwen3-4b-qlora-manual/runs
```

## 评测（基座 vs 微调）
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --dataset_name mandarjoshi/trivia_qa \
  --dataset_config rc.nocontext \
  --eval_split validation \
  --max_eval_samples 20 \
  --output_file ./outputs/eval_compare_results.json
```

低显存模式：
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --load_in_4bit
```

LLM 裁判打分模式：
```bash
python eval_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq \
  --judge_model_name Qwen/Qwen3-4B \
  --max_eval_samples 20 \
  --output_file ./outputs/eval_compare_with_judge.json
```

## 交互式单条问答对比
```bash
python chat_compare_qwen3_qlora.py \
  --adapter_path ./outputs/qwen3-4b-qlora-nq
```

输入 `exit` 或 `quit` 退出。

OpenR1 推理对比：
```bash
python chat_openr1_compare.py \
  --adapter_path ./outputs/qwen3-4b-qlora-openr1-math
```

## QLoRA 核心流程
1. 加载基座模型：`Qwen/Qwen3-4B`
2. 开启 4-bit 量化（`nf4` + double quant）
3. 注入 LoRA 到 attention/MLP 投影层
4. 从 TriviaQA 中构造 `question-answer` 样本
5. 训练格式：`Question: ...` + `Answer: ...`
6. 将 prompt 标签置 `-100`，只对答案 token 计算 loss
