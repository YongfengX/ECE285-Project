# ECE285 项目
[English](./README.md) | 中文

这个仓库目前是一个围绕 `Qwen/Qwen3-4B` 的本地实验集合，主要包括 QLoRA 微调、LoRA 适配器对比、以及基于 prompt 的线性代数 skill 注入。

下面的说明以当前目录里真实存在的文件为准。

## 仓库目前包含什么

- `nq/` 下的 TriviaQA 风格问答 QLoRA 微调
- 根目录下的 OpenR1 数学/推理 QLoRA 微调
- 基础模型和 LoRA 适配器之间的交互式对比聊天
- 使用外部 LLM judge 的数据集评测脚本
- `skills/linear-algebra-solver` 下的本地线性代数 skill 包

## 当前需要注意的点

- 现在建议统一用 `python 脚本路径.py` 的方式运行。
- [`pyproject.toml`](./pyproject.toml) 里的 console script 和模块路径还是旧布局，和当前目录结构不一致。
- 仓库里已经提交了一个适配器 checkpoint，在 `output/qwen3-4b-qlora-openr1-math/checkpoint-2400`。
- 训练脚本默认一般输出到 `./outputs/...`，所以当前项目里 `output/` 和 `outputs/` 两种路径都需要区分。

## 目录说明

- [`nq/train_qwen3_qlora_nq.py`](./nq/train_qwen3_qlora_nq.py)：基于 `Trainer` 的 TriviaQA 风格问答微调
- [`nq/train_qwen3_qlora_manual.py`](./nq/train_qwen3_qlora_manual.py)：手写单卡 QLoRA 训练循环
- [`nq/chat_compare_qwen3_qlora.py`](./nq/chat_compare_qwen3_qlora.py)：问答场景下的 base vs adapter 交互对比
- [`train_qwen3_qlora_openr1_math.py`](./train_qwen3_qlora_openr1_math.py)：在 `oieieio/OpenR1-Math-220k` 上做推理式微调
- [`chatbot.py`](./chatbot.py)：基础模型和 OpenR1 适配器的推理对比聊天
- [`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py)：自动注入线性代数 skill 的聊天入口
- [`eval_compare_with_minimax.py`](./eval_compare_with_minimax.py)：用 MiniMax 作为 judge，对 base 和两个 adapter 做对比评测
- [`chat_openr1_dataset_eval.py`](./chat_openr1_dataset_eval.py)：数据集驱动的三路对比评测脚本
- [`level5compare.py`](./level5compare.py)：针对 `competition_math` Level 5 的专项对比脚本
- [`skills/linear-algebra-solver`](./skills/linear-algebra-solver)：skill 文本、参考资料和验证脚本
- [`output/`](./output)：仓库中已存在的实验产物目录

## 环境准备

建议使用 Python `3.10+`。

创建虚拟环境：

```bash
uv venv -p 3.10
```

在 Windows PowerShell 中激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装这些脚本实际依赖的核心包：

```bash
uv pip install torch transformers datasets accelerate peft bitsandbytes sentencepiece tensorboard openai python-dotenv
```

如果要运行 judge 相关评测脚本，先设置 [`.env.example`](./.env.example) 里对应的环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

## 主要工作流

### 1. TriviaQA 问答微调

`Trainer` 版本：

```bash
python nq/train_qwen3_qlora_nq.py
```

手写训练循环版本：

```bash
python nq/train_qwen3_qlora_manual.py --use_tensorboard
```

这两条训练线的默认设置：

- 基础模型：`Qwen/Qwen3-4B`
- 数据集：`mandarjoshi/trivia_qa`
- 配置：`rc.nocontext`
- 训练格式：`Question: ...` 然后 `Answer: ...`

### 2. OpenR1 数学 / 推理微调

```bash
python train_qwen3_qlora_openr1_math.py
```

常用变体：

```bash
python train_qwen3_qlora_openr1_math.py --use_early_stopping
python train_qwen3_qlora_openr1_math.py --auto_resume
```

默认设置：

- 基础模型：`Qwen/Qwen3-4B`
- 数据集：`oieieio/OpenR1-Math-220k`
- 训练 split：`default`
- 验证 split：`extended`
- 输出目录：`./outputs/qwen3-4b-qlora-openr1-math`

### 3. 交互式聊天

问答 adapter 和基础模型对比：

```bash
python nq/chat_compare_qwen3_qlora.py --adapter_path .\outputs\qwen3-4b-qlora-nq\checkpoint-200 --load_in_4bit
```

使用仓库里现成 checkpoint 的推理对比：

```bash
python chatbot.py --adapter_path .\output\qwen3-4b-qlora-openr1-math\checkpoint-2400 --load_in_4bit
```

线性代数 skill 聊天：

```bash
python chat_linear_algebra_skill.py --load_in_4bit
```

强制每个问题都注入 skill：

```bash
python chat_linear_algebra_skill.py --load_in_4bit --always_use_skill
```

### 4. 数据集评测

使用 MiniMax judge 的主评测脚本：

```bash
python eval_compare_with_minimax.py --adapter_path .\output\qwen3-4b-qlora-openr1-math\checkpoint-2400 --judge_api_key YOUR_API_KEY --load_in_4bit
```

这个脚本会做的事情：

- 加载 base model
- 加载 `adapter_path`
- 加载 `adapter2_path`
- 把支持的 Hugging Face 数据集统一成公共问答格式
- 调用外部 judge 模型打分

脚本内置支持的数据格式：

- `gsm8k`
- `competition_math`
- `svamp`
- `commonsense_qa`
- `arc`

另外还有两个本地实验性质更强的评测脚本：

- [`chat_openr1_dataset_eval.py`](./chat_openr1_dataset_eval.py)：更通用的三路数据集评测
- [`level5compare.py`](./level5compare.py)：面向 `qwedsacf/competition_math` Level 5 的专项评测

这两个脚本更偏实验脚本，运行前建议先检查默认参数。

## 线性代数 Skill 包

[`skills/linear-algebra-solver`](./skills/linear-algebra-solver) 里包含：

- `SKILL.md`：主 skill 工作流
- `references/methods.md`：按题型整理的方法说明
- `references/checklist.md`：常见错误检查清单
- `scripts/verify_linear_algebra.py`：符号验证辅助脚本

[`chat_linear_algebra_skill.py`](./chat_linear_algebra_skill.py) 会在问题像线性代数时自动加载这个 skill，或者通过 `--always_use_skill` 对所有问题强制注入。

## 仓库中已有的产物

当前仓库里已经包含这个 checkpoint：

- [`output/qwen3-4b-qlora-openr1-math/checkpoint-2400`](./output/qwen3-4b-qlora-openr1-math/checkpoint-2400)

里面包含 adapter 权重、tokenizer 文件、优化器状态、调度器状态和训练元数据。

## 已知不一致之处

- `pyproject.toml` 仍然描述的是旧的根目录模块布局。
- 仓库里有 `output/`，而训练脚本默认写入的是 `outputs/`。
- 部分评测脚本是为某次实验临时扩展的，默认参数假设比主训练脚本更强。

如果你接下来还想把安装命令和 console scripts 也整理通，我建议下一步同步修正 [`pyproject.toml`](./pyproject.toml)。
