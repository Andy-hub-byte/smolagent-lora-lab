# smolagent-lora-lab

> 用 smolagents + LoRA(QLoRA) 在**自己电脑上**微调一个"懂你数据"的本地智能体。
> Fine-tune your own local AI agent with LoRA on your own machine — no cloud, no API key.

**三大件**：`train_lora.py`(训练器) + `agent.py`(本地智能体) + `datasets/`(训练数据)
**Two model sources**: ModelScope(魔搭, 国内默认) ↔ Hugging Face(海外) — 一行参数切换

---

## 1. 这是什么 / What is this

一个**开箱可用的 LoRA 微调工作台**：把基座大模型(Qwen 系列等)用你自己的对话数据微调，
产出一个几十~几百 MB 的 LoRA 适配器，挂回基座就是一个"学过你数据"的本地智能体
(基于 smolagents 的 CodeAgent，会写 Python 代码调工具完成任务)。

整个流程**全本地运行**：数据不出本机、无 API key、无云端费用。
适配器与基座分离，换机器、换下载源都不影响使用。

```
datasets/*.jsonl  ──►  train_lora.py  ──►  lora_output/ (LoRA 适配器)
   你的对话数据          QLoRA 训练              │
                                                ▼
   你的任务/问题  ◄──  agent.py(CodeAgent + 工具 + 记忆)  ◄──  基座模型 + 适配器
```

## 2. 环境要求 / Requirements

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| 系统 | Windows 10 / macOS / Linux | Linux |
| Python | 3.10+ | 3.11 |
| 内存 | 16 GB | 32 GB |
| 显卡 | 可选(CPU 也能跑小模型) | NVIDIA 8GB+ 显存 |

- **NVIDIA 显卡**：4bit QLoRA，7B 模型约需 18GB 显存
- **纯 CPU / ARM 平板**：自动降级 fp32 LoRA，仅支持 ≤4B 模型(脚本会阻止 7B/14B 启动)

```bash
pip install -r requirements.txt
```

## 3. 快速开始 / Quick start

```bash
# ① 合并 datasets/ 下全部数据集, 从魔搭拉 Qwen2.5-7B, 开始训练(默认配置)
python train_lora.py --model qwen2.5-7b

# ② 训练完成后, 用本地 Agent 加载适配器, 进入交互模式
python agent.py --mode A --model qwen2.5-7b --adapter ./lora_output

# ③ 或者单次任务模式(问完即退)
python agent.py --mode A --model qwen2.5-7b --adapter ./lora_output --task "帮我写一首关于春天的短诗"
```

新手只需改 `train_lora.py` 顶部的 **CONFIG 集中配置区**(学习率/轮数/LoRA 秩等都有中文注释)，
所有配置项都能用命令行参数覆盖，两者等价。

## 4. 三种使用模式 / Three modes

| 模式 | 命令 | 说明 |
|---|---|---|
| **A** 在线基座 + LoRA | `agent.py --mode A --model qwen2.5-7b --adapter ./lora_output` | 从魔搭/HF 拉基座, 挂适配器(最常用) |
| **B** Ollama | `agent.py --mode B` | 对接本地 Ollama 服务(需先 `ollama pull qwen2.5:7b-instruct`) |
| **C** 本地基座 + LoRA | `agent.py --mode C --model-path ./models/xxx --adapter ./lora_output` | 基座已在本地文件夹(如手动下载过) |

## 5. 国内 / 海外使用说明 / China & overseas usage

本项目内置**双模型源**，同一份训练代码/数据集/产出的 LoRA 完全兼容，只是下载渠道不同：

| 用户 | 命令 | 说明 |
|---|---|---|
| 国内(默认) | `python train_lora.py --model qwen2.5-7b` | 走**魔搭 ModelScope**，速度快，无需登录 |
| 海外 | `python train_lora.py --model qwen2.5-7b --model-source huggingface` | 走 **Hugging Face Hub** |

内置短名映射表(`--list-models` 查看，短名不区分大小写)：

| 短名 | 魔搭 ModelScope | Hugging Face |
|---|---|---|
| `qwen2.5-1.5b` | qwen/Qwen2.5-1.5B-Instruct | Qwen/Qwen2.5-1.5B-Instruct |
| `qwen2.5-3b` | qwen/Qwen2.5-3B-Instruct | Qwen/Qwen2.5-3B-Instruct |
| `qwen2.5-7b` | qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct |
| `qwen2.5-14b` | qwen/Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct |
| `qwen3-1.7b` | qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B |
| `qwen3-4b` | qwen/Qwen3-4B | Qwen/Qwen3-4B |
| `qwen3-8b` | qwen/Qwen3-8B | Qwen/Qwen3-8B |
| `qwen3-14b` | qwen/Qwen3-14B | Qwen/Qwen3-14B |
| `glm4-9b-chat` | ZhipuAI/glm-4-9b-chat | THUDM/glm-4-9b-chat |
| `yi-1.5-6b-chat` | 01ai/Yi-1.5-6B-Chat | 01-ai/Yi-1.5-6B-Chat |
| `yi-1.5-9b-chat` | 01ai/Yi-1.5-9B-Chat | 01-ai/Yi-1.5-9B-Chat |
| `internlm2.5-7b-chat` | Shanghai_AI_Laboratory/internlm2_5-7b-chat | internlm/internlm2_5-7b-chat |
| `llama3.1-8b-instruct` | LLM-Research/Meta-Llama-3.1-8B-Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct |

也可以直接传**完整仓库ID**(含 `/`)，按当前源原样下载。

> ⚠️ **Llama 系列门控提醒 / Llama gated-model notice**
> `meta-llama/Meta-Llama-3.1-8B-Instruct` 在 Hugging Face 上是**门控模型**，需要：
> 1. 在模型页点击 "Agree and access" 同意协议；
> 2. 到 https://huggingface.co/settings/tokens 创建 access token；
> 3. `pip install -U huggingface_hub && hf auth login` 完成登录。
> 国内用户可改用 `--model-source modelscope`(LLM-Research 镜像无需登录)。

> ✅ **两源兼容性说明 / Cross-source compatibility**
> 两个模型源只是**下载渠道不同**：训练参数、数据集、最终产出的 LoRA 完全通用。
> 在魔练下载的基座训出的适配器，挂到 HF 下载的同一模型上同样有效。

## 6. 数据集 / Datasets

`datasets/` 目录存放 jsonl 训练数据，每行一条对话：

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！有什么可以帮你？"}]}
```

| 文件 | 条数 | 内容 |
|---|---|---|
| `dataset_daily.jsonl` | 40 | 日常对话风格(语气/自我介绍/常见问答) |
| `dataset_study.jsonl` | 52 | 学习辅导场景(概念讲解/解题思路) |
| `dataset_agent_tool.jsonl` | 24 | Agent 工具调用范例(Thought→`<code>`→Observation→final_answer) |
| `dataset_custom_sample.jsonl` | 2 | 自定义示例(示范如何添加你自己的数据) |

- `--dataset-mode all`(默认)：合并全部 4 个文件，按内容 md5 去重
- `--dataset-mode single --dataset-file datasets/dataset_daily.jsonl`：只用一个文件
- **想教模型新知识？往 `datasets/` 里按同样格式加数据，重训即可。** 数据质量 > 数据数量。

## 7. 训练参数速查 / Training options

```bash
python train_lora.py --model qwen2.5-7b \
    --lora-rank 16 \            # LoRA 秩: 越大学习能力越强(常用 8/16/32/64)
    --lora-alpha 32 \           # LoRA 缩放: 常用设为 rank 的 2 倍
    --learning-rate 2e-4 \      # LoRA 可用比全参大得多的学习率
    --num-epochs 3 \            # 训练轮数(数据少可加大, 过拟合就减小)
    --batch-size 2 --grad-accum 8 \   # 等效批量 = 2×8 = 16
    --max-seq-len 1024 \        # 最大序列长度(CPU 模式自动降为 512)
    --output-dir ./lora_output  # 适配器输出目录
```

常用组合：

```bash
# 小数据快跑验证流程(几分钟)
python train_lora.py --model qwen2.5-7b --max-steps 20 --save-steps 10

# 断点续训(自动检测 output-dir 下最新快照; 挂机请显式指定)
python train_lora.py --model qwen2.5-7b --resume

# 忽略快照, 从头再来
python train_lora.py --model qwen2.5-7b --no-resume
```

## 8. Agent 能力 / Agent capabilities

`agent.py` 基于 smolagents `CodeAgent`(模型写 Python 代码调工具)：

- **文件工具**：`read_file` / `write_file` / `list_folder`(带工作目录安全边界，拒绝 `../` 越界)
- **长期记忆**：任务与结果自动存入 `agent_memory.json`，重启后注入系统指令(保留最近 50 条)
- **安全护栏**：单任务最大步数(默认 8)防失控烧时间；代码块标签显式锁定 `<code>...</code>`
- **中英双语**系统指令，适配 Qwen 系聊天模板

## 9. 目录结构 / Project layout

```
smolagent-lora-lab/
├── train_lora.py          # 训练器(CONFIG 集中配置 + 双模型源 + 断点续训)
├── agent.py               # 本地智能体(模式 A/B/C + 工具 + 记忆)
├── requirements.txt       # 依赖清单(含 CUDA / ARM 双说明)
├── datasets/              # 训练数据(118 条, 4 个文件)
│   ├── dataset_daily.jsonl
│   ├── dataset_study.jsonl
│   ├── dataset_agent_tool.jsonl
│   └── dataset_custom_sample.jsonl
├── lora_output/           # 训练产出的适配器(首次训练后生成)
├── models/                # 模型下载缓存(首次运行后生成)
└── agent_memory.json      # Agent 记忆(首次交互后生成)
```

## 10. 常见问题 / FAQ

**Q: 训练中断了怎么办？**
A: 直接重跑同一条命令加 `--resume`，自动从 `lora_output/checkpoint-*` 最新快照继续。
Ctrl+C 中断时脚本会先保存 `checkpoint-interrupt` 再退出。

**Q: 显存不足 / 内存爆了？**
A: NVIDIA 用户装 `bitsandbytes` 启用 4bit；或换小模型(`--model qwen2.5-1.5b`)；
CPU/ARM 用户请使用 ≤4B 模型。

**Q: 适配器能单独拷走吗？**
A: 不能单独用。适配器只含增量权重(几十~几百 MB)，必须搭配对应基座模型加载。

**Q: 模型回答没有变好？**
A: 检查数据集是否覆盖目标场景；数据少时先 `--num-epochs 5~10`；避免用大模型生成的低质数据。

## 11. 测试流程清单 / Testing checklist

```bash
# 1) 语法与 CLI 自检
python train_lora.py --help && python agent.py --help

# 2) 数据集加载验证(合并去重, 不训练)—— 用 --max-steps 1 快速跑通
python train_lora.py --model-source local --model <小模型路径> --dataset-mode all --max-steps 1

# 3) 微型冒烟(CPU 几分钟): 训练 3 步 → 保存快照 → --resume 续训 → Agent 挂适配器
#    详见 dev_note.md【冒烟测试记录】

# 4) 真实训练(按显存选模型), 观察 loss 下降
python train_lora.py --model qwen2.5-7b

# 5) 验证微调效果: 挂适配器问训练数据里覆盖过的问题, 对比不挂适配器的回答
python agent.py --mode A --model qwen2.5-7b --adapter ./lora_output
python agent.py --mode A --model qwen2.5-7b          # 不挂适配器作对照

# 6) 断点续训: 训练中 Ctrl+C → 重跑加 --resume → 确认从断点继续

# 7) 工具与记忆: Agent 对话几轮 → 检查 agent_memory.json 是否落盘
```

## 12. 技术栈 / Tech stack

[smolagents](https://github.com/huggingface/smolagents) · [transformers](https://github.com/huggingface/transformers) · [peft](https://github.com/huggingface/peft) · [trl](https://github.com/huggingface/trl) · [ModelScope](https://modelscope.cn) · [Hugging Face](https://huggingface.co)
