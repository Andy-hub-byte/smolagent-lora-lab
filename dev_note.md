# dev_note.md — 开发说明 / Developer notes

> 记录两件事：**哪些能力来自框架、哪些是本项目的自研改造**，以及**冒烟测试记录**。
> 换框架、升级依赖或接手维护前，先读这份文件。

---

## 1. 框架能力 vs 自研改造 / Framework vs custom

| 模块 | 来源 | 说明 |
|---|---|---|
| `SFTTrainer` / `SFTConfig` | 【框架能力】trl 1.16 | 训练循环、checkpoint、断点续训均由框架提供 |
| `LoraConfig` / `PeftModel` | 【框架能力】peft 0.21 | LoRA 注入与适配器加载 |
| `CodeAgent` / `@tool` / `TransformersModel` | 【框架能力】smolagents 1.26 | ReAct 循环、工具包装、模型封装 |
| `TrainerCallback` 机制 | 【框架能力】transformers | 进度回调基类 |
| 硬件检测 + CPU/ARM 降参 | 【自研改造】 | `detect_hardware()` / `check_hardware()` / batch 与 seq_len 自动降级 |
| 双模型源加载层 | 【自研改造·阶段二】 | `MODEL_REGISTRY` + `resolve_repo_id()` + `download_model()`，纯加载层，与训练逻辑解耦 |
| 数据集合并去重 | 【自研改造】 | `load_dataset_rows()`：md5 去重、坏行跳过告警 |
| 断点续训决策 | 【自研改造】 | `find_checkpoints()` / `decide_resume()`：非 TTY 有快照未指定时**报错**，杜绝挂机隐晦默认 |
| 进度输出格式 | 【自研改造】 | `ProgressCallback`：step/loss/lr/已用/预计剩余 |
| **smolagents 挂 LoRA** | 【自研改造·关键】 | smolagents 的 `TransformersModel` **不原生支持 PEFT**，需加载后手动包装：`tm.model = PeftModel.from_pretrained(tm.model, adapter)` |
| 文件工具安全边界 | 【自研改造】 | `_safe_path()` 拒绝 `../` 与绝对路径越界 |
| 记忆存储与注入 | 【自研改造】 | `MemoryStore`：json 持久化、保留 50 条、`summary()` 注入 instructions |
| 代码块标签锁定 | 【自研改造】 | `CODE_TAGS = ("<code>", "</code>")` 显式传入 CodeAgent，防版本默认值漂移 |
| **WebUI 图形界面** | 【自研改造·阶段三】 | `webui/`：Flask 零构建单页应用，把 train_lora.py / agent.py 包装成表单 + 实时 loss 曲线 + SSE 终端日志；只做进程管理与展示，训练/推理逻辑仍归两个脚本 |

### 关键坑位记录 / Pitfalls hit

1. **trl 1.16**：`SFTConfig` 字段名是 `max_length`(旧版 `max_seq_length`)；`formatting_func` 不在 Config 里，走 `SFTTrainer(...)` 构造参数。
2. **trl 1.16**：`train_dataset` 必须是 `datasets.Dataset`，裸 list 报 TypeError → 用 `Dataset.from_list(rows)`。
3. **smolagents 1.26**：`TransformersModel` 无 PEFT 支持；内部仍用 `torch_dtype=` 透传(transformers 5.x 已弃用，agent.py 已对该特定告警定向抑制)。
4. **bitsandbytes 无 ARM64 构建**：平板 CPU 无法 4bit QLoRA → 脚本自动降级 fp32 LoRA 并阻止 7B/14B。
5. **CONFIG 与 argparse 键必须一一对应**：CLI 覆盖只遍历 CONFIG 键，漏定义(如 `resume`/`task`)会 KeyError——冒烟测试抓到过两次。
6. **transformers≥4.56**：`from_pretrained` 推荐 `dtype=`(弃用 `torch_dtype=`)，train_lora.py 已用新名。
7. **WebUI 子进程相对路径**：`--dataset-file` 必须传 `datasets/xxx.jsonl`(相对项目根)，只传文件名会在 cwd 下找不到；`--model ./models/tiny-qwen-test` 同理。
8. **WebUI 产出扫描**：用户可用 `--output-dir` 指定任意目录，`/api/outputs` 不能写死 `lora_output/`，需扫描根目录下所有含 `adapter_config.json` 或 `checkpoint-*` 的目录。
9. **Job 单例互斥**：同 kind 任务运行中拒绝重复启动；页面刷新后 `resumeJobStates()` 自动重连 SSE，避免"僵尸训练看不见"。
10. **SSE 日志缓冲**：`deque(maxlen=5000)` 环形缓冲 + `queue.Queue` 订阅广播，晚连上的客户端先补发缓冲末尾，断线重连不丢上下文。

## 2. 冒烟测试记录 / Smoke test record

测试环境：CPU(x86_64) / torch 2.10 / transformers 5.1 / trl 1.16 / smolagents 1.26。
方法：**微型随机权重 Qwen2**(真实 Qwen2.5 tokenizer，9.74M 参数)验证全链路，不下载 GB 级权重。

| # | 用例 | 命令要点 | 结果 |
|---|---|---|---|
| 1 | 单文件数据集 | `--dataset-mode single --dataset-file datasets/dataset_agent_tool.jsonl` | ✅ +24 条 |
| 2 | 全量合并去重 | `--dataset-mode all` | ✅ 24+2+40+52=118 条 |
| 3 | CPU LoRA 训练 | `--max-steps 3 --save-steps 2` | ✅ loss 11.87→11.84 |
| 4 | 快照生成 | save_steps=2 | ✅ checkpoint-2 / checkpoint-3 + 适配器 19KB |
| 5 | 断点续训 | `--resume --max-steps 5` | ✅ 从 step 3 续跑到 step 5 |
| 6 | Agent 挂适配器 | `agent.py --mode C --model-source local --adapter ...` | ✅ CodeAgent 跑完 max steps 正常退出 |
| 7 | 记忆持久化 | 任务后检查 json | ✅ 落盘 + 重载 2 条 |
| 8 | 工具与安全边界 | `smoke_tools.py` 直接调工具 | ✅ 读写列全过；`../`、绝对路径越界均被拒 |
| 9 | WebUI 训练链路 | `/api/train/start`(local + tiny, 3 步) | ✅ returncode 0，3 个 loss 点，适配器落盘 |
| 10 | WebUI loss 曲线 | `/api/train/loss` | ✅ step/loss/ts 三点齐全 |
| 11 | WebUI Agent 测试台 | 模式 C 挂适配器单任务 | ✅ 加载基座+适配器，跑完 max steps 正常退出 |
| 12 | WebUI 产出扫描 | 自定义 `--output-dir` | ✅ 扫到适配器目录与 checkpoint-3(修复了只认 `lora_output/` 的漏洞) |

冒烟中修复的 bug：`train_dataset` 类型、`cfg["resume"]` / `cfg["task"]` 缺失、
local 模式误解析仓库 ID、docstring 参数名不一致。修复后全绿。

### 复现冒烟 / Reproduce

```bash
python make_tiny_model.py                 # 下载 tokenizer + 构造微型随机权重模型(约 12MB)
python train_lora.py --model-source local --model ./models/tiny-qwen-test \
    --dataset-mode single --dataset-file datasets/dataset_agent_tool.jsonl \
    --max-steps 3 --save-steps 2 --num-epochs 1 --batch-size 1 --grad-accum 1 \
    --lora-rank 4 --lora-alpha 8 --max-seq-len 128 --output-dir ./outputs/smoke-adapter
python train_lora.py ...(同上) --resume --max-steps 5     # 断点续训
python agent.py --mode C --model-source local --model-path ./models/tiny-qwen-test \
    --adapter ./outputs/smoke-adapter --task "你好，请用一句话介绍你自己" --quiet
python smoke_tools.py                     # 工具/记忆/安全边界(不加载模型, 秒级)
```

WebUI 冒烟(上述 1~11 的图形版复现)：

```bash
pip install flask
python make_tiny_model.py --skip-download   # 有 tokenizer 缓存时离线生成微型模型
python webui/app.py                         # 打开 http://127.0.0.1:7860
# 环境体检全绿 → 训练控制台填: 源=local, 模型=./models/tiny-qwen-test,
#   数据集=datasets/dataset_agent_tool.jsonl, max-steps=3 → 开始训练
# → loss 曲线出现 3 点 → Agent 测试台选模式 C, 适配器填刚产出的目录 → 运行
```

## 3. 版本策略 / Versioning

- **完整版(本目录)**：带全部中文详解注释、dev_note、冒烟脚本，适合学习与二次开发。
- **自用版**：同逻辑精简注释 + `simplified_requirements.txt` + `simple_readme.md`，适合直接开跑。
