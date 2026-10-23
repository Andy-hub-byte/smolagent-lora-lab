/* ============================================================
 * smolagent-lora-lab Web 微调工作台 · 中英双语模块
 * 用法:
 *   t("tab.train")                    取当前语言文本
 *   t("job.metaPid", {pid: 123})      {var} 插值
 *   applyStatic()                     扫描 data-i18n 属性刷静态文本
 *   onLangChange(fn)                  语言切换时重渲染动态内容
 * 语言持久化: localStorage "lora-lab-lang-v1"
 * ============================================================ */
"use strict";

const DICT = {
  zh: {
    "app.title": "smolagent-lora-lab · Web 微调工作台",
    "app.subtitle": "Web 微调工作台 · QLoRA / LoRA 训练 + 模型对话",
    "app.envDetecting": "环境检测中…",
    "app.footer": "smolagent-lora-lab · 本地微调工作台 — 所有数据仅在本机处理，不上传任何服务器。",
    "app.adopted": "★ 已接上仍在运行的作业：{kinds}（服务器重启过，任务没断）",

    "tab.env": "🩺 环境体检",
    "tab.import": "📥 模型导入",
    "tab.train": "🚀 训练",
    "tab.chat": "💬 模型对话",
    "tab.data": "📚 数据集",
    "tab.out": "📦 产出管理",

    "nav.chat": "模型对话",
    "nav.train": "训练",
    "nav.import": "模型导入",
    "nav.data": "数据集",
    "nav.out": "产出管理",
    "nav.env": "环境体检",
    "chat.paramsBtn": "⚙ 参数",
    "chat.send": "发送",
    "chat.emptyHint": "在下方输入任务，按回车开始对话",
    "imp.searchPh": "搜索已导入的模型…",

    "env.title": "运行环境",
    "env.deps": "依赖情况",
    "env.quick": "快速自检（无网络 / 无显卡也能用）",
    "env.tinyHint": "生成一个微型随机权重 Qwen 模型（约 10MB，真实 tokenizer），用于在本地完整跑通「训练 → 保存适配器 → 对话」全链路，无需下载 GB 级权重。模型文件在 models/ 目录，仅用于测试，不要用于正式对话。",
    "env.btnTiny": "生成微型模型",
    "env.py": "Python",
    "env.os": "系统",
    "env.device": "设备",
    "env.project": "项目目录",
    "env.cpu": "CPU（无 CUDA GPU）",
    "env.gpu": "CUDA GPU（{name}，{vram} GB）",
    "dep.torch": "PyTorch",
    "dep.transformers": "Transformers",
    "dep.peft": "PEFT",
    "dep.trl": "TRL",
    "dep.datasets": "Datasets",
    "dep.smolagents": "smolagents",
    "dep.modelscope": "ModelScope",
    "dep.accelerate": "Accelerate",
    "dep.flask": "Flask（本工作台）",
    "dep.missing": "未安装",
    "env.detectFail": "环境检测失败",

    "imp.title": "导入外部模型",
    "imp.intro": "把已经下载好的模型加进来，之后训练和对话都能直接选它。支持三种方式，任选其一。",
    "imp.wayPath": "方式一 · 本机文件夹",
    "imp.wayPathHint": "填写模型文件夹的绝对路径（在运行本服务的电脑上），会拷贝到项目的 models/ 目录。适合模型已经在你电脑硬盘里的情况。",
    "imp.pathLabel": "模型文件夹绝对路径",
    "imp.nameLabel": "目录名（可选）",
    "imp.nameLabelReq": "目录名（必填）",
    "imp.namePh": "默认用原文件夹名",
    "imp.overwrite": "覆盖已存在的同名目录",
    "imp.btnPath": "导入到 models/",
    "imp.wayRepo": "方式二 · Hugging Face / ModelScope 仓库",
    "imp.wayRepoHint": "填写仓库 ID 直接下载到 models/ 目录。国内推荐魔搭；Llama 等门控模型需先在 HF 官网同意协议并运行 hf auth login。支持内置短名（如 qwen2.5-1.5b）。",
    "imp.repoSource": "来源",
    "imp.repoLabel": "仓库 ID",
    "imp.repoPh": "Qwen/Qwen2.5-1.5B-Instruct 或 qwen2.5-1.5b（内置短名）",
    "imp.btnRepo": "开始下载",
    "imp.wayUpload": "方式三 · 网页上传",
    "imp.wayUploadHint": "适合平板 / 手机：直接把模型文件（可多选，或打包成一个 zip）传上来，自动落到 models/ 目录。",
    "imp.uploadLabel": "选择文件（可多选 / zip）",
    "imp.btnUpload": "上传并导入",
    "imp.uploading": "上传中…",
    "imp.listTitle": "已导入的模型",
    "imp.listHint": "这些模型位于 models/ 目录，训练和对话都能直接选。",
    "imp.colName": "目录",
    "imp.colArch": "架构",
    "imp.colWeights": "权重文件",
    "imp.colValid": "格式校验",
    "imp.colSize": "大小",
    "imp.colTime": "导入时间",
    "imp.useBase": "用作基座",
    "imp.valid": "✓ 完整",
    "imp.invalid": "✗ 不完整",
    "imp.noTok": "（缺 tokenizer）",
    "imp.empty": "还没有导入模型。用上面三种方式之一添加，或到环境体检页生成微型测试模型。",
    "imp.noMatch": "没有匹配的模型",
    "imp.loadFail": "加载失败：",
    "imp.confirmDelete": "确定删除 models/{name}？此操作不可恢复（之后可以重新导入）。",
    "imp.imported": "已导入 models/{name}（{size} MB）",
    "imp.partial": "文件已收到，但 models/{name} 不像完整模型（缺 config.json 或权重文件）",
    "imp.deleted": "已删除 models/{name}",
    "imp.baseFilled": "已设置基座 models/{name}（来源=本地路径），去训练页开始吧",
    "imp.repoStarted": "下载任务已开始，进度见下方日志",
    "imp.needPath": "请填写模型文件夹路径",
    "imp.needRepo": "请填写仓库 ID",
    "imp.needName": "请填写模型名称（字母/数字/._-）",
    "imp.needFiles": "请先选择要上传的文件",
    "imp.repoDone": "下载完成：models/{name}",
    "imp.repoErr": "下载失败，请查看日志末尾",
    "imp.logTitle": "下载日志",

    "train.config": "训练配置",
    "train.model": "模型",
    "train.source": "来源",
    "src.modelscope": "魔搭 ModelScope（国内推荐）",
    "src.huggingface": "Hugging Face（Llama 需门控授权）",
    "src.local": "本地路径（已导入的模型）",
    "train.base": "基座模型",
    "train.or": "或",
    "train.modelPh": "自定义短名 / 仓库ID / 本地路径",
    "train.cache": "模型缓存目录",
    "train.onlineGroup": "在线短名（自动下载）",
    "train.localGroup": "本地模型（models/ 目录，来自导入）",
    "train.dataset": "数据集",
    "train.dsAll": "合并 datasets 目录全部 jsonl",
    "train.dsSingle": "只用下面选中的一个文件",
    "train.file": "文件",
    "train.lora": "LoRA 参数",
    "train.rank": "秩 r",
    "train.alpha": "alpha",
    "train.dropout": "dropout",
    "train.hyper": "训练超参",
    "train.bs": "batch",
    "train.ga": "梯度累积",
    "train.seq": "最大序列长",
    "train.lr": "学习率",
    "train.ep": "轮数 epochs",
    "train.steps": "固定步数(-1)",
    "train.output": "输出与快照",
    "train.outDir": "输出目录",
    "train.save": "每 N 步保存",
    "train.limit": "保留快照数",
    "train.resume": "断点续训（从最新快照继续）",
    "train.force": "硬件不足仍强制启动",
    "train.start": "🚀 开始训练",
    "train.stop": "⏹ 停止训练",
    "train.reset": "重置为推荐值",
    "train.cmdPreview": "等价命令行：",
    "train.lossTitle": "Loss 曲线",
    "train.logTitle": "实时日志",
    "train.errToast": "训练进程出错，请查看日志末尾定位问题",
    "train.nextTip": "训练完成后，产出会出现在「产出管理」；想看看模型变得多聪明？去「💬 模型对话」挂上适配器问它几个问题。",
    "train.gotoChat": "去对话 →",

    "chat.title": "和你的模型对话",
    "chat.intro": "两种用法，都在这里：① 训练好的模型——选模型 + 选适配器；② 刚导入、没训练过的模型——只选模型，适配器选「不使用」。",
    "chat.step1": "第一步 · 选模型",
    "chat.modelLabel": "模型",
    "chat.localGroup": "本地模型（models/ 目录）",
    "chat.onlineGroup": "在线模型（自动下载）",
    "chat.ollamaOpt": "Ollama 本地服务",
    "chat.manualOpt": "✏️ 手动输入路径…",
    "chat.noLocal": "（models/ 目录还没有模型 — 去「模型导入」添加）",
    "chat.tipLocal": "该模型来自 models/ 目录（你导入的）。<strong>没训练过也能直接对话</strong> — 适配器选「不使用」即可。",
    "chat.tipManual": "请在下方「高级设置」里填写模型文件夹路径。",
    "chat.tipOnline": "在线模型会在首次对话时自动下载到 models/ 目录（之后就是本地模型）。",
    "chat.tipOllama": "使用本机 Ollama 服务（需另装 Ollama 并拉取模型），不占用本项目的显存。",
    "chat.step2": "第二步 · 选适配器（可选）",
    "chat.adapterLabel": "LoRA 适配器",
    "chat.noAdapter": "不使用适配器（原版模型直接对话）",
    "chat.adapterGroup": "训练产出",
    "chat.manualAdapterOpt": "✏️ 手动输入路径…",
    "chat.noAdapters": "（还没有训练产出 — 没训练过就选「不使用适配器」）",
    "chat.tipAdapter": "💡 适配器是微调产生的「补丁」，只有训练过的模型才需要。",
    "chat.advanced": "高级设置（手动路径 / Ollama）",
    "chat.manualModel": "模型路径（手动）",
    "agent.modelPathPh": "./models/Qwen2.5-1.5B-Instruct",
    "chat.manualAdapter": "适配器路径（手动）",
    "chat.ollamaUrl": "Ollama 服务地址",
    "chat.ollamaModel": "Ollama 模型名",
    "chat.task": "任务内容",
    "chat.taskPh": "例如：用三句话介绍杭州\n或：帮我把 /etc/hosts 读一下（Agent 会调用文件工具）",
    "chat.start": "▶ 开始对话",
    "chat.stop": "⏹ 停止",
    "chat.stopping": "正在停止…",
    "chat.needTask": "请先填写任务内容",
    "chat.needModel": "请先选择模型",
    "chat.genParams": "生成参数",
    "chat.tokens": "单步生成上限",
    "chat.steps": "Agent 最大步数",
    "chat.logTitle": "运行日志",

    "data.title": "数据集文件",
    "data.colFile": "文件",
    "data.colCount": "条数",
    "data.colSize": "大小",
    "data.colOps": "操作",
    "data.preview": "预览",
    "data.empty": "datasets 目录下没有 jsonl 文件",
    "data.previewTitle": "预览：",
    "data.loading": "加载中…",
    "data.emptyFile": "空文件",
    "data.previewFail": "预览失败：",
    "data.rowsSuffix": " 条",

    "out.title": "训练产出（LoRA 适配器 / 快照）",
    "out.hint": "训练产生的适配器与 checkpoint 会列在这里。点「用作适配器」可快速填入模型对话页。",
    "out.useAdapter": "用作适配器",
    "out.empty": "暂无产出。训练完成后，适配器会出现在 lora_output/ 目录。",
    "out.loadFail": "加载失败：",
    "out.filled": "已填入适配器路径：{path}",

    "job.idle": "空闲",
    "job.running": "运行中",
    "job.done": "已完成",
    "job.error": "出错",
    "job.stopped": "已停止",
    "job.stopping": "停止中…",
    "job.metaPid": "PID {pid}",
    "job.metaSteps": "共 {n} 步",

    "chart.empty": "开始训练后显示 loss 变化",
    "chart.current": "当前",
    "chart.min": "最低",

    "tiny.done": "微型模型已生成，训练和对话都能选到它",
    "job.tiny": "微型模型",

    "err.path_required": "请填写模型文件夹的绝对路径",
    "err.path_not_found": "路径不存在或不是文件夹",
    "err.bad_name": "名称只能包含字母、数字、._-",
    "err.not_a_model": "该文件夹不是 HuggingFace 格式模型（需要 config.json + .safetensors/.bin 权重）",
    "err.already_exists": "同名目录已存在（勾选「覆盖」可重导）",
    "err.copy_failed": "拷贝失败",
    "err.no_files": "没有收到文件",
    "err.bad_zip": "压缩包中含有非法路径",
    "err.upload_failed": "上传失败",
    "err.repo_required": "请填写仓库 ID（如 Qwen/Qwen2.5-1.5B-Instruct）",
  },

  en: {
    "app.title": "smolagent-lora-lab · Web Fine-Tuning Workbench",
    "app.subtitle": "Web Fine-Tuning Workbench · QLoRA / LoRA training + model chat",
    "app.envDetecting": "Detecting environment…",
    "app.footer": "smolagent-lora-lab · Local fine-tuning workbench — all data stays on your machine, nothing is uploaded.",
    "app.adopted": "★ Reconnected to running job(s): {kinds} (the server restarted; the job never stopped)",

    "tab.env": "🩺 Environment",
    "tab.import": "📥 Import Models",
    "tab.train": "🚀 Training",
    "tab.chat": "💬 Chat with Model",
    "tab.data": "📚 Datasets",
    "tab.out": "📦 Outputs",

    "nav.chat": "Chat",
    "nav.train": "Training",
    "nav.import": "Import Models",
    "nav.data": "Datasets",
    "nav.out": "Outputs",
    "nav.env": "Environment",
    "chat.paramsBtn": "⚙ Parameters",
    "chat.send": "Send",
    "chat.emptyHint": "Type a task below and press Enter to start chatting",
    "imp.searchPh": "Search imported models…",

    "env.title": "Environment",
    "env.deps": "Dependencies",
    "env.quick": "Quick self-test (works offline / without a GPU)",
    "env.tinyHint": "Generate a tiny random-weight Qwen model (~10MB, real tokenizer) to walk the full pipeline — train → save adapter → chat — locally, without downloading GBs of weights. It lands in models/ and is for pipeline testing only; do not use it for real conversations.",
    "env.btnTiny": "Generate Tiny Model",
    "env.py": "Python",
    "env.os": "OS",
    "env.device": "Device",
    "env.project": "Project root",
    "env.cpu": "CPU (no CUDA GPU)",
    "env.gpu": "CUDA GPU ({name}, {vram} GB)",
    "dep.torch": "PyTorch",
    "dep.transformers": "Transformers",
    "dep.peft": "PEFT",
    "dep.trl": "TRL",
    "dep.datasets": "Datasets",
    "dep.smolagents": "smolagents",
    "dep.modelscope": "ModelScope",
    "dep.accelerate": "Accelerate",
    "dep.flask": "Flask (this workbench)",
    "dep.missing": "not installed",
    "env.detectFail": "Environment check failed",

    "imp.title": "Import External Models",
    "imp.intro": "Add models you already have — training and chat can both pick them up right away. Three ways, use whichever fits:",
    "imp.wayPath": "Option 1 · Local folder",
    "imp.wayPathHint": "Enter the absolute path to a model folder on the machine running this server; it is copied into the project's models/ directory. Best when the model is already on your disk.",
    "imp.pathLabel": "Absolute path to model folder",
    "imp.nameLabel": "Directory name (optional)",
    "imp.nameLabelReq": "Directory name (required)",
    "imp.namePh": "Defaults to the folder name",
    "imp.overwrite": "Overwrite an existing directory with the same name",
    "imp.btnPath": "Import to models/",
    "imp.wayRepo": "Option 2 · Hugging Face / ModelScope repo",
    "imp.wayRepoHint": "Enter a repo ID to download straight into models/. ModelScope is faster in China; gated models like Llama require accepting terms on HF and running hf auth login. Built-in short names (e.g. qwen2.5-1.5b) also work.",
    "imp.repoSource": "Source",
    "imp.repoLabel": "Repo ID",
    "imp.repoPh": "Qwen/Qwen2.5-1.5B-Instruct or qwen2.5-1.5b (built-in short name)",
    "imp.btnRepo": "Start Download",
    "imp.wayUpload": "Option 3 · Browser upload",
    "imp.wayUploadHint": "Great for tablets/phones: upload model files directly (multi-select, or a single zip); they land in models/ automatically.",
    "imp.uploadLabel": "Choose files (multi-select / zip)",
    "imp.btnUpload": "Upload & Import",
    "imp.uploading": "Uploading…",
    "imp.listTitle": "Imported Models",
    "imp.listHint": "These live in models/ and can be picked directly for training or chat.",
    "imp.colName": "Directory",
    "imp.colArch": "Architecture",
    "imp.colWeights": "Weights",
    "imp.colValid": "Format check",
    "imp.colSize": "Size",
    "imp.colTime": "Imported",
    "imp.useBase": "Use as base",
    "imp.valid": "✓ Valid",
    "imp.invalid": "✗ Incomplete",
    "imp.noTok": "(no tokenizer)",
    "imp.empty": "No models imported yet. Add one with any of the three options above, or generate a tiny test model on the Environment tab.",
    "imp.noMatch": "No matching models",
    "imp.loadFail": "Load failed: ",
    "imp.confirmDelete": "Delete models/{name}? This cannot be undone (you can re-import later).",
    "imp.imported": "Imported to models/{name} ({size} MB)",
    "imp.partial": "Files received, but models/{name} is not a complete model (missing config.json or weights)",
    "imp.deleted": "Deleted models/{name}",
    "imp.baseFilled": "Base model set to models/{name} (source=local) — head to the training tab",
    "imp.repoStarted": "Download started — see the log below",
    "imp.needPath": "Please enter the model folder path",
    "imp.needRepo": "Please enter a repo ID",
    "imp.needName": "Please enter a model name (letters/digits/._-)",
    "imp.needFiles": "Please choose files to upload first",
    "imp.repoDone": "Download complete: models/{name}",
    "imp.repoErr": "Download failed — check the end of the log",
    "imp.logTitle": "Download Log",

    "train.config": "Training Configuration",
    "train.model": "Model",
    "train.source": "Source",
    "src.modelscope": "ModelScope (recommended in China)",
    "src.huggingface": "Hugging Face (Llama needs gated access)",
    "src.local": "Local path (imported models)",
    "train.base": "Base model",
    "train.or": "or",
    "train.modelPh": "short name / repo ID / local path",
    "train.cache": "Model cache dir",
    "train.onlineGroup": "Online short names (auto-download)",
    "train.localGroup": "Local models (models/, from imports)",
    "train.dataset": "Dataset",
    "train.dsAll": "Merge all jsonl files in datasets/",
    "train.dsSingle": "Use only the selected file below",
    "train.file": "File",
    "train.lora": "LoRA Parameters",
    "train.rank": "Rank r",
    "train.alpha": "alpha",
    "train.dropout": "dropout",
    "train.hyper": "Hyperparameters",
    "train.bs": "batch",
    "train.ga": "grad accum",
    "train.seq": "max seq len",
    "train.lr": "learning rate",
    "train.ep": "epochs",
    "train.steps": "max steps (-1)",
    "train.output": "Output & Snapshots",
    "train.outDir": "output dir",
    "train.save": "save every N steps",
    "train.limit": "keep snapshots",
    "train.resume": "Resume from the latest snapshot",
    "train.force": "Force start even if hardware is insufficient",
    "train.start": "🚀 Start Training",
    "train.stop": "⏹ Stop Training",
    "train.reset": "Reset to defaults",
    "train.cmdPreview": "Equivalent command: ",
    "train.lossTitle": "Loss Curve",
    "train.logTitle": "Live Log",
    "train.errToast": "Training failed — check the end of the log for the cause",
    "train.nextTip": "After training, outputs appear under Outputs; to see how much smarter your model got, open Chat with Model and attach the adapter.",
    "train.gotoChat": "Go to Chat →",

    "chat.title": "Chat with Your Model",
    "chat.intro": "Two use cases, both here: ① A fine-tuned model — pick a model + an adapter; ② A freshly imported, untrained model — pick a model only, adapter = No adapter.",
    "chat.step1": "Step 1 · Pick a model",
    "chat.modelLabel": "Model",
    "chat.localGroup": "Local models (models/)",
    "chat.onlineGroup": "Online models (auto-download)",
    "chat.ollamaOpt": "Local Ollama server",
    "chat.manualOpt": "✏️ Enter a path manually…",
    "chat.noLocal": "(no models in models/ yet — add some in Import Models)",
    "chat.tipLocal": "This model lives in your models/ directory. <strong>Untrained models work too</strong> — just leave the adapter at No adapter.",
    "chat.tipManual": "Fill in the model folder path under Advanced settings below.",
    "chat.tipOnline": "Online models download to models/ on first use (after that they count as local).",
    "chat.tipOllama": "Uses your local Ollama server (install Ollama and pull a model separately); it does not use this project's VRAM.",
    "chat.step2": "Step 2 · Pick an adapter (optional)",
    "chat.adapterLabel": "LoRA adapter",
    "chat.noAdapter": "No adapter (chat with the base model)",
    "chat.adapterGroup": "Training outputs",
    "chat.manualAdapterOpt": "✏️ Enter a path manually…",
    "chat.noAdapters": "(no training outputs yet — untrained? choose No adapter)",
    "chat.tipAdapter": "💡 An adapter is the \"patch\" produced by fine-tuning — only fine-tuned models need one.",
    "chat.advanced": "Advanced (manual paths / Ollama)",
    "chat.manualModel": "Model path (manual)",
    "agent.modelPathPh": "./models/Qwen2.5-1.5B-Instruct",
    "chat.manualAdapter": "Adapter path (manual)",
    "chat.ollamaUrl": "Ollama service URL",
    "chat.ollamaModel": "Ollama model name",
    "chat.task": "Task",
    "chat.taskPh": "e.g. Introduce Hangzhou in three sentences\nOr: read /etc/hosts for me (the agent will use file tools)",
    "chat.start": "▶ Start Chat",
    "chat.stop": "⏹ Stop",
    "chat.stopping": "Stopping…",
    "chat.needTask": "Please enter a task first",
    "chat.needModel": "Please pick a model first",
    "chat.genParams": "Generation",
    "chat.tokens": "max new tokens",
    "chat.steps": "max agent steps",
    "chat.logTitle": "Run Log",

    "data.title": "Dataset Files",
    "data.colFile": "File",
    "data.colCount": "Rows",
    "data.colSize": "Size",
    "data.colOps": "Actions",
    "data.preview": "Preview",
    "data.empty": "No jsonl files in datasets/",
    "data.previewTitle": "Preview: ",
    "data.loading": "Loading…",
    "data.emptyFile": "Empty file",
    "data.previewFail": "Preview failed: ",
    "data.rowsSuffix": " rows",

    "out.title": "Training Outputs (LoRA Adapters / Snapshots)",
    "out.hint": "Adapters and checkpoints from your runs appear here. Click Use as adapter to fill the chat form instantly.",
    "out.useAdapter": "Use as adapter",
    "out.empty": "No outputs yet. After a training run, adapters show up in lora_output/.",
    "out.loadFail": "Load failed: ",
    "out.filled": "Adapter path filled: {path}",

    "job.idle": "Idle",
    "job.running": "Running",
    "job.done": "Done",
    "job.error": "Error",
    "job.stopped": "Stopped",
    "job.stopping": "Stopping…",
    "job.metaPid": "PID {pid}",
    "job.metaSteps": "{n} steps total",

    "chart.empty": "Loss curve appears after training starts",
    "chart.current": "current",
    "chart.min": "min",

    "tiny.done": "Tiny model generated — it is now selectable for training and chat",
    "job.tiny": "Tiny model",

    "err.path_required": "Please enter the absolute path to the model folder",
    "err.path_not_found": "Path does not exist or is not a folder",
    "err.bad_name": "Name may only contain letters, digits, . _ -",
    "err.not_a_model": "That folder is not a HuggingFace-format model (needs config.json + .safetensors/.bin weights)",
    "err.already_exists": "A directory with that name exists (check Overwrite to re-import)",
    "err.copy_failed": "Copy failed",
    "err.no_files": "No files received",
    "err.bad_zip": "The zip contains illegal paths",
    "err.upload_failed": "Upload failed",
    "err.repo_required": "Please enter a repo ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
  },
};

/* ---------------- 核心 ---------------- */
function getLang() {
  return localStorage.getItem("lora-lab-lang-v1") || "zh";
}

function detectLang() {
  const l = (navigator.language || "zh").toLowerCase();
  return l.startsWith("zh") ? "zh" : "en";
}

function t(key, vars) {
  const lang = getLang();
  let s = (DICT[lang] && DICT[lang][key]) ?? (DICT.zh[key]) ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replaceAll(`{${k}}`, v);
    }
  }
  return s;
}

/** 把后端错误消息翻译成当前语言（有 error_code 走词典，否则原样） */
function tErr(data) {
  if (data && data.error_code && DICT[getLang()][`err.${data.error_code}`]) {
    return t(`err.${data.error_code}`);
  }
  return (data && data.error) || "Unknown error";
}

const _langListeners = [];
function onLangChange(fn) { _langListeners.push(fn); }

function setLang(lang) {
  if (!DICT[lang]) lang = "zh";
  localStorage.setItem("lora-lab-lang-v1", lang);
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  applyStatic();
  updateLangButton();
  for (const fn of _langListeners) fn();
}

function updateLangButton() {
  const btn = document.getElementById("btn-lang");
  if (btn) btn.textContent = getLang() === "zh" ? "🌐 EN" : "🌐 中文";
}

/** 扫描 data-i18n / data-i18n-ph / data-i18n-title 刷新静态文本 */
function applyStatic() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    // 含子元素的(如包着 input 的 label)不处理——文本应包在 span 里，
    // 直接设 textContent 会把输入框等子元素清掉。
    if (el.children.length === 0) el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPh);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
  });
  document.title = t("app.title");
}

/** 初始化：首访按浏览器语言，之后随时可切 */
function initLang() {
  if (!localStorage.getItem("lora-lab-lang-v1")) {
    localStorage.setItem("lora-lab-lang-v1", detectLang());
  }
  document.documentElement.lang = getLang() === "zh" ? "zh-CN" : "en";
  const btn = document.getElementById("btn-lang");
  if (btn) {
    btn.addEventListener("click", () => setLang(getLang() === "zh" ? "en" : "zh"));
  }
  applyStatic();
  updateLangButton();
}
