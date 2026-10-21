#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora.py — QLoRA LoRA 微调训练器 / QLoRA LoRA Fine-tuning Trainer
=====================================================================

中文:
  基于 HuggingFace TRL SFTTrainer 的 QLoRA(4bit 量化) LoRA 微调脚本。
  只训练 LoRA 适配器（约占基座 1% 参数量），不做全参数训练。

  功能特性:
    1. 多尺寸基座: 2B / 3B / 4B / 7B / 14B 均可训练
    2. 双模型源切换: modelscope(魔搭, 国内默认) / huggingface(HF Hub, 海外)
       —— 内置模型短名映射表, 同一个短名自动路由到对应平台仓库
    3. 两种基座来源: 在线拉取(魔搭/HF) 或 本地文件夹加载
    4. 自动设备识别: NVIDIA CUDA → GPU 高速训练; 无 GPU → CPU 模式
       (平板 AidLearning ARM64 仅稳定支持 2B/4B; 选 7B/14B 会阻止启动)
    5. 断点续训: 定时自动保存快照, 中断后可从上次进度继续, 不用从头再来
    6. 数据集: 指定单个 jsonl, 或扫描 datasets 目录全部 jsonl 自动合并去重

  框架边界说明(重要):
    · 训练循环/日志/断点保存由 [框架能力] HuggingFace TRL SFTTrainer 提供;
    · 标记 [自研改造点·阶段二] 的部分, 是脱离第三方框架后需要自己
      重新实现的逻辑(详见 dev_note.md)。

English:
  QLoRA (4-bit quantized) LoRA fine-tuning script built on HuggingFace TRL
  SFTTrainer. Only the LoRA adapter is trained, never the full weights.

  Dual model sources: ModelScope (default, fast in China) / Hugging Face Hub.
  A built-in registry maps short model names to per-platform repo IDs.
  Everything runs locally; the only network access is model downloading.

用法 / Usage:
  # 列出内置模型短名
  python train_lora.py --list-models

  # 最常用: 魔搭拉取 Qwen2.5-7B, 合并训练 datasets 下全部数据集
  python train_lora.py --model qwen2.5-7b

  # 使用本地已下载的基座 + 单个数据集
  python train_lora.py --model-source local --model ./models/Qwen2.5-7B-Instruct \\
      --dataset-mode single --dataset-file datasets/dataset_daily.jsonl

  # 海外用户切换 HF 源
  python train_lora.py --model qwen2.5-7b --model-source huggingface

  # 断点续训(检测到快照时询问; 挂机建议显式指定)
  python train_lora.py --model qwen2.5-7b --resume
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import sys
import time

# ============================================================
# 集中配置区（新手只需要改这里）/ Central config (edit me)
# ============================================================
CONFIG = {
    # —— 模型来源 ——
    "model_source": "modelscope",     # modelscope(魔搭,国内) | huggingface(HF) | local(本地)
    "model": "qwen2.5-7b",            # 内置短名(见 MODEL_REGISTRY)或完整仓库ID; local 模式下为本地路径
    "model_cache_dir": "./models",    # 在线拉取模型的本地缓存目录

    # —— 数据集 ——
    "dataset_mode": "all",            # all=合并 datasets 目录全部 jsonl | single=只用一个文件
    "dataset_file": "datasets/dataset_daily.jsonl",
    "datasets_dir": "datasets",

    # —— LoRA 参数 ——
    "lora_rank": 16,                  # LoRA 秩 r: 越大学习能力越强、越占显存(常用 8/16/32/64)
    "lora_alpha": 32,                 # LoRA 缩放 alpha: 常用设为 rank 的 2 倍
    "lora_dropout": 0.05,             # LoRA dropout, 防止过拟合

    # —— 训练超参 ——
    "batch_size": 2,                  # 每卡批量(CPU 模式会自动降为 1)
    "grad_accum": 8,                  # 梯度累积: 等效批量 = batch_size × grad_accum
    "learning_rate": 2e-4,            # 学习率(LoRA 可用比全参数大得多的学习率)
    "num_epochs": 3,                  # 训练轮数
    "max_steps": -1,                  # 最大步数: -1=按 epochs 计算; >0 则固定步数(与 epochs 取先到者)
    "max_seq_len": 1024,              # 最大序列长度(CPU 模式默认降为 512)
    "seed": 42,

    # —— 输出与快照 ——
    "output_dir": "./lora_output",    # LoRA 适配器输出目录
    "save_steps": 100,                # 每多少步保存一次快照(断点续训靠它)
    "save_total_limit": 3,            # 最多保留几个快照(防磁盘占满)
    "resume": None,                   # 断点续训: True=--resume / False=--no-resume / None=检测到快照时询问
}

# ============================================================
# 模型短名映射表 / Model registry (short name → repo id per source)
# ============================================================
# 【自研改造点·阶段二】此表是纯粹的"加载层"配置, 与训练逻辑完全解耦:
#   换框架(脱离 trl)时这张表可以原样保留, 只需要改写 download_model()。
# 同一个短名, 按 --model-source 自动路由到对应平台, 用户无需手改长链接。
MODEL_REGISTRY = {
    # —— Qwen2.5 系列(最常用的中文基座)——
    "qwen2.5-1.5b": {"modelscope": "qwen/Qwen2.5-1.5B-Instruct", "huggingface": "Qwen/Qwen2.5-1.5B-Instruct"},
    "qwen2.5-3b":   {"modelscope": "qwen/Qwen2.5-3B-Instruct",   "huggingface": "Qwen/Qwen2.5-3B-Instruct"},
    "qwen2.5-7b":   {"modelscope": "qwen/Qwen2.5-7B-Instruct",   "huggingface": "Qwen/Qwen2.5-7B-Instruct"},
    "qwen2.5-14b":  {"modelscope": "qwen/Qwen2.5-14B-Instruct",  "huggingface": "Qwen/Qwen2.5-14B-Instruct"},
    # —— Qwen3 系列(新版, 思考/非思考双模式)——
    "qwen3-1.7b":   {"modelscope": "qwen/Qwen3-1.7B",            "huggingface": "Qwen/Qwen3-1.7B"},
    "qwen3-4b":     {"modelscope": "qwen/Qwen3-4B",              "huggingface": "Qwen/Qwen3-4B"},
    "qwen3-8b":     {"modelscope": "qwen/Qwen3-8B",              "huggingface": "Qwen/Qwen3-8B"},
    "qwen3-14b":    {"modelscope": "qwen/Qwen3-14B",             "huggingface": "Qwen/Qwen3-14B"},
    # —— GLM / Yi / InternLM(中英文均可)——
    "glm4-9b-chat":       {"modelscope": "ZhipuAI/glm-4-9b-chat",        "huggingface": "THUDM/glm-4-9b-chat"},
    "yi-1.5-6b-chat":     {"modelscope": "01ai/Yi-1.5-6B-Chat",          "huggingface": "01-ai/Yi-1.5-6B-Chat"},
    "yi-1.5-9b-chat":     {"modelscope": "01ai/Yi-1.5-9B-Chat",          "huggingface": "01-ai/Yi-1.5-9B-Chat"},
    "internlm2.5-7b-chat": {"modelscope": "Shanghai_AI_Laboratory/internlm2_5-7b-chat", "huggingface": "internlm/internlm2_5-7b-chat"},
    # —— Llama 系列(注意: HF 上为门控模型, 需提前同意协议!)——
    "llama3.1-8b-instruct": {"modelscope": "LLM-Research/Meta-Llama-3.1-8B-Instruct", "huggingface": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
}

# ============================================================
# 硬件需求表 / Hardware requirements
# ============================================================
# 4bit QLoRA 训练的大致显存/内存需求(GB), 含优化器与激活值余量。
MODEL_VRAM_REQ_GB = {"1.5": 6, "2": 6, "3": 8, "4": 12, "7": 18, "8": 20, "9": 20, "14": 30}
# CPU(尤其 ARM64 平板)仅稳定支持的尺寸; 7B/14B 直接阻止启动, 避免 OOM 崩溃。
CPU_SAFE_SIZES = {"1.5", "2", "3", "4"}

BANNER = """
============================================================
  train_lora.py — QLoRA LoRA 微调训练器
  模型源: {source}    设备: {device}    架构: {arch}
============================================================"""


# ============================================================
# 硬件检测 / Hardware detection
# ============================================================
def detect_hardware():
    """检测运行环境: CUDA显卡/显存, 内存, CPU架构。"""
    hw = {
        "device": "cpu", "gpu_name": None, "vram_gb": 0.0,
        "ram_gb": 0.0, "arch": platform.machine().lower(),
        "is_arm": False, "cuda_ver": None,
    }
    hw["is_arm"] = hw["arch"] in ("aarch64", "arm64", "armv8l", "armv7l")
    try:
        import torch
        if torch.cuda.is_available():
            hw["device"] = "cuda"
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            hw["cuda_ver"] = torch.version.cuda
    except Exception as e:  # torch 未安装时给出可读信息
        print(f"[错误] 无法导入 PyTorch: {e}\n请先安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    # 内存检测: psutil 优先, 回退 /proc/meminfo(Linux)
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
            hw["ram_gb"] = round(kb / 1e6, 1)
        except Exception:
            hw["ram_gb"] = -1
    return hw


def parse_model_size(model_name):
    """从模型名解析参数量(B)。取所有 '数字+B' 匹配中的最大值。
    例: Qwen2.5-7B-Instruct → 7; Qwen3-30B-A3B → 30; qwen2.5-7b → 7。"""
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if not nums:
        return None
    return max(float(n) for n in nums)


def check_hardware(hw, model_name, size_b):
    """硬件充足性检查。返回 (size_key, proceed)。"""
    size_key = None
    if size_b is not None:
        # 归并到需求表的键(1.7→2, 6→7 这类就近归并)
        size_key = str(int(size_b)) if size_b >= 1 else str(size_b)
        if size_key not in MODEL_VRAM_REQ_GB:
            near = min(MODEL_VRAM_REQ_GB, key=lambda k: abs(float(k) - size_b))
            size_key = near

    avail = hw["vram_gb"] if hw["device"] == "cuda" else hw["ram_gb"]
    avail_label = "显存" if hw["device"] == "cuda" else "内存"

    if hw["device"] == "cpu":
        if size_key and size_key not in CPU_SAFE_SIZES:
            print(f"""
[阻止启动] 当前为 CPU 环境({hw['arch']}), 仅稳定支持 2B/4B 模型。
  你选择的模型约 {size_b}B, 在 CPU 上训练会因内存不足崩溃或慢到不可用。
  建议: 换用 qwen2.5-1.5b / qwen2.5-3b / qwen3-1.7b / qwen3-4b,
        或到 NVIDIA 显卡电脑上训练 7B/14B。""")
            return size_key, False
        if hw["is_arm"]:
            print(f"[提示] ARM64 CPU 模式({hw['arch']}): 仅支持 ≤4B; 4bit 量化不可用, "
                  f"将以 LoRA(fp32) 方式训练, 速度较慢, 适合验证流程。")
        else:
            print(f"[提示] CPU 模式: 4bit 量化(bitsandbytes)不可用, 将以 LoRA(fp32) 方式训练。")

    if hw["device"] == "cuda" and size_key:
        need = MODEL_VRAM_REQ_GB[size_key]
        if avail > 0 and avail < need:
            print(f"""
[硬件警告] {avail_label} {avail}GB 可能不足以训练约 {size_b}B 模型(建议 ≥{need}GB)。
  可能遇到 CUDA OOM。建议: 换更小模型 / 减小 batch_size / 开启 gradient_accumulation,
  确认要继续请加 --force 参数。""")
            return size_key, "need_force"
        print(f"[硬件] {hw['gpu_name']} | {avail_label} {avail}GB | CUDA {hw['cuda_ver']}")
    return size_key, True


# ============================================================
# 模型加载层 / Model loading layer
# ============================================================
def resolve_repo_id(model, source):
    """把用户输入(短名或完整ID)解析成当前源上的仓库ID。
    【加载层】只负责"去哪里下载", 不接触任何训练逻辑。"""
    key = model.strip().lower()
    # 1) 命中内置映射表 → 按源路由
    if key in MODEL_REGISTRY:
        repo = MODEL_REGISTRY[key].get(source)
        if repo:
            return repo
        raise SystemExit(f"[错误] 模型 '{model}' 在 {source} 源上没有登记, 可选源: "
                         f"{list(MODEL_REGISTRY[key].keys())}")
    # 2) 完整仓库ID(含 /) → 按源原样使用
    if "/" in model:
        print(f"[加载层] 使用自定义仓库ID: {model} (源: {source})")
        return model
    # 3) 无法识别
    raise SystemExit(f"[错误] 无法识别模型 '{model}'。\n"
                     f"请使用 --list-models 查看内置短名, 或传完整仓库ID(如 Qwen/Qwen2.5-7B-Instruct)。")


def download_model(model, source, cache_dir):
    """双模型源下载。modelscope: 国内快; huggingface: 海外。
    返回本地模型目录路径。魔搭/HF 只负责下载权重, 训练推理全部在本机。"""
    repo_id = resolve_repo_id(model, source)
    os.makedirs(cache_dir, exist_ok=True)
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError:
            raise SystemExit("[错误] 未安装 modelscope: pip install modelscope")
        print(f"[魔搭 ModelScope] 正在拉取 {repo_id} → {cache_dir}")
        # 国内源, 速度快; 断点续传由 modelscope 自动处理
        return snapshot_download(repo_id, cache_dir=cache_dir)
    if source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("[错误] 未安装 huggingface_hub: pip install huggingface_hub")
        print(f"[Hugging Face Hub] 正在拉取 {repo_id} → {cache_dir}")
        try:
            return snapshot_download(repo_id, cache_dir=cache_dir)
        except Exception as e:
            msg = str(e)
            if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
                raise SystemExit(
                    f"[错误] {repo_id} 是 Hugging Face 门控模型: 需要先在该模型页面同意协议并绑定账号。\n"
                    f"  1) 浏览器打开 https://huggingface.co/{repo_id} 点击 Agree and access;\n"
                    f"  2) 创建 access token: https://huggingface.co/settings/tokens ;\n"
                    f"  3) 安装 CLI 后登录: pip install -U huggingface_hub && hf auth login ;\n"
                    f"本项目只提供加载能力, 不负责绕过权限。国内用户可改用 --model-source modelscope。")
            raise
    raise SystemExit(f"[错误] 未知模型源: {source}(可选 modelscope / huggingface / local)")


def resolve_base_model(cfg):
    """确定基座模型本地路径: local 直接用; 在线源走下载层。"""
    if cfg["model_source"] == "local":
        path = cfg["model"]
        if not os.path.isdir(path):
            raise SystemExit(f"[错误] 本地模型路径不存在: {path}")
        print(f"[本地] 加载基座: {path}")
        return path
    return download_model(cfg["model"], cfg["model_source"], cfg["model_cache_dir"])


def load_tokenizer(model_path):
    """加载分词器; 补齐 pad_token(缺失是训练崩溃的常见原因)。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("[提示] 分词器无 pad_token, 已使用 eos_token 代替。")
    return tok


def build_bnb_config(hw):
    """CUDA 下构造 4bit QLoRA 量化配置; CPU/ARM 返回 None(自动降级 LoRA fp32)。"""
    if hw["device"] != "cuda":
        return None
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        raise SystemExit(
            "[错误] CUDA 环境缺少 bitsandbytes(4bit 量化依赖)。\n"
            "  pip install bitsandbytes\n"
            "  注意: bitsandbytes 仅支持 NVIDIA CUDA, ARM 平板无法安装, 属正常现象。")
    import torch
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[QLoRA] 4bit 量化(nf4, 双重量化, compute={compute_dtype})")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def guess_target_modules(model):
    """自适应 LoRA 注入位置: 优先常见投影层, 模型结构不匹配时扫描实际模块名兜底。
    【框架能力】peft 提供 LoRA 注入; 这里是本项目的适配层。"""
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj",
                 "c_attn", "c_proj", "W_pack"]  # 覆盖 Qwen/Llama/GLM/ChatGLM/Yi
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    hit = [m for m in preferred if m in present]
    if hit:
        return hit
    # 兜底: 所有 Linear 层(排除 lm_head)
    hit = [name.split(".")[-1] for name, mod in model.named_modules()
           if type(mod).__name__ == "Linear" and "lm_head" not in name]
    return sorted(set(hit))[:8]


# ============================================================
# 数据集加载 / Dataset loading
# ============================================================
def _iter_jsonl(path):
    """逐行读 jsonl, 坏行跳过并告警(避免半路崩溃)。"""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] {path}:{lineno} JSON 解析失败, 已跳过: {e}")


def _validate_messages(obj, path, lineno):
    """校验单条样本必须是 {"messages":[{role,content}...]}。"""
    if not isinstance(obj, dict) or "messages" not in obj:
        return None
    msgs = obj["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return None
        if not isinstance(m["content"], str):
            return None
    return msgs


def load_dataset_rows(cfg):
    """按模式收集样本: single=单文件; all=扫描目录全部 jsonl 合并+去重。"""
    rows, seen, bad = [], set(), 0

    def add(msgs, src):
        fp = hashlib.md5(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if fp in seen:
            return
        seen.add(fp)
        rows.append({"messages": msgs, "_src": src})

    if cfg["dataset_mode"] == "single":
        path = cfg["dataset_file"]
        if not os.path.isfile(path):
            raise SystemExit(f"[错误] 数据集文件不存在: {path}")
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(cfg["datasets_dir"], "*.jsonl")))
        if not files:
            raise SystemExit(f"[错误] {cfg['datasets_dir']} 目录下没有 jsonl 数据集")

    for path in files:
        n0 = len(rows)
        for lineno, obj in _iter_jsonl(path):
            msgs = _validate_messages(obj, path, lineno)
            if msgs is None:
                bad += 1
                continue
            add(msgs, os.path.basename(path))
        print(f"[数据集] {os.path.basename(path)}: +{len(rows) - n0} 条")
    if bad:
        print(f"[警告] 共 {bad} 条样本格式不符已跳过(需要 messages 字段且至少 2 轮对话)。")
    if not rows:
        raise SystemExit("[错误] 没有可用训练样本。")
    print(f"[数据集] 合并后共 {len(rows)} 条(已去重)。")
    return rows


def make_formatting_func(tokenizer):
    """把 messages 列表渲染成训练文本: 优先用模型自带聊天模板,
    没有模板的模型回退到通用格式。【框架能力】SFTTrainer 按此函数取文本。"""
    has_template = bool(getattr(tokenizer, "chat_template", None))

    def fallback(msgs):
        parts = []
        for m in msgs:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def fmt(example):
        msgs = example["messages"]
        if has_template:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:
                pass
        return fallback(msgs)

    if not has_template:
        print("[提示] 该模型无聊天模板, 使用通用对话格式(可能影响微调效果)。")
    return fmt


# ============================================================
# 断点续训 / Checkpoint resume
# ============================================================
def find_checkpoints(output_dir):
    """扫描 output_dir 下 checkpoint-* 目录, 按步数排序返回。"""
    cks = []
    for p in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m:
            cks.append((int(m.group(1)), p))
    cks.sort()
    return cks


def decide_resume(cfg):
    """决定是否续训。返回 checkpoint 路径或 None。
    设计原则: 挂机场景绝不做隐晦默认 —— 有快照又没给参数时,
    TTY 下询问, 非 TTY 直接报错要求显式 --resume / --no-resume。"""
    cks = find_checkpoints(cfg["output_dir"])
    if not cks:
        return None
    latest_step, latest = cks[-1]
    print(f"[断点] 发现 {len(cks)} 个快照, 最新: {latest}(step {latest_step})")
    if cfg["resume"] is True:
        print(f"[断点] 按 --resume 从 step {latest_step} 继续训练。")
        return latest
    if cfg["resume"] is False:
        print("[断点] 按 --no-resume 开始全新训练(旧快照保留不删)。")
        return None
    if sys.stdin.isatty():
        ans = input(f"    检测到已有训练快照(step {latest_step}), 是否继续上次进度? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            print(f"[断点] 从 step {latest_step} 继续训练。")
            return latest
        print("[断点] 开始全新训练。")
        return None
    raise SystemExit(
        f"[错误] 检测到快照但未指定续训方式(非交互环境)。\n"
        f"  继续训练请加: --resume\n  重新开始请加: --no-resume")


# ============================================================
# 训练进度回调 / Progress callback (挂机友好: 实时 loss + 预计剩余)
# ============================================================
from transformers import TrainerCallback


class ProgressCallback(TrainerCallback):
    """【框架能力】TrainerCallback 机制由 transformers 提供;
    这里自定义输出格式: step / loss / lr / 已用 / 预计剩余。"""
    def __init__(self, total_steps):
        self.total = max(total_steps, 1)
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        print(f"[训练] 开始, 共 {self.total} 步。", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        el = time.time() - self.t0
        done = max(state.global_step, 1)
        eta = el / done * (self.total - done)
        print(f"[训练] step {state.global_step}/{self.total} "
              f"loss={logs['loss']:.4f} lr={logs.get('learning_rate', 0):.2e} "
              f"已用 {_fmt_dur(el)} 预计剩余 {_fmt_dur(eta)}", flush=True)


def _fmt_dur(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# 主流程 / Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="QLoRA LoRA 微调训练器(支持魔搭/HF 双模型源, 断点续训)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-source", choices=["modelscope", "huggingface", "local"], default=None,
                    help="模型源: modelscope(魔搭,国内默认) / huggingface(HF Hub) / local(本地)")
    ap.add_argument("--model", default=None, help="内置短名(见 --list-models)或完整仓库ID; local 模式下为本地路径")
    ap.add_argument("--model-cache-dir", default=None, help="模型下载缓存目录")
    ap.add_argument("--dataset-mode", choices=["all", "single"], default=None, help="all=合并全部 / single=单文件")
    ap.add_argument("--dataset-file", default=None, help="single 模式下的 jsonl 文件")
    ap.add_argument("--datasets-dir", default=None, help="all 模式下的数据集目录")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--num-epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="固定训练步数(-1=按 epochs)")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--save-total-limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=None, help="从最新快照继续训练")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="忽略快照, 开始全新训练")
    ap.add_argument("--force", action="store_true", help="硬件不足时仍强制继续")
    ap.add_argument("--list-models", action="store_true", help="列出内置模型短名映射表后退出")
    args = ap.parse_args()

    if args.list_models:
        print("内置模型短名映射表(按 --model-source 自动路由):\n")
        for short, m in MODEL_REGISTRY.items():
            print(f"  {short:22s} → 魔搭: {m['modelscope']}")
            print(f"  {'':22s}   HF: {m['huggingface']}")
        print("\n用法: python train_lora.py --model qwen2.5-7b [--model-source huggingface]")
        return

    # CLI 覆盖 CONFIG
    cfg = dict(CONFIG)
    for k in cfg:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    # ---- 1. 硬件检测 ----
    hw = detect_hardware()
    size_b = parse_model_size(cfg["model"])
    if size_b is None and cfg["model_source"] != "local":
        # 短名解析成仓库ID后再试一次(如 qwen2.5-7b → Qwen/Qwen2.5-7B-Instruct)
        size_b = parse_model_size(resolve_repo_id(cfg["model"], cfg["model_source"]))
    size_key, verdict = check_hardware(hw, cfg["model"], size_b)
    if verdict is False:
        sys.exit(2)
    if verdict == "need_force" and not args.force:
        sys.exit(2)

    # ---- 2. CPU 模式自动降参(防止 OOM 与漫长等待) ----
    if hw["device"] == "cpu":
        if cfg["batch_size"] > 1:
            print(f"[提示] CPU 模式 batch_size {cfg['batch_size']}→1")
            cfg["batch_size"] = 1
        if cfg["max_seq_len"] > 512:
            print(f"[提示] CPU 模式 max_seq_len {cfg['max_seq_len']}→512")
            cfg["max_seq_len"] = 512

    print(BANNER.format(source=cfg["model_source"], device=hw["device"], arch=hw["arch"]))
    print(f"[配置] 模型: {cfg['model']}(约 {size_b}B) | 源: {cfg['model_source']} | "
          f"LoRA r={cfg['lora_rank']} α={cfg['lora_alpha']} | "
          f"bs={cfg['batch_size']}× accum {cfg['grad_accum']} | lr={cfg['learning_rate']} | "
          f"epochs={cfg['num_epochs']} | max_len={cfg['max_seq_len']}")

    # ---- 3. 基座模型(加载层) ----
    model_path = resolve_base_model(cfg)
    tokenizer = load_tokenizer(model_path)

    # ---- 4. 数据集(合并去重) ----
    rows = load_dataset_rows(cfg)

    # ---- 5. 断点续训决策 ----
    resume_ckpt = decide_resume(cfg)

    # ---- 6. 构建模型 + LoRA ----
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig

    bnb_config = build_bnb_config(hw)
    load_kwargs = {"trust_remote_code": True}
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float32  # CPU/ARM 稳定优先(transformers>=4.56 字段名)

    print("[加载] 正在加载基座权重…")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # 【框架能力】peft: kbit 训练准备(梯度检查点兼容 + 层冻结)
    if bnb_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.enable_input_require_grads()  # 防 "tensor does not require grad" 崩溃

    target_modules = guess_target_modules(model)
    print(f"[LoRA] 注入位置: {target_modules}")
    lora_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )

    # ---- 7. Trainer(框架能力: TRL SFTTrainer) ----
    from trl import SFTConfig, SFTTrainer

    use_bf16 = hw["device"] == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = hw["device"] == "cuda" and not use_bf16
    sft_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        logging_steps=1,
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        max_length=cfg["max_seq_len"],          # trl>=1.0 字段名(旧版为 max_seq_length)
        packing=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to=[],
        seed=cfg["seed"],
        optim="paged_adamw_8bit" if bnb_config is not None else "adamw_torch",
    )
    if cfg["max_steps"] and cfg["max_steps"] > 0:   # -1=按 epochs; >0 固定步数(先到者生效)
        sft_kwargs["max_steps"] = cfg["max_steps"]
    sft_config = SFTConfig(**sft_kwargs)
    # trl 1.16: formatting_func 走 SFTTrainer 构造参数, 不在 SFTConfig 里
    # 注意: train_dataset 必须是 datasets.Dataset, 裸 list 会报 TypeError
    from datasets import Dataset
    train_dataset = Dataset.from_list(rows)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        formatting_func=make_formatting_func(tokenizer),
    )

    total_steps = (len(rows) // (cfg["batch_size"] * cfg["grad_accum"]) + 1) * int(cfg["num_epochs"])
    trainer.add_callback(ProgressCallback(total_steps))

    # ---- 8. 开练(支持断点续训 + Ctrl+C 安全保存) ----
    print(f"[训练] 启动! 等效批量 = {cfg['batch_size'] * cfg['grad_accum']}, "
          f"样本 {len(rows)} 条, 预计 {total_steps} 步/轮×{int(cfg['num_epochs'])}轮。")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C, 正在保存当前进度…")
        trainer.save_model(os.path.join(cfg["output_dir"], "checkpoint-interrupt"))
        print(f"[中断] 已保存到 {cfg['output_dir']}/checkpoint-interrupt。")
        print("       重新运行并加 --resume 即可从上次快照继续(自动跳过 interrupt 快照, 请用 checkpoint-* 命名)。")
        return

    # ---- 9. 收尾: 保存适配器 + 使用说明 ----
    out = cfg["output_dir"]
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"""
[完成] LoRA 适配器已保存到: {out}
  文件: adapter_config.json / adapter_model.safetensors / tokenizer 文件

  用本项目的 Agent 加载它:
    python agent.py --mode A --model qwen2.5-7b --adapter {out}
    python agent.py --mode C --model-source local --model-path <基座路径> --adapter {out}

  适配器只含增量权重(几十~几百 MB), 与基座搭配使用, 不要单独拷贝走。
  换机器/换源(modelscope↔huggingface)不影响适配器兼容性。""")


if __name__ == "__main__":
    main()
