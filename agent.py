#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — 本地智能体(LoRA 版) / Local Agent with LoRA adapter
=============================================================

中文:
  基于 smolagents CodeAgent 的本地智能体。支持三种模型接入模式:

    模式A  魔搭/HF 在线基座 + 我训练的 LoRA 适配器
           (模型源可切换 modelscope / huggingface, 见 train_lora.py 映射表)
    模式B  Ollama 本地服务(http://localhost:11434/v1), 兼容 GGUF 全尺寸模型
    模式C  Transformers 原生加载本地文件夹基座 + 我训练的 LoRA 适配器

  内置工具: read_file / write_file / list_folder(文件读写被限制在工作目录内,
  防止 Agent 误操作工作目录之外的文件)。工具扩展接口见文件末尾注释。
  长期记忆: 任务与对话记录保存在本地 json, 重启后自动读取并注入指令。

  框架边界说明(重要):
    · 思考-代码-观察循环由 [框架能力] smolagents CodeAgent 提供;
    · LoRA 适配器挂载(TransformersModel 不原生支持 peft)是
      [自研改造点·阶段二] 需要重新实现的部分, 详见 dev_note.md。

English:
  A local agent built on smolagents CodeAgent, with three model backends:
  online base (ModelScope/HF) + your LoRA adapter, Ollama, or a local folder
  base + LoRA adapter. Ships file tools sandboxed to a working directory and
  a simple JSON long-term memory that survives restarts.

用法 / Usage:
  # 模式A: 魔搭拉 Qwen2.5-7B + 加载训练好的 LoRA(最常用)
  python agent.py --mode A --model qwen2.5-7b --adapter ./lora_output

  # 模式A 海外源
  python agent.py --mode A --model qwen2.5-7b --model-source huggingface --adapter ./lora_output

  # 模式B: Ollama(先确保 ollama serve 已启动)
  python agent.py --mode B --ollama-model qwen2.5:7b-instruct

  # 模式C: 纯本地(完全离线)
  python agent.py --mode C --model-path ./models/Qwen2.5-7B-Instruct --adapter ./lora_output

  # 单次任务模式(不进交互循环)
  python agent.py --mode C --model-path ./models/Qwen2.5-7B-Instruct --task "你好"
"""

import argparse
import json
import os
import platform
import sys
import time

# 加载层与 train_lora.py 共享(同一套代码基底的单一事实来源):
# 模型短名映射表、双源下载逻辑。改这里两边同时生效。
from train_lora import MODEL_REGISTRY, download_model  # noqa: E402

# ============================================================
# 配置 / Config
# ============================================================
CONFIG = {
    "mode": "A",                    # A=在线基座+LoRA | B=Ollama | C=本地基座+LoRA
    "model_source": "modelscope",   # A/C 模式的模型源: modelscope / huggingface / local
    "model": "qwen2.5-7b",          # A 模式: 内置短名或完整仓库ID
    "model_path": "",               # C 模式: 本地基座文件夹路径
    "adapter": "./lora_output",     # LoRA 适配器目录(train_lora.py 的输出)
    "ollama_url": "http://localhost:11434/v1",
    "ollama_model": "qwen2.5:7b-instruct",
    "max_new_tokens": 1024,         # 单步生成上限(平板建议 512)
    "max_steps": 8,                 # Agent 单任务最大步数(防失控烧时间)
    "work_dir": ".",                # 文件工具的工作目录(安全边界)
    "memory_file": "./agent_memory.json",
    "task": None,                   # 单次任务模式: 给定任务串, 完成后退出(交互模式为 None)
}

# CodeAgent 输出格式标记。训练数据(dataset_agent_tool.jsonl)使用同一组标记,
# 显式指定可避免 smolagents 版本默认值变化导致的解析失败。
CODE_TAGS = ("<code>", "</code>")

BASE_INSTRUCTIONS = """你是运行在本地设备上的 AI 助手, 通过编写 Python 代码调用工具来完成任务。
规则:
1. 每一步先输出 "Thought:" 说明思路, 然后输出代码块(以 <code> 开始、</code> 结束);
2. 代码中直接用参数名调用工具, 如 read_file(path="a.txt"), 不要传字典;
3. 需要中间结果时用 print() 输出, 观察结果后再决定下一步;
4. 任务完成后必须调用 final_answer(...) 返回最终结果;
5. 工具不可用时如实说明能力边界, 不要编造结果;
6. 使用中文回答, 保持简洁清晰。
You are a local AI assistant. Solve tasks by writing Python code that calls the
provided tools. Always start with "Thought:", wrap code in <code>...</code>,
use print() for intermediate values, and finish with final_answer(...)."""


# ============================================================
# 设备自适应 / Runtime detection
# ============================================================
def detect_runtime():
    """CUDA → GPU 推理; 否则 CPU(平板 AidLearning 自动走这条)。"""
    rt = {"device": "cpu", "dtype": "float32"}
    try:
        import torch
        if torch.cuda.is_available():
            rt["device"] = "cuda"
            rt["dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    except Exception as e:
        print(f"[错误] 无法导入 PyTorch: {e}\n请先安装依赖: pip install -r requirements.txt")
        sys.exit(1)
    return rt


# ============================================================
# 文件工具(带工作目录安全边界) / File tools (sandboxed)
# ============================================================
def _safe_path(root, path):
    """把路径限制在 root 内, 阻止 ../ 越界访问。"""
    p = path if os.path.isabs(path) else os.path.join(root, path)
    real = os.path.realpath(p)
    root_real = os.path.realpath(root)
    if real != root_real and not real.startswith(root_real + os.sep):
        raise ValueError(f"路径越界: '{path}' 不在工作目录 '{root}' 内, 已阻止。")
    return real


def build_tools(work_dir):
    """构造 Agent 工具集。
    【框架能力】smolagents @tool 装饰器负责把函数包装成模型可调用的工具,
    工具的描述(文档字符串)会自动进入系统提示词。"""
    from smolagents import tool

    @tool
    def read_file(path: str) -> str:
        """读取文本文件内容 / Read a text file.
        Args:
            path: 文件路径(相对工作目录或绝对路径) / File path.
        """
        p = _safe_path(work_dir, path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()

    @tool
    def write_file(path: str, content: str) -> str:
        """把文本内容写入文件(自动创建父目录) / Write text to a file.
        Args:
            path: 目标文件路径 / Target file path.
            content: 要写入的文本内容 / Text content to write.
        """
        p = _safe_path(work_dir, path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {os.path.basename(p)}({len(content)} 字符)"

    @tool
    def list_folder(path: str) -> list:
        """列出文件夹下的所有文件名 / List file names in a folder.
        Args:
            path: 文件夹路径 / Folder path.
        """
        p = _safe_path(work_dir, path)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"文件夹不存在: {path}")
        return sorted(os.listdir(p))

    # 【预留扩展接口·阶段二之后】ADB 安卓设备控制工具(截图/点击/滑动)可在此追加:
    #   @tool
    #   def adb_screenshot(save_path: str) -> str:
    #       """截图并保存 / Take a screenshot."""
    #       import subprocess
    #       subprocess.run(["adb", "shell", "screencap", "-p", ...], check=True)
    #       return f"截图已保存到 {save_path}"
    # 注意: 新工具加入后, 建议在 dataset_agent_tool.jsonl 补充对应示范数据再微调。

    return [read_file, write_file, list_folder]


# ============================================================
# 长期记忆 / Long-term memory (json 持久化)
# ============================================================
class MemoryStore:
    """任务-结果记录保存为 json, 重启 Agent 后读取并注入系统指令。
    【自研改造点·阶段二】自研版本需要重新实现记忆的存储格式与检索策略。"""

    def __init__(self, path, keep=50):
        self.path = path
        self.keep = keep
        self.data = {"version": 1, "records": []}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
                print(f"[记忆] 已加载 {len(self.data.get('records', []))} 条历史记录。")
            except Exception:
                print("[记忆] 历史记忆文件损坏, 从空白开始。")

    def add(self, task, answer):
        self.data["records"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task[:500],
            "answer": str(answer)[:500],
        })
        self.data["records"] = self.data["records"][-self.keep:]
        self.save()

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[记忆] 保存失败: {e}")

    def summary(self, n=5):
        """把最近 n 条记录格式化为可注入 instructions 的文本。"""
        recs = self.data.get("records", [])[-n:]
        if not recs:
            return ""
        lines = ["【历史记忆 / Memory】最近完成的任务:"]
        for r in recs:
            lines.append(f"- 任务: {r['task'][:60]} → 结果: {r['answer'][:60]}")
        return "\n".join(lines)


# ============================================================
# 三种模型模式 / Model construction
# ============================================================
def build_model(cfg, rt):
    """按模式构造 smolagents 模型对象。
    【自研改造点·阶段二】LoRA 挂载部分(PeftModel.from_pretrained 包住
    TransformersModel.model)是自研胶水层, 换框架时需重新实现。"""
    if cfg["mode"] == "B":
        # —— 模式B: Ollama(OpenAI 兼容接口, GGUF 模型由 Ollama 管理)——
        from smolagents import OpenAIServerModel
        print(f"[模式B] Ollama: {cfg['ollama_url']} 模型 {cfg['ollama_model']}")
        return OpenAIServerModel(
            model_id=cfg["ollama_model"],
            api_base=cfg["ollama_url"],
            api_key="ollama",  # Ollama 不校验密钥, 占位即可
        )

    # —— 模式A/C: TransformersModel + LoRA 适配器 ——
    from smolagents import TransformersModel

    if cfg["mode"] == "A":
        base_path = download_model(cfg["model"], cfg["model_source"], "./models")
    else:
        base_path = cfg["model_path"]
        if not os.path.isdir(base_path):
            raise SystemExit(f"[错误] 本地基座路径不存在: {base_path}")

    # CUDA 下 4bit 量化推理(省显存); CPU/ARM 用 fp32(稳定优先)
    model_kwargs = {}
    if rt["device"] == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=__import__("torch").bfloat16)
            print("[模式A/C] CUDA 4bit 量化推理。")
        except ImportError:
            print("[提示] 未安装 bitsandbytes, 以全精度加载(显存吃紧请安装)。")
    else:
        print(f"[模式A/C] CPU 推理({rt['dtype']}), 速度较慢属正常现象。")

    # 说明: smolagents 的 TransformersModel 内部仍以 torch_dtype= 透传给
    # transformers(5.x 已弃用该参数名), 这里只抑制这一条已知弃用提示,
    # 不影响其他告警。
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")
        tm = TransformersModel(
            model_id=base_path,
            torch_dtype=rt["dtype"],
            model_kwargs=model_kwargs or None,
            max_new_tokens=cfg["max_new_tokens"],
            trust_remote_code=True,
        )

    # ——【自研改造点·阶段二】LoRA 适配器挂载 ——
    # smolagents TransformersModel 不原生支持 peft 适配器, 这里在模型加载
    # 完成后把 PeftModel 包到 tm.model 上(PeftModel 透明转发 generate)。
    if cfg["adapter"] and os.path.isdir(cfg["adapter"]):
        from peft import PeftModel
        print(f"[LoRA] 挂载适配器: {cfg['adapter']}")
        tm.model = PeftModel.from_pretrained(tm.model, cfg["adapter"])
    elif cfg["adapter"]:
        print(f"[警告] 适配器目录不存在: {cfg['adapter']}, 将使用原始基座(未微调)推理。")
    return tm


# ============================================================
# 主流程 / Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="本地智能体(三模式: 在线基座+LoRA / Ollama / 本地基座+LoRA)")
    ap.add_argument("--mode", choices=["A", "B", "C"], default=None)
    ap.add_argument("--model-source", choices=["modelscope", "huggingface", "local"], default=None)
    ap.add_argument("--model", default=None, help="模式A: 内置短名或完整仓库ID")
    ap.add_argument("--model-path", default=None, help="模式C: 本地基座文件夹")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--work-dir", default=None, help="文件工具工作目录(安全边界)")
    ap.add_argument("--memory-file", default=None)
    ap.add_argument("--task", default=None, help="单次任务模式: 直接给一个任务, 完成后退出")
    ap.add_argument("--quiet", action="store_true", help="减少日志输出")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    for k in cfg:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.mode == "C" and not cfg["model_path"]:
        raise SystemExit("[错误] 模式C 需要 --model-path 指定本地基座文件夹。")

    rt = detect_runtime()
    print(f"""
============================================================
  agent.py — 本地智能体
  模式: {cfg['mode']}   设备: {rt['device']}({rt['dtype']})   架构: {platform.machine()}
============================================================""")

    model = build_model(cfg, rt)

    from smolagents import CodeAgent
    memory = MemoryStore(cfg["memory_file"])
    instructions = BASE_INSTRUCTIONS
    mem_text = memory.summary()
    if mem_text:
        instructions += "\n\n" + mem_text

    agent = CodeAgent(
        tools=build_tools(cfg["work_dir"]),
        model=model,
        max_steps=cfg["max_steps"],
        instructions=instructions,
        code_block_tags=CODE_TAGS,   # 与训练数据格式对齐
        verbosity_level=0 if args.quiet else 1,
    )

    def run_one(task):
        print(f"\n[任务] {task}")
        t0 = time.time()
        try:
            result = agent.run(task)
            print(f"[完成] 耗时 {time.time() - t0:.1f}s\n[结果] {result}")
            memory.add(task, result)
            return result
        except KeyboardInterrupt:
            print("\n[中断] 用户中止。")
        except Exception as e:
            print(f"[失败] {type(e).__name__}: {e}")
            print("       可尝试: 换更小模型 / 增大 max_new_tokens / 检查适配器与基座是否匹配。")
        return None

    if cfg["task"]:
        run_one(cfg["task"])
        return

    print("进入交互模式。输入任务后回车执行; 输入 quit / exit / q 退出。")
    while True:
        try:
            task = input("\n任务> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break
        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            print("[退出] 记忆已保存:", cfg["memory_file"])
            break
        run_one(task)


if __name__ == "__main__":
    main()
