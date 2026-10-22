#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_model.py — 把外部模型导入本项目 models/ 目录

三种用法（自动识别）：

  # 1) 本地文件夹（拷到你电脑上的模型目录）
  python import_model.py /path/to/Qwen2.5-1.5B-Instruct
  python import_model.py /path/to/model --name my-qwen        # 自定义目录名

  # 2) Hugging Face 仓库（海外源）
  python import_model.py Qwen/Qwen2.5-1.5B-Instruct --source huggingface

  # 3) ModelScope 仓库（国内源，支持内置短名）
  python import_model.py qwen2.5-1.5b --source modelscope

  # 覆盖已存在的同名目录
  python import_model.py <模型> --overwrite

导入后即可用于训练 / Agent 测试：
  python train_lora.py --model-source local --model ./models/<目录名>
  python agent.py --mode C --model-path ./models/<目录名>

Web 界面用户无需使用本脚本——webui/ 的「模型导入」页签就是它的图形版。
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHT_EXTS = (".safetensors", ".bin")
TOKENIZER_HINTS = ("tokenizer", "vocab.txt", "merges.txt", "spiece.model", "tokenizer.model")


def log(msg: str = ""):
    print(msg, flush=True)


def sanitize_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    if not n or n in (".", ".."):
        raise SystemExit("[错误] 非法的模型目录名（只允许字母/数字/._-）")
    return n[:80]


def inspect_dir(p: Path) -> dict:
    """HuggingFace 格式校验：config.json + 权重文件（+ tokenizer 提示）。"""
    if not p.is_dir():
        return {"valid": False, "reason": "not_a_dir"}
    files, weights = [], []
    for f in sorted(p.iterdir()):
        if f.is_file():
            files.append(f.name)
            if f.name.endswith(WEIGHT_EXTS):
                weights.append(f.name)
    has_config = "config.json" in files
    has_tok = any(any(h in n for h in TOKENIZER_HINTS) for n in files)
    arch = None
    if has_config:
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            arch = cfg.get("model_type") or (cfg.get("architectures") or [None])[0]
        except Exception:
            pass
    return {
        "valid": has_config and bool(weights),
        "has_config": has_config, "has_tokenizer": has_tok,
        "arch": arch, "weight_count": len(weights),
        "weights": weights[:4], "files": len(files),
    }


def dir_size_mb(p: Path) -> float:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return round(total / 1e6, 1)


def import_from_path(src: Path, name: str, overwrite: bool):
    if not src.is_dir():
        raise SystemExit(f"[错误] 路径不存在或不是文件夹: {src}")
    info = inspect_dir(src)
    if not info["valid"]:
        raise SystemExit(
            "[错误] 该文件夹不像 HuggingFace 格式模型。\n"
            "  需要包含: config.json + 权重文件(*.safetensors / *.bin)\n"
            "  （tokenizer 文件建议一并放入，否则训练前需重新指定分词器）")
    dest = MODELS_DIR / name
    if dest.exists():
        if not overwrite:
            raise SystemExit(f"[错误] models/{name} 已存在（加 --overwrite 覆盖重导）")
        log(f"[清理] 删除已存在的 models/{name}")
        shutil.rmtree(dest)
    log(f"[拷贝] {src} → models/{name}（约 {dir_size_mb(src)} MB，大文件请耐心等待）")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def _resolve_short_name(model: str, source: str) -> str:
    """内置短名（如 qwen2.5-1.5b）→ 仓库ID；复用 train_lora 的注册表（惰性 import，避免拖慢 torch）。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import train_lora
        return train_lora.resolve_repo_id(model, source)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"[错误] 无法解析短名 '{model}': {e}\n"
                         f"请改传完整仓库ID（如 Qwen/Qwen2.5-1.5B-Instruct）")


def import_from_repo(repo: str, source: str, name: str, overwrite: bool):
    dest = MODELS_DIR / name
    if dest.exists():
        if not overwrite:
            raise SystemExit(f"[错误] models/{name} 已存在（加 --overwrite 覆盖重下）")
        log(f"[清理] 删除已存在的 models/{name}")
        shutil.rmtree(dest)

    # 完整仓库ID（含 /）或内置短名都走这里；短名走 train_lora 注册表解析
    repo_id = repo if "/" in repo else _resolve_short_name(repo, source)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if source == "huggingface":
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise SystemExit("[错误] 未安装 huggingface_hub: pip install huggingface_hub")
            log(f"[Hugging Face] 正在拉取 {repo_id} → models/{name}")
            log("[提示] 大模型下载耗时较长；进度条在部分终端不显示，属正常现象")
            try:
                snapshot_download(repo_id, local_dir=str(dest))
            except Exception as e:
                msg = str(e)
                if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
                    raise SystemExit(
                        f"[错误] {repo_id} 是 Hugging Face 门控模型: 需要先同意协议并登录。\n"
                        f"  1) 浏览器打开 https://huggingface.co/{repo_id} 点击 Agree and access;\n"
                        f"  2) 创建 access token: https://huggingface.co/settings/tokens ;\n"
                        f"  3) 登录: pip install -U huggingface_hub && hf auth login ;\n"
                        f"国内用户可改用 --source modelscope（同名短名通常有镜像）。")
                raise
        else:
            try:
                from modelscope import snapshot_download
            except ImportError:
                raise SystemExit("[错误] 未安装 modelscope: pip install modelscope")
            log(f"[魔搭 ModelScope] 正在拉取 {repo_id} → models/{name}")
            log("[提示] 大模型下载耗时较长；进度条在部分终端不显示，属正常现象")
            try:
                snapshot_download(repo_id, local_dir=str(dest))
            except TypeError:
                # 老版本 modelscope 不支持 local_dir：先下到缓存再整目录拷贝
                log("[提示] 当前 modelscope 版本不支持 local_dir，改用缓存+拷贝")
                cache = snapshot_download(repo_id, cache_dir=str(MODELS_DIR / "_cache"))
                shutil.copytree(cache, dest)
                shutil.rmtree(MODELS_DIR / "_cache", ignore_errors=True)
            except Exception as e:
                msg = str(e)
                if "not found" in msg.lower() or "404" in msg:
                    raise SystemExit(
                        f"[错误] ModelScope 上找不到仓库: {repo_id}\n"
                        f"  请到 https://modelscope.cn 搜索确认模型 ID；或改用 --source huggingface。")
                raise
    except BaseException:
        # 下载失败：清掉半成品目录，避免 models/ 里留下无效条目
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            log(f"[清理] 已删除未完成的 models/{name}")
        raise
    return dest


def _parse_args():
    ap = argparse.ArgumentParser(
        description="导入外部模型到 models/ 目录（本地路径 / Hugging Face / ModelScope）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python import_model.py /path/to/Qwen2.5-1.5B-Instruct\n"
               "  python import_model.py Qwen/Qwen2.5-1.5B-Instruct --source huggingface\n"
               "  python import_model.py qwen2.5-1.5b --source modelscope\n")
    ap.add_argument("model", nargs="?", help="本地文件夹绝对路径，或仓库ID（如 Qwen/Qwen2.5-1.5B-Instruct）/ 内置短名")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "huggingface", "modelscope"],
                    help="模型来源（默认 auto：是本地路径就走拷贝，否则按 huggingface）")
    ap.add_argument("--name", default="", help="导入后的目录名（默认用原文件夹名/仓库名）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名目录")
    return ap.parse_args()


def main():
    args = _parse_args()
    if not args.model:
        raise SystemExit("请提供模型路径或仓库ID，详见 python import_model.py --help")

    src = Path(args.model).expanduser()
    is_local = src.is_absolute() and src.is_dir()

    if args.source == "auto":
        source = "local" if is_local else "huggingface"

    if is_local:
        name = sanitize_name(args.name or src.name)
        log(f"[导入] 本地路径: {src}")
        dest = import_from_path(src.resolve(), name, args.overwrite)
    else:
        source = "huggingface" if args.source == "local" else args.source
        name = sanitize_name(args.name or args.model.split("/")[-1])
        log(f"[导入] 仓库: {args.model}（源: {source}）")
        dest = import_from_repo(args.model, source, name, args.overwrite)

    # 下载/拷贝后再校验一次，缺文件给明确警告
    info = inspect_dir(dest)
    size = dir_size_mb(dest)
    log("")
    log("=" * 60)
    log(f"  导入完成: models/{name}（{size} MB）")
    log(f"  架构: {info['arch'] or '未知'} · 权重文件: {info['weight_count']} 个 · 文件数: {info['files']}")
    if not info["has_tokenizer"]:
        log("  [警告] 未发现 tokenizer 文件，训练/推理可能无法加载分词器")
    if not info["valid"]:
        log("  [警告] 缺少 config.json 或权重文件，这不是一份完整模型")
    log("=" * 60)
    log("")
    log("下一步（训练）:")
    log(f"  python train_lora.py --model-source local --model ./models/{name}")
    log("下一步（Agent 测试）:")
    log(f"  python agent.py --mode C --model-path ./models/{name}")
    log("或在 Web 界面「模型导入」页签中直接选用。")


if __name__ == "__main__":
    main()
