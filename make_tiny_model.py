# -*- coding: utf-8 -*-
"""冒烟测试准备: 构造微型随机权重 Qwen2 模型(真实 tokenizer, 极小权重)。
用途: 在 CPU 上快速验证 train_lora.py / agent.py 全链路, 不下载 GB 级权重。"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "models", "_cache")
TINY = os.path.join(BASE, "models", "tiny-qwen-test")
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "generation_config.json",
]


def fetch_tokenizer():
    from modelscope import snapshot_download
    path = snapshot_download(
        "qwen/Qwen2.5-0.5B-Instruct",
        cache_dir=CACHE,
        allow_patterns=TOKENIZER_FILES,
    )
    print(f"[ok] tokenizer 源目录: {path}")
    return path


def build_tiny(tok_src):
    from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_src)
    vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
    # 与 Qwen2.5 对齐的特殊 token id
    special = {
        "bos_token_id": 151643, "eos_token_id": 151645,
        "pad_token_id": 151643, "vocab_size": 151936,
    }
    cfg = Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=False, **special,
    )
    model = Qwen2ForCausalLM(cfg)
    os.makedirs(TINY, exist_ok=True)
    model.save_pretrained(TINY, safe_serialization=True)
    for name in TOKENIZER_FILES:
        src = os.path.join(tok_src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(TINY, name))
    # 删除 tokenizer_config 里指向旧架构的字段, 避免自动加载类型歧义
    total = sum(p.numel() for p in model.parameters())
    print(f"[ok] 微型模型已写入 {TINY} (参数量 {total/1e6:.2f}M, vocab 151936)")


if __name__ == "__main__":
    if "--skip-download" not in sys.argv:
        src = fetch_tokenizer()
    else:
        src = os.path.join(CACHE, "qwen", "Qwen2.5-0.5B-Instruct")
    build_tiny(src)
    print("[done] 冒烟测试模型就绪")
