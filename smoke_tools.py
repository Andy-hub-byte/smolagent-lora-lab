# -*- coding: utf-8 -*-
"""冒烟测试: 直接调用 agent.py 的工具与记忆模块(不加载模型, 秒级完成)。"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as A  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(f"{'[PASS]' if cond else '[FAIL]'} {name} {extra}")
    if not cond:
        FAILED.append(name)


WORK = "/tmp/tool_smoke"
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK, exist_ok=True)

# 1. 工具集构造
tools = A.build_tools(WORK)
names = {t.name for t in tools}
check("工具集包含 read_file/write_file/list_folder",
      {"read_file", "write_file", "list_folder"} <= names, str(sorted(names)))

# 2. 写入 + 读取 + 列举
w = next(t for t in tools if t.name == "write_file")
r = next(t for t in tools if t.name == "read_file")
ls = next(t for t in tools if t.name == "list_folder")
check("write_file 写入", "已写入" in w(path="sub/hello.txt", content="你好, 冒烟测试"))
check("read_file 读回", r(path="sub/hello.txt") == "你好, 冒烟测试")
check("list_folder 列出", "hello.txt" in ls(path="sub"))

# 3. 安全边界: ../ 越界必须被拒
for bad in ["../../etc/passwd", "/etc/passwd", "sub/../../../../etc/hostname"]:
    try:
        r(path=bad)
        check(f"拒绝越界读取 {bad}", False, "未拦截!")
    except Exception as e:
        check(f"拒绝越界读取 {bad}", "越界" in str(e) or "outside" in str(e).lower() or "不允许" in str(e), type(e).__name__)

# 4. 记忆模块
mem_file = os.path.join(WORK, "mem.json")
mem = A.MemoryStore(mem_file)
mem.add("任务A", "答案A")
mem.add("任务B", "答案B")
check("记忆持久化落盘", os.path.exists(mem_file))
mem2 = A.MemoryStore(mem_file)
check("记忆可从磁盘恢复", len(mem2.data["records"]) == 2)
s = mem2.summary(n=1)
check("summary 注入格式", "任务B" in s and "答案B" in s, repr(s[:40]))

print("\n结果:", "全部通过 ✅" if not FAILED else f"失败 {len(FAILED)} 项: {FAILED}")
sys.exit(1 if FAILED else 0)
