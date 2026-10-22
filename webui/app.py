#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — smolagent-lora-lab Web 微调工作台（后端 / Flask backend）

把训练脚本 train_lora.py 与 Agent 测试脚本 agent.py 包装成一个本地 Web 界面：
  - 环境体检：依赖 / GPU / 设备类型一键诊断
  - 训练控制台：表单配置 → 启动 train_lora.py → 实时日志(SSE) → loss 曲线 → 随时停止
  - Agent 测试台：加载微调出的 LoRA 适配器，直接对话验证效果
  - 数据集浏览 / 产出(checkpoint)管理 / 微型测试模型一键生成

设计约束：
  - 零构建：Flask + 原生 HTML/JS/CSS，pip install flask 即可跑
  - 跨平台：Windows / macOS / Linux / 安卓 Termux（Proot 容器）
  - 安全：训练/测试进程互斥单例；所有路径限制在项目目录内解析
  - 所有子进程 cwd = 项目根目录，与命令行手动运行完全等价
"""
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# ============================================================
# 路径与常量 / Paths
# ============================================================
WEBUI_DIR = Path(__file__).resolve().parent          # .../smolagent-lora-lab/webui
PROJECT_ROOT = WEBUI_DIR.parent                       # .../smolagent-lora-lab
STATIC_DIR = WEBUI_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# 日志格式（train_lora.py 自定义回调输出）:
#   [训练] step 3/10 loss=11.8700 lr=2.00e-04
LOSS_RE = re.compile(r"\[训练\]\s*step\s*(\d+)\s*/\s*(\d+)\s+loss=([0-9.]+)")
STEP_RE = re.compile(r"\[训练\]\s*开始,\s*共\s*(\d+)\s*步")


# ============================================================
# 进程作业管理 / Job manager（训练与 Agent 测试共用一套机制）
# ============================================================
class Job:
    """一个后台子进程作业：进程句柄 + 日志广播 + 状态。"""

    def __init__(self, kind: str):
        self.kind = kind                    # 'train' | 'agent' | 'tiny'
        self.proc = None                    # subprocess.Popen
        self.state = "idle"                 # idle|running|done|error|stopped
        self.started_at = None
        self.finished_at = None
        self.returncode = None
        self.cmd = []
        self.logs = deque(maxlen=5000)      # 环形日志缓冲（防止长跑爆内存）
        self.loss_history = []              # [{step, total, loss, ts}]
        self.total_steps = None
        self._subscribers = []              # [queue.Queue] SSE 订阅者
        self._lock = threading.Lock()
        self._stop_requested = False

    # ---------- 订阅广播 ----------
    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue[str]" = queue.Queue(maxsize=2000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, line: str) -> None:
        self.logs.append(line)
        # 解析 loss / 总步数（画曲线与进度条用）
        m = LOSS_RE.search(line)
        if m:
            self.loss_history.append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
                "ts": time.time(),
            })
        else:
            ms = STEP_RE.search(line)
            if ms:
                self.total_steps = int(ms.group(1))
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    # ---------- 进程控制 ----------
    def start(self, argv: list) -> bool:
        if self.proc and self.proc.poll() is None:
            return False  # 已在运行（单例）
        self.__init__(self.kind)  # 复位状态
        self.cmd = argv
        self.state = "running"
        self.started_at = time.time()
        self._stop_requested = False
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,          # 合并 stderr，日志不丢
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:  # 可执行文件不存在等
            self.state = "error"
            self._emit(f"[WebUI] 启动失败: {e}")
            return False
        self._emit(f"[WebUI] 已启动 (PID {self.proc.pid}): {' '.join(argv)}")
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self) -> None:
        """后台线程：逐行读子进程输出 → 广播 + 解析。"""
        assert self.proc is not None
        try:
            for line in self.proc.stdout:
                self._emit(line.rstrip("\n"))
        except Exception as e:
            self._emit(f"[WebUI] 日志读取异常: {e}")
        finally:
            self.returncode = self.proc.wait()
            self.finished_at = time.time()
            if self._stop_requested:
                self.state = "stopped"
                self._emit(f"[WebUI] 已停止（返回码 {self.returncode}）。")
            elif self.returncode == 0:
                self.state = "done"
                self._emit("[WebUI] 进程结束（成功，返回码 0）。")
            else:
                self.state = "error"
                self._emit(f"[WebUI] 进程结束（失败，返回码 {self.returncode}）。")

    def stop(self) -> bool:
        if not self.proc or self.proc.poll() is not None:
            return False
        self._stop_requested = True
        try:
            if os.name == "nt":
                self.proc.terminate()                     # Windows
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)  # POSIX: 整组
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                return False
        return True

    # ---------- 状态快照 ----------
    def status(self) -> dict:
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "kind": self.kind,
            "state": self.state,
            "pid": self.proc.pid if self.proc else None,
            "returncode": self.returncode,
            "elapsed": elapsed,
            "cmd": self.cmd,
            "total_steps": self.total_steps,
            "loss_count": len(self.loss_history),
        }


JOBS = {k: Job(k) for k in ("train", "agent", "tiny")}


# ============================================================
# 工具函数 / Helpers
# ============================================================
def _run_quick(argv: list, timeout: int = 30) -> dict:
    """同步执行一个短命令（环境体检等），返回输出。"""
    try:
        r = subprocess.run(
            argv, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        return {"ok": r.returncode == 0, "out": (r.stdout or "") + (r.stderr or "")}
    except Exception as e:
        return {"ok": False, "out": str(e)}


def _pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _read_registry() -> dict:
    """从 train_lora.py 读取模型短名注册表（import 安全：main 有 __name__ 守卫）。"""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import train_lora
        return {
            "registry": train_lora.MODEL_REGISTRY,
            "defaults": train_lora.CONFIG,
            "cpu_safe": sorted(train_lora.CPU_SAFE_SIZES),
        }
    except Exception as e:
        return {"registry": {}, "defaults": {}, "cpu_safe": [], "error": str(e)}


def _count_jsonl(path: Path) -> int:
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except Exception:
        pass
    return n


def _dir_size(p: Path) -> int:
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


# ============================================================
# 页面路由 / Pages
# ============================================================
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ============================================================
# API：环境体检 / Environment
# ============================================================
@app.route("/api/env")
def api_env():
    import platform
    py = sys.version.split()[0]
    deps = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets",
                "smolagents", "modelscope", "flask", "accelerate"):
        deps[pkg] = _pkg_version(pkg)

    device = "cpu"
    gpu_name = None
    vram_gb = None
    torch_ver = deps.get("torch")
    if torch_ver:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            pass

    missing = [k for k, v in deps.items() if v is None and k != "flask"]
    advice = []
    if missing:
        advice.append(f"缺少依赖: {', '.join(missing)} → 运行 pip install -r requirements.txt")
    if device == "cpu":
        advice.append("CPU 模式：7B/14B 会被阻止启动；≤4B 自动以 fp32 LoRA 降级训练（慢但可跑）。")
    if deps.get("torch") is None:
        advice.append("未安装 PyTorch：CPU 版可装 torch（体积较小）；GPU 版请按官网选 CUDA 版本。")

    return jsonify({
        "python": py,
        "platform": platform.platform(),
        "device": device,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "deps": deps,
        "missing": missing,
        "advice": advice,
        "project_root": str(PROJECT_ROOT),
    })


# ============================================================
# API：模型注册表 / 数据集 / 产出
# ============================================================
@app.route("/api/models")
def api_models():
    return jsonify(_read_registry())


@app.route("/api/datasets")
def api_datasets():
    reg = _read_registry()
    ds_dir = PROJECT_ROOT / (reg["defaults"].get("datasets_dir") or "datasets")
    files = []
    if ds_dir.is_dir():
        for f in sorted(ds_dir.glob("*.jsonl")):
            files.append({
                "name": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)),
                "count": _count_jsonl(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return jsonify({"dir": str(ds_dir.relative_to(PROJECT_ROOT)), "files": files})


@app.route("/api/datasets/preview")
def api_dataset_preview():
    rel = request.args.get("path", "")
    limit = min(int(request.args.get("limit", 5)), 20)
    p = (PROJECT_ROOT / rel).resolve()
    if not str(p).startswith(str(PROJECT_ROOT)) or not p.is_file():
        return jsonify({"ok": False, "error": "非法路径"}), 400
    rows = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    rows.append({"raw": line.strip()[:200]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "rows": rows})


@app.route("/api/outputs")
def api_outputs():
    """扫描项目根目录下含适配器/checkpoint 的目录（含自定义 --output-dir）。"""
    out_dirs = []
    for c in sorted(PROJECT_ROOT.iterdir(), key=lambda x: x.name):
        if not c.is_dir() or c.name.startswith((".", "_")):
            continue
        if c.name in ("webui", "datasets", "models", "node_modules", ".git"):
            continue
        entries = []
        # 目录本身就是适配器（含 adapter_config.json）
        if (c / "adapter_config.json").is_file():
            entries.append({
                "name": c.name,
                "path": str(c.relative_to(PROJECT_ROOT)),
                "size_mb": round(_dir_size(c) / 1e6, 1),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(c.stat().st_mtime)),
            })
        # 目录下的 checkpoint-* 子目录
        for sub in sorted(c.iterdir(), key=lambda x: x.name, reverse=True):
            if sub.is_dir() and (sub.name.startswith("checkpoint-") or (sub / "adapter_config.json").is_file()):
                entries.append({
                    "name": f"{c.name}/{sub.name}" if sub.name != c.name else sub.name,
                    "path": str(sub.relative_to(PROJECT_ROOT)),
                    "size_mb": round(_dir_size(sub) / 1e6, 1),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(sub.stat().st_mtime)),
                })
        if entries:
            out_dirs.append({"dir": c.name, "entries": entries})
    return jsonify({"dirs": out_dirs})


# ============================================================
# API：训练控制台 / Training
# ============================================================
@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    body = request.get_json(force=True) or {}
    reg = _read_registry()
    defaults = reg["defaults"]

    def g(key, cast, fallback):
        v = body.get(key, fallback)
        try:
            return cast(v)
        except (TypeError, ValueError):
            return fallback

    argv = [sys.executable, "train_lora.py"]
    # —— 模型来源与基座 ——
    source = g("model_source", str, defaults.get("model_source", "modelscope"))
    if source not in ("modelscope", "huggingface", "local"):
        return jsonify({"ok": False, "error": f"非法模型来源: {source}"}), 400
    argv += ["--model-source", source]
    argv += ["--model", g("model", str, defaults.get("model", "qwen2.5-7b"))]
    if body.get("model_cache_dir"):
        argv += ["--model-cache-dir", body["model_cache_dir"]]
    # —— 数据集 ——
    mode = g("dataset_mode", str, defaults.get("dataset_mode", "all"))
    argv += ["--dataset-mode", mode]
    if mode == "single":
        df = g("dataset_file", str, defaults.get("dataset_file", ""))
        if not df:
            return jsonify({"ok": False, "error": "single 模式必须选择数据集文件"}), 400
        argv += ["--dataset-file", df]
    if body.get("datasets_dir"):
        argv += ["--datasets-dir", body["datasets_dir"]]
    # —— LoRA 超参 ——
    for key, cast, flag in (
        ("lora_rank", int, "--lora-rank"),
        ("lora_alpha", int, "--lora-alpha"),
        ("lora_dropout", float, "--lora-dropout"),
        ("batch_size", int, "--batch-size"),
        ("grad_accum", int, "--grad-accum"),
        ("learning_rate", float, "--learning-rate"),
        ("num_epochs", float, "--num-epochs"),
        ("max_steps", int, "--max-steps"),
        ("max_seq_len", int, "--max-seq-len"),
        ("output_dir", str, "--output-dir"),
        ("save_steps", int, "--save-steps"),
        ("save_total_limit", int, "--save-total-limit"),
    ):
        v = body.get(key)
        if v not in (None, ""):
            argv += [flag, str(cast(v))]
    # —— 断点续训 ——
    if body.get("resume") is True:
        argv += ["--resume"]
    elif body.get("resume") is False:
        argv += ["--no-resume"]
    if body.get("force"):
        argv += ["--force"]

    ok = JOBS["train"].start(argv)
    return jsonify({"ok": ok, "cmd": " ".join(argv),
                    "error": None if ok else "训练进程已在运行"})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    return jsonify({"ok": JOBS["train"].stop()})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(JOBS["train"].status())


@app.route("/api/train/logs")
def api_train_logs():
    """SSE：实时推送训练日志（新客户端先补发缓冲末尾）。"""
    job = JOBS["train"]

    def gen():
        q = job.subscribe()
        try:
            # 补发最近 80 行，刷新页面不丢上下文
            for line in list(job.logs)[-80:]:
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            yield "data: {\"hello\": true}\n\n"
            while True:
                try:
                    line = q.get(timeout=15)
                    yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"  # 注释行保活
        finally:
            job.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/train/loss")
def api_train_loss():
    return jsonify({"points": JOBS["train"].loss_history[-2000:],
                    "total_steps": JOBS["train"].total_steps})


# ============================================================
# API：Agent 测试台 / Agent playground
# ============================================================
@app.route("/api/agent/run", methods=["POST"])
def api_agent_run():
    body = request.get_json(force=True) or {}
    mode = body.get("mode", "C")
    if mode not in ("A", "B", "C"):
        return jsonify({"ok": False, "error": f"非法模式: {mode}"}), 400
    task = (body.get("task") or "").strip()
    if not task:
        return jsonify({"ok": False, "error": "任务内容不能为空"}), 400

    argv = [sys.executable, "agent.py", "--mode", mode, "--quiet", "--task", task]
    if mode in ("A", "C"):
        if body.get("model_source"):
            argv += ["--model-source", body["model_source"]]
        if mode == "A" and body.get("model"):
            argv += ["--model", body["model"]]
        if mode == "C" and body.get("model_path"):
            argv += ["--model-path", body["model_path"]]
        if body.get("adapter"):
            argv += ["--adapter", body["adapter"]]
    if mode == "B":
        if body.get("ollama_url"):
            argv += ["--ollama-url", body["ollama_url"]]
        if body.get("ollama_model"):
            argv += ["--ollama-model", body["ollama_model"]]
    if body.get("max_new_tokens"):
        argv += ["--max-new-tokens", str(int(body["max_new_tokens"]))]
    if body.get("max_steps"):
        argv += ["--max-steps", str(int(body["max_steps"]))]

    ok = JOBS["agent"].start(argv)
    return jsonify({"ok": ok, "cmd": " ".join(argv),
                    "error": None if ok else "Agent 测试进程已在运行"})


@app.route("/api/agent/stop", methods=["POST"])
def api_agent_stop():
    return jsonify({"ok": JOBS["agent"].stop()})


@app.route("/api/agent/status")
def api_agent_status():
    return jsonify(JOBS["agent"].status())


@app.route("/api/agent/logs")
def api_agent_logs():
    job = JOBS["agent"]

    def gen():
        q = job.subscribe()
        try:
            for line in list(job.logs)[-120:]:
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            yield "data: {\"hello\": true}\n\n"
            while True:
                try:
                    line = q.get(timeout=15)
                    yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            job.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================
# API：微型测试模型 / Tiny model (端到端自检，无需下载)
# ============================================================
@app.route("/api/tiny/start", methods=["POST"])
def api_tiny_start():
    argv = [sys.executable, "make_tiny_model.py"]
    out = (request.get_json(force=True) or {}).get("output", "./models/tiny-qwen-test")
    if out:
        argv += ["--output", out]
    ok = JOBS["tiny"].start(argv)
    return jsonify({"ok": ok, "cmd": " ".join(argv),
                    "error": None if ok else "进程已在运行"})


@app.route("/api/tiny/status")
def api_tiny_status():
    return jsonify(JOBS["tiny"].status())


@app.route("/api/tiny/stop", methods=["POST"])
def api_tiny_stop():
    return jsonify({"ok": JOBS["tiny"].stop()})


@app.route("/api/tiny/logs")
def api_tiny_logs():
    job = JOBS["tiny"]

    def gen():
        q = job.subscribe()
        try:
            for line in list(job.logs)[-80:]:
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            yield "data: {\"hello\": true}\n\n"
            while True:
                try:
                    line = q.get(timeout=15)
                    yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            job.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================
# 启动 / Entry
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="smolagent-lora-lab Web 微调工作台")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机；平板/局域网可设 0.0.0.0）")
    ap.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    # 云端/容器部署兼容: 注入 PORT 时自动改为监听 0.0.0.0 且不自动开浏览器
    env_port = os.environ.get("PORT")
    cloud = bool(env_port)
    if cloud:
        args.port = int(env_port)
        args.host = "0.0.0.0"
        args.no_browser = True

    url = f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}/"
    print("=" * 60)
    print("  smolagent-lora-lab · Web 微调工作台")
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  访问地址: {url}")
    print("  停止: Ctrl+C")
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(1.0, lambda: _open_browser(url)).start()
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        # 退出时尽量收拾子进程
        for j in JOBS.values():
            if j.proc and j.proc.poll() is None:
                j.stop()
        print("\n[WebUI] 已退出。")


def _open_browser(url: str):
    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()
