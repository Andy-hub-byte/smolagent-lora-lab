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
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# ============================================================
# 路径与常量 / Paths
# ============================================================
WEBUI_DIR = Path(__file__).resolve().parent          # .../smolagent-lora-lab/webui
PROJECT_ROOT = WEBUI_DIR.parent                       # .../smolagent-lora-lab
STATIC_DIR = WEBUI_DIR / "static"
MODELS_DIR = PROJECT_ROOT / "models"                  # 导入/下载的模型都落在这里

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# 日志格式（train_lora.py 自定义回调输出）:
#   [训练] step 3/10 loss=11.8700 lr=2.00e-04
LOSS_RE = re.compile(r"\[训练\]\s*step\s*(\d+)\s*/\s*(\d+)\s+loss=([0-9.]+)")
STEP_RE = re.compile(r"\[训练\]\s*开始,\s*共\s*(\d+)\s*步")


# ============================================================
# 进程作业管理 / Job manager
# 「关屏幕 / 关终端，训练继续跑」的关键设计:
#   1. 子进程输出写入 runtime/<kind>_<ts>.log 文件（不是管道）
#      —— WebUI 服务器被杀，子进程不会因管道断裂(SIGPIPE)而死
#   2. POSIX start_new_session / Windows DETACHED_PROCESS
#      —— 子进程脱离服务器的进程组与控制台，关掉终端窗口不连带杀死训练
#   3. runtime/<kind>.json 记录 {pid, cmd, started_at, log}
#      —— 服务器重启后 adopt_orphans() 发现 pid 还活着就「领养」该作业，
#         重新 tail 日志文件，前端自动接上（断线不丢训练）
#   4. 日志文件是唯一事实源: _emit 写文件, tailer 线程读文件并广播给 SSE
# ============================================================
RUNTIME_DIR = WEBUI_DIR / "runtime"


def _pid_alive(pid) -> bool:
    """跨平台判断进程是否存活（领养作业用）。"""
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        return False


class Job:
    """一个后台作业（训练 / Agent / 微型模型 / 导入）: 独立子进程 + 文件日志 + 可领养。"""

    KINDS = ("train", "agent", "tiny", "import")

    def __init__(self, kind: str):
        self.kind = kind
        self.proc = None                    # 服务器亲自启动时的 Popen（领养的作业为 None）
        self.pid = None
        self.state = "idle"                 # idle|running|done|error|stopped
        self.started_at = None
        self.finished_at = None
        self.returncode = None
        self.cmd = []
        self.logs = deque(maxlen=5000)      # 环形日志缓冲（防止长跑爆内存）
        self.loss_history = []              # [{step, total, loss, ts}]
        self.total_steps = None
        self.adopted = False                # 是否为重启后领养的作业
        self.log_path = None
        self._subscribers = []              # [queue.Queue] SSE 订阅者
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._stop_requested = False
        self._tail_thread = None

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

    # ---------- 日志：写文件（唯一事实源） ----------
    def _write_file(self, line: str) -> None:
        if not self.log_path:
            return
        try:
            with self._file_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            pass

    def _emit(self, line: str) -> None:
        """服务器自己的注解行：写入日志文件，由 tailer 统一广播。"""
        self._write_file(line)

    def _broadcast(self, line: str) -> None:
        """tailer 调用：进环形缓冲 + 解析 loss + 推给所有 SSE 订阅者。"""
        self.logs.append(line)
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
    def is_running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return _pid_alive(self.pid)

    def _state_path(self) -> Path:
        return RUNTIME_DIR / f"{self.kind}.json"

    def _save_state(self) -> None:
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path().with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "pid": self.pid, "cmd": self.cmd, "started_at": self.started_at,
                "log": str(self.log_path),
            }), encoding="utf-8")
            tmp.replace(self._state_path())
        except OSError:
            pass

    def _clear_state(self) -> None:
        try:
            self._state_path().unlink()
        except OSError:
            pass

    def start(self, argv: list) -> bool:
        if self.is_running():
            return False  # 已在运行（单例）
        self.__init__(self.kind)  # 复位状态
        self.cmd = argv
        self.state = "running"
        self.started_at = time.time()
        self._stop_requested = False
        try:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            # 只保留最近 10 个日志，防磁盘堆积
            old = sorted(RUNTIME_DIR.glob(f"{self.kind}_*.log"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            for p in old[10:]:
                try:
                    p.unlink()
                except OSError:
                    pass
            self.log_path = RUNTIME_DIR / f"{self.kind}_{int(self.started_at)}.log"
            self.log_path.touch()
            popen_kw = dict(
                cwd=str(PROJECT_ROOT),
                stdout=open(self.log_path, "a", encoding="utf-8"),
                stderr=subprocess.STDOUT,          # 合并 stderr，日志不丢
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            )
            if os.name == "nt":
                # 脱离父控制台 + 新进程组：关掉终端窗口不会杀死训练
                popen_kw["creationflags"] = 0x00000008 | 0x00000200
            else:
                # 独立会话/进程组：免受 SIGHUP 与误用 killpg 波及
                popen_kw["start_new_session"] = True
            self.proc = subprocess.Popen(argv, **popen_kw)
            self.pid = self.proc.pid
        except Exception as e:  # 可执行文件不存在等
            self.state = "error"
            self._write_file(f"[WebUI] 启动失败: {e}")
            self._broadcast(f"[WebUI] 启动失败: {e}")
            return False
        self._save_state()
        self._emit(f"[WebUI] 已启动 (PID {self.pid}): {' '.join(argv)}")
        self._start_tailer()
        return True

    def _start_tailer(self) -> None:
        t = threading.Thread(target=self._tail_loop, daemon=True)
        t.start()
        self._tail_thread = t

    def _tail_loop(self) -> None:
        """follow 日志文件 → 广播 + 解析 loss；进程消失后收尾。"""
        try:
            f = open(self.log_path, "r", encoding="utf-8", errors="replace")
        except OSError:
            return
        with f:
            while True:
                line = f.readline()
                if line:
                    self._broadcast(line.rstrip("\n"))
                    continue
                if not self.is_running():
                    break
                time.sleep(0.25)
            # 进程已结束，读净尾部残留
            for line in f:
                self._broadcast(line.rstrip("\n"))
        if self.proc is not None:
            try:
                self.returncode = self.proc.wait()
            except Exception:
                pass
        self._finish()

    def _infer_returncode(self) -> int:
        """返回码：亲自启动的用 wait()；领养的读退出码文件，再不行按日志启发式推断。"""
        if self.returncode is not None:
            return self.returncode
        try:
            return int(Path(str(self.log_path) + ".exit").read_text(encoding="utf-8").strip())
        except Exception:
            pass
        text = "\n".join(list(self.logs)[-400:])
        if re.search(r"\[错误\]|Traceback|CUDA error|Killed|OutOfMemory", text):
            return 1
        return 0

    def _finish(self) -> None:
        if self.finished_at:
            return  # 防重入
        rc = self._infer_returncode()
        self.returncode = rc
        self.finished_at = time.time()
        if self._stop_requested:
            self.state = "stopped"
            msg = f"[WebUI] 已停止（返回码 {rc}）。"
        elif rc == 0:
            self.state = "done"
            msg = "[WebUI] 进程结束（成功，返回码 0）。"
        else:
            self.state = "error"
            msg = f"[WebUI] 进程结束（失败，返回码 {rc}）。"
        self._clear_state()
        try:
            Path(str(self.log_path) + ".exit").write_text(str(rc), encoding="utf-8")
        except OSError:
            pass
        # tailer 此时已退出，直接写文件 + 广播（不会重复）
        self._write_file(msg)
        self._broadcast(msg)

    def stop(self) -> bool:
        if not self.is_running():
            return False
        self._stop_requested = True
        try:
            if os.name == "nt":
                if self.proc is not None:
                    self.proc.terminate()
                else:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.pid)],
                                   capture_output=True)
            else:
                # 子进程在独立进程组里，pgid == pid，只杀它自己
                os.killpg(self.pid, signal.SIGTERM)
        except Exception:
            try:
                if self.proc is not None:
                    self.proc.terminate()
            except Exception:
                return False
        # 5 秒内没退出就升级强杀，防止训练卡在校验点关不掉
        def _escalate():
            try:
                if self.is_running():
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.pid)],
                                       capture_output=True)
                    else:
                        os.killpg(self.pid, signal.SIGKILL)
                    self._emit("[WebUI] 进程未响应停止信号，已强制结束。")
            except Exception:
                pass
        threading.Timer(5.0, _escalate).start()
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
            "pid": self.pid,
            "alive": self.is_running(),
            "returncode": self.returncode,
            "elapsed": elapsed,
            "cmd": self.cmd,
            "total_steps": self.total_steps,
            "loss_count": len(self.loss_history),
            "adopted": self.adopted,
        }

    # ---------- 领养：服务器重启后接上仍在跑的作业 ----------
    @classmethod
    def adopt_orphans(cls) -> list:
        adopted = []
        for kind in cls.KINDS:
            state_path = RUNTIME_DIR / f"{kind}.json"
            if not state_path.is_file():
                continue
            try:
                st = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                try:
                    state_path.unlink()
                except OSError:
                    pass
                continue
            pid = st.get("pid")
            log_path = Path(st.get("log") or "")
            if not _pid_alive(pid) or not log_path.is_file():
                # 进程已不在：清理状态文件（日志保留供查看）
                try:
                    state_path.unlink()
                except OSError:
                    pass
                continue
            job = JOBS[kind]
            job.__init__(kind)
            job.pid = pid
            job.cmd = st.get("cmd") or []
            job.started_at = st.get("started_at") or time.time()
            job.log_path = log_path
            job.state = "running"
            job.adopted = True
            job._start_tailer()   # 从 0 读日志：重建 loss 历史 + 填日志缓冲
            adopted.append(kind)
        return adopted


JOBS = {k: Job(k) for k in ("train", "agent", "tiny", "import")}

# 服务器启动即领养：上次运行遗留的存活作业自动接上（断线续跑核心）
ADOPTED_KINDS = Job.adopt_orphans()


def _sse_response(job: "Job"):
    """SSE 响应：先补发日志缓冲末尾（缓冲空则读日志文件末尾），再实时推送。"""

    def gen():
        q = job.subscribe()
        try:
            lines = list(job.logs)[-120:]
            if not lines and job.log_path and Path(job.log_path).is_file():
                # 领养的作业：tailer 还没读到缓冲，直接从文件补发
                try:
                    with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = [l.rstrip("\n") for l in f.readlines()[-120:]]
                except OSError:
                    pass
            for line in lines:
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
# API：导入外部模型 / Import models (本地路径 · HF/ModelScope 仓库 · 网页上传)
# ============================================================
WEIGHT_EXTS = (".safetensors", ".bin")
TOKENIZER_HINTS = ("tokenizer", "vocab.txt", "merges.txt", "spiece.model", "tokenizer.model")


class _BadZip(Exception):
    """上传的压缩包含非法路径（Zip Slip 防护）。"""


def _sanitize_name(name: str) -> str:
    """模型目录名净化：只允许安全字符，防路径逃逸。"""
    n = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    if not n or n in (".", ".."):
        raise ValueError("bad_name")
    return n[:80]


def _inspect_model_dir(p: Path) -> dict:
    """检查目录是否像一份 HuggingFace 格式模型（config.json + 权重 + tokenizer）。"""
    if not p.is_dir():
        return {"valid": False, "reason": "not_a_dir"}
    files, weights = [], []
    try:
        for f in sorted(p.iterdir()):
            if f.is_file():
                files.append(f.name)
                if f.name.endswith(WEIGHT_EXTS):
                    weights.append(f.name)
    except OSError:
        pass
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
        "has_config": has_config,
        "has_tokenizer": has_tok,
        "arch": arch,
        "weight_count": len(weights),
        "weights": weights[:4],
        "files": len(files),
        "reason": None if (has_config and weights) else "missing_files",
    }


def _model_entry(p: Path, display: str = None) -> dict:
    st = p.stat()
    info = _inspect_model_dir(p)
    name = display or p.name
    try:
        rel = os.path.relpath(p, PROJECT_ROOT)
    except ValueError:
        rel = str(p)
    info.update({
        "name": name,
        "path": rel,
        "delete_name": "/".join(rel.replace("\\", "/").split("/")[1:]),  # 去掉开头的 models/
        "abs_path": str(p),
        "size_mb": round(_dir_size(p) / 1e6, 1),
        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
    })
    return info


def _iter_model_dirs():
    """遍历 models/ 下的模型目录，支持三种布局:
    1) 导入产生:              models/<name>/
    2) cache_dir 下载产生:    models/<namespace>/<Model>/
    3) HF hub 缓存产生:       models/<hub>/<repo>--<name>/snapshots/<rev>/
    返回 (display_name, Path)。容器层(本身不是模型的目录)不单独列出。"""
    if not MODELS_DIR.is_dir():
        return

    def kids(p):
        try:
            return [c for c in sorted(p.iterdir())
                    if c.is_dir() and not c.name.startswith((".", "_"))]
        except OSError:
            return []

    def emit(c):
        if _inspect_model_dir(c)["valid"]:
            yield c.name, c
            return
        subs = kids(c)
        found = False
        for s in subs:
            # HF hub 缓存: 再向下找一层 snapshots/<rev>/
            if (s / "snapshots").is_dir():
                for snap in kids(s / "snapshots"):
                    if _inspect_model_dir(snap)["valid"]:
                        found = True
                        yield f"{c.name}/{s.name}@{snap.name}", snap
                continue
            if _inspect_model_dir(s)["valid"]:
                found = True
                yield f"{c.name}/{s.name}", s
        if not found:
            # 下层也没有合法模型：原样列出(表格里会显示✗不完整, 可删除清理)
            yield c.name, c

    for c in sorted(MODELS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not c.is_dir() or c.name.startswith((".", "_")):
            continue
        yield from emit(c)


@app.route("/api/models/imported")
def api_models_imported():
    """列出 models/ 下已导入/已下载的模型目录（含格式校验状态）。"""
    entries = []
    for display, p in _iter_model_dirs():
        try:
            entries.append(_model_entry(p, display))
        except OSError:
            pass
    return jsonify({"dir": "models", "entries": entries})


@app.route("/api/models/import/path", methods=["POST"])
def api_models_import_path():
    """方式一：从本机绝对路径拷贝模型文件夹 → models/<name>/。"""
    body = request.get_json(force=True) or {}
    src = (body.get("path") or "").strip()
    if not src:
        return jsonify({"ok": False, "error_code": "path_required",
                        "error": "请填写模型文件夹的绝对路径"}), 400
    sp = Path(src).expanduser()
    try:
        sp = sp.resolve()
    except OSError:
        pass
    if not sp.is_dir():
        return jsonify({"ok": False, "error_code": "path_not_found",
                        "error": f"路径不存在或不是文件夹: {src}"}), 400
    try:
        name = _sanitize_name(body.get("name") or sp.name)
    except ValueError:
        return jsonify({"ok": False, "error_code": "bad_name",
                        "error": "模型名称只能包含字母/数字/._-"}), 400
    info = _inspect_model_dir(sp)
    if not info["valid"]:
        return jsonify({"ok": False, "error_code": "not_a_model",
                        "error": "该文件夹不像 HuggingFace 格式模型（需要 config.json + .safetensors/.bin 权重）"}), 400
    dest = MODELS_DIR / name
    if dest.exists() and not body.get("overwrite"):
        return jsonify({"ok": False, "error_code": "already_exists",
                        "error": f"models/{name} 已存在（勾选覆盖可重导）"}), 409
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(sp, dest)
    except OSError as e:
        return jsonify({"ok": False, "error_code": "copy_failed",
                        "error": f"拷贝失败: {e}"}), 500
    entry = _model_entry(dest)
    return jsonify({"ok": True, "entry": entry,
                    "message": f"已导入到 models/{name}（{entry['size_mb']} MB）"})


@app.route("/api/models/import/upload", methods=["POST"])
def api_models_import_upload():
    """方式二：浏览器上传模型文件（多文件或单个 zip）→ models/<name>/。"""
    name = request.form.get("name", "").strip()
    try:
        name = _sanitize_name(name)
    except ValueError:
        return jsonify({"ok": False, "error_code": "bad_name",
                        "error": "请填写合法的模型名称（字母/数字/._-）"}), 400
    if not request.files:
        return jsonify({"ok": False, "error_code": "no_files",
                        "error": "没有收到文件"}), 400
    dest = MODELS_DIR / name
    if dest.exists() and request.form.get("overwrite") != "1":
        return jsonify({"ok": False, "error_code": "already_exists",
                        "error": f"models/{name} 已存在（勾选覆盖可重传）"}), 409
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        zips = []
        for key in request.files.keys():
            for f in request.files.getlist(key):
                fn = Path(f.filename or "").name  # 只取 basename，防路径逃逸
                if not fn:
                    continue
                if fn.lower().endswith(".zip"):
                    zips.append(f)
                else:
                    f.save(dest / fn)
        # zip 包：解压并防 Zip Slip（成员路径不得逃出目标目录）
        for f in zips:
            with zipfile.ZipFile(f.stream) as zf:
                for member in zf.namelist():
                    target = (dest / member).resolve()
                    if not str(target).startswith(str(dest.resolve())):
                        raise _BadZip(f"压缩包含非法路径: {member}")
                zf.extractall(dest)
        # zip 内可能还有一层同名文件夹，向下刨一层
        sub_dirs = [c for c in dest.iterdir() if c.is_dir()]
        if zips and len(sub_dirs) == 1 and not (dest / "config.json").is_file():
            inner = sub_dirs[0]
            for item in inner.iterdir():
                shutil.move(str(item), str(dest / item.name))
            inner.rmdir()
    except _BadZip as e:
        shutil.rmtree(dest, ignore_errors=True)  # 清理半成品
        return jsonify({"ok": False, "error_code": "bad_zip", "error": str(e)}), 400
    except OSError as e:
        shutil.rmtree(dest, ignore_errors=True)
        return jsonify({"ok": False, "error_code": "upload_failed",
                        "error": f"上传失败: {e}"}), 500
    entry = _model_entry(dest)
    if not entry["valid"]:
        return jsonify({"ok": True, "ok_valid": False, "entry": entry,
                        "message": f"文件已收到，但 models/{name} 不像完整模型（缺 config.json 或权重）"}), 200
    return jsonify({"ok": True, "entry": entry,
                    "message": f"已导入到 models/{name}（{entry['size_mb']} MB）"})


@app.route("/api/models/import/repo", methods=["POST"])
def api_models_import_repo():
    """方式三：从 Hugging Face / ModelScope 下载仓库 → models/<name>/（后台作业 + SSE 日志）。"""
    body = request.get_json(force=True) or {}
    source = body.get("source", "huggingface")
    if source not in ("huggingface", "modelscope"):
        return jsonify({"ok": False, "error": "非法模型源"}), 400
    repo = (body.get("repo") or "").strip()
    if not repo:
        return jsonify({"ok": False, "error_code": "repo_required",
                        "error": "请填写仓库 ID（如 Qwen/Qwen2.5-1.5B-Instruct）"}), 400
    try:
        name = _sanitize_name(body.get("name") or repo.split("/")[-1])
    except ValueError:
        return jsonify({"ok": False, "error_code": "bad_name", "error": "非法的模型名称"}), 400
    argv = [sys.executable, "import_model.py", repo, "--source", source, "--name", name]
    if body.get("overwrite"):
        argv.append("--overwrite")
    ok = JOBS["import"].start(argv)
    return jsonify({"ok": ok, "cmd": " ".join(argv),
                    "error": None if ok else "导入作业已在运行"})


@app.route("/api/models/import/stop", methods=["POST"])
def api_models_import_stop():
    return jsonify({"ok": JOBS["import"].stop()})


@app.route("/api/models/import/status")
def api_models_import_status():
    return jsonify(JOBS["import"].status())


@app.route("/api/models/import/logs")
def api_models_import_logs_logs():
    """SSE：实时推送导入作业日志。"""
    return _sse_response(JOBS["import"])


@app.route("/api/models/imported/delete", methods=["POST"])
def api_models_imported_delete():
    """删除 models/ 下的模型目录。name 可带一层命名空间(如 qwen/Qwen2.5-1.5B-Instruct)。"""
    name = (request.get_json(force=True) or {}).get("name", "")
    parts = [p for p in str(name).split("/") if p]
    if (not parts or len(parts) > 2
            or any(p in (".", "..") or p.startswith(".") for p in parts)):
        return jsonify({"ok": False, "error": "非法名称"}), 400
    target = MODELS_DIR.joinpath(*parts).resolve()
    models_root = MODELS_DIR.resolve()
    if not str(target).startswith(str(models_root)) or not target.is_dir():
        return jsonify({"ok": False, "error": "目录不存在"}), 404
    try:
        shutil.rmtree(target)
        # 命名空间层(两层布局的父目录)空了就一并清掉
        parent = target.parent
        if parent != models_root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


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
def api_train_logs_logs():
    """SSE：实时推送训练日志（新客户端先补发缓冲末尾）。"""
    return _sse_response(JOBS["train"])


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
        if "adapter" in body:
            # 显式传递(含空串=不使用适配器), 避免回退到 CONFIG 默认 ./lora_output
            argv += ["--adapter", body.get("adapter") or ""]
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
def api_agent_logs_logs():
    """SSE：实时推送 Agent 运行日志。"""
    return _sse_response(JOBS["agent"])


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
def api_tiny_logs_logs():
    """SSE：实时推送微型模型生成日志。"""
    return _sse_response(JOBS["tiny"])


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
    print("  提示: 训练/对话在后台独立运行 — 关掉网页或本窗口都不会中断，")
    print("        重新打开本程序会自动接上仍在运行的任务。")
    if ADOPTED_KINDS:
        print(f"  ★ 已接上仍在运行的作业: {', '.join(ADOPTED_KINDS)}")
    print("=" * 60)
    if not args.no_browser:
        threading.Timer(1.0, lambda: _open_browser(url)).start()
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        # Ctrl+C 只停服务器；子进程按设计继续跑（下次启动自动领养）
        print("\n[WebUI] 服务器已停止。后台作业如仍在运行会继续，重新启动将自动接上。")


def _open_browser(url: str):
    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()
