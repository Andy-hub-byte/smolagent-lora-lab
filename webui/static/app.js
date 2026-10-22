/* ============================================================
 * smolagent-lora-lab Web 微调工作台 · 前端逻辑
 * 原生 JS，无依赖。与后端 app.py 的 REST/SSE 接口对接。
 * ============================================================ */
"use strict";

/* ---------------- 工具 ---------------- */
const $ = (id) => document.getElementById(id);

function toast(msg, ms = 3200) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function fmtDur(sec) {
  if (sec == null) return "";
  if (sec < 60) return `${sec | 0}s`;
  return `${(sec / 60) | 0}m${(sec % 60) | 0}s`;
}

/* ---------------- 页签 ---------------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".page").forEach((p) =>
      p.classList.toggle("active", p.id === `page-${btn.dataset.tab}`));
    if (btn.dataset.tab === "out") loadOutputs();
    if (btn.dataset.tab === "data") loadDatasets();
  });
});

/* ---------------- 通用作业面板（训练 / Agent / Tiny 共用模式） ----------------
 * 每个作业 = 一个 SSE 日志流 + 一个状态轮询 + 开始/停止按钮
 */
class JobView {
  constructor({ logEl, stateEl, metaEl, startBtn, stopBtn, progressWrap, progressBar }) {
    Object.assign(this, { logEl, stateEl, metaEl, startBtn, stopBtn, progressWrap, progressBar });
    this.es = null;
    this.poll = null;
    if (startBtn) startBtn.addEventListener("click", () => this.onStart && this.onStart());
    if (stopBtn) stopBtn.addEventListener("click", () => this.onStop && this.onStop());
  }

  connectSSE(url) {
    this.disconnectSSE();
    this.es = new EventSource(url);
    this.es.onmessage = (ev) => {
      let d = {};
      try { d = JSON.parse(ev.data); } catch { return; }
      if (d.hello) return;
      this.appendLog(d.line);
    };
    this.es.onerror = () => { /* 浏览器自动重连 */ };
  }

  disconnectSSE() {
    if (this.es) { this.es.close(); this.es = null; }
  }

  appendLog(line) {
    const el = this.logEl;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    el.textContent += (el.textContent ? "\n" : "") + line;
    // 终端上限 6000 行，防止超长训练爆内存
    if (el.textContent.length > 600000) el.textContent = el.textContent.slice(-400000);
    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  clearLog() { this.logEl.textContent = ""; }

  setState(state) {
    this.stateEl.textContent = {
      idle: "空闲", running: "运行中", done: "已完成", error: "出错", stopped: "已停止"
    }[state] || state;
    this.stateEl.className = `job-state ${state}`;
    const running = state === "running";
    if (this.startBtn) this.startBtn.hidden = running;
    if (this.stopBtn) this.stopBtn.hidden = !running;
    if (!running && this.progressWrap) this.progressWrap.hidden = true;
  }

  setMeta(text) { if (this.metaEl) this.metaEl.textContent = text; }

  /** 轮询状态；state 变化时触发回调 */
  watchStatus(url, interval = 2000, onChange = () => {}) {
    this.stopWatch();
    const tick = async () => {
      try {
        const s = await api(url);
        this.setState(s.state);
        const meta = [
          s.pid ? `PID ${s.pid}` : "",
          s.elapsed != null ? fmtDur(s.elapsed) : "",
          s.total_steps ? `共 ${s.total_steps} 步` : "",
        ].filter(Boolean).join(" · ");
        this.setMeta(meta);
        onChange(s);
        if (s.state !== "running") this.stopWatch();
      } catch { /* 忽略瞬时错误 */ }
    };
    tick();
    this.poll = setInterval(tick, interval);
  }

  stopWatch() {
    if (this.poll) { clearInterval(this.poll); this.poll = null; }
  }
}

/* ============================================================
 * 环境体检
 * ============================================================ */
async function loadEnv() {
  const chip = $("env-chip"), chipText = $("env-chip-text");
  try {
    const env = await api("/api/env");
    const dev = env.device === "cuda" ? `GPU ${env.gpu_name || ""}` : "CPU";
    chip.className = "chip " + (env.missing.length ? "chip-warn" : "chip-ok");
    chipText.textContent = `Python ${env.python} · ${dev}${env.vram_gb ? ` · ${env.vram_gb}GB` : ""}`;
    chip.title = env.platform;

    const rows = [
      ["Python", env.python],
      ["系统", env.platform],
      ["设备", env.device === "cuda" ? `CUDA GPU（${env.gpu_name}，${env.vram_gb} GB）` : "CPU（无 CUDA GPU）"],
      ["项目目录", env.project_root],
    ];
    $("env-table").querySelector("tbody").innerHTML =
      rows.map(([k, v]) => `<tr><td>${k}</td><td>${v || "—"}</td></tr>`).join("");

    const depNames = {
      torch: "PyTorch", transformers: "Transformers", peft: "PEFT", trl: "TRL",
      datasets: "Datasets", smolagents: "smolagents", modelscope: "ModelScope",
      accelerate: "Accelerate", flask: "Flask（本工作台）",
    };
    $("deps-table").querySelector("tbody").innerHTML = Object.entries(env.deps)
      .map(([k, v]) => `<tr><td>${depNames[k] || k}</td><td class="${v ? "ok" : "miss"}">${v || "未安装"}</td></tr>`)
      .join("");

    const adv = $("env-advice");
    adv.innerHTML = "";
    for (const a of env.advice || []) {
      const div = document.createElement("div");
      div.className = "advice-item";
      div.textContent = a;
      adv.appendChild(div);
    }
  } catch (e) {
    chip.className = "chip chip-err";
    chipText.textContent = "环境检测失败";
  }
}

/* ============================================================
 * 模型注册表 / 数据集 / 产出
 * ============================================================ */
let REGISTRY = {};

async function loadModels() {
  const data = await api("/api/models");
  REGISTRY = data;
  const sel = $("f-model-preset");
  sel.innerHTML = "";
  for (const short of Object.keys(data.registry || {})) {
    const opt = document.createElement("option");
    opt.value = short;
    opt.textContent = short;
    sel.appendChild(opt);
  }
  // Agent 测试台的基座短名补全
  const ml = $("model-list");
  ml.innerHTML = "";
  for (const short of Object.keys(data.registry || {})) {
    const opt = document.createElement("option");
    opt.value = short;
    ml.appendChild(opt);
  }
  // 用脚本默认值预填表单（与命令行默认一致）
  const d = data.defaults || {};
  if (d.model) { $("f-model").value = ""; sel.value = d.model; }
  if (d.model_cache_dir) $("f-cache").value = d.model_cache_dir;
  if (d.lora_rank) $("f-rank").value = d.lora_rank;
  if (d.lora_alpha) $("f-alpha").value = d.lora_alpha;
  if (d.lora_dropout != null) $("f-dropout").value = d.lora_dropout;
  if (d.batch_size) $("f-bs").value = d.batch_size;
  if (d.grad_accum) $("f-ga").value = d.grad_accum;
  if (d.learning_rate) $("f-lr").value = d.learning_rate;
  if (d.num_epochs) $("f-ep").value = d.num_epochs;
  if (d.max_steps != null) $("f-steps").value = d.max_steps;
  if (d.max_seq_len) $("f-seq").value = d.max_seq_len;
  if (d.output_dir) $("f-out").value = d.output_dir;
  if (d.save_steps) $("f-save").value = d.save_steps;
  if (d.save_total_limit) $("f-limit").value = d.save_total_limit;
  updateCmdPreview();
}

async function loadDatasets() {
  try {
    const data = await api("/api/datasets");
    $("ds-dir").textContent = data.dir;
    const sel = $("f-ds-file");
    sel.innerHTML = "";
    const tbody = $("ds-table").querySelector("tbody");
    tbody.innerHTML = "";
    for (const f of data.files) {
      const opt = document.createElement("option");
      opt.value = f.path;
      opt.textContent = `${f.name}（${f.count} 条）`;
      sel.appendChild(opt);
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="num">${f.name}</td><td class="num">${f.count}</td><td class="num">${f.size_kb} KB</td>`;
      const td = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = "btn sm";
      btn.textContent = "预览";
      btn.addEventListener("click", () => previewDataset(f));
      td.appendChild(btn);
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
    if (!data.files.length) {
      tbody.innerHTML = `<tr><td colspan="4">datasets 目录下没有 jsonl 文件</td></tr>`;
    }
  } catch (e) { /* 静默 */ }
}

async function previewDataset(f) {
  const card = $("ds-preview-card");
  $("ds-preview-name").textContent = `${f.name}（${f.count} 条）`;
  card.hidden = false;
  const box = $("ds-preview");
  box.innerHTML = "<p class='hint'>加载中…</p>";
  try {
    const data = await api(`/api/datasets/preview?path=${encodeURIComponent(f.path)}&limit=5`);
    box.innerHTML = "";
    (data.rows || []).forEach((row, i) => {
      const div = document.createElement("div");
      div.className = "preview-row";
      const head = document.createElement("div");
      head.className = "head";
      head.textContent = `#${i + 1}`;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(row, null, 2);
      div.append(head, pre);
      box.appendChild(div);
    });
    if (!data.rows || !data.rows.length) box.innerHTML = "<p class='hint'>空文件</p>";
  } catch (e) {
    box.innerHTML = `<p class="hint">预览失败：${e.message}</p>`;
  }
}

async function loadOutputs() {
  const box = $("out-list");
  try {
    const data = await api("/api/outputs");
    // 适配器自动补全列表
    const dl = $("adapter-list");
    dl.innerHTML = "";
    let any = false;
    for (const dir of data.dirs) {
      for (const e of dir.entries) {
        any = true;
        const opt = document.createElement("option");
        opt.value = e.path;
        dl.appendChild(opt);
        const item = document.createElement("div");
        item.className = "out-item";
        item.innerHTML = `
          <div class="meta">
            <span class="name">${dir.dir}/${e.name}</span>
            <span class="sub">${e.size_mb} MB · ${e.mtime}</span>
          </div>`;
        const btn = document.createElement("button");
        btn.className = "btn sm";
        btn.textContent = "用作适配器";
        btn.addEventListener("click", () => {
          const mode = document.querySelector('input[name="a-mode"]:checked').value;
          $(mode === "A" ? "a-adapter2" : "a-adapter").value = e.path;
          document.querySelector('.tab[data-tab="agent"]').click();
          toast(`已填入适配器路径：${e.path}`);
        });
        item.appendChild(btn);
        box.appendChild(item);
      }
    }
    if (!any) box.innerHTML = "<p class='hint'>暂无产出。训练完成后，适配器会出现在 lora_output/ 目录。</p>";
  } catch (e) {
    box.innerHTML = `<p class="hint">加载失败：${e.message}</p>`;
  }
}

/* ============================================================
 * 训练控制台
 * ============================================================ */
const trainView = new JobView({
  logEl: $("train-log"), stateEl: $("train-state"), metaEl: $("train-meta"),
  startBtn: $("btn-train-start"), stopBtn: $("btn-train-stop"),
  progressWrap: $("train-progress"), progressBar: $("train-progress-bar"),
});

function buildTrainPayload() {
  const preset = $("f-model-preset").value;
  const custom = $("f-model").value.trim();
  return {
    model_source: $("f-source").value,
    model: custom || preset,
    model_cache_dir: $("f-cache").value.trim() || undefined,
    dataset_mode: $("f-ds-mode").value,
    dataset_file: $("f-ds-mode").value === "single" ? $("f-ds-file").value : undefined,
    lora_rank: +$("f-rank").value,
    lora_alpha: +$("f-alpha").value,
    lora_dropout: +$("f-dropout").value,
    batch_size: +$("f-bs").value,
    grad_accum: +$("f-ga").value,
    learning_rate: +$("f-lr").value,
    num_epochs: +$("f-ep").value,
    max_steps: +$("f-steps").value,
    max_seq_len: +$("f-seq").value,
    output_dir: $("f-out").value.trim() || "./lora_output",
    save_steps: +$("f-save").value,
    save_total_limit: +$("f-limit").value,
    resume: $("f-resume").checked ? true : undefined,
    force: $("f-force").checked ? true : undefined,
  };
}

/** 生成等价的命令行（展示给用户，便于复现/学习） */
function updateCmdPreview() {
  const p = buildTrainPayload();
  let cmd = "python train_lora.py";
  cmd += ` --model-source ${p.model_source} --model ${p.model}`;
  cmd += ` --dataset-mode ${p.dataset_mode}`;
  if (p.dataset_file) cmd += ` --dataset-file ${p.dataset_file}`;
  cmd += ` --lora-rank ${p.lora_rank} --lora-alpha ${p.lora_alpha} --lora-dropout ${p.lora_dropout}`;
  cmd += ` --batch-size ${p.batch_size} --grad-accum ${p.grad_accum}`;
  cmd += ` --learning-rate ${p.learning_rate} --num-epochs ${p.num_epochs}`;
  cmd += ` --max-steps ${p.max_steps} --max-seq-len ${p.max_seq_len}`;
  cmd += ` --output-dir ${p.output_dir}`;
  if (p.resume) cmd += " --resume";
  if (p.force) cmd += " --force";
  $("train-cmd-preview").innerHTML = `等价命令行：<code>${cmd}</code>`;
}

trainView.onStart = async () => {
  try {
    const r = await api("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildTrainPayload()),
    });
    trainView.clearLog();
    trainView.appendLog(`$ ${r.cmd}`);
    trainView.setState("running");
    trainView.connectSSE("/api/train/logs");
    trainView.watchStatus("/api/train/status", 2000, onTrainStatus);
    startLossPolling();
  } catch (e) {
    toast(e.message);
  }
};

trainView.onStop = async () => {
  try {
    await api("/api/train/stop", { method: "POST" });
    toast("已发送停止信号，训练进程正在退出…");
  } catch (e) { toast(e.message); }
};

function onTrainStatus(s) {
  // 进度条：loss 点数 / 总步数
  if (s.total_steps && s.loss_count) {
    const pct = Math.min(100, (s.loss_count / s.total_steps) * 100);
    $("train-progress").hidden = false;
    $("train-progress-bar").style.width = pct + "%";
  }
  if (s.state !== "running") {
    trainView.stopWatch();
    stopLossPolling();
    loadOutputs(); // 训练结束刷新产出列表
    if (s.state === "error") toast("训练进程出错，请查看日志末尾定位问题");
  }
}

/* ---- loss 曲线（SVG 折线图） ---- */
let lossPoll = null;

function startLossPolling() {
  stopLossPolling();
  const tick = async () => {
    try {
      const d = await api("/api/train/loss");
      drawLoss(d.points || [], d.total_steps);
    } catch { /* 忽略 */ }
  };
  tick();
  lossPoll = setInterval(tick, 3000);
}

function stopLossPolling() {
  if (lossPoll) { clearInterval(lossPoll); lossPoll = null; }
}

function drawLoss(points, totalSteps) {
  const box = $("loss-chart");
  if (!points.length) {
    box.className = "chart-empty";
    box.textContent = "开始训练后显示 loss 变化";
    return;
  }
  box.className = "";
  const W = box.clientWidth || 640, H = 170;
  const pad = { l: 46, r: 12, t: 12, b: 22 };
  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 1e-6) { y0 -= 0.5; y1 += 0.5; }
  const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);

  const path = points.map((p, i) => `${i ? "L" : "M"}${X(p.step).toFixed(1)},${Y(p.loss).toFixed(1)}`).join("");
  const area = `${path}L${X(xs[xs.length - 1]).toFixed(1)},${Y(y0).toFixed(1)}L${X(xs[0]).toFixed(1)},${Y(y0).toFixed(1)}Z`;
  const last = points[points.length - 1];
  const min = Math.min(...ys);

  box.innerHTML = `
  <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="loss 曲线">
    <defs>
      <linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(79,143,247,0.35)"/>
        <stop offset="100%" stop-color="rgba(79,143,247,0)"/>
      </linearGradient>
    </defs>
    <line x1="${pad.l}" y1="${Y(y1)}" x2="${W - pad.r}" y2="${Y(y1)}" stroke="#22304f" stroke-width="1"/>
    <line x1="${pad.l}" y1="${Y(y0)}" x2="${W - pad.r}" y2="${Y(y0)}" stroke="#22304f" stroke-width="1"/>
    <path d="${area}" fill="url(#lg)"/>
    <path d="${path}" fill="none" stroke="#4f8ff7" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${X(last.step).toFixed(1)}" cy="${Y(last.loss).toFixed(1)}" r="3.5" fill="#4f8ff7"/>
    <text x="${pad.l - 6}" y="${Y(y1) + 4}" fill="#5d6f92" font-size="10" text-anchor="end">${y1.toFixed(2)}</text>
    <text x="${pad.l - 6}" y="${Y(y0) + 4}" fill="#5d6f92" font-size="10" text-anchor="end">${y0.toFixed(2)}</text>
    <text x="${W - pad.r}" y="${H - 6}" fill="#5d6f92" font-size="10" text-anchor="end">step ${last.step}${totalSteps ? `/${totalSteps}` : ""}</text>
    <text x="${pad.l}" y="${H - 6}" fill="#5d6f92" font-size="10">当前 ${last.loss.toFixed(4)} · 最低 ${min.toFixed(4)}</text>
  </svg>`;
}

// 表单变化即时刷新等价命令预览
$("train-form").addEventListener("input", updateCmdPreview);
$("btn-train-reset").addEventListener("click", () => { loadModels(); toast("已重置为脚本默认推荐值"); });

/* ============================================================
 * Agent 测试台
 * ============================================================ */
const agentView = new JobView({
  logEl: $("agent-log"), stateEl: $("agent-state"), metaEl: $("agent-meta"),
  startBtn: $("btn-agent-run"), stopBtn: $("btn-agent-stop"),
});

function currentAgentMode() {
  return document.querySelector('input[name="a-mode"]:checked').value;
}

document.querySelectorAll('input[name="a-mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    const m = currentAgentMode();
    $("a-fields-c").hidden = m !== "C";
    $("a-fields-a").hidden = m !== "A";
    $("a-fields-b").hidden = m !== "B";
  });
});

agentView.onStart = async () => {
  const task = $("a-task").value.trim();
  if (!task) { toast("请先填写任务内容"); return; }
  const mode = currentAgentMode();
  const body = {
    mode,
    task,
    max_new_tokens: +$("a-tokens").value,
    max_steps: +$("a-steps").value,
  };
  if (mode === "C") {
    body.model_path = $("a-model-path").value.trim();
    body.adapter = $("a-adapter").value.trim();
    if (!body.model_path) { toast("请填写本地基座模型路径"); return; }
  } else if (mode === "A") {
    body.model_source = $("a-source").value;
    body.model = $("a-model").value.trim();
    body.adapter = $("a-adapter2").value.trim();
  } else {
    body.ollama_url = $("a-ollama-url").value.trim();
    body.ollama_model = $("a-ollama-model").value.trim();
  }
  try {
    const r = await api("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    agentView.clearLog();
    agentView.appendLog(`$ ${r.cmd}`);
    agentView.setState("running");
    agentView.connectSSE("/api/agent/logs");
    agentView.watchStatus("/api/agent/status", 2000, (s) => {
      if (s.state !== "running") agentView.stopWatch();
    });
  } catch (e) { toast(e.message); }
};

agentView.onStop = async () => {
  try { await api("/api/agent/stop", { method: "POST" }); }
  catch (e) { toast(e.message); }
};

/* ============================================================
 * 微型测试模型
 * ============================================================ */
const tinyView = new JobView({
  logEl: $("tiny-log"), stateEl: $("tiny-state"),
  startBtn: $("btn-tiny"), stopBtn: $("btn-tiny-stop"),
});

tinyView.onStart = async () => {
  try {
    const r = await api("/api/tiny/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output: $("tiny-out").value.trim() || "./models/tiny-qwen-test" }),
    });
    tinyView.clearLog();
    tinyView.appendLog(`$ ${r.cmd}`);
    tinyView.setState("running");
    tinyView.connectSSE("/api/tiny/logs");
    tinyView.watchStatus("/api/tiny/status", 2000, (s) => {
      if (s.state !== "running") {
        tinyView.stopWatch();
        if (s.state === "done") toast("微型模型已生成，可在训练控制台用「本地路径」来源选择它");
      }
    });
  } catch (e) { toast(e.message); }
};

tinyView.onStop = async () => {
  try { await api("/api/tiny/stop", { method: "POST" }); }
  catch (e) { toast(e.message); }
};

// 页面加载时探测一次各作业状态（刷新页面不丢进行中的任务）
async function resumeJobStates() {
  for (const [path, view, sse] of [
    ["/api/train/status", trainView, "/api/train/logs"],
    ["/api/agent/status", agentView, "/api/agent/logs"],
    ["/api/tiny/status", tinyView, "/api/tiny/logs"],
  ]) {
    try {
      const s = await api(path);
      view.setState(s.state);
      if (s.state === "running") {
        view.connectSSE(sse);
        view.watchStatus(path, 2000, (st) => { if (st.state !== "running") view.stopWatch(); });
      }
    } catch { /* 忽略 */ }
  }
}

/* ============================================================
 * 启动
 * ============================================================ */
loadEnv();
loadModels();
loadDatasets();
resumeJobStates();
