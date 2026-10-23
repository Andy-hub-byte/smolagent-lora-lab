/* ============================================================
 * smolagent-lora-lab Web 微调工作台 · 前端逻辑
 * 原生 JS，无依赖。与后端 app.py 的 REST/SSE 接口对接。
 * 中英双语: t()/tErr() 见 i18n.js; 语言切换时 onLangChange 重渲染动态内容。
 * ============================================================ */
"use strict";

/* ---------------- 工具 ---------------- */
const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function toast(msg, ms = 3600) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(tErr(data) || `HTTP ${r.status}`);
  return data;
}

function fmtDur(sec) {
  if (sec == null) return "";
  if (sec < 60) return `${sec | 0}s`;
  return `${(sec / 60) | 0}m${(sec % 60) | 0}s`;
}

/** 记录 select 当前值，重建选项后若仍存在则恢复（语言切换不丢选择） */
function withPreservedValue(sel, rebuild) {
  const prev = sel.value;
  rebuild();
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

/* ---------------- 页签（LM Studio 侧边栏导航） ---------------- */
function switchTab(tab) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".page").forEach((p) =>
    p.classList.toggle("active", p.id === `page-${tab}`));
  if (tab === "out") loadOutputs();
  if (tab === "data") loadDatasets();
  if (tab === "import") loadImportedModels();
  if (tab === "chat") buildChatSelects();
}

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

/* ============================================================
 * 通用作业面板（训练 / Agent 对话 / Tiny / 导入 共用模式）
 * 每个作业 = 一个 SSE 日志流 + 一个状态轮询 + 开始/停止按钮
 * ============================================================ */
const JOB_STATE_KEYS = {
  idle: "job.idle", running: "job.running", done: "job.done",
  error: "job.error", stopped: "job.stopped",
};

class JobView {
  constructor({ logEl, stateEl, metaEl, startBtn, stopBtn, progressWrap, progressBar, jobCard, emptyEl }) {
    Object.assign(this, { logEl, stateEl, metaEl, startBtn, stopBtn, progressWrap, progressBar, jobCard, emptyEl });
    this.es = null;
    this.poll = null;
    // 停止按钮文案键（语言切换时随 setState 刷新）
    this._stopKey = stopBtn ? stopBtn.dataset.stopLabel : null;
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
    if (this.emptyEl) {
      this.emptyEl.hidden = true;
      this.logEl.hidden = false;
    }
    const el = this.logEl;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    el.textContent += (el.textContent ? "\n" : "") + line;
    // 终端上限，防止超长训练爆内存
    if (el.textContent.length > 600000) el.textContent = el.textContent.slice(-400000);
    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  clearLog() {
    this.logEl.textContent = "";
    if (this.emptyEl) {
      this.emptyEl.hidden = false;
      this.logEl.hidden = true;
    }
  }

  setState(state) {
    this.state = state;
    this.stateEl.textContent = t(JOB_STATE_KEYS[state] || state);
    this.stateEl.className = `job-state ${state}`;
    const running = state === "running";
    if (this.startBtn) this.startBtn.hidden = running;
    if (this.stopBtn) {
      this.stopBtn.hidden = !running;
      this.stopBtn.disabled = false;
      // 刷新停止按钮文案（语言切换后也正确）
      if (this._stopKey) this.stopBtn.textContent = t(this._stopKey);
    }
    if (this.jobCard && state !== "idle") this.jobCard.hidden = false;
    if (!running && this.progressWrap) this.progressWrap.hidden = true;
  }

  /** 点击停止后的即时反馈：按钮变"停止中…"并禁用，直到状态回来 */
  setStopping() {
    if (!this.stopBtn || this.stopBtn.hidden) return;
    this.stopBtn.textContent = t("job.stopping");
    this.stopBtn.disabled = true;
    setTimeout(() => { if (this.stopBtn) this.stopBtn.disabled = false; }, 8000);
  }

  setMeta(text) { if (this.metaEl) this.metaEl.textContent = text; }

  /** 轮询状态；state 变化时触发回调 */
  watchStatus(url, interval = 2000, onChange = () => {}) {
    this.stopWatch();
    const tick = async () => {
      try {
        const s = await api(url);
        if (this.state !== s.state) this.setState(s.state);
        const meta = [
          s.pid ? t("job.metaPid", { pid: s.pid }) : "",
          s.elapsed != null ? fmtDur(s.elapsed) : "",
          s.total_steps ? t("job.metaSteps", { n: s.total_steps }) : "",
          s.adopted ? (getLang() === "zh" ? "已接上" : "adopted") : "",
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
      [t("env.py"), env.python],
      [t("env.os"), env.platform],
      [t("env.device"), env.device === "cuda"
        ? t("env.gpu", { name: env.gpu_name, vram: env.vram_gb })
        : t("env.cpu")],
      [t("env.project"), env.project_root],
    ];
    $("env-table").querySelector("tbody").innerHTML =
      rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v) || "—"}</td></tr>`).join("");

    $("deps-table").querySelector("tbody").innerHTML = Object.entries(env.deps)
      .map(([k, v]) => `<tr><td>${esc(t(`dep.${k}`) || k)}</td><td class="${v ? "ok" : "miss"}">${esc(v) || t("dep.missing")}</td></tr>`)
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
    chipText.textContent = t("env.detectFail");
  }
}

/* ============================================================
 * 模型注册表 / 已导入模型 / 数据集 / 产出
 * ============================================================ */
let REGISTRY = {};
let IMPORTED = [];   // /api/models/imported 的 entries
let OUTPUTS = [];    // /api/outputs 的扁平适配器列表

async function loadModels() {
  const data = await api("/api/models");
  REGISTRY = data;
  // —— 训练页: 基座下拉（在线短名 + 本地模型 两个分组）——
  const sel = $("f-model-preset");
  withPreservedValue(sel, () => {
    sel.innerHTML = "";
    const gOnline = document.createElement("optgroup");
    gOnline.label = t("train.onlineGroup");
    for (const short of Object.keys(data.registry || {})) {
      const opt = document.createElement("option");
      opt.value = short;
      opt.textContent = short;
      gOnline.appendChild(opt);
    }
    sel.appendChild(gOnline);
    const gLocal = document.createElement("optgroup");
    gLocal.label = t("train.localGroup");
    const locals = IMPORTED.filter((e) => e.valid);
    if (!locals.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = t("chat.noLocal");
      opt.disabled = true;
      gLocal.appendChild(opt);
    }
    for (const e of locals) {
      const opt = document.createElement("option");
      opt.value = e.path;
      opt.textContent = e.name;
      opt.dataset.local = "1";
      gLocal.appendChild(opt);
    }
    sel.appendChild(gLocal);
  });

  // 用脚本默认值预填表单（与命令行默认一致；仅首次）
  if (!loadModels._filled) {
    loadModels._filled = true;
    const d = data.defaults || {};
    if (d.model) sel.value = d.model;
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
    // 本地来源时同步来源下拉
    syncSourceWithPreset();
  }
  updateCmdPreview();
}

/** 训练页: 下拉选了本地模型 → 来源自动切 local；选在线短名 → 离开 local */
function syncSourceWithPreset() {
  const sel = $("f-model-preset");
  const opt = sel.selectedOptions[0];
  const isLocal = opt && opt.dataset.local === "1";
  const src = $("f-source");
  if (isLocal) src.value = "local";
  else if (src.value === "local") src.value = "modelscope";
}

/* ---- 已导入模型（模型导入页，支持搜索过滤） ---- */
async function loadImportedModels() {
  const tbody = $("imp-table").querySelector("tbody");
  try {
    const data = await api("/api/models/imported");
    IMPORTED = data.entries || [];
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="hint">${esc(t("imp.loadFail"))}${esc(e.message)}</td></tr>`;
    return;
  }
  renderImportedRows();
}

function renderImportedRows() {
  const tbody = $("imp-table").querySelector("tbody");
  if (!IMPORTED.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="hint">${esc(t("imp.empty"))}</td></tr>`;
    return;
  }
  const q = ($("imp-filter").value || "").trim().toLowerCase();
  const rows = q ? IMPORTED.filter((e) => e.name.toLowerCase().includes(q)) : IMPORTED;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="hint">${esc(t("imp.noMatch"))}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((e) => {
    const zh = getLang() === "zh";
    const filesTxt = zh ? "个" : (e.weight_count === 1 ? "file" : "files");
    return `
    <tr>
      <td><code>${esc(e.name)}</code></td>
      <td>${esc(e.arch) || "—"}</td>
      <td>${e.weight_count} ${esc(filesTxt)}${e.has_tokenizer ? "" : " " + esc(t("imp.noTok"))}</td>
      <td class="${e.valid ? "ok" : "miss"}">${e.valid ? esc(t("imp.valid")) : esc(t("imp.invalid"))}</td>
      <td class="num">${e.size_mb}</td>
      <td class="num">${esc(e.mtime)}</td>
      <td class="ops"></td>
    </tr>`;
  }).join("");
  // 操作列按钮（事件绑定，避免内联 onclick）
  [...tbody.querySelectorAll("tr")].forEach((tr, i) => {
    const e = rows[i];
    const td = tr.querySelector("td.ops");
    if (e.valid) {
      const b1 = document.createElement("button");
      b1.className = "btn sm";
      b1.textContent = t("imp.useBase");
      b1.addEventListener("click", () => useAsBase(e));
      td.appendChild(b1);
    }
    const b2 = document.createElement("button");
    b2.className = "btn danger sm";
    b2.textContent = getLang() === "zh" ? "删除" : "Delete";
    b2.addEventListener("click", () => deleteImported(e));
    td.appendChild(b2);
  });
}

$("imp-filter").addEventListener("input", renderImportedRows);

function useAsBase(e) {
  $("f-source").value = "local";
  $("f-model-preset").value = e.path;
  $("f-model").value = "";
  syncSourceWithPreset();
  updateCmdPreview();
  switchTab("train");
  toast(t("imp.baseFilled", { name: e.name }));
}

async function deleteImported(e) {
  if (!confirm(t("imp.confirmDelete", { name: e.name }))) return;
  try {
    await api("/api/models/imported/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: e.delete_name || e.name }),
    });
    toast(t("imp.deleted", { name: e.name }));
    loadImportedModels();
    buildChatSelects();
    loadModels();  // 训练页本地分组可能空了
  } catch (err) { toast(err.message); }
}

/* ---- 数据集 ---- */
async function loadDatasets() {
  try {
    const data = await api("/api/datasets");
    $("ds-dir").textContent = data.dir;
    const sel = $("f-ds-file");
    withPreservedValue(sel, () => {
      sel.innerHTML = "";
      for (const f of data.files) {
        const opt = document.createElement("option");
        opt.value = f.path;
        opt.textContent = `${f.name}（${f.count} ${t("data.rowsSuffix").trim()}）`;
        sel.appendChild(opt);
      }
    });
    const tbody = $("ds-table").querySelector("tbody");
    tbody.innerHTML = "";
    for (const f of data.files) {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td");
      td1.textContent = f.name;
      const td2 = document.createElement("td");
      td2.className = "num";
      td2.textContent = f.count;
      const td3 = document.createElement("td");
      td3.className = "num";
      td3.textContent = `${f.size_kb} KB`;
      const td4 = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = "btn sm";
      btn.textContent = t("data.preview");
      btn.addEventListener("click", () => previewDataset(f));
      td4.appendChild(btn);
      tr.append(td1, td2, td3, td4);
      tbody.appendChild(tr);
    }
    if (!data.files.length) {
      tbody.innerHTML = `<tr><td colspan="4" class="hint">${esc(t("data.empty"))}</td></tr>`;
    }
  } catch (e) { /* 静默 */ }
}

async function previewDataset(f) {
  const card = $("ds-preview-card");
  $("ds-preview-name").textContent = `${f.name}（${f.count} ${t("data.rowsSuffix").trim()}）`;
  card.hidden = false;
  const box = $("ds-preview");
  box.innerHTML = `<p class="hint">${esc(t("data.loading"))}</p>`;
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
    if (!data.rows || !data.rows.length) box.innerHTML = `<p class="hint">${esc(t("data.emptyFile"))}</p>`;
  } catch (e) {
    box.innerHTML = `<p class="hint">${esc(t("data.previewFail"))}${esc(e.message)}</p>`;
  }
}

/* ---- 产出（适配器列表 + 对话页适配器下拉共用） ---- */
async function loadOutputs() {
  try {
    const data = await api("/api/outputs");
    OUTPUTS = [];
    for (const dir of data.dirs) {
      for (const e of dir.entries) {
        OUTPUTS.push({ ...e, path: e.path, dir: dir.dir });
      }
    }
  } catch { OUTPUTS = []; }
  renderOutputsList();
  fillAdapterSelect();
}

function renderOutputsList() {
  const box = $("out-list");
  box.innerHTML = "";
  if (!OUTPUTS.length) {
    box.innerHTML = `<p class="hint">${esc(t("out.empty"))}</p>`;
    return;
  }
  for (const e of OUTPUTS) {
    const item = document.createElement("div");
    item.className = "out-item";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `
      <span class="name">${esc(e.dir)}/${esc(e.name)}</span>
      <span class="sub">${e.size_mb} MB · ${esc(e.mtime)}</span>`;
    const btn = document.createElement("button");
    btn.className = "btn sm";
    btn.textContent = t("out.useAdapter");
    btn.addEventListener("click", () => {
      const sel = $("chat-adapter");
      sel.value = e.path;
      if (sel.value !== e.path) {   // 选项不在（刚训练完）→ 重建后再选
        fillAdapterSelect(e.path);
      }
      switchTab("chat");
      toast(t("out.filled", { path: e.path }));
    });
    item.append(meta, btn);
    box.appendChild(item);
  }
}

/* ============================================================
 * 模型导入（三种方式）
 * ============================================================ */
const importView = new JobView({
  logEl: $("imp-log"), stateEl: $("imp-state"), metaEl: $("imp-meta"),
  stopBtn: $("btn-imp-stop"), jobCard: $("imp-job-card"),
});

$("btn-imp-path").addEventListener("click", async () => {
  const path = $("imp-path").value.trim();
  if (!path) { toast(t("imp.needPath")); return; }
  try {
    const r = await api("/api/models/import/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path,
        name: $("imp-path-name").value.trim() || undefined,
        overwrite: $("imp-path-overwrite").checked ? true : undefined,
      }),
    });
    toast(r.message || t("imp.imported", { name: r.entry.name, size: r.entry.size_mb }));
    $("imp-path").value = "";
    $("imp-path-name").value = "";
    loadImportedModels();
    buildChatSelects();
    loadModels();
  } catch (e) { toast(e.message); }
});

$("btn-imp-repo").addEventListener("click", async () => {
  const repo = $("imp-repo").value.trim();
  if (!repo) { toast(t("imp.needRepo")); return; }
  try {
    const r = await api("/api/models/import/repo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: $("imp-source").value,
        repo,
        name: $("imp-repo-name").value.trim() || undefined,
        overwrite: $("imp-repo-overwrite").checked ? true : undefined,
      }),
    });
    importView.clearLog();
    importView.appendLog(`$ ${r.cmd}`);
    importView.setState("running");
    importView.connectSSE("/api/models/import/logs");
    importView.watchStatus("/api/models/import/status", 2000, onImportStatus);
    toast(t("imp.repoStarted"));
  } catch (e) { toast(e.message); }
});

function onImportStatus(s) {
  if (s.state === "running") return;
  importView.stopWatch();
  if (s.state === "done") {
    toast(t("imp.repoDone", { name: lastRepoName() }));
    loadImportedModels();
    buildChatSelects();
    loadModels();
  } else if (s.state === "error") {
    toast(t("imp.repoErr"));
  }
}

function lastRepoName() {
  return $("imp-repo-name").value.trim() || $("imp-repo").value.trim().split("/").pop() || "?";
}

$("btn-imp-upload").addEventListener("click", async () => {
  const name = $("imp-upload-name").value.trim();
  if (!name) { toast(t("imp.needName")); return; }
  const files = $("imp-files").files;
  if (!files.length) { toast(t("imp.needFiles")); return; }
  const btn = $("btn-imp-upload");
  btn.disabled = true;
  btn.textContent = t("imp.uploading");
  try {
    const fd = new FormData();
    fd.append("name", name);
    if ($("imp-upload-overwrite").checked) fd.append("overwrite", "1");
    for (const f of files) fd.append("files", f);
    const r = await fetch("/api/models/import/upload", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(tErr(data) || `HTTP ${r.status}`);
    toast(data.message || t("imp.imported", { name, size: data.entry ? data.entry.size_mb : "?" }));
    $("imp-upload-name").value = "";
    $("imp-files").value = "";
    loadImportedModels();
    buildChatSelects();
    loadModels();
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = t("imp.btnUpload");
  }
});

importView.onStop = async () => {
  importView.setStopping();
  try { await api("/api/models/import/stop", { method: "POST" }); }
  catch (e) { toast(e.message); }
};

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
    learning_rate: $("f-lr").value,
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
  $("train-cmd-preview").innerHTML = `${esc(t("train.cmdPreview"))} <code>${esc(cmd)}</code>`;
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
  trainView.setStopping();
  try {
    await api("/api/train/stop", { method: "POST" });
    toast(getLang() === "zh" ? "已发送停止信号，训练进程正在退出…" : "Stop signal sent — training process is exiting…");
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
    if (s.state === "error") toast(t("train.errToast"));
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
    box.textContent = t("chart.empty");
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
  <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="loss">
    <defs>
      <linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="rgba(91,141,239,0.35)"/>
        <stop offset="100%" stop-color="rgba(91,141,239,0)"/>
      </linearGradient>
    </defs>
    <line x1="${pad.l}" y1="${Y(y1)}" x2="${W - pad.r}" y2="${Y(y1)}" stroke="#2a2a34" stroke-width="1"/>
    <line x1="${pad.l}" y1="${Y(y0)}" x2="${W - pad.r}" y2="${Y(y0)}" stroke="#2a2a34" stroke-width="1"/>
    <path d="${area}" fill="url(#lg)"/>
    <path d="${path}" fill="none" stroke="#5b8def" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${X(last.step).toFixed(1)}" cy="${Y(last.loss).toFixed(1)}" r="3.5" fill="#5b8def"/>
    <text x="${pad.l - 6}" y="${Y(y1) + 4}" fill="#66666f" font-size="10" text-anchor="end">${y1.toFixed(2)}</text>
    <text x="${pad.l - 6}" y="${Y(y0) + 4}" fill="#66666f" font-size="10" text-anchor="end">${y0.toFixed(2)}</text>
    <text x="${W - pad.r}" y="${H - 6}" fill="#66666f" font-size="10" text-anchor="end">step ${last.step}${totalSteps ? `/${totalSteps}` : ""}</text>
    <text x="${pad.l}" y="${H - 6}" fill="#66666f" font-size="10">${esc(t("chart.current"))} ${last.loss.toFixed(4)} · ${esc(t("chart.min"))} ${min.toFixed(4)}</text>
  </svg>`;
}

// 表单变化即时刷新等价命令预览
$("train-form").addEventListener("input", () => { syncSourceWithPreset(); updateCmdPreview(); });
$("f-model-preset").addEventListener("change", () => { syncSourceWithPreset(); updateCmdPreview(); });
$("btn-train-reset").addEventListener("click", () => {
  loadModels._filled = false;
  loadModels().then(() => toast(getLang() === "zh" ? "已重置为脚本默认推荐值" : "Reset to script defaults"));
});
$("btn-goto-chat").addEventListener("click", () => {
  switchTab("chat");
  // 预选最新产出作为适配器，省一步操作
  if (OUTPUTS.length) fillAdapterSelect(OUTPUTS[0].path);
});

/* ============================================================
 * 模型对话（两步选择器：选模型 → 选适配器）
 * ============================================================ */
const agentView = new JobView({
  logEl: $("agent-log"), stateEl: $("agent-state"), metaEl: $("agent-meta"),
  startBtn: $("btn-agent-run"), stopBtn: $("btn-agent-stop"),
  emptyEl: $("chat-empty"),
});

const CHAT_MANUAL = "__manual__";

/** 重建对话页两个下拉（语言切换 / 数据变化时调用） */
function buildChatSelects(preferAdapterPath) {
  // —— 第一步: 模型 ——
  const mSel = $("chat-model");
  withPreservedValue(mSel, () => {
    mSel.innerHTML = "";
    // 本地模型组
    const gLocal = document.createElement("optgroup");
    gLocal.label = t("chat.localGroup");
    const locals = IMPORTED.filter((e) => e.valid);
    if (!locals.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = t("chat.noLocal");
      opt.disabled = true;
      gLocal.appendChild(opt);
    }
    for (const e of locals) {
      const opt = document.createElement("option");
      opt.value = e.path;
      opt.textContent = e.name;
      opt.dataset.kind = "local";
      opt.dataset.path = e.path;
      gLocal.appendChild(opt);
    }
    mSel.appendChild(gLocal);
    // 在线模型组
    const gOnline = document.createElement("optgroup");
    gOnline.label = t("chat.onlineGroup");
    for (const [short, srcs] of Object.entries(REGISTRY.registry || {})) {
      const src = srcs.modelscope ? "modelscope" : "huggingface";
      const opt = document.createElement("option");
      opt.value = short;
      opt.textContent = short;
      opt.dataset.kind = "online";
      opt.dataset.source = src;
      gOnline.appendChild(opt);
    }
    mSel.appendChild(gOnline);
    // Ollama / 手动
    const oOllama = document.createElement("option");
    oOllama.value = "__ollama__";
    oOllama.textContent = "🦙 " + t("chat.ollamaOpt");
    oOllama.dataset.kind = "ollama";
    mSel.appendChild(oOllama);
    const oManual = document.createElement("option");
    oManual.value = CHAT_MANUAL;
    oManual.textContent = t("chat.manualOpt");
    oManual.dataset.kind = "manual";
    mSel.appendChild(oManual);
  });
  updateChatTip();
  // —— 第二步: 适配器 ——
  fillAdapterSelect(preferAdapterPath);
}

/** 适配器下拉：不使用 / 训练产出 / 手动 */
function fillAdapterSelect(preferPath) {
  const sel = $("chat-adapter");
  withPreservedValue(sel, () => {
    sel.innerHTML = "";
    const oNone = document.createElement("option");
    oNone.value = "";
    oNone.textContent = t("chat.noAdapter");
    sel.appendChild(oNone);
    if (OUTPUTS.length) {
      const g = document.createElement("optgroup");
      g.label = t("chat.adapterGroup");
      for (const e of OUTPUTS) {
        const opt = document.createElement("option");
        opt.value = e.path;
        opt.textContent = `${e.dir}/${e.name}`;
        opt.dataset.kind = "path";
        g.appendChild(opt);
      }
      sel.appendChild(g);
    } else {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = t("chat.noAdapters");
      opt.disabled = true;
      sel.appendChild(opt);
    }
    const oManual = document.createElement("option");
    oManual.value = CHAT_MANUAL;
    oManual.textContent = t("chat.manualAdapterOpt");
    oManual.dataset.kind = "manual";
    sel.appendChild(oManual);
  });
  if (preferPath) sel.value = preferPath;
  updateChatTip();
}

/** 第一步/第二步选中项变化时，更新提示文案 */
function updateChatTip() {
  const tip = $("chat-tip");
  const mOpt = $("chat-model").selectedOptions[0];
  const aOpt = $("chat-adapter").selectedOptions[0];
  let html = "";
  if (mOpt) {
    if (mOpt.dataset.kind === "local") html = t("chat.tipLocal");
    else if (mOpt.dataset.kind === "online") html = t("chat.tipOnline");
    else if (mOpt.dataset.kind === "ollama") html = t("chat.tipOllama");
    else if (mOpt.dataset.kind === "manual") html = t("chat.tipManual");
  }
  if (aOpt && aOpt.dataset.kind === "manual") html += (html ? "<br>" : "") + t("chat.tipManual");
  tip.innerHTML = html;
  tip.hidden = !html;
}

/* ---- 聊天页：参数面板开关（LM Studio 右侧设置栏） ---- */
$("btn-chat-params").addEventListener("click", () => {
  $("chat-params").classList.toggle("open");
});
// 宽屏默认展开参数面板，与 LM Studio 一致
if (window.innerWidth > 1180) $("chat-params").classList.add("open");

$("chat-model").addEventListener("change", updateChatTip);
$("chat-adapter").addEventListener("change", () => {
  updateChatTip();
  // 选手动路径时展开高级设置，直接把光标送进去
  const aOpt = $("chat-adapter").selectedOptions[0];
  if (aOpt && aOpt.dataset.kind === "manual") {
    const det = document.querySelector("details.advanced");
    if (det) det.open = true;
    $("a-adapter").focus();
  }
});

/* ---- 回车发送（Shift+Enter 换行），与 LM Studio 一致 ---- */
$("a-task").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    if (!$("btn-agent-run").hidden) agentView.onStart();
  }
});

agentView.onStart = async () => {
  const task = $("a-task").value.trim();
  if (!task) { toast(t("chat.needTask")); return; }
  const mOpt = $("chat-model").selectedOptions[0];
  if (!mOpt || !mOpt.value) { toast(t("chat.needModel")); return; }
  const kind = mOpt.dataset.kind;
  if (kind === "manual" && !$("a-model-path").value.trim()) {
    document.querySelector("details.advanced").open = true;
    $("a-model-path").focus();
    toast(t("chat.needModel"));
    return;
  }

  // 用户消息气泡（LM Studio 聊天风格）
  const userEl = $("chat-user");
  userEl.textContent = task;
  userEl.hidden = false;

  const body = {
    task,
    max_new_tokens: +$("a-tokens").value,
    max_steps: +$("a-steps").value,
  };
  if (kind === "ollama") {
    body.mode = "B";
    body.ollama_url = $("a-ollama-url").value.trim();
    body.ollama_model = $("a-ollama-model").value.trim();
  } else {
    if (kind === "online") {
      body.mode = "A";
      body.model_source = mOpt.dataset.source || "modelscope";
      body.model = mOpt.value;
    } else {
      body.mode = "C";
      body.model_path = kind === "manual" ? $("a-model-path").value.trim() : mOpt.dataset.path;
    }
    // 适配器：空串 = 不使用（显式传空，避免回退默认 ./lora_output）
    const aOpt = $("chat-adapter").selectedOptions[0];
    body.adapter = aOpt && aOpt.dataset.kind === "manual"
      ? $("a-adapter").value.trim()
      : (aOpt ? aOpt.value : "");
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
  agentView.setStopping();
  try { await api("/api/agent/stop", { method: "POST" }); }
  catch (e) { toast(e.message); }
};

/* ============================================================
 * 微型测试模型
 * ============================================================ */
const tinyView = new JobView({
  logEl: $("tiny-log"), stateEl: $("tiny-state"),
  stopBtn: $("btn-tiny-stop"),
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
        if (s.state === "done") {
          toast(t("tiny.done"));
          loadImportedModels();
          buildChatSelects();
          loadModels();
        }
      }
    });
  } catch (e) { toast(e.message); }
};

tinyView.onStop = async () => {
  tinyView.setStopping();
  try { await api("/api/tiny/stop", { method: "POST" }); }
  catch (e) { toast(e.message); }
};

/* ============================================================
 * 断线续跑：页面加载时探测各作业状态
 * （服务器重启过也能接上仍在跑的作业）
 * ============================================================ */
const JOB_TAB_KEYS = { train: "tab.train", agent: "tab.chat", tiny: "job.tiny", import: "tab.import" };

async function resumeJobStates() {
  const adopted = [];
  for (const [path, view, sse] of [
    ["/api/train/status", trainView, "/api/train/logs"],
    ["/api/agent/status", agentView, "/api/agent/logs"],
    ["/api/tiny/status", tinyView, "/api/tiny/logs"],
    ["/api/models/import/status", importView, "/api/models/import/logs"],
  ]) {
    try {
      const s = await api(path);
      view.setState(s.state);
      if (s.adopted) adopted.push(s.kind);
      if (s.state === "running") {
        view.connectSSE(sse);
        view.watchStatus(path, 2000, (st) => { if (st.state !== "running") view.stopWatch(); });
      }
    } catch { /* 忽略 */ }
  }
  if (adopted.length) {
    const kinds = adopted.map((k) => t(JOB_TAB_KEYS[k] || k)).join("、");
    toast(t("app.adopted", { kinds }), 6000);
  }
}

/* ============================================================
 * 启动
 * ============================================================ */
async function boot() {
  initLang();
  // 先拿已导入模型（训练/对话下拉的本地组依赖它），再并行加载其余
  await loadImportedModels();
  await Promise.all([loadEnv(), loadModels(), loadDatasets(), loadOutputs()]);
  buildChatSelects();
  resumeJobStates();
  // 语言切换：重渲染所有动态内容
  onLangChange(() => {
    loadEnv();
    loadModels();
    loadDatasets();
    loadOutputs();
    loadImportedModels();
    buildChatSelects();
  });
}

boot();
