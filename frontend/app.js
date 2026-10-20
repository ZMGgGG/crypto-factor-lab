const viewMeta = {
  overview: { title: "研究总览", description: "从真实历史数据开始，审计每个因子在样本外是否成立。" },
  data: { title: "数据检查", description: "查看固定数据版本的完整性、连续性和校验结果。" },
  factors: { title: "因子库", description: "生成和筛选候选因子，冻结参数后加入独立观察。" },
  observations: { title: "观察跟踪", description: "查看冻结候选在独立月度数据上的成本后表现。" },
  runs: { title: "实验记录", description: "每次实验绑定数据清单、标签定义和代码版本。" },
};

const toast = document.querySelector("#toast");
const experimentModal = document.querySelector("#experiment-modal");
const experimentForm = document.querySelector("#experiment-form");
const experimentStatus = document.querySelector("#experiment-status");
const miningModal = document.querySelector("#mining-modal");
const miningForm = document.querySelector("#mining-form");
const miningStatus = document.querySelector("#mining-status");
const candidateModal = document.querySelector("#candidate-modal");
let currentMiningRunId = null;
let toastTimer;

const factorTypeLabels = { time_series: "时间序列", volatility: "波动率", liquidity: "流动性", trend: "趋势", derivatives: "衍生品", direction: "方向", risk: "风险", condition: "条件" };
const semanticsLabels = { direct: "正向", inverse: "反向", neutral: "中性" };
let aiConfig = null;

document.querySelector(".page-wrap").insertAdjacentHTML("beforeend", `<div id="view-observations" class="view-panel"><section class="panel detail-panel observation-panel"><div class="panel-header"><div><h2>观察列表</h2><p>冻结候选后，只用独立归档数据跟踪成本后收益。</p></div><span class="badge badge-blue" id="observation-range">读取中</span></div><div class="table-wrap"><table><thead><tr><th>因子</th><th>观察区间</th><th>净收益</th><th>BTC收益</th><th>超额收益</th><th>最大回撤</th><th>换手</th><th>操作</th></tr></thead><tbody id="observation-results-body"><tr><td colspan="8" class="empty-cell">暂无观察中的候选。</td></tr></tbody></table></div></section></div>`);
document.querySelector("#view-data .panel-header").insertAdjacentHTML("beforeend", `<button class="primary-button small" id="update-data-button" type="button">更新数据</button>`);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 2400);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `API ${response.status}`);
  return payload;
}

async function waitForTask(taskId) {
  for (;;) {
    const task = await api(`/api/tasks/${taskId}`);
    if (task.status === "succeeded") return task.result;
    if (task.status === "failed") throw new Error(task.error || "实验失败");
    await new Promise((resolve) => window.setTimeout(resolve, 700));
  }
}

function renderChart(points) {
  const valid = points.filter((point) => point.momentum !== null);
  if (valid.length < 2) return;
  const width = 760;
  const height = 250;
  const prices = valid.map((point) => point.close);
  const factors = valid.map((point) => point.momentum);
  const pathFor = (values, top, bottom) => { const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1; return values.map((value, index) => `${index ? "L" : "M"}${(index / (values.length - 1)) * width} ${bottom - ((value - min) / span) * (bottom - top)}`).join(" "); };
  document.querySelector(".price-path").setAttribute("d", pathFor(prices, 35, 180));
  document.querySelector(".factor-path").setAttribute("d", pathFor(factors, 90, 220));
  document.querySelector("#chart-desc").textContent = `真实 BTCUSDT 价格与 24 小时动量，共 ${valid.length} 个采样点。`;
}

function setView(view) {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("is-active", item.dataset.view === view));
  document.querySelectorAll(".view-panel").forEach((panel) => panel.classList.toggle("is-visible", panel.id === `view-${view}`));
  document.querySelector("#view-title").textContent = viewMeta[view].title;
  document.querySelector("#page-heading").textContent = viewMeta[view].title;
  document.querySelector("#page-description").textContent = viewMeta[view].description;
}

function openExperiment() { experimentModal.hidden = false; experimentStatus.textContent = ""; experimentStatus.className = "form-status"; experimentForm.querySelector("input").focus(); }
function closeExperiment() { experimentModal.hidden = true; }
function setExperimentStatus(message, kind = "") { experimentStatus.textContent = message; experimentStatus.className = `form-status ${kind}`; }
function openMining() { miningModal.hidden = false; miningStatus.textContent = ""; miningStatus.className = "form-status"; miningForm.querySelector("input").focus(); }
function closeMining() { miningModal.hidden = true; }
function setMiningStatus(message, kind = "") { miningStatus.textContent = message; miningStatus.className = `form-status ${kind}`; }
function closeCandidate() { candidateModal.hidden = true; }

function dateControl(name, value, min = "", max = "") {
  return `<span class="date-input-group"><input name="${name}" type="text" inputmode="numeric" pattern="\\d{4}-\\d{2}-\\d{2}" placeholder="YYYY-MM-DD" value="${escapeHtml(value)}" data-date-text required><input type="date" min="${escapeHtml(min)}" max="${escapeHtml(max)}" value="${escapeHtml(value)}" data-date-picker aria-label="选择日期"></span><small>可直接输入 YYYY-MM-DD，也可以点击日历</small>`;
}

function validateDateFields(form) {
  for (const input of form.querySelectorAll("[data-date-text]")) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(input.value)) {
      input.focus();
      throw new Error("日期格式请使用 YYYY-MM-DD");
    }
    const picker = input.closest(".date-input-group")?.querySelector("[data-date-picker]");
    if (!picker || picker.value !== input.value || !picker.validity.valid) {
      input.focus();
      throw new Error("请输入有效且在可选范围内的日期");
    }
  }
}

document.addEventListener("input", (event) => {
  if (!event.target.matches("[data-date-text]")) return;
  const picker = event.target.closest(".date-input-group")?.querySelector("[data-date-picker]");
  if (picker && /^\d{4}-\d{2}-\d{2}$/.test(event.target.value)) picker.value = event.target.value;
});
document.addEventListener("change", (event) => {
  if (!event.target.matches("[data-date-picker]")) return;
  const text = event.target.closest(".date-input-group")?.querySelector("[data-date-text]");
  if (text) text.value = event.target.value;
});

function updateDatasetSummary(dataset, observationDataset) {
  document.querySelector("#scope-symbol").textContent = dataset.symbol;
  document.querySelector("#scope-range").textContent = `${dataset.first_timestamp.slice(0, 10)} → ${dataset.last_timestamp.slice(0, 10)}`;
  document.querySelector("#scope-observation-end").textContent = observationDataset ? observationDataset.last_timestamp.slice(0, 10) : "未配置";
  document.querySelector(".metric-grid .metric-card:first-child .metric-value").textContent = dataset.bars.toLocaleString();
  const observationCard = document.querySelector("#view-data .check-card:last-child");
  if (observationCard && observationDataset) {
    observationCard.querySelector("strong").textContent = "观察数据";
    observationCard.querySelector("small").textContent = `${observationDataset.bars.toLocaleString()} bars · 至 ${observationDataset.last_timestamp.slice(0, 10)}`;
  }
}

function updateResearchDateBounds(dataset) {
  const min = dataset.first_timestamp.slice(0, 10);
  const max = dataset.last_timestamp.slice(0, 10);
  for (const name of ["research_start", "research_end"]) {
    const text = miningForm.querySelector(`[name="${name}"]`);
    const picker = text?.closest(".date-input-group")?.querySelector("[data-date-picker]");
    if (!text || !picker) continue;
    picker.min = min;
    picker.max = max;
    if (name === "research_end" && text.value < max) {
      text.value = max;
      picker.value = max;
    }
  }
}

function renderAiStatus(status) {
  aiConfig = status || null;
  const label = document.querySelector("#ai-mode-status");
  if (!label) return;
  label.textContent = status?.configured
    ? `已配置真实 ${status.provider} API：${status.model}（首次挖掘时验证）`
    : "未配置真实 API；选择此模式会回退公式挖掘";
  label.className = status?.configured ? "ai-status-ready" : "ai-status-fallback";
}

function renderLatestReport(rows, tasks = []) {
  const body = document.querySelector(".table-panel tbody");
  if (!body) return;
  const historical = Array.isArray(rows) ? rows.slice(0, 6).map((row) => `<tr><td><strong>${row.factor}</strong><small>历史报告 · BTCUSDT</small></td><td><span class="mono">${row.factor}</span></td><td>forward_return_4</td><td>${Number(row.observations).toLocaleString()}</td><td class="${Number(row.mean_test_rank_ic) < 0 ? "negative" : ""}">${Number(row.mean_test_rank_ic).toFixed(4)}</td><td>${Number.isFinite(Number(row.net_return)) ? `${(Number(row.net_return) * 100).toFixed(2)}%` : "—"}</td><td><span class="badge ${Number(row.mean_test_rank_ic) > 0 ? "badge-green" : "badge-amber"}">${Number(row.mean_test_rank_ic) > 0 ? "候选" : "基线"}</span></td></tr>`).join("") : "";
  const taskRows = Array.isArray(tasks) ? tasks.slice(0, 4).map((task) => { const statusLabel = task.status === "succeeded" ? "已完成" : task.status === "failed" ? "失败" : task.status === "running" ? "运行中" : "排队中"; const badge = task.status === "succeeded" ? "badge-green" : task.status === "failed" ? "badge-amber" : "badge-blue"; const combo = task.result && Array.isArray(task.result.combination) ? task.result.combination.find((row) => row.combination === "equal_weight_all") : null; const isMining = task.payload && task.payload.task_type === "mining"; return `<tr><td><strong>页面任务 ${task.task_id.slice(0, 8)}</strong><small>页面提交 · ${task.submitted_at.replace("T", " ").slice(0, 19)}</small></td><td><span class="mono">${isMining ? `${task.payload.max_candidates || "—"} candidates` : `${task.payload.momentum_window}h suite`}</span></td><td>${isMining ? "forward_return" : `forward_return_${task.payload.forward_horizon}`}</td><td>后台计算</td><td>—</td><td>${combo && Number.isFinite(Number(combo.net_return)) ? `${(Number(combo.net_return) * 100).toFixed(2)}%` : "—"}</td><td><span class="badge ${badge}">${statusLabel}</span></td></tr>`; }).join("") : "";
  body.innerHTML = taskRows + historical;
}

function renderFactors(factors) {
  const list = document.querySelector(".factor-list");
  if (!list) return;
  list.innerHTML = factors.map((factor) => `<div class="factor-item"><div class="factor-symbol">${escapeHtml(factor.factor_id.slice(0, 1).toUpperCase())}</div><div><strong>${escapeHtml(factor.factor_id)}</strong><p>${escapeHtml(factor.description)} · ${escapeHtml(factorTypeLabels[factor.factor_type] || factor.factor_type)}${factor.signal_semantics ? ` · ${escapeHtml(semanticsLabels[factor.signal_semantics] || factor.signal_semantics)}` : ""}</p></div><span class="badge ${factor.source === "mined" ? "badge-green" : "badge-gray"}">${factor.source === "mined" ? "挖掘入库" : "手工注册"}</span></div>`).join("");
}

function renderObservations(tracks) {
  const body = document.querySelector("#observation-results-body");
  if (!body) return;
  if (!Array.isArray(tracks) || !tracks.length) {
    body.innerHTML = `<tr><td colspan="8" class="empty-cell">暂无观察中的候选。</td></tr>`;
    return;
  }
  body.innerHTML = tracks.map((track) => {
    const result = track.result || {};
    const percent = (value) => value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
    const range = result.tracking_status === "waiting_data" ? `计划从 ${(result.planned_observation_start || "").slice(0, 10)} 开始 · 等待数据` : result.observation_start ? `${result.observation_start.slice(0, 10)} → ${result.observation_end.slice(0, 10)}` : "待更新";
    return `<tr><td><strong>${escapeHtml(track.formula_zh || track.formula || track.expression_id)}</strong><small>${escapeHtml(track.track_id.slice(0, 8))}</small></td><td>${range}</td><td>${percent(result.net_return)}</td><td>${percent(result.buy_hold_net_return)}</td><td>${percent(result.excess_net_return_vs_buy_hold)}</td><td>${percent(result.max_drawdown)}</td><td>${Number.isFinite(Number(result.total_turnover)) ? Number(result.total_turnover).toFixed(1) : "—"}</td><td><button class="ghost-button small observation-update-button" data-track-id="${escapeHtml(track.track_id)}" type="button">更新观察</button></td></tr>`;
  }).join("");
}

function renderMiningCandidates(candidates, meta = {}) {
  const body = document.querySelector("#mining-results-body");
  const summary = document.querySelector("#mining-results-meta");
  if (!body || !summary) return;
  currentMiningRunId = meta.run_id || currentMiningRunId;
  const rows = Array.isArray(candidates) ? candidates.filter((row) => ["valid", "duplicate", "fdr_rejected"].includes(row.status)).slice(0, 50) : [];
  const aiSummary = meta.ai_used ? ` · AI有效提议 ${meta.ai_audit?.valid || 0} 个` : meta.mining_mode_requested === "hybrid" ? " · AI已回退" : "";
  summary.textContent = meta.run_id ? `运行 ${meta.run_id.slice(0, 8)} · ${meta.candidate_count} 个候选 · ${meta.selected_count} 个验证入选${aiSummary}` : "尚未从页面提交自动挖掘任务。";
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="7" class="empty-cell">本次任务没有可展示候选。</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((row) => {
    const statusLabel = row.status === "duplicate" ? "重复" : row.status === "fdr_rejected" ? "FDR拒绝" : "候选";
    const screen = row.screen_status === "selected" ? "验证入选" : row.screen_status === "eligible" ? "待设门槛" : row.screen_status === "screen_rejected" ? "验证未入选" : "—";
    const stability = Number.isFinite(Number(row.validation_stability_directional_ratio)) ? `${(Number(row.validation_stability_directional_ratio) * 100).toFixed(1)}%` : "—";
    const suggestion = semanticsLabels[row.suggested_signal_semantics] || "";
    const hacPvalue = Number.isFinite(Number(row.validation_hac_pvalue)) ? Number(row.validation_hac_pvalue).toFixed(4) : "—";
    const promoted = row.promotion_status === "promoted" && row.promotion_role;
    const unavailable = meta.final_confirmation_available === false;
    const promote = row.status === "valid" && row.screen_status === "selected" && (promoted || !unavailable) ? `<button class="ghost-button small promote-button" data-run-id="${escapeHtml(meta.run_id)}" data-expression-id="${escapeHtml(row.expression_id)}" type="button" ${promoted ? "disabled" : ""}>${promoted ? "已最终确认" : "运行最终确认"}</button>` : "";
    const observed = Boolean(row.observation_track);
    const observe = row.status === "valid" && row.screen_status === "selected" ? `<button class="primary-button small candidate-observe-button" data-run-id="${escapeHtml(meta.run_id)}" data-expression-id="${escapeHtml(row.expression_id)}" type="button" ${observed ? "disabled" : ""}>${observed ? "已观察" : "加入观察"}</button>` : "";
    const detail = `<button class="ghost-button small candidate-detail-button" data-run-id="${escapeHtml(meta.run_id)}" data-expression-id="${escapeHtml(row.expression_id)}" type="button">详情</button>`;
    const source = row.source === "ai" ? "AI 提议" : row.also_proposed_by_ai ? "公式 + AI" : "公式生成";
    return `<tr><td><strong class="formula-cell">${escapeHtml(row.formula_zh || row.formula || "—")}</strong><small>${escapeHtml(row.expression_id || "")} · ${source}</small></td><td><span class="badge ${row.status === "valid" ? "badge-green" : "badge-amber"}">${statusLabel}</span></td><td>${Number.isFinite(Number(row.validation_rank_ic)) ? Number(row.validation_rank_ic).toFixed(4) : "—"}</td><td>${stability}${suggestion ? `<small>建议${suggestion}</small>` : ""}</td><td>${hacPvalue}</td><td>${screen}</td><td><div class="topbar-actions">${detail}${observe}${promote}</div></td></tr>`;
  }).join("");
}

async function loadMiningRun(runId) {
  if (!runId) return;
  const run = await api(`/api/mining/${runId}`);
  renderMiningCandidates(run.candidates, run);
}

async function loadMiningHistory() {
  const history = await api("/api/mining");
  const select = document.querySelector("#mining-run-select");
  const runs = Array.isArray(history.runs) ? history.runs : [];
  select.innerHTML = runs.length ? runs.map((run) => `<option value="${escapeHtml(run.run_id)}">${run.created_at.replace("T", " ").slice(0, 19)} · ${escapeHtml(run.protocol_version)} · ${run.candidate_count} 候选 · ${run.promoted_count} 晋级</option>`).join("") : `<option value="">暂无历史运行</option>`;
  if (runs.length) await loadMiningRun(runs[0].run_id);
}

function metric(label, value, percent = false) {
  const number = Number(value);
  const text = Number.isFinite(number) ? (percent ? `${(number * 100).toFixed(2)}%` : number.toFixed(4)) : "—";
  return `<div class="candidate-metric"><span>${label}</span><strong>${text}</strong></div>`;
}

async function openCandidateDetail(runId, expressionId) {
  const candidate = await api(`/api/mining/${runId}/candidates/${expressionId}`);
  document.querySelector("#candidate-modal-subtitle").textContent = `${candidate.expression_id} · ${candidate.status} · ${candidate.screen_status || "未筛选"}`;
  const suggestedSemantics = candidate.suggested_signal_semantics === "inverse" ? "inverse" : "direct";
  const promotionForm = `<form id="candidate-promotion-form" class="experiment-form inline-register-form" data-run-id="${escapeHtml(runId)}" data-expression-id="${escapeHtml(expressionId)}" data-suggested-semantics="${suggestedSemantics}"><input name="factor_role" type="hidden" value="direction"><label>信号语义<select name="signal_semantics"><option value="direct" ${suggestedSemantics === "direct" ? "selected" : ""}>正向</option><option value="inverse" ${suggestedSemantics === "inverse" ? "selected" : ""}>反向</option></select></label><div class="candidate-actions"><button class="primary-button" type="submit">运行最终确认</button></div><p class="form-status" id="candidate-promotion-status"></p></form>`;
  const registration = candidate.library_factor_id
    ? `<p class="form-status success">正式因子库：${escapeHtml(candidate.library_factor_id)} · ${escapeHtml(candidate.library_factor_role)} · ${escapeHtml(candidate.library_signal_semantics)}</p>`
    : candidate.promotion_status === "promoted" && candidate.promotion_role && candidate.promotion_result?.confirmation_status === "confirmed"
      ? `<form id="candidate-register-form" class="experiment-form inline-register-form" data-run-id="${escapeHtml(runId)}" data-expression-id="${escapeHtml(expressionId)}"><label>正式因子 ID<input name="factor_id" value="mined_${escapeHtml(expressionId.slice(0, 8))}" pattern="[a-z][a-z0-9_]{2,63}" required></label><div class="candidate-actions"><button class="primary-button" type="submit">加入正式因子库</button></div><p class="form-status" id="candidate-register-status"></p></form>`
      : candidate.status === "valid" && candidate.screen_status === "selected" && candidate.final_confirmation_available
        ? promotionForm
        : candidate.promotion_status === "promoted"
          ? `<p class="form-status">最终确认未通过，不能加入正式因子库。</p>`
          : `<p class="form-status">${escapeHtml(candidate.final_confirmation_reason || "只有验证集筛选入选的候选可以运行最终确认。")}</p>`;
  const observation = candidate.observation_track
    ? `<p class="form-status success">已加入观察：${escapeHtml(candidate.observation_track.track_id.slice(0, 8))}</p>`
    : candidate.status === "valid" && candidate.screen_status === "selected"
      ? (() => { const start = candidate.earliest_observation_start || ""; const endDate = start ? new Date(`${start}T00:00:00Z`) : null; if (endDate) endDate.setUTCDate(endDate.getUTCDate() + 89); const end = endDate ? endDate.toISOString().slice(0, 10) : ""; return `<form id="candidate-observation-form" class="experiment-form inline-register-form" data-run-id="${escapeHtml(runId)}" data-expression-id="${escapeHtml(expressionId)}"><label>观察开始日期${dateControl("observation_start", start, start)}</label><label>观察结束日期${dateControl("observation_end", end, start)}</label><div class="candidate-actions"><button class="primary-button" type="submit">确认加入观察</button></div><p class="form-status" id="candidate-observation-status"></p></form>`; })()
      : "";
  const promotionResult = candidate.promotion_result || {};
  const confirmationLabel = promotionResult.confirmation_status === "confirmed" ? "最终确认通过" : promotionResult.confirmation_status === "not_confirmed" ? "最终确认未通过" : promotionResult.confirmation_status === "awaiting_specialized_protocol" ? "等待专用评价协议" : "尚未最终确认";
  const sourceLabel = candidate.source === "ai" ? "AI 提议" : candidate.also_proposed_by_ai ? "公式生成，同时被 AI 提议" : "公式生成";
  const aiHypothesis = candidate.ai_hypothesis ? `<p class="form-status">AI 研究假设：${escapeHtml(candidate.ai_hypothesis)}${candidate.ai_family ? `<br>因子族：${escapeHtml(candidate.ai_family)}` : ""}</p>` : "";
  document.querySelector("#candidate-detail").innerHTML = `<div class="candidate-formula">${escapeHtml(candidate.formula_zh || candidate.formula)}</div><p class="form-status">原始 DSL：${escapeHtml(candidate.formula)}<br>候选来源：${escapeHtml(sourceLabel)}</p>${aiHypothesis}<div class="candidate-metrics">${metric("训练 Rank IC", candidate.train_rank_ic)}${metric("验证 |Rank IC|", candidate.validation_abs_rank_ic)}${metric("验证 Rank IC", candidate.validation_rank_ic)}${metric("方向一致性", candidate.validation_stability_directional_ratio, true)}${candidate.promotion_status === "promoted" ? metric("留出集对齐 Rank IC", promotionResult.holdout_aligned_rank_ic) : ""}${candidate.promotion_status === "promoted" ? metric("随机 IC 95% 阈值", promotionResult.random_null_abs_rank_ic_95) : ""}${candidate.promotion_status === "promoted" && promotionResult.evaluation_mode === "direction_backtest" ? metric("有效持仓比例", promotionResult.active_position_ratio, true) : ""}${candidate.promotion_status === "promoted" && promotionResult.evaluation_mode === "direction_backtest" ? metric("留出集净收益", promotionResult.net_return, true) : ""}${candidate.promotion_status === "promoted" && promotionResult.evaluation_mode === "direction_backtest" ? metric("买入持有净收益", promotionResult.buy_hold_net_return, true) : ""}${candidate.promotion_status === "promoted" && promotionResult.evaluation_mode === "direction_backtest" ? metric("相对买入持有", promotionResult.excess_net_return_vs_buy_hold, true) : ""}${candidate.promotion_status === "promoted" && promotionResult.evaluation_mode === "direction_backtest" ? metric("留出集 Sharpe", promotionResult.annualized_sharpe) : ""}</div><p class="form-status">研究阶段：${escapeHtml(confirmationLabel)}<br>建议语义：${escapeHtml(semanticsLabels[candidate.suggested_signal_semantics] || "—")}<br>验证筛选：${escapeHtml(candidate.screen_status || "—")} ${candidate.screen_reason ? `· ${escapeHtml(candidate.screen_reason)}` : ""}<br>最终确认：${escapeHtml(candidate.promotion_status || "candidate")}${candidate.promotion_role ? ` · ${escapeHtml(factorTypeLabels[candidate.promotion_role] || candidate.promotion_role)} / ${escapeHtml(semanticsLabels[candidate.promotion_signal_semantics] || candidate.promotion_signal_semantics)}` : ""}${candidate.promotion_task_id ? ` · 任务 ${escapeHtml(candidate.promotion_task_id.slice(0, 8))}` : ""}</p>${observation}${registration}`;
  candidateModal.hidden = false;
}

async function loadResearchState() {
  const [state, factorState, observationState] = await Promise.all([api("/api/status"), api("/api/factors"), api("/api/observations")]);
  updateDatasetSummary(state.dataset, state.observation_dataset);
  updateResearchDateBounds(state.dataset);
  renderAiStatus(state.ai);
  document.querySelector("#observation-range").textContent = state.observation_dataset ? `${state.observation_dataset.first_timestamp.slice(0, 10)} → ${state.observation_dataset.last_timestamp.slice(0, 10)}` : "未配置观察数据";
  renderFactors(factorState.factors);
  renderObservations(observationState.tracks);
  renderLatestReport(state.latest_report, state.tasks);
  const chart = await api("/api/chart");
  renderChart(chart.points);
  await loadMiningHistory();
  return state;
}

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
document.querySelectorAll("[data-view-target]").forEach((item) => item.addEventListener("click", () => setView(item.dataset.viewTarget)));
document.querySelector("#refresh-button").addEventListener("click", async () => { try { await loadResearchState(); showToast("数据状态已从后端刷新"); } catch (error) { showToast(`刷新失败：${error.message}`); } });
document.querySelector("#update-data-button").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "更新中…";
  try {
    const submitted = await api("/api/data/update", { method: "POST", body: "{}" });
    const result = await waitForTask(submitted.task_id);
    await loadResearchState();
    showToast(`研究与观察数据已更新至 ${result.observation_dataset.last_timestamp.slice(0, 10)}，已重算 ${result.updated_tracks} 个观察；服务将加载最新研究快照`);
  } catch (error) {
    showToast(`数据更新失败：${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "更新数据";
  }
});
document.querySelector("#new-run-button").addEventListener("click", openExperiment);
document.querySelector("#new-run-button-secondary").addEventListener("click", openExperiment);
document.querySelector("#new-factor-button").addEventListener("click", () => showToast("因子通过 FactorRegistry 注册，下一步开放在线注册"));
document.querySelector("#new-mining-button").addEventListener("click", openMining);
document.querySelector("#mining-results-body").addEventListener("click", async (event) => {
  const detailButton = event.target.closest(".candidate-detail-button");
  if (detailButton) {
    try { await openCandidateDetail(detailButton.dataset.runId, detailButton.dataset.expressionId); } catch (error) { showToast(`详情加载失败：${error.message}`); }
    return;
  }
  const observeButton = event.target.closest(".candidate-observe-button");
  if (observeButton) {
    try {
      await openCandidateDetail(observeButton.dataset.runId, observeButton.dataset.expressionId);
    } catch (error) {
      showToast(`详情加载失败：${error.message}`);
    }
    return;
  }
  const button = event.target.closest(".promote-button");
  if (!button) return;
  try { await openCandidateDetail(button.dataset.runId, button.dataset.expressionId); } catch (error) { showToast(`详情加载失败：${error.message}`); }
});
document.querySelector("#mining-run-select").addEventListener("change", async (event) => { try { await loadMiningRun(event.target.value); } catch (error) { showToast(`历史加载失败：${error.message}`); } });
document.querySelector("#close-candidate-button").addEventListener("click", closeCandidate);
candidateModal.addEventListener("click", (event) => { if (event.target === candidateModal) closeCandidate(); });
candidateModal.addEventListener("change", (event) => {
  if (event.target.name !== "factor_role") return;
  const semantics = candidateModal.querySelector('[name="signal_semantics"]');
  if (!semantics) return;
  if (event.target.value === "direction") {
    semantics.disabled = false;
    semantics.value = event.target.form.dataset.suggestedSemantics || "direct";
  } else {
    semantics.value = "neutral";
    semantics.disabled = true;
  }
});
candidateModal.addEventListener("submit", async (event) => {
  if (!["candidate-register-form", "candidate-promotion-form", "candidate-observation-form"].includes(event.target.id)) return;
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const payload = Object.fromEntries(data.entries());
  if (payload.factor_role !== "direction") payload.signal_semantics = "neutral";
  const isPromotion = form.id === "candidate-promotion-form";
  const isObservation = form.id === "candidate-observation-form";
  const status = form.querySelector(isPromotion ? "#candidate-promotion-status" : isObservation ? "#candidate-observation-status" : "#candidate-register-status");
  try { validateDateFields(form); } catch (error) { status.textContent = error.message; status.className = "form-status error"; return; }
  const submit = form.querySelector('button[type="submit"]');
  submit.disabled = true;
  status.textContent = isPromotion ? "正在运行最终留出确认…" : isObservation ? "正在冻结参数并计算观察收益…" : "正在加入正式因子库…";
  try {
    if (isPromotion) {
      const submitted = await api(`/api/mining/${form.dataset.runId}/candidates/${form.dataset.expressionId}/promote`, { method: "POST", body: JSON.stringify(payload) });
      const result = await waitForTask(submitted.task_id);
      status.textContent = result.confirmation_status === "confirmed" ? "最终确认通过" : "最终确认完成，但未达到入库标准";
      showToast(status.textContent);
    } else if (isObservation) {
      const observed = await api(`/api/mining/${form.dataset.runId}/candidates/${form.dataset.expressionId}/observe`, { method: "POST", body: JSON.stringify(payload) });
      status.textContent = `已加入观察：${observed.track_id.slice(0, 8)}`;
      showToast("候选已加入观察");
    } else {
      const registered = await api(`/api/mining/${form.dataset.runId}/candidates/${form.dataset.expressionId}/register`, { method: "POST", body: JSON.stringify(payload) });
      status.textContent = `已加入：${registered.factor_id}`;
      showToast("候选已加入正式因子库");
    }
    status.className = "form-status success";
    await loadResearchState();
    await openCandidateDetail(form.dataset.runId, form.dataset.expressionId);
  } catch (error) {
    submit.disabled = false;
    status.textContent = `操作失败：${error.message}`;
    status.className = "form-status error";
  }
});
document.querySelector("#view-observations").addEventListener("click", async (event) => {
  const button = event.target.closest(".observation-update-button");
  if (!button) return;
  button.disabled = true;
  try {
    await api(`/api/observations/${button.dataset.trackId}/update`, { method: "POST", body: "{}" });
    const observations = await api("/api/observations");
    renderObservations(observations.tracks);
    showToast("观察收益已更新");
  } catch (error) {
    showToast(`更新失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
});
document.querySelector("#open-progress-button").addEventListener("click", () => showToast("开发进度请查看项目根目录的开发进度.md"));
document.querySelector("#close-experiment-button").addEventListener("click", closeExperiment);
document.querySelector("#cancel-experiment-button").addEventListener("click", closeExperiment);
experimentModal.addEventListener("click", (event) => { if (event.target === experimentModal) closeExperiment(); });
miningModal.addEventListener("click", (event) => { if (event.target === miningModal) closeMining(); });
experimentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(experimentForm);
  const payload = Object.fromEntries([...form.entries()].map(([key, value]) => [key, Number(value)]));
  setExperimentStatus("正在运行后端实验…");
  const submit = experimentForm.querySelector("button[type=submit]");
  submit.disabled = true;
  try {
    const submitted = await api("/api/experiments", { method: "POST", body: JSON.stringify(payload) });
    setExperimentStatus("任务已提交，后台计算中…");
    const result = await waitForTask(submitted.task_id);
    setExperimentStatus(`完成：${result.factors.length} 个注册因子已写入报告。`, "success");
    await loadResearchState();
    showToast("实验完成，报告已保存");
    window.setTimeout(closeExperiment, 900);
  } catch (error) {
    setExperimentStatus(`运行失败：${error.message}`, "error");
  } finally {
    submit.disabled = false;
  }
});
document.querySelector("#close-mining-button").addEventListener("click", closeMining);
document.querySelector("#cancel-mining-button").addEventListener("click", closeMining);
miningForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try { validateDateFields(miningForm); } catch (error) { setMiningStatus(error.message, "error"); return; }
  const form = new FormData(miningForm);
  const stringFields = new Set(["research_start", "research_end", "mining_mode"]);
  const payload = Object.fromEntries([...form.entries()].filter(([, value]) => value !== "").map(([key, value]) => [key, stringFields.has(key) ? value : Number(value)]));
  setMiningStatus("正在生成并评价候选因子…");
  const submit = miningForm.querySelector("button[type=submit]");
  submit.disabled = true;
  try {
    const submitted = await api("/api/mining", { method: "POST", body: JSON.stringify(payload) });
    setMiningStatus("任务已提交，后台挖掘中…");
    const task = await waitForTask(submitted.task_id);
    renderMiningCandidates(task.candidates, task);
    await loadMiningHistory();
    const modeNote = task.mining_mode_requested === "hybrid" && !task.ai_used ? `；AI未启用，已回退公式模式：${task.ai_fallback_reason}` : "";
    setMiningStatus(`完成：${task.selected_count} 个候选进入最终确认${modeNote}`, "success");
    showToast("自动挖掘完成，候选已写入目录");
    window.setTimeout(closeMining, 1000);
  } catch (error) {
    setMiningStatus(`挖掘失败：${error.message}`, "error");
  } finally {
    submit.disabled = false;
  }
});
document.querySelector("#dataset-select").addEventListener("change", (event) => { showToast(event.target.value === "synthetic" ? "合成面板将在多资产 API 接入后显示" : "已选择 BTCUSDT 多年基线"); });
loadResearchState().catch((error) => showToast(`后端连接失败：${error.message}`));
