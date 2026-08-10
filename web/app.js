const state = {
  models: [], images: [], results: [], summaries: [], pool: {}, benchmark: {},
  slide: 0, activeModelId: "", benchmarkRuns: [], annotation: null
};
const $ = (selector) => document.querySelector(selector);

function checkedValues(selector) {
  return [...document.querySelectorAll(`${selector}:checked`)].map((input) => input.value);
}

function updateCounts() {
  $("#model-count").textContent = `${checkedValues('.model-check').length} seleccionados`;
  $("#image-count").textContent = `${checkedValues('.image-check').length} seleccionadas`;
}

function renderModels() {
  $("#models").innerHTML = state.models.map((model) => `
    <label class="model-card">
      <input class="model-check" type="checkbox" value="${model.id}" ${model.recommended ? "checked" : ""}>
      <span class="model-name">${model.name}</span>
      ${model.checkpoint_status?.includes("verified") ? '<span class="badge">CHECKPOINT VERIFICADO</span>' : ""}
      <span class="model-meta">${model.year} · ${model.license} · ${model.framework}</span>
      <span class="model-meta">Perfil: ${model.resource_profile}</span>
      <span class="model-meta">Datos: ${model.datasets.join(", ")}</span>
      <span class="model-reason">${model.description}</span>
      <a class="repo-link" href="${model.repository}" target="_blank" rel="noreferrer" onclick="event.stopPropagation()">Abrir fuente ↗</a>
    </label>`).join("");
  document.querySelectorAll(".model-check").forEach((input) => input.addEventListener("change", updateCounts));
}

function renderPoolStatus() {
  const counts = state.pool?.fitzpatrick_counts ?? {};
  const distribution = ["I", "II", "III", "IV", "V", "VI"]
    .map((type) => `${type}: ${counts[type] ?? 0}`).join(" · ");
  const metadata = state.pool?.metadata_available ? "metadatos disponibles" : "falta el CSV de metadatos";
  $("#pool-status").textContent = `${state.pool?.total_local_images ?? 0} imágenes locales · ${metadata} · ${distribution}`;
}

function renderBenchmarkState() {
  const evaluations = Object.fromEntries((state.benchmark?.evaluations ?? []).map((item) => [item.id, item]));
  const resources = state.benchmark?.resources ?? {};
  const common = $("#common-state");
  common.innerHTML = `
    <article><strong>B1</strong><span>${evaluations.B1?.available ? "Disponible para smoke/compatibilidad" : "No disponible"}</span></article>
    <article><strong>B2</strong><span>${evaluations.B2?.available ? "Disponible" : evaluations.B2?.reason ?? "No disponible"}</span></article>
    <article><strong>YOLOv3</strong><span>${resources.yolov3?.message ?? "Sin estado"}</span></article>
    <article><strong>ISIC 2018</strong><span>${resources.isic2018_task1_manifest?.available ? "Manifest local detectado" : "Falta manifest oficial registrado"}</span></article>`;
  $("#ablation-state").textContent = "C0–C3 disponibles para S01–S16; los resultados ausentes se muestran como no ejecutados.";
  $("#sealed-state").textContent = `Checkpoints B2 ausentes: ${resources.b2_checkpoints?.missing_count ?? 15}. La preparación valida hashes y árbol Git limpio.`;
  $("#common-evaluation").addEventListener("change", (event) => {
    if (event.target.value === "B2" && !evaluations.B2?.available) {
      $("#common-state").setAttribute("data-warning", evaluations.B2?.reason ?? "B2 no disponible");
    } else {
      $("#common-state").removeAttribute("data-warning");
    }
  });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

async function loadRunIndex() {
  const payload = await (await fetch("/api/benchmark/runs")).json();
  state.benchmarkRuns = payload.runs ?? [];
  const options = (filter) => state.benchmarkRuns.filter(filter).map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${escapeHtml(run.status)}</option>`).join("");
  $("#common-run").innerHTML = options((run) => ["B1", "B2"].includes(run.evaluation));
  $("#ablation-run").innerHTML = options((run) => run.evaluation === "ablation");
  const sealed = state.benchmarkRuns.filter((run) => String(run.run_id).startsWith("sealed-"));
  $("#sealed-runs").innerHTML = sealed.length ? `<h3>Runs sellados</h3>${sealed.map((run) => `<p><strong>${escapeHtml(run.run_id)}</strong> · ${escapeHtml(run.status)}</p>`).join("")}` : "<p>No existe ninguna ejecución sellada.</p>";
}

function renderRun(payload, target) {
  const results = payload.results ?? [];
  if (!results.length) { target.innerHTML = "<p class='notice'>El run no contiene resultados legibles.</p>"; return; }
  const methods = [...new Set(results.map((item) => item.method_id))];
  const images = [...new Set(results.map((item) => item.image_id))];
  const conditions = [...new Set(results.map((item) => item.condition ?? item.evaluation))];
  const fitzpatrick = [...new Set(results.map((item) => item.metadata?.fitzpatrick).filter(Boolean))].sort();
  const metrics = ["threshold_jaccard", "jaccard", "dice", "sensitivity", "specificity", "precision", "boundary_f1", "hd95_normalized", "fov_leak", "clean_skin_contamination"];
  const isAblation = conditions.some((value) => /^C[0-3]$/.test(value));
  target.innerHTML = `<p class="notice">${escapeHtml(payload.manifest?.dataset)} · ${escapeHtml(payload.manifest?.split)} · ${escapeHtml(payload.manifest?.configuration_hash ?? "sin hash")}</p><div class="run-controls"><label>Método <select class="run-method">${methods.map((value) => `<option>${escapeHtml(value)}</option>`).join("")}</select></label><label>Imagen <select class="run-image">${images.map((value) => `<option>${escapeHtml(value)}</option>`).join("")}</select></label><label>Condición/ranking <select class="run-condition">${conditions.map((value) => `<option>${escapeHtml(value)}</option>`).join("")}</select></label><label>Métrica <select class="run-metric">${metrics.map((value) => `<option>${escapeHtml(value)}</option>`).join("")}</select></label><label>Fitzpatrick <select class="run-fitzpatrick"><option value="">Todos</option>${fitzpatrick.map((value) => `<option>${escapeHtml(value)}</option>`).join("")}</select></label><label>YOLO/fallback <select class="run-fallback"><option value="">Todos</option><option value="success">Sin fallback</option><option value="fallback">Con fallback</option></select></label></div><div class="run-result"></div><div class="table-wrap"><table><thead><tr><th>Método</th><th>N</th><th>Fallos</th><th>Media</th><th>Mediana</th><th>IC 95 %</th></tr></thead><tbody class="run-ranking"></tbody></table></div>`;
  const selectedResults = () => results.filter((item) => {
    const fitz = target.querySelector(".run-fitzpatrick").value;
    const fallback = target.querySelector(".run-fallback").value;
    return (!fitz || item.metadata?.fitzpatrick === fitz) && (!fallback || Boolean(item.p0_fallback_used) === (fallback === "fallback"));
  });
  const renderRanking = () => {
    const metric = target.querySelector(".run-metric").value, condition = target.querySelector(".run-condition").value;
    const filtered = Boolean(target.querySelector(".run-fitzpatrick").value || target.querySelector(".run-fallback").value);
    let rows = (payload.report?.summaries ?? []).filter((row) => row.condition === condition);
    if (filtered) rows = methods.map((method) => {
      const subset = selectedResults().filter((item) => item.method_id === method && (item.condition ?? item.evaluation) === condition);
      const values = subset.map((item) => item.metrics?.[metric]).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      const middle = values.length ? values.length % 2 ? values[(values.length - 1) / 2] : (values[values.length / 2 - 1] + values[values.length / 2]) / 2 : NaN;
      return {method_id:method, n:subset.length, failures:subset.filter((item) => item.failure_code).length, [`${metric}_mean`]:values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : NaN, [`${metric}_median`]:middle};
    });
    target.querySelector(".run-ranking").innerHTML = rows.map((row) => `<tr><td>${escapeHtml(row.method_id)}</td><td>${row.n}</td><td>${row.failures}</td><td>${Number(row[`${metric}_mean`] ?? NaN).toFixed(4)}</td><td>${Number(row[`${metric}_median`] ?? NaN).toFixed(4)}</td><td>${filtered ? "ND en filtro interactivo" : `${Number(row[`${metric}_ci95_low`] ?? NaN).toFixed(4)}–${Number(row[`${metric}_ci95_high`] ?? NaN).toFixed(4)}`}</td></tr>`).join("") || `<tr><td colspan="6">Sin métricas autorizadas para esta condición.</td></tr>`;
  };
  const render = () => {
    const matches = selectedResults().filter((value) => value.method_id === target.querySelector(".run-method").value && value.image_id === target.querySelector(".run-image").value);
    const item = matches.find((value) => (value.condition ?? value.evaluation) === target.querySelector(".run-condition").value) ?? matches[0];
    if (!item) { target.querySelector(".run-result").innerHTML = "<p class='notice'>No hay resultado con estos filtros.</p>"; renderRanking(); return; }
    if (isAblation) {
      const comparisons = matches.sort((left, right) => String(left.condition).localeCompare(String(right.condition))).map((value) => `<article><h4>${escapeHtml(value.condition)}</h4>${resultPanel(`Máscara final ${value.condition}`, value.artifacts?.["final_mask.png"], `Máscara ${value.condition}`)}<p>${value.metrics ? `TJ ${Number(value.metrics.threshold_jaccard).toFixed(4)} · Dice ${Number(value.metrics.dice).toFixed(4)}` : "GT no disponible"} · ${value.failure_code ?? "sin fallo"} · fallback ${value.p0_fallback_used ? "sí" : "no"}</p></article>`).join("");
      target.querySelector(".run-result").innerHTML = `<h3>${escapeHtml(item.method_id)} · ${escapeHtml(item.image_id)} · comparación C0–C3</h3><div class="artifact-grid">${comparisons}</div>`;
      renderRanking(); return;
    }
    const cards = Object.entries(item.artifacts ?? {}).filter(([, url]) => !url.endsWith(".json") && !url.endsWith(".npy")).map(([name, url]) => resultPanel(escapeHtml(name), url, name)).join("");
    const metrics = item.metrics ? Object.entries(item.metrics).filter(([, value]) => typeof value === "number" || value === null).map(([key, value]) => `${escapeHtml(key)}: ${value === null ? "ND" : Number(value).toFixed(4)}`).join(" · ") : "Ground truth no autorizado/disponible";
    const resources = state.benchmark?.resources?.backends?.[item.method_id] ?? {};
    target.querySelector(".run-result").innerHTML = `<h3>${escapeHtml(item.condition ?? item.evaluation)} · ${escapeHtml(item.method_id)} · ${escapeHtml(item.image_id)}</h3><div class="artifact-grid">${cards}</div><p class="runtime">Backend: ${Number(item.backend?.backend_time_ms ?? 0).toFixed(1)} ms · End-to-end: ${Number(item.end_to_end_time_ms ?? 0).toFixed(1)} ms · Parámetros: ${Number(resources.parameter_count ?? 0).toLocaleString()} · Pesos: ${(Number(resources.checkpoint_bytes ?? 0) / 1048576).toFixed(1)} MiB · RAM pico: ${Number(item.backend?.details?.peak_ram_mb ?? 0).toFixed(1)} MiB · Fallo: ${escapeHtml(item.failure_code ?? "ninguno")} · Fallback: ${item.p0_fallback_used ? "sí" : "no"}</p><p class="stats">${metrics}</p>`;
    renderRanking();
  };
  target.querySelectorAll("select").forEach((select) => select.addEventListener("change", render)); render();
}

async function loadSelectedRun(selectId, targetId) {
  const runId = $(selectId).value;
  if (!runId) { $(targetId).innerHTML = "<p class='notice'>No hay un run compatible todavía.</p>"; return; }
  const response = await fetch(`/api/benchmark/run?run_id=${encodeURIComponent(runId)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error);
  renderRun(payload, $(targetId));
}

let annotationDrawing = false;
function annotationImageChanged() {
  const image = state.annotation?.items?.find((item) => item.image_id === $("#annotation-image").value);
  if (!image) return;
  const source = new Image();
  source.onload = () => {
    for (const id of ["#annotation-reference", "#annotation-mask"]) { const canvas = $(id); canvas.width = source.naturalWidth; canvas.height = source.naturalHeight; }
    $("#annotation-reference").getContext("2d").drawImage(source, 0, 0);
    $("#annotation-mask").getContext("2d").fillStyle = "black"; $("#annotation-mask").getContext("2d").fillRect(0, 0, source.naturalWidth, source.naturalHeight);
  };
  if (!image.url) { $("#annotation-status").textContent = `No se encontró localmente la imagen ${image.image_id}`; return; }
  source.src = image.url;
}

function drawAnnotation(event) {
  if (!annotationDrawing) return;
  const canvas = $("#annotation-mask"), rect = canvas.getBoundingClientRect(), context = canvas.getContext("2d");
  context.fillStyle = "white"; context.beginPath(); context.arc((event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height, Number($("#annotation-brush").value) * canvas.width / rect.width, 0, Math.PI * 2); context.fill();
}

async function initializeAnnotations() {
  const payload = await (await fetch("/api/annotations/state")).json(); state.annotation = payload;
  $("#annotation-status").textContent = payload.available ? `Pendientes: ${payload.counts.pending} · doble anotación: ${payload.counts.double_annotated} · adjudicadas: ${payload.counts.complete}` : payload.message;
  if (!payload.available) return;
  $("#annotation-workspace").hidden = false;
  $("#annotation-image").innerHTML = payload.items.map((item) => `<option value="${escapeHtml(item.image_id)}">${escapeHtml(item.image_id)} · ${escapeHtml(item.status)}</option>`).join("");
  annotationImageChanged();
}

async function saveAnnotation() {
  const response = await fetch("/api/annotations/save", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({image_id:$("#annotation-image").value, role:$("#annotation-role").value, mask_type:$("#annotation-type").value, actor:$("#annotation-actor").value, png_data_url:$("#annotation-mask").toDataURL("image/png")})});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error); $("#annotation-status").textContent = `Versión ${payload.saved.version} guardada · SHA-256 ${payload.saved.sha256.slice(0, 12)}…`;
}

function selectTab(name) {
  document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== `tab-${name}`; });
  document.querySelectorAll(".tab").forEach((button) => { button.classList.toggle("active", button.dataset.tab === name); });
}

function renderImages(selectAll = true) {
  $("#images").innerHTML = state.images.map((image) => `
    <label class="image-card">
      <input class="image-check" type="checkbox" value="${image.id}" ${selectAll ? "checked" : ""}>
      <img src="${image.url}" alt="${image.id}" loading="lazy">
      <span title="${image.id}">${image.id}</span>
      <small>Fitzpatrick ${image.fitzpatrick_skin_type || "sin dato"}${image.diagnosis_1 ? ` · ${image.diagnosis_1}` : ""}</small>
    </label>`).join("");
  document.querySelectorAll(".image-check").forEach((input) => input.addEventListener("change", updateCounts));
  renderPoolStatus();
  updateCounts();
}

async function sampleImages() {
  const total = Number($("#sample-total").value);
  const seed = Number($("#sample-seed").value);
  $("#pool-status").textContent = "Construyendo muestra equilibrada…";
  const response = await fetch("/api/sample", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ total, seed })
  });
  const payload = await response.json();
  if (!response.ok) {
    $("#pool-status").textContent = payload.error;
    return;
  }
  state.images = payload.images;
  state.pool = payload.pool;
  renderImages(true);
  $("#pool-status").textContent += ` · muestra: ${payload.per_type} por tipo · semilla ${payload.seed}`;
}

function resultPanel(title, url, alt) {
  return url
    ? `<figure class="panel"><img src="${url}?v=${Date.now()}" alt="${alt}"><figcaption>${title}</figcaption></figure>`
    : `<div class="missing"><strong>${title}</strong><br>Resultado todavía no disponible.</div>`;
}

function renderSummaries() {
  const container = $("#benchmark-summary");
  if (!state.summaries.length) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  container.hidden = false;
  container.innerHTML = `
    <h3>Resumen de rendimiento</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Modelo</th><th>Imágenes</th><th>Tiempo total</th><th>Promedio/imagen</th><th>CPU usada</th><th>Núcleos efectivos</th><th>RAM máxima</th></tr></thead>
      <tbody>${state.summaries.map((summary) => {
        const model = state.models.find((item) => item.id === summary.model_id);
        const effectiveCores = summary.average_effective_cpu_cores ?? (summary.average_cpu_percent_single_core_equivalent / 100);
        return `<tr><td>${model?.name ?? summary.model_id}</td><td>${summary.images_timed}</td><td>${summary.total_wall_seconds} s</td><td>${summary.average_wall_seconds} s</td><td>${summary.average_cpu_percent_system_capacity}% del equipo</td><td>${effectiveCores.toFixed(2)}</td><td>${summary.peak_ram_mb_max} MB</td></tr>`;
      }).join("")}</tbody>
    </table></div>`;
}

function runtimeDetails(runtime) {
  if (!runtime) return "Sin medición de recursos para esta máscara.";
  const effectiveCores = runtime.effective_cpu_cores ?? (runtime.cpu_percent_single_core_equivalent / 100);
  return `Tiempo: ${runtime.wall_seconds} s · CPU: ${runtime.cpu_seconds} s (${runtime.cpu_percent_system_capacity}% del equipo; ${effectiveCores.toFixed(2)} núcleos efectivos) · RAM máxima: ${runtime.peak_ram_mb} MB`;
}

function activeResults() {
  return state.results.filter((result) => result.model_id === state.activeModelId);
}

function populateResultModels() {
  const available = [...new Set(state.results.map((result) => result.model_id))];
  $("#result-model").innerHTML = available.map((modelId) => {
    const model = state.models.find((item) => item.id === modelId);
    return `<option value="${modelId}">${model?.name ?? modelId}</option>`;
  }).join("");
  state.activeModelId = available[0] ?? "";
  $("#result-model").value = state.activeModelId;
}

function renderSlide() {
  const results = activeResults();
  if (!results.length) return;
  const result = results[state.slide];
  const model = state.models.find((item) => item.id === result.model_id);
  $("#slide-position").textContent = `${state.slide + 1} / ${results.length}`;
  $("#previous").disabled = state.slide === 0;
  $("#next").disabled = state.slide === results.length - 1;
  const stats = result.stats?.skin_colour;
  const imageMetadata = result.image_metadata ?? {};
  $("#slide").innerHTML = `
    <h3>${model?.name ?? result.model_id} · ${result.image_id}</h3>
    <p class="result-metadata">Fitzpatrick ${imageMetadata.fitzpatrick_skin_type || "sin dato"}${imageMetadata.diagnosis_1 ? ` · ${imageMetadata.diagnosis_1}` : ""}${imageMetadata.image_type ? ` · ${imageMetadata.image_type}` : ""}</p>
    <div class="triptych">
      ${resultPanel("Imagen original", result.original_url, "Imagen original")}
      ${resultPanel("Máscara 1: lesión", result.lesion_mask_url, "Máscara binaria de lesión")}
      ${resultPanel("Máscara 2: piel limpia", result.clean_skin_mask_url, "Máscara de piel circundante")}
    </div>
    <p class="runtime"><strong>Rendimiento por imagen:</strong> ${runtimeDetails(result.runtime)}</p>
    <p class="stats">${stats
      ? `Píxeles de piel: ${stats.pixel_count} · RGB mediana: ${stats.rgb_median.join(", ")} · Lab mediana: ${stats.lab_median.join(", ")} · ITA: ${stats.ita_degrees}°`
      : result.stats?.skin_colour_error ?? result.message ?? "Este modelo necesita un adaptador y su checkpoint antes de inferir."}</p>`;
}

async function runReview() {
  const modelIds = checkedValues(".model-check");
  const imageIds = checkedValues(".image-check");
  $("#status").textContent = "Comprobando máscaras e inferencias disponibles…";
  const response = await fetch("/api/infer", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_ids: modelIds, image_ids: imageIds })
  });
  const payload = await response.json();
  if (!response.ok) { $("#status").textContent = payload.error; return; }
  state.results = payload.results;
  state.summaries = payload.summaries ?? [];
  state.slide = 0;
  populateResultModels();
  $("#status").textContent = `${state.results.filter((r) => r.status === "ready").length} resultados listos de ${state.results.length}.`;
  $("#carousel").hidden = false;
  renderSummaries();
  renderSlide();
}

async function initialize() {
  const response = await fetch("/api/state");
  Object.assign(state, await response.json());
  renderModels(); renderImages(true); renderBenchmarkState(); updateCounts();
  await Promise.all([loadRunIndex(), initializeAnnotations()]);
}

document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => selectTab(button.dataset.tab)));

$("#sample-images").addEventListener("click", () => sampleImages().catch((error) => { $("#pool-status").textContent = error.message; }));
$("#select-all-models").addEventListener("click", () => {
  document.querySelectorAll(".model-check").forEach((input) => { input.checked = true; });
  updateCounts();
});
$("#clear-models").addEventListener("click", () => {
  document.querySelectorAll(".model-check").forEach((input) => { input.checked = false; });
  updateCounts();
});
$("#clear-images").addEventListener("click", () => { document.querySelectorAll(".image-check").forEach((i) => { i.checked = false; }); updateCounts(); });
$("#run").addEventListener("click", runReview);
$("#previous").addEventListener("click", () => { state.slide -= 1; renderSlide(); });
$("#next").addEventListener("click", () => { state.slide += 1; renderSlide(); });
$("#result-model").addEventListener("change", (event) => { state.activeModelId = event.target.value; state.slide = 0; renderSlide(); });
$("#load-common-run").addEventListener("click", () => loadSelectedRun("#common-run", "#common-run-view").catch((error) => { $("#common-run-view").textContent = error.message; }));
$("#load-ablation-run").addEventListener("click", () => loadSelectedRun("#ablation-run", "#ablation-run-view").catch((error) => { $("#ablation-run-view").textContent = error.message; }));
$("#annotation-image").addEventListener("change", annotationImageChanged);
$("#annotation-mask").addEventListener("pointerdown", (event) => { annotationDrawing = true; event.target.setPointerCapture(event.pointerId); drawAnnotation(event); });
$("#annotation-mask").addEventListener("pointermove", drawAnnotation);
$("#annotation-mask").addEventListener("pointerup", () => { annotationDrawing = false; });
$("#annotation-clear").addEventListener("click", annotationImageChanged);
$("#annotation-save").addEventListener("click", () => saveAnnotation().catch((error) => { $("#annotation-status").textContent = error.message; }));
initialize().catch((error) => { $("#status").textContent = `No se pudo iniciar: ${error.message}`; });
