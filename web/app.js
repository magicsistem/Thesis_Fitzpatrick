const state = { models: [], images: [], results: [], slide: 0 };
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
      ${model.recommended ? '<span class="badge">TOP 5</span>' : ""}
      <span class="model-meta">${model.framework} · ${model.datasets.join(", ")}</span>
      <span class="model-reason">${model.reason}</span>
    </label>`).join("");
  document.querySelectorAll(".model-check").forEach((input) => input.addEventListener("change", updateCounts));
}

function renderImages() {
  $("#images").innerHTML = state.images.map((image) => `
    <label class="image-card">
      <input class="image-check" type="checkbox" value="${image.id}">
      <img src="${image.url}" alt="${image.id}" loading="lazy">
      <span title="${image.id}">${image.id}</span>
    </label>`).join("");
  const range = $("#image-limit");
  range.max = Math.max(1, state.images.length);
  range.value = Math.min(10, Math.max(1, state.images.length));
  $("#limit-value").textContent = range.value;
  document.querySelectorAll(".image-check").forEach((input) => input.addEventListener("change", updateCounts));
}

function applyImageLimit() {
  const limit = Number($("#image-limit").value);
  document.querySelectorAll(".image-check").forEach((input, index) => { input.checked = index < limit; });
  updateCounts();
}

function resultPanel(title, url, alt) {
  return url
    ? `<figure class="panel"><img src="${url}?v=${Date.now()}" alt="${alt}"><figcaption>${title}</figcaption></figure>`
    : `<div class="missing"><strong>${title}</strong><br>Resultado todavía no disponible.</div>`;
}

function renderSlide() {
  if (!state.results.length) return;
  const result = state.results[state.slide];
  const model = state.models.find((item) => item.id === result.model_id);
  $("#slide-position").textContent = `${state.slide + 1} / ${state.results.length}`;
  $("#previous").disabled = state.slide === 0;
  $("#next").disabled = state.slide === state.results.length - 1;
  const stats = result.stats?.skin_colour;
  $("#slide").innerHTML = `
    <h3>${model?.name ?? result.model_id} · ${result.image_id}</h3>
    <div class="triptych">
      ${resultPanel("Imagen original", result.original_url, "Imagen original")}
      ${resultPanel("Máscara 1: lesión", result.lesion_mask_url, "Máscara binaria de lesión")}
      ${resultPanel("Máscara 2: piel limpia", result.clean_skin_mask_url, "Máscara de piel circundante")}
    </div>
    <p class="stats">${stats
      ? `Píxeles de piel: ${stats.pixel_count} · RGB mediana: ${stats.rgb_median.join(", ")} · Lab mediana: ${stats.lab_median.join(", ")} · ITA: ${stats.ita_degrees}°`
      : result.message ?? "Este modelo necesita un adaptador y su checkpoint antes de inferir."}</p>`;
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
  state.slide = 0;
  $("#status").textContent = `${state.results.filter((r) => r.status === "ready").length} resultados listos de ${state.results.length}.`;
  $("#carousel").hidden = false;
  renderSlide();
}

async function initialize() {
  const response = await fetch("/api/state");
  Object.assign(state, await response.json());
  renderModels(); renderImages(); updateCounts(); applyImageLimit();
}

$("#image-limit").addEventListener("input", (event) => { $("#limit-value").textContent = event.target.value; });
$("#select-visible").addEventListener("click", applyImageLimit);
$("#clear-images").addEventListener("click", () => { document.querySelectorAll(".image-check").forEach((i) => { i.checked = false; }); updateCounts(); });
$("#run").addEventListener("click", runReview);
$("#previous").addEventListener("click", () => { state.slide -= 1; renderSlide(); });
$("#next").addEventListener("click", () => { state.slide += 1; renderSlide(); });
initialize().catch((error) => { $("#status").textContent = `No se pudo iniciar: ${error.message}`; });
