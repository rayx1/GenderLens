// GenderLens v1.0 - github.com/your-username/gender-lens
import { loadDictionary, loadModel, predictBatch, getModelStatus } from "./src/predictor.js";
import { parseFile, addPredictionColumns, exportToCSV } from "./src/filehandler.js";

const state = {
  phase: "idle",
  file: null,
  parsed: null,
  selectedColumn: null,
  predictions: null,
  output: null
};

const el = {
  uploadZone: document.getElementById("uploadZone"),
  fileInput: document.getElementById("fileInput"),
  uploadIdle: document.getElementById("uploadIdle"),
  uploadLoaded: document.getElementById("uploadLoaded"),
  fileMeta: document.getElementById("fileMeta"),
  changeFileBtn: document.getElementById("changeFileBtn"),
  columnSection: document.getElementById("columnSection"),
  nameColumnSelect: document.getElementById("nameColumnSelect"),
  namePreview: document.getElementById("namePreview"),
  predictBtn: document.getElementById("predictBtn"),
  predictBtnLabel: document.getElementById("predictBtnLabel"),
  predictSpinner: document.getElementById("predictSpinner"),
  statsBar: document.getElementById("statsBar"),
  previewSection: document.getElementById("previewSection"),
  previewTable: document.getElementById("previewTable"),
  downloadBtn: document.getElementById("downloadBtn"),
  errorBanner: document.getElementById("errorBanner"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText")
};

function setPhase(phase) {
  state.phase = phase;
}

function showError(message) {
  el.errorBanner.textContent = message;
  el.errorBanner.classList.remove("hidden");
}

function clearError() {
  el.errorBanner.classList.add("hidden");
  el.errorBanner.textContent = "";
}

function updateModelStatusUI() {
  const status = getModelStatus();
  if (status === "model+dict") {
    el.statusText.textContent = "Dictionary + Conv1D Model";
    el.statusDot.style.background = "#10B981";
  } else if (status === "dict+rules") {
    el.statusText.textContent = "Dictionary + Rules only";
    el.statusDot.style.background = "#F59E0B";
  } else {
    el.statusText.textContent = "Rules only";
    el.statusDot.style.background = "#EF4444";
  }
}

function resetResultsUI() {
  el.statsBar.classList.add("hidden");
  el.previewSection.classList.add("hidden");
  el.downloadBtn.classList.add("hidden");
  el.previewTable.innerHTML = "";
}

function setPredictLoading(active, done = 0, total = 0) {
  el.predictBtn.disabled = active || state.selectedColumn === null;
  el.predictSpinner.classList.toggle("hidden", !active);
  el.predictBtnLabel.textContent = active ? (total > 0 ? `Predicting ${done}/${total}` : "Predicting...") : "Predict Gender";
}

function renderColumnOptions(headers) {
  el.nameColumnSelect.innerHTML = '<option value="">Choose a column</option>';
  headers.forEach((header, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = `${idx + 1}. ${header}`;
    el.nameColumnSelect.appendChild(opt);
  });
}

function renderNamePreview() {
  if (!state.parsed || state.selectedColumn === null) {
    el.namePreview.textContent = "";
    return;
  }
  const sample = state.parsed.rows
    .slice(0, 3)
    .map((row) => String(row[state.selectedColumn] || "").trim())
    .filter(Boolean)
    .join(" | ");
  el.namePreview.textContent = sample ? `Sample: ${sample}` : "Selected column has empty sample rows.";
}

function renderStats(predictions) {
  const total = predictions.length;
  const male = predictions.filter((p) => p.gender === "Male").length;
  const female = predictions.filter((p) => p.gender === "Female").length;
  const unknown = total - male - female;

  el.statsBar.innerHTML = "";

  [
    { label: "Total", value: total, className: "" },
    { label: "Male", value: male, className: "stat-male" },
    { label: "Female", value: female, className: "stat-female" },
    { label: "Unknown", value: unknown, className: "stat-unknown" }
  ].forEach((pill) => {
    const node = document.createElement("div");
    node.className = `stat-pill ${pill.className}`.trim();
    node.textContent = `${pill.label}: ${pill.value}`;
    el.statsBar.appendChild(node);
  });

  el.statsBar.classList.remove("hidden");
}

function chipClass(gender) {
  if (gender === "Male") return "chip chip-male";
  if (gender === "Female") return "chip chip-female";
  return "chip chip-unknown";
}

function confClass(gender) {
  if (gender === "Male") return "conf-male";
  if (gender === "Female") return "conf-female";
  return "conf-unknown";
}

function renderPreview(headers, rows) {
  const previewRows = rows.slice(0, 10);
  const thead = document.createElement("thead");
  const headTr = document.createElement("tr");

  headers.forEach((header, idx) => {
    const th = document.createElement("th");
    th.textContent = header;
    if (idx === headers.length - 2) th.classList.add("col-pred");
    if (idx === headers.length - 1) th.classList.add("col-conf");
    headTr.appendChild(th);
  });

  thead.appendChild(headTr);

  const tbody = document.createElement("tbody");

  previewRows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value, idx) => {
      const td = document.createElement("td");
      if (idx === headers.length - 2) {
        td.classList.add("col-pred");
        const span = document.createElement("span");
        span.className = chipClass(String(value));
        span.textContent = String(value);
        td.appendChild(span);
      } else if (idx === headers.length - 1) {
        td.classList.add("col-conf", confClass(String(row[headers.length - 2])));
        td.textContent = `${value}%`;
      } else {
        td.textContent = String(value);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  el.previewTable.innerHTML = "";
  el.previewTable.appendChild(thead);
  el.previewTable.appendChild(tbody);
  el.previewSection.classList.remove("hidden");
}

async function handleSelectedFile(file) {
  clearError();
  resetResultsUI();

  try {
    const parsed = await parseFile(file);
    state.file = file;
    state.parsed = parsed;
    state.selectedColumn = null;
    state.predictions = null;
    state.output = null;
    setPhase("file_loaded");

    el.fileMeta.textContent = `${file.name} | ${parsed.rows.length} rows`;
    el.uploadIdle.classList.add("hidden");
    el.uploadLoaded.classList.remove("hidden");
    el.columnSection.classList.remove("hidden");

    renderColumnOptions(parsed.headers);
    el.namePreview.textContent = "";
    el.nameColumnSelect.value = "";
    el.predictBtn.disabled = true;
  } catch (err) {
    showError(err.message || "Failed to parse file.");
    setPhase("idle");
  }
}

function getSelectedNames() {
  if (!state.parsed || state.selectedColumn === null) return [];
  return state.parsed.rows.map((r) => String(r[state.selectedColumn] || ""));
}

async function runPrediction() {
  clearError();
  if (!state.parsed || state.selectedColumn === null) {
    showError("Please upload a file and choose a name column first.");
    return;
  }

  try {
    setPhase("predicting");
    const names = getSelectedNames();
    setPredictLoading(true, 0, names.length);

    const predictions = await predictBatch(names, (done, total) => setPredictLoading(true, done, total));
    const output = addPredictionColumns(state.parsed.headers, state.parsed.rows, state.selectedColumn, predictions);

    state.predictions = predictions;
    state.output = output;

    renderStats(predictions);
    renderPreview(output.headers, output.rows);
    el.downloadBtn.classList.remove("hidden");

    setPhase("results_ready");
  } catch (err) {
    showError(err.message || "Prediction failed. Please retry.");
    setPhase("column_selected");
  } finally {
    setPredictLoading(false);
  }
}

function attachUploadEvents() {
  el.uploadZone.addEventListener("click", () => el.fileInput.click());
  el.uploadZone.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      el.fileInput.click();
    }
  });

  el.fileInput.addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) handleSelectedFile(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    el.uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.uploadZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    el.uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.uploadZone.classList.remove("dragover");
    });
  });

  el.uploadZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleSelectedFile(file);
  });

  el.changeFileBtn.addEventListener("click", () => {
    el.fileInput.value = "";
    el.fileInput.click();
  });
}

function attachActionEvents() {
  el.nameColumnSelect.addEventListener("change", (ev) => {
    const val = ev.target.value;
    state.selectedColumn = val === "" ? null : Number(val);
    renderNamePreview();
    if (state.selectedColumn !== null) {
      setPhase("column_selected");
      el.predictBtn.disabled = false;
    } else {
      setPhase("file_loaded");
      el.predictBtn.disabled = true;
    }
  });

  el.predictBtn.addEventListener("click", runPrediction);

  el.downloadBtn.addEventListener("click", () => {
    clearError();
    try {
      if (!state.output) throw new Error("No prediction output available.");
      exportToCSV(state.output.headers, state.output.rows, state.file ? state.file.name : "genderlens_result.csv");
    } catch (err) {
      showError(err.message || "Failed to export CSV");
    }
  });
}

async function init() {
  clearError();
  setPredictLoading(false);
  attachUploadEvents();
  attachActionEvents();
  updateModelStatusUI();

  const [dictOk, modelOk] = await Promise.all([loadDictionary(), loadModel()]);
  updateModelStatusUI();

  if (!dictOk && !modelOk) {
    showError("Dictionary and model failed to load. Extension will use heuristic rules only.");
  } else if (!modelOk) {
    showError("Model failed to load. Using Dictionary + Rules only.");
  }
}

init().catch((err) => showError(err.message || "Initialization failed"));
