// GenderLens v1.0 - github.com/your-username/gender-lens
let nameDict = null;
let model = null;
let modelStatus = "rules-only";
let modelLoadError = "";
let manualModelWeights = null;

const FEMALE_SUFFIXES = [
  "shree", "priya", "devi", "vati", "vathi", "mati", "bai", "lata", "wati",
  "deep", "ita", "ika", "iya", "ina", "ine", "ette", "elle",
  "ia", "na", "ie", "ee", "athi", "sri", "puja",
  "isha", "eeka", "ini", "ani", "oni", "angi", "shri", "mala", "rani", "shika"
].sort((a, b) => b.length - a.length);

const MALE_SUFFIXES = [
  "esh", "raj", "vik", "son", "man", "ren", "han", "jin",
  "oud", "ud", "ith", "ath", "jit", "nath", "das", "dev",
  "dra", "ndra", "nkar", "swar", "pati", "nand", "veer", "pal", "kar", "ram",
  "an", "on", "in", "er", "ar", "or", "ir"
].sort((a, b) => b.length - a.length);

const FEMALE_PREFIXES = ["su", "sa", "ra", "la", "ka", "ma", "pa", "na", "aa", "an", "sh"];
const MALE_PREFIXES = ["bi", "vi", "ni", "pr", "br", "tr", "kr", "dh", "ra", "sh", "bh"];

const ODIA_FEMALE = new Set([
  "subhashree", "lipsa", "smruti", "sasmita", "jyotirmayee", "rashmita",
  "sujata", "lopamudra", "sarada", "itishree", "debasmita", "pallabi",
  "ipsita", "pratima", "nibedita", "sanghamitra", "chitralekha", "sucharita",
  "hemalata", "lilavati", "malati", "kuntala", "binodini"
]);

const ODIA_MALE = new Set([
  "biswajit", "satyajit", "debasis", "sipun", "niladri", "trilochan",
  "pranab", "subrat", "biswaranjan", "bichitrananda", "hrudananda",
  "sudhansu", "amarendra", "rabindra", "jatindra", "patitapaban",
  "brundaban", "madhusudan", "jagannath", "pitambar", "laxmidhar"
]);

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeName(rawName) {
  return String(rawName || "")
    .trim()
    .toLowerCase()
    .split(/[\s,.-]+/)
    .map((x) => x.replace(/[^a-z]/g, ""))
    .filter(Boolean)[0] || "";
}

function buildNameCandidates(rawName) {
  const parts = String(rawName || "")
    .trim()
    .toLowerCase()
    .split(/[\s,.-]+/)
    .map((x) => x.replace(/[^a-z]/g, ""))
    .filter(Boolean);

  const first = parts[0] || "";
  const combinedTwo = parts.slice(0, 2).join("");
  const combinedAll = parts.join("");

  const seen = new Set();
  const candidates = [];
  [combinedAll, combinedTwo, first].forEach((k) => {
    if (k && !seen.has(k)) {
      seen.add(k);
      candidates.push(k);
    }
  });

  return { first, candidates, parts };
}


function buildFallbackModel() {
  return tf.sequential({
    layers: [
      tf.layers.inputLayer({ inputShape: [20, 28] }),
      tf.layers.conv1d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu", name: "conv1d" }),
      tf.layers.conv1d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu", name: "conv1d_1" }),
      tf.layers.globalMaxPooling1d({ name: "global_max_pooling1d" }),
      tf.layers.dense({ units: 32, activation: "relu", name: "dense" }),
      tf.layers.dropout({ rate: 0.3, name: "dropout" }),
      tf.layers.dense({ units: 1, activation: "sigmoid", name: "dense_1" })
    ]
  });
}

async function loadWeightsIntoFallbackModel(targetModel) {
  const weightsUrl = chrome.runtime.getURL("model/weights.bin");
  const res = await fetch(weightsUrl);
  if (!res.ok) throw new Error(`weights.bin fetch failed: ${res.status}`);

  const buffer = await res.arrayBuffer();
  const raw = new Float32Array(buffer);

  // Warm up to create layer weights.
  tf.tidy(() => {
    const x = tf.zeros([1, 20, 28]);
    targetModel.predict(x);
  });

  const template = targetModel.getWeights();
  const loaded = [];
  let offset = 0;

  for (const w of template) {
    const size = w.size;
    const next = offset + size;
    if (next > raw.length) {
      template.forEach((t) => t.dispose());
      throw new Error("weights.bin length mismatch");
    }
    const slice = raw.slice(offset, next);
    loaded.push(tf.tensor(slice, w.shape, "float32"));
    offset = next;
  }

  if (offset !== raw.length) {
    template.forEach((t) => t.dispose());
    loaded.forEach((t) => t.dispose());
    throw new Error("weights.bin has trailing values");
  }

  targetModel.setWeights(loaded);
  template.forEach((t) => t.dispose());
  loaded.forEach((t) => t.dispose());
}


function relu(x) {
  return x > 0 ? x : 0;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function readSegment(raw, offset, size, shape) {
  const next = offset + size;
  if (next > raw.length) throw new Error("weights.bin length mismatch");
  const data = raw.slice(offset, next);
  return { data, next, shape };
}

function reshape2D(flat, rows, cols) {
  const out = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols);
    const base = r * cols;
    for (let c = 0; c < cols; c++) row[c] = flat[base + c];
    out[r] = row;
  }
  return out;
}

function parseManualWeights(raw) {
  let offset = 0;
  const conv1Kernel = readSegment(raw, offset, 3 * 28 * 64, [3, 28, 64]); offset = conv1Kernel.next;
  const conv1Bias = readSegment(raw, offset, 64, [64]); offset = conv1Bias.next;
  const conv2Kernel = readSegment(raw, offset, 3 * 64 * 32, [3, 64, 32]); offset = conv2Kernel.next;
  const conv2Bias = readSegment(raw, offset, 32, [32]); offset = conv2Bias.next;
  const dense1Kernel = readSegment(raw, offset, 32 * 32, [32, 32]); offset = dense1Kernel.next;
  const dense1Bias = readSegment(raw, offset, 32, [32]); offset = dense1Bias.next;
  const dense2Kernel = readSegment(raw, offset, 32 * 1, [32, 1]); offset = dense2Kernel.next;
  const dense2Bias = readSegment(raw, offset, 1, [1]); offset = dense2Bias.next;

  if (offset !== raw.length) throw new Error("weights.bin trailing values");

  return {
    conv1Kernel: conv1Kernel.data,
    conv1Bias: conv1Bias.data,
    conv2Kernel: conv2Kernel.data,
    conv2Bias: conv2Bias.data,
    dense1Kernel: dense1Kernel.data,
    dense1Bias: dense1Bias.data,
    dense2Kernel: dense2Kernel.data,
    dense2Bias: dense2Bias.data,
  };
}

function conv1dSame(input, inChannels, outChannels, kernelFlat, biasFlat) {
  const seqLen = 20;
  const out = new Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const row = new Array(outChannels);
    for (let oc = 0; oc < outChannels; oc++) {
      let sum = biasFlat[oc];
      for (let k = 0; k < 3; k++) {
        const src = i + k - 1;
        if (src < 0 || src >= seqLen) continue;
        const inVec = input[src];
        const base = ((k * inChannels) * outChannels) + oc;
        for (let ic = 0; ic < inChannels; ic++) {
          sum += inVec[ic] * kernelFlat[base + ic * outChannels];
        }
      }
      row[oc] = relu(sum);
    }
    out[i] = row;
  }
  return out;
}

function globalMaxPool(input, channels) {
  const out = new Array(channels).fill(-Infinity);
  for (let i = 0; i < input.length; i++) {
    for (let c = 0; c < channels; c++) {
      if (input[i][c] > out[c]) out[c] = input[i][c];
    }
  }
  return out;
}

function dense(input, inUnits, outUnits, kernelFlat, biasFlat, activationRelu) {
  const out = new Array(outUnits);
  for (let o = 0; o < outUnits; o++) {
    let sum = biasFlat[o];
    for (let i = 0; i < inUnits; i++) {
      sum += input[i] * kernelFlat[i * outUnits + o];
    }
    out[o] = activationRelu ? relu(sum) : sum;
  }
  return out;
}

function manualPredictProbability(encoded, weights) {
  const layer1 = conv1dSame(encoded, 28, 64, weights.conv1Kernel, weights.conv1Bias);
  const layer2 = conv1dSame(layer1, 64, 32, weights.conv2Kernel, weights.conv2Bias);
  const pooled = globalMaxPool(layer2, 32);
  const dense1 = dense(pooled, 32, 32, weights.dense1Kernel, weights.dense1Bias, true);
  const dense2 = dense(dense1, 32, 1, weights.dense2Kernel, weights.dense2Bias, false);
  return sigmoid(dense2[0]);
}

async function loadManualModelFromWeights() {
  const res = await fetch(chrome.runtime.getURL("model/weights.bin"));
  if (!res.ok) throw new Error(`weights.bin fetch failed: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const raw = new Float32Array(buffer);
  manualModelWeights = parseManualWeights(raw);
}

function getVowelRatio(name) {
  if (!name.length) return 0;
  const vowels = (name.match(/[aeiou]/g) || []).length;
  return vowels / name.length;
}

export function encodeName(name) {
  const clean = name.trim().toLowerCase().split(/[\s,.-]+/)[0].slice(0, 20);
  const tensor = [];
  for (let i = 0; i < 20; i++) {
    const row = new Array(28).fill(0);
    if (i < clean.length) {
      const code = clean.charCodeAt(i) - 96;
      if (code >= 1 && code <= 26) row[code] = 1;
      else if (clean[i] === " ") row[27] = 1;
    }
    tensor.push(row);
  }
  return tensor;
}

export async function loadDictionary() {
  try {
    const url = chrome.runtime.getURL("data/name_gender_dict.json");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Dictionary fetch failed: ${res.status}`);
    nameDict = await res.json();
    if (!nameDict || typeof nameDict !== "object") {
      throw new Error("Dictionary JSON is invalid");
    }
    if (modelStatus === "rules-only") modelStatus = "dict+rules";
    return true;
  } catch {
    nameDict = null;
    modelStatus = "rules-only";
    return false;
  }
}

export async function loadModel() {
  modelLoadError = "";
  model = null;
  manualModelWeights = null;

  const hasTfLoader = !!(globalThis.tf && typeof tf.loadLayersModel === "function");
  const hasTfLayers = !!(globalThis.tf && tf.layers && tf.sequential);

  if (hasTfLoader) {
    try {
      model = await tf.loadLayersModel(chrome.runtime.getURL("model/model.json"));
      modelStatus = nameDict ? "model+dict" : "rules-only";
      return true;
    } catch (primaryErr) {
      if (hasTfLayers) {
        try {
          const fallback = buildFallbackModel();
          await loadWeightsIntoFallbackModel(fallback);
          model = fallback;
          modelStatus = nameDict ? "model+dict" : "rules-only";
          return true;
        } catch (fallbackErr) {
          modelLoadError = `model.json load failed: ${primaryErr?.message || primaryErr}; tf-layers fallback failed: ${fallbackErr?.message || fallbackErr}`;
        }
      } else {
        modelLoadError = `model.json load failed: ${primaryErr?.message || primaryErr}; tf-layers fallback unavailable`;
      }
    }
  }

  try {
    await loadManualModelFromWeights();
    modelStatus = nameDict ? "model+dict" : "rules-only";
    if (!hasTfLoader) {
      const tfState = globalThis.tf ? `tf_version=${globalThis.tf.version?.tfjs || "unknown"}, loadLayersModel=${typeof globalThis.tf.loadLayersModel}` : "tf undefined";
      modelLoadError = `Using manual Conv1D fallback (${tfState})`;
    }
    return true;
  } catch (manualErr) {
    const tfState = globalThis.tf ? `tf_version=${globalThis.tf.version?.tfjs || "unknown"}, loadLayersModel=${typeof globalThis.tf.loadLayersModel}` : "tf undefined";
    model = null;
    manualModelWeights = null;
    modelStatus = nameDict ? "dict+rules" : "rules-only";
    modelLoadError = `${modelLoadError ? modelLoadError + "; " : ""}manual fallback failed: ${manualErr?.message || manualErr} (${tfState})`;
    return false;
  }
}


export function getModelStatus() {
  return modelStatus;
}

export async function predictGender(rawName) {
  try {
    const { first: name, candidates, parts } = buildNameCandidates(rawName);
    if (!name) return { gender: "Unknown", confidence: 0 };

    if (parts.includes("ranjan")) return { gender: "Male", confidence: 95 };

    for (const key of candidates) {
      if (ODIA_FEMALE.has(key)) return { gender: "Female", confidence: 95 };
      if (ODIA_MALE.has(key)) return { gender: "Male", confidence: 95 };
    }

    if (nameDict) {
      for (const key of candidates) {
        const entry = nameDict[key];
        if (!entry) continue;
        const conf = Number(entry.confidence || 0);
        if (conf >= 0.8) {
          return {
            gender: entry.gender === "Male" ? "Male" : entry.gender === "Female" ? "Female" : "Unknown",
            confidence: clamp(Math.round(conf * 100), 0, 100)
          };
        }
      }
    }

    if (model) {
      try {
        const prediction = tf.tidy(() => {
          const input = tf.tensor([encodeName(name)], [1, 20, 28], "float32");
          const out = model.predict(input);
          return out.dataSync()[0];
        });
        const prob = clamp(Number(prediction), 0, 1);
        return {
          gender: prob > 0.5 ? "Male" : "Female",
          confidence: clamp(Math.round(Math.abs(prob - 0.5) * 200), 0, 100)
        };
      } catch {
      }
    }

    if (manualModelWeights) {
      try {
        const prob = clamp(Number(manualPredictProbability(encodeName(name), manualModelWeights)), 0, 1);
        return {
          gender: prob > 0.5 ? "Male" : "Female",
          confidence: clamp(Math.round(Math.abs(prob - 0.5) * 200), 0, 100)
        };
      } catch {
      }
    }

    for (const suffix of FEMALE_SUFFIXES) {
      if (name.endsWith(suffix)) return { gender: "Female", confidence: clamp(60 + Math.min(30, suffix.length * 4), 0, 100) };
    }

    for (const suffix of MALE_SUFFIXES) {
      if (name.endsWith(suffix)) return { gender: "Male", confidence: clamp(60 + Math.min(30, suffix.length * 4), 0, 100) };
    }

    const ratio = getVowelRatio(name);
    const femalePrefix = FEMALE_PREFIXES.some((p) => name.startsWith(p));
    const malePrefix = MALE_PREFIXES.some((p) => name.startsWith(p));

    if (femalePrefix && ratio > 0.55) return { gender: "Female", confidence: 65 };
    if (malePrefix && ratio < 0.35) return { gender: "Male", confidence: 65 };

    return { gender: "Unknown", confidence: 30 };
  } catch {
    return { gender: "Unknown", confidence: 0 };
  }
}

export async function predictBatch(namesArr, onProgress) {
  const results = [];
  for (let i = 0; i < namesArr.length; i++) {
    results.push(await predictGender(namesArr[i]));
    if (typeof onProgress === "function") onProgress(i + 1, namesArr.length);
    if ((i + 1) % 50 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return results;
}

export function getModelLoadError() {
  return modelLoadError;
}
