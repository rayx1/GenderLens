// GenderLens v1.0 - github.com/your-username/gender-lens
let nameDict = null;
let model = null;
let modelStatus = "rules-only";

const FEMALE_SUFFIXES = [
  "shree", "priya", "devi", "vati", "mati", "bai", "lata", "wati",
  "deep", "ita", "ika", "iya", "ina", "ine", "ette", "elle",
  "ia", "na", "ie", "ee", "athi", "sri", "puja",
  "isha", "eeka", "ini", "ani", "oni"
].sort((a, b) => b.length - a.length);

const MALE_SUFFIXES = [
  "esh", "raj", "vik", "son", "man", "ren", "han", "jin",
  "oud", "ud", "ith", "ath", "jit", "nath", "das", "dev",
  "dra", "ndra", "nkar", "swar", "pati", "nand",
  "an", "on", "in", "er", "ar", "or", "ir"
].sort((a, b) => b.length - a.length);

const FEMALE_PREFIXES = ["su", "sa", "ra", "la", "ka", "ma", "pa", "na"];
const MALE_PREFIXES = ["bi", "vi", "ni", "pr", "br", "tr", "kr", "dh"];

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
    .filter(Boolean)[0] || "";
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
  try {
    if (!globalThis.tf || !tf.loadLayersModel) {
      model = null;
      modelStatus = nameDict ? "dict+rules" : "rules-only";
      return false;
    }
    model = await tf.loadLayersModel(chrome.runtime.getURL("model/model.json"));
    modelStatus = nameDict ? "model+dict" : "rules-only";
    return true;
  } catch {
    model = null;
    modelStatus = nameDict ? "dict+rules" : "rules-only";
    return false;
  }
}

export function getModelStatus() {
  return modelStatus;
}

export async function predictGender(rawName) {
  try {
    const name = normalizeName(rawName);
    if (!name) return { gender: "Unknown", confidence: 0 };

    if (ODIA_FEMALE.has(name)) return { gender: "Female", confidence: 95 };
    if (ODIA_MALE.has(name)) return { gender: "Male", confidence: 95 };

    if (nameDict && nameDict[name]) {
      const entry = nameDict[name];
      const conf = Number(entry.confidence || 0);
      if (conf >= 0.8) {
        return {
          gender: entry.gender === "Male" ? "Male" : entry.gender === "Female" ? "Female" : "Unknown",
          confidence: clamp(Math.round(conf * 100), 0, 100)
        };
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
