// GenderLens v1.0 - github.com/your-username/gender-lens
function normalizeCell(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function isEmptyRow(row) {
  return row.every((cell) => String(cell).trim() === "");
}

export async function parseFile(file) {
  if (!file) throw new Error("No file selected");
  if (!globalThis.XLSX) throw new Error("SheetJS library is missing (lib/xlsx.full.min.js)");

  const lower = file.name.toLowerCase();
  let workbook;

  if (lower.endsWith(".csv")) {
    const text = await file.text();
    workbook = XLSX.read(text, { type: "string" });
  } else if (lower.endsWith(".xls") || lower.endsWith(".xlsx")) {
    const buffer = await file.arrayBuffer();
    workbook = XLSX.read(buffer, { type: "array" });
  } else {
    throw new Error("Unsupported file type. Please upload .csv, .xls, or .xlsx");
  }

  const sheetName = workbook.SheetNames[0];
  if (!sheetName) throw new Error("No worksheet found in file");

  const sheet = workbook.Sheets[sheetName];
  const aoa = XLSX.utils.sheet_to_json(sheet, { header: 1, raw: false, defval: "" });
  const cleaned = aoa
    .map((row) => (Array.isArray(row) ? row.map(normalizeCell) : []))
    .filter((row) => row.length > 0 && !isEmptyRow(row));

  if (cleaned.length === 0) throw new Error("File is empty or contains no usable rows");

  const headers = cleaned[0].map((h, i) => String(h || "").trim() || `Column ${i + 1}`);
  const rows = cleaned.slice(1).map((row) => {
    const padded = [...row];
    while (padded.length < headers.length) padded.push("");
    return padded.slice(0, headers.length);
  });

  return { headers, rows, sheetName };
}

export function addPredictionColumns(headers, rows, nameColIndex, predictions) {
  const outHeaders = [...headers, "Predicted Gender", "Confidence %"];
  const outRows = rows.map((row, idx) => {
    const prediction = predictions[idx] || { gender: "Unknown", confidence: 0 };
    return [
      ...row,
      prediction.gender,
      Number.isFinite(prediction.confidence) ? Math.max(0, Math.min(100, Math.round(prediction.confidence))) : 0
    ];
  });
  return { headers: outHeaders, rows: outRows };
}

export function exportToCSV(headers, rows, filename) {
  if (!globalThis.XLSX) throw new Error("SheetJS library is missing (lib/xlsx.full.min.js)");

  const ws = XLSX.utils.aoa_to_sheet([headers, ...rows]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "GenderLens Result");

  const csvArray = XLSX.write(wb, { type: "array", bookType: "csv" });
  const blob = new Blob([csvArray], { type: "text/csv;charset=utf-8;" });
  const safeName = (filename || "genderlens_result.csv").replace(/\.(xlsx|xls|csv)$/i, "") + "_genderlens.csv";
  const url = URL.createObjectURL(blob);

  if (chrome && chrome.downloads && chrome.downloads.download) {
    chrome.downloads.download({ url, filename: safeName, saveAs: true }, () => {
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    });
    return;
  }

  const a = document.createElement("a");
  a.href = url;
  a.download = safeName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
