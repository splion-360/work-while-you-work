function scorePresentation(score, band, fitLabel) {
  const numericScore = Number(score);
  if (!Number.isFinite(numericScore) || numericScore < 0 || numericScore > 100) {
    throw new Error("Score must be between 0 and 100");
  }
  const normalizedFitLabel = String(fitLabel || "").trim().toLowerCase();
  if (normalizedFitLabel) {
    if (!["good fit", "no fit"].includes(normalizedFitLabel)) {
      throw new Error("Unknown fit label");
    }
    const goodFit = normalizedFitLabel === "good fit";
    return {
      className: goodFit ? "score-high" : "score-low",
      text: `${numericScore}/100 (${goodFit ? "Good Fit" : "No Fit"})`
    };
  }
  const normalized = String(band || "").trim().toLowerCase();
  const bands = {
    strong: { className: "score-high", label: "High" },
    high: { className: "score-high", label: "High" },
    moderate: { className: "score-moderate", label: "Moderate" },
    medium: { className: "score-moderate", label: "Moderate" },
    weak: { className: "score-low", label: "Low" },
    low: { className: "score-low", label: "Low" }
  };
  const presentation = bands[normalized];
  if (!presentation) throw new Error("Unknown score band");
  return {
    className: presentation.className,
    text: `${numericScore}/100 (${presentation.label})`
  };
}

function classifierBreakdownPresentation(result) {
  const probability = Number(result?.fit_probability);
  const threshold = Number(result?.decision_threshold);
  if (!Number.isFinite(probability) || !Number.isFinite(threshold)) return [];
  return [
    { label: "Decision", value: result.fit_label },
    { label: "Good Fit probability", value: `${Math.round(probability * 1000) / 10}%` },
    { label: "Decision threshold", value: `${Math.round(threshold * 1000) / 10}%` }
  ];
}

function scoreSourceLabel(source) {
  const labels = { cached: "Cached", fetched: "Fetched", computed: "Computed" };
  if (!labels[source]) throw new Error("Unknown score source");
  return labels[source];
}

function scoreBreakdownPresentation(categories) {
  const labels = {
    required: "Required",
    core: "Core",
    responsibilities: "Duties",
    preferred: "Preferred",
    domain: "Domain",
    bonus: "Bonus"
  };
  return Object.entries(labels).flatMap(([name, label]) => {
    const value = Number(categories?.[name]);
    if (!Number.isFinite(value)) return [];
    const rounded = Math.round(value * 10) / 10;
    const band = rounded >= 75 ? "High" : rounded >= 55 ? "Moderate" : "Low";
    return [{
      className: scorePresentation(rounded, band).className,
      label,
      value: rounded
    }];
  });
}

if (typeof module !== "undefined") {
  module.exports = { classifierBreakdownPresentation, scoreBreakdownPresentation, scorePresentation, scoreSourceLabel };
}
