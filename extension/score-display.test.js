const assert = require("node:assert/strict");
const test = require("node:test");
const {
  classifierBreakdownPresentation,
  scoreBreakdownPresentation,
  scorePresentation,
  scoreSourceLabel
} = require("./score-display.js");

test("maps backend score bands to user-facing labels and colors", () => {
  assert.deepEqual(scorePresentation(81.2, "Strong"), {
    className: "score-high",
    text: "81.2/100 (High)"
  });
  assert.deepEqual(scorePresentation(66.6, "Moderate"), {
    className: "score-moderate",
    text: "66.6/100 (Moderate)"
  });
  assert.deepEqual(scorePresentation(47.2, "Weak"), {
    className: "score-low",
    text: "47.2/100 (Low)"
  });
});

test("accepts equivalent API band names", () => {
  assert.equal(scorePresentation(80, "high").text, "80/100 (High)");
  assert.equal(scorePresentation(60, "medium").text, "60/100 (Moderate)");
  assert.equal(scorePresentation(40, "low").text, "40/100 (Low)");
});

test("presents the binary classifier decision instead of a legacy band", () => {
  assert.deepEqual(scorePresentation(63.2, "Moderate", "Good Fit"), {
    className: "score-high",
    text: "63.2/100 (Good Fit)"
  });
  assert.equal(scorePresentation(35, "Weak", "No Fit").className, "score-low");
});

test("formats calibrated classifier details", () => {
  assert.deepEqual(classifierBreakdownPresentation({
    fit_label: "Good Fit",
    fit_probability: 0.6321,
    decision_threshold: 0.394191
  }), [
    { label: "Decision", value: "Good Fit" },
    { label: "Good Fit probability", value: "63.2%" },
    { label: "Decision threshold", value: "39.4%" }
  ]);
});

test("labels the score result source", () => {
  assert.equal(scoreSourceLabel("cached"), "Cached");
  assert.equal(scoreSourceLabel("fetched"), "Fetched");
  assert.equal(scoreSourceLabel("computed"), "Computed");
});

test("rejects invalid scores and unknown bands", () => {
  assert.throws(() => scorePresentation("missing", "Strong"), /score/i);
  assert.throws(() => scorePresentation(70, "Unknown"), /band/i);
});

test("formats available score axes in a stable order", () => {
  assert.deepEqual(scoreBreakdownPresentation({
    bonus: 33.3,
    required: 83.2,
    responsibilities: 49,
    core: 67,
    ignored: 100
  }), [
    { className: "score-high", label: "Required", value: 83.2 },
    { className: "score-moderate", label: "Core", value: 67 },
    { className: "score-low", label: "Duties", value: 49 },
    { className: "score-low", label: "Bonus", value: 33.3 }
  ]);
});
