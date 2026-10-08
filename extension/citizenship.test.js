const assert = require("node:assert/strict");
const test = require("node:test");
const { citizenshipSignal } = require("./citizenship.js");

test("detects sponsorship availability", () => {
  assert.deepEqual(
    citizenshipSignal("Visa sponsorship is available for this role."),
    { text: "Sponsorship Available", className: "clear" }
  );
});

test("detects US citizenship requirements", () => {
  assert.deepEqual(
    citizenshipSignal("Applicants must be U.S. citizens for this role."),
    { text: "Sponsorship Not Available", className: "restricted" }
  );
});

test("detects permanent resident and green card requirements", () => {
  assert.deepEqual(
    citizenshipSignal("Must be a US citizen or permanent resident."),
    { text: "Sponsorship Not Available", className: "restricted" }
  );
  assert.deepEqual(
    citizenshipSignal("Green card holders are eligible."),
    { text: "Sponsorship Not Available", className: "restricted" }
  );
});

test("detects no sponsorship language", () => {
  assert.deepEqual(
    citizenshipSignal("We cannot sponsor visas for this position."),
    { text: "Sponsorship Not Available", className: "restricted" }
  );
});

test("returns a clear state when no signal is present", () => {
  assert.deepEqual(
    citizenshipSignal("Build machine learning systems with Python."),
    { text: "No information", className: "warning" }
  );
});
