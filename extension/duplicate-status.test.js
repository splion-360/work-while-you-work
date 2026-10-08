const assert = require("node:assert/strict");
const test = require("node:test");
const { duplicateApplicationStatus } = require("./duplicate-status.js");

test("shows rejected for rejected duplicate applications", () => {
  assert.deepEqual(
    duplicateApplicationStatus({ exists: true, status: "Rejected" }),
    { text: "Rejected", className: "rejected" }
  );
});

test("keeps already applied for non-rejected duplicate applications", () => {
  assert.deepEqual(
    duplicateApplicationStatus({ exists: true, status: "Applied" }),
    { text: "Already applied", className: "exists" }
  );
});

test("shows clear state when no application exists", () => {
  assert.deepEqual(
    duplicateApplicationStatus({ exists: false }),
    { text: "Not applied yet", className: "clear" }
  );
});
