const assert = require("node:assert/strict");
const test = require("node:test");
const { isGitHubUrl, isLinkedInUrl, normalizeHttpUrl } = require("./settings.js");

test("normalizes valid saved URLs", () => {
  assert.equal(normalizeHttpUrl(" https://splion-360.github.io/website "), "https://splion-360.github.io/website");
  assert.equal(normalizeHttpUrl(""), "");
});

test("rejects non-http URLs", () => {
  assert.throws(() => normalizeHttpUrl("javascript:alert(1)"), /http/);
});

test("accepts only LinkedIn hosts for the LinkedIn field", () => {
  assert.equal(isLinkedInUrl("https://www.linkedin.com/in/example"), true);
  assert.equal(isLinkedInUrl("https://linkedin.example.com/in/example"), false);
});

test("accepts only GitHub hosts for the GitHub field", () => {
  assert.equal(isGitHubUrl("https://github.com/example"), true);
  assert.equal(isGitHubUrl("https://github.example.com/example"), false);
});
