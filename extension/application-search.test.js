const assert = require("node:assert/strict");
const test = require("node:test");
const { matchingApplications } = require("./application-search.js");

const applications = [
  { application_key: "1", company: "Gray Swan", job_title: "Machine Learning Engineer", job_source: "LinkedIn", status: "Applied" },
  { application_key: "2", company: "Acme", job_title: "AI Research Engineer", job_source: "Greenhouse", status: "In Progress" },
  { application_key: "3", company: "Elsewhere", job_title: "Backend Engineer", job_source: "Lever", status: "Rejected" },
];

test("matches typeahead text across company, title, and source", () => {
  assert.deepEqual(matchingApplications(applications, "gray machine").map((item) => item.application_key), ["1"]);
  assert.deepEqual(matchingApplications(applications, "greenhouse").map((item) => item.application_key), ["2"]);
});

test("returns recent entries for an empty query and respects the limit", () => {
  assert.deepEqual(matchingApplications(applications, "", 2), applications.slice(0, 2));
});

test("filters applications by status", () => {
  assert.deepEqual(matchingApplications(applications, "", 8, "Rejected").map((item) => item.application_key), ["3"]);
});

test("combines text search and status filtering", () => {
  assert.deepEqual(matchingApplications(applications, "engineer", 8, "In Progress").map((item) => item.application_key), ["2"]);
});
