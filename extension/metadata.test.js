const assert = require("node:assert/strict");
const test = require("node:test");
const { extractMetadata } = require("./metadata.js");

function documentFixture(values) {
  const element = (value, selector) => ({
    textContent: value,
    innerText: values[`${selector}:innerText`] || value,
    getAttribute(attribute) { return values[`${selector}:${attribute}`] || ""; }
  });
  return {
    title: values.documentTitle || "Fallback title",
    querySelector(selector) {
      const value = values[selector];
      if (value === undefined) return null;
      return element(Array.isArray(value) ? value[0] : value, selector);
    },
    querySelectorAll(selector) {
      const value = values[selector];
      if (value === undefined) return [];
      return (Array.isArray(value) ? value : [value]).map((item) => element(item, selector));
    }
  };
}

test("extracts LinkedIn metadata with adapter selectors", () => {
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "Machine Learning Engineer",
    ".job-details-jobs-unified-top-card__company-name": "TurboHome",
    ".jobs-description__content": "Build and deploy production machine learning systems."
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/123" });
  assert.deepEqual(metadata, {
    company: "TurboHome",
    job_title: "Machine Learning Engineer",
    job_url: "https://www.linkedin.com/jobs/view/123",
    job_source: "LinkedIn",
    canonical_url: "https://www.linkedin.com/jobs/view/123",
    source_job_id: "123",
    job_description: "Build and deploy production machine learning systems.",
    description_provenance: "original_saved",
    extraction_quality: "complete"
  });
});

test("uses the longest populated LinkedIn description node", () => {
  const fullDescription = "Build production ML systems with Python, PyTorch, monitoring, and reliable deployment pipelines.";
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "Machine Learning Engineer",
    ".job-details-jobs-unified-top-card__company-name": "Example AI",
    ".jobs-description__content": ["About the job", fullDescription]
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/123" });

  assert.equal(metadata.job_description, fullDescription);
});

test("preserves rendered job description section boundaries", () => {
  const structured = "About us\n\nWhat You'll Do\nBuild production ML systems.";
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "Machine Learning Engineer",
    ".job-details-jobs-unified-top-card__company-name": "Example AI",
    ".jobs-description__content": "About us What You'll Do Build production ML systems.",
    ".jobs-description__content:innerText": structured
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/123" });

  assert.equal(metadata.job_description, structured);
});

test("finds LinkedIn description from the About the job section", () => {
  const fullDescription = ("About the job " + "Build and deploy reliable production machine learning systems. ".repeat(5)).trim();
  const section = { textContent: fullDescription, parentElement: null };
  const heading = { textContent: "About the job", parentElement: section };
  const fixture = documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "Machine Learning Engineer",
    ".job-details-jobs-unified-top-card__company-name": "Example AI"
  });
  const querySelectorAll = fixture.querySelectorAll;
  fixture.querySelectorAll = (selector) => (
    selector === "h1,h2,h3,h4,[role='heading']" ? [heading] : querySelectorAll(selector)
  );

  const metadata = extractMetadata(fixture, {
    hostname: "www.linkedin.com",
    href: "https://www.linkedin.com/jobs/view/123"
  });

  assert.equal(metadata.job_description, fullDescription);
});

test("canonicalizes SPA job URLs and reports unavailable descriptions", () => {
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "ML Engineer",
    ".job-details-jobs-unified-top-card__company-name": "Example AI"
  }), {
    hostname: "www.linkedin.com",
    href: "https://www.linkedin.com/jobs/view/987?trackingId=abc#details"
  });
  assert.equal(metadata.canonical_url, "https://www.linkedin.com/jobs/view/987");
  assert.equal(metadata.source_job_id, "987");
  assert.equal(metadata.job_description, "");
  assert.equal(metadata.description_provenance, "");
  assert.equal(metadata.extraction_quality, "unavailable");
});

test("extracts LinkedIn company from the primary-description link", () => {
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "ML Platform Engineer",
    ".job-details-jobs-unified-top-card__primary-description a": "Stitch Fix"
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/456" });
  assert.equal(metadata.company, "Stitch Fix");
});

test("keeps only the LinkedIn job title when the heading includes metadata", () => {
  const metadata = extractMetadata(documentFixture({
    h1: "ML Platform Engineer | Stitch Fix | LinkedIn"
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/789" });
  assert.equal(metadata.job_title, "ML Platform Engineer");
});

test("extracts LinkedIn company from the top-card tracking link", () => {
  const metadata = extractMetadata(documentFixture({
    ".job-details-jobs-unified-top-card__job-title": "ML Platform Engineer",
    "[data-tracking-control-name='public_jobs_topcard-org-name']": "Stitch Fix"
  }), { hostname: "www.linkedin.com", href: "https://www.linkedin.com/jobs/view/999" });
  assert.equal(metadata.company, "Stitch Fix");
});

test("extracts Greenhouse metadata", () => {
  const metadata = extractMetadata(documentFixture({
    "[data-qa='job-title']": "Applied ML Engineer",
    "[data-qa='company-name']": "Example AI"
  }), { hostname: "boards.greenhouse.io", href: "https://boards.greenhouse.io/example/jobs/1" });
  assert.equal(metadata.company, "Example AI");
  assert.equal(metadata.job_title, "Applied ML Engineer");
  assert.equal(metadata.job_source, "Greenhouse");
});

test("extracts Lever, Indeed, and Workday metadata", () => {
  const cases = [
    {
      hostname: "jobs.lever.co",
      values: { ".posting-headline h2": "ML Engineer", ".main-header-logo img": "", ".main-header-logo img:alt": "Example Labs" },
      source: "Lever"
    },
    {
      hostname: "www.indeed.com",
      values: { "[data-testid='jobsearch-JobInfoHeader-title']": "Applied ML Engineer", "[data-testid='inlineHeader-companyName']": "Example AI" },
      source: "Indeed"
    },
    {
      hostname: "example.wd5.myworkdayjobs.com",
      values: { "[data-automation-id='jobPostingHeader']": "Machine Learning Engineer", "[data-automation-id='jobPostingCompany']": "Example Corp" },
      source: "Workday"
    }
  ];
  for (const entry of cases) {
    const metadata = extractMetadata(documentFixture(entry.values), {
      hostname: entry.hostname,
      href: `https://${entry.hostname}/job/1`
    });
    assert.equal(metadata.job_source, entry.source);
    assert.ok(metadata.company);
    assert.ok(metadata.job_title);
  }
});

test("retains the Indeed job key in the canonical URL", () => {
  const metadata = extractMetadata(documentFixture({
    "[data-testid='jobsearch-JobInfoHeader-title']": "ML Engineer",
    "[data-testid='inlineHeader-companyName']": "Example AI",
    "#jobDescriptionText": "Develop production models."
  }), {
    hostname: "www.indeed.com",
    href: "https://www.indeed.com/viewjob?jk=abc123&from=search"
  });
  assert.equal(metadata.source_job_id, "abc123");
  assert.equal(metadata.canonical_url, "https://www.indeed.com/viewjob?jk=abc123");
});

test("uses generic extraction for unknown sites", () => {
  const metadata = extractMetadata(documentFixture({
    h1: "ML Engineer",
    "a[href*='/company/']": "Example Labs"
  }), { hostname: "jobs.example.com", href: "https://jobs.example.com/1" });
  assert.equal(metadata.company, "Example Labs");
  assert.equal(metadata.job_title, "ML Engineer");
  assert.equal(metadata.job_source, "jobs.example.com");
});
