const ADAPTERS = [
  { matches: /(^|\.)linkedin\.com$/i, source: "LinkedIn", title: [".job-details-jobs-unified-top-card__job-title", "h1"], company: [".job-details-jobs-unified-top-card__company-name", "[data-tracking-control-name='public_jobs_topcard-org-name']", ".job-details-jobs-unified-top-card__primary-description a", ".topcard__org-name-link", "a[href*='/company/']"], description: [".jobs-description__content", ".jobs-description-content__text", "#job-details", ".jobs-box__html-content", ".jobs-description", "[class*='jobs-description']"] },
  { matches: /(^|\.)greenhouse\.io$/i, source: "Greenhouse", title: ["[data-qa='job-title']", "h1.app-title", "h1"], company: ["[data-qa='company-name']", ".company-name", "#header .company-name"], description: ["[data-qa='job-description']", "#content", ".content"] },
  { matches: /(^|\.)lever\.co$/i, source: "Lever", title: [".posting-headline h2", "h1"], company: [".main-header-logo img", "meta[property='og:site_name']"], description: [".posting-page .content", ".section-wrapper.page-full-width", ".posting"] },
  { matches: /(^|\.)indeed\.com$/i, source: "Indeed", title: ["[data-testid='jobsearch-JobInfoHeader-title']", "h1"], company: ["[data-testid='inlineHeader-companyName']", "[data-testid='jobsearch-InlineCompanyReview']"], description: ["#jobDescriptionText", "[data-testid='jobDescriptionText']"] },
  { matches: /(^|\.)myworkdayjobs\.com$/i, source: "Workday", title: ["[data-automation-id='jobPostingHeader']", "h1"], company: ["[data-automation-id='jobPostingCompany']", "[data-automation-id='requisitionDescription']"], description: ["[data-automation-id='jobPostingDescription']"] }
];

function textFrom(document, selectors, attributes = ["textContent"]) {
  for (const selector of selectors) {
    const element = document.querySelector(selector);
    if (!element) continue;
    for (const attribute of attributes) {
      const value = attribute === "textContent" ? element.textContent : element.getAttribute(attribute);
      if (value?.trim()) return value.trim();
    }
  }
  return "";
}

function longestTextFrom(document, selectors) {
  let longest = "";
  for (const selector of selectors) {
    const elements = document.querySelectorAll?.(selector) || [];
    for (const element of elements) {
      const value = (element.innerText || element.textContent)?.trim() || "";
      if (value.length > longest.length) longest = value;
    }
  }
  return longest;
}

function textNearHeading(document, labels, minimumLength = 200) {
  const expected = new Set(labels.map((label) => label.toLowerCase()));
  const headings = document.querySelectorAll?.("h1,h2,h3,h4,[role='heading']") || [];
  for (const heading of headings) {
    if (!expected.has((heading.textContent || "").trim().toLowerCase())) continue;
    let container = heading.parentElement;
    for (let depth = 0; container && depth < 6; depth += 1) {
      const value = (container.innerText || container.textContent)?.trim() || "";
      if (value.length >= minimumLength) return value;
      container = container.parentElement;
    }
  }
  return "";
}

function adapterFor(hostname) {
  return ADAPTERS.find((adapter) => adapter.matches.test(hostname)) || null;
}

function normalizeTitle(title, adapter) {
  if (adapter?.source === "LinkedIn") {
    return title.split(/\s*\|\s*/)[0].trim();
  }
  return title;
}

function jobIdentity(location, adapter) {
  const url = new URL(location.href);
  url.search = "";
  url.hash = "";
  let sourceJobId = "";
  if (adapter?.source === "LinkedIn") {
    const original = new URL(location.href);
    sourceJobId = location.href.match(/\/jobs\/view\/(?:[^/?#]+-)?(\d+)/)?.[1] || original.searchParams.get("currentJobId") || "";
    if (sourceJobId) url.pathname = `/jobs/view/${sourceJobId}`;
  } else if (adapter?.source === "Indeed") {
    const original = new URL(location.href);
    sourceJobId = original.searchParams.get("jk") || original.searchParams.get("vjk") || "";
    if (sourceJobId) url.search = `?jk=${encodeURIComponent(sourceJobId)}`;
  } else if (adapter?.source === "Greenhouse") {
    sourceJobId = url.pathname.match(/\/jobs\/(\d+)/)?.[1] || "";
  } else if (["Lever", "Workday"].includes(adapter?.source)) {
    sourceJobId = url.pathname.split("/").filter(Boolean).at(-1) || "";
  }
  return { canonical_url: url.toString(), source_job_id: sourceJobId };
}

function extractMetadata(document, location) {
  const adapter = adapterFor(location.hostname);
  const titleSelectors = adapter?.title || ["[data-testid='job-title']", "[data-automation='job-title']", "h1"];
  const companySelectors = adapter?.company || ["[data-testid='company-name']", "[data-automation='company-name']", ".topcard__org-name-link", "a[href*='/company/']"];
  const descriptionSelectors = adapter?.description || ["[data-testid='job-description']", "[data-automation='job-description']", "article"];
  let jobDescription = longestTextFrom(document, descriptionSelectors);
  if (adapter?.source === "LinkedIn" && jobDescription.length < 200) {
    const semanticDescription = textNearHeading(document, ["About the job"]);
    if (semanticDescription.length > jobDescription.length) jobDescription = semanticDescription;
  }
  return {
    company: textFrom(document, companySelectors, ["textContent", "aria-label", "alt"]),
    job_title: normalizeTitle(textFrom(document, titleSelectors) || document.title, adapter),
    job_url: location.href,
    job_source: adapter?.source || location.hostname,
    ...jobIdentity(location, adapter),
    job_description: jobDescription,
    description_provenance: jobDescription ? "original_saved" : "",
    extraction_quality: jobDescription ? "complete" : "unavailable"
  };
}

if (typeof module !== "undefined") module.exports = { adapterFor, extractMetadata, jobIdentity };
