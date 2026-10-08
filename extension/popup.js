const config = globalThis.JOB_TRACKER_CONFIG || { service: "http://127.0.0.1:8765", apiKey: "" };
const service = config.service;
const requestHeaders = config.apiKey ? { "X-Job-Tracker-Key": config.apiKey } : {};
const resumeSelect = document.querySelector("#resume");
const resumePicker = document.querySelector("#resume-picker");
const resumePickerButton = document.querySelector("#resume-picker-button");
const resumePickerLabel = document.querySelector("#resume-picker-label");
const resumeMenu = document.querySelector("#resume-menu");
const companyInput = document.querySelector("#company");
const titleInput = document.querySelector("#title");
const sourceInput = document.querySelector("#source");
const applicationFields = document.querySelector("#application-fields");
const descriptionEditor = document.querySelector("#description-editor");
const descriptionInput = document.querySelector("#description");
const dateInput = document.querySelector("#date");
const statusInput = document.querySelector("#status");
const sponsorshipInput = document.querySelector("#sponsorship");
const referralInput = document.querySelector("#referral");
const generatedPasswordInput = document.querySelector("#generated-password");
const generatePasswordButton = document.querySelector("#generate-password");
const copyPasswordButton = document.querySelector("#copy-password");
const message = document.querySelector("#message");
const messageText = document.querySelector("#message-text");
const logView = document.querySelector("#log-view");
const applicationsTab = document.querySelector("#applications-tab");
const settingsView = document.querySelector("#settings-view");
const utilitiesView = document.querySelector("#utilities-view");
const dashboardView = document.querySelector("#dashboard-view");
const searchView = document.querySelector("#search-view");
const searchButton = document.querySelector("#search");
const dashboardButton = document.querySelector("#dashboard");
const refreshContextButton = document.querySelector("#refresh-context");
const refreshContextIcon = document.querySelector("#refresh-context-icon");
const utilitiesButton = document.querySelector("#utilities-tab");
const settingsButton = document.querySelector("#settings");
const linkedinUrlInput = document.querySelector("#linkedin-url");
const websiteUrlInput = document.querySelector("#website-url");
const githubUrlInput = document.querySelector("#github-url");
const settingsMessage = document.querySelector("#settings-message");
const copyLinkedInButton = document.querySelector("#copy-linkedin");
const copyWebsiteButton = document.querySelector("#copy-website");
const copyGitHubButton = document.querySelector("#copy-github");
const duplicateStatus = document.querySelector("#duplicate-status");
const citizenshipStatus = document.querySelector("#citizenship-status");
const dashboardMessage = document.querySelector("#dashboard-message");
const dayCount = document.querySelector("#day-count");
const weekCount = document.querySelector("#week-count");
const monthCount = document.querySelector("#month-count");
const totalCount = document.querySelector("#total-count");
const monthLabel = document.querySelector("#month-label");
const dashboardMonth = document.querySelector("#dashboard-month");
const statusAppliedCount = document.querySelector("#status-applied-count");
const statusProgressCount = document.querySelector("#status-progress-count");
const statusRejectedCount = document.querySelector("#status-rejected-count");
const statusCancelledCount = document.querySelector("#status-cancelled-count");
const applicationSearchInput = document.querySelector("#application-search");
const applicationStatusFilter = document.querySelector("#application-status-filter");
const applicationResults = document.querySelector("#application-results");
const searchMessage = document.querySelector("#search-message");
const applicationInsights = document.querySelector("#application-insights");
const insightTitle = document.querySelector("#insight-title");
const insightCompany = document.querySelector("#insight-company");
const insightStatus = document.querySelector("#insight-status");
const insightDate = document.querySelector("#insight-date");
const insightSource = document.querySelector("#insight-source");
const insightResume = document.querySelector("#insight-resume");
const insightSponsorship = document.querySelector("#insight-sponsorship");
const insightReferral = document.querySelector("#insight-referral");
const insightScore = document.querySelector("#insight-score");
const insightAxes = document.querySelector("#insight-axes");
const insightTerms = document.querySelector("#insight-terms");
const insightUpdated = document.querySelector("#insight-updated");
const insightLink = document.querySelector("#insight-link");
const saveInsightStatusButton = document.querySelector("#save-insight-status");
const statusMessage = document.querySelector("#status-message");
const deleteApplicationButton = document.querySelector("#delete-application");
const deleteConfirmation = document.querySelector("#delete-confirmation");
const cancelDeleteButton = document.querySelector("#cancel-delete");
const confirmDeleteButton = document.querySelector("#confirm-delete");
const insightDetailsEditor = document.querySelector("#insight-details-editor");
const insightCompanyInput = document.querySelector("#insight-company-input");
const insightTitleInput = document.querySelector("#insight-title-input");
const insightSourceInput = document.querySelector("#insight-source-input");
const insightDateInput = document.querySelector("#insight-date-input");
const insightUrlInput = document.querySelector("#insight-url-input");
const insightSponsorshipInput = document.querySelector("#insight-sponsorship-input");
const insightReferralInput = document.querySelector("#insight-referral-input");
const insightDescriptionInput = document.querySelector("#insight-description-input");
const saveInsightDetailsButton = document.querySelector("#save-insight-details");
const insightBreakdown = document.querySelector("#insight-breakdown");
const scoreStatus = document.querySelector("#score-status");
const refreshScoreButton = document.querySelector("#refresh-score");
const scoreValue = document.querySelector("#score-value");
const scoreSpinner = document.querySelector("#score-spinner");
const scoreControl = document.querySelector("#score-control");
const scorePopover = document.querySelector("#score-popover");
const scoreAxes = document.querySelector("#score-axes");
const profileUrls = { linkedinUrl: "", websiteUrl: "", githubUrl: "" };
let messageTimer;
let settingsMessageTimer;
let scoreRequest;
let scoreGeneration = 0;
let latestScoreBreakdown;
let capturedMetadata = {};
let descriptionEdited = false;
let applicationCatalog = [];
let selectedInsights;
let statusMessageTimer;
let duplicateCheckTimer;
let duplicateCheckGeneration = 0;
let searchMessageTimer;
let captureStateTimer;
const logStateTimers = new Map();

const actionIcons = {
  idle: "icons/refresh-cw.svg",
  loading: "icons/refresh-cw.svg",
  success: "icons/check.svg",
  error: "icons/circle-alert.svg"
};

function setCaptureState(state, detail = "") {
  clearTimeout(captureStateTimer);
  refreshContextIcon.src = actionIcons[state];
  refreshContextIcon.classList.toggle("action-state-loading", state === "loading");

  const labels = {
    idle: "Capture current page",
    loading: "Capturing current page",
    success: "Current page captured",
    error: detail || "Current page could not be captured"
  };
  refreshContextButton.title = labels[state];
  refreshContextButton.setAttribute("aria-label", labels[state]);

  if (state === "success" || state === "error") {
    captureStateTimer = setTimeout(() => setCaptureState("idle"), state === "success" ? 2000 : 3000);
  }
}

function setLogActionState(button, state, detail = "") {
  clearTimeout(logStateTimers.get(button));
  const icon = button.querySelector(".log-state-icon");
  const label = button.querySelector("span").textContent;
  icon.hidden = state === "idle";
  icon.src = actionIcons[state];
  icon.classList.toggle("action-state-loading", state === "loading");

  const accessibleLabel = state === "error" && detail
    ? `${label}: ${detail}`
    : `${label}: ${state}`;
  button.title = state === "error" ? detail : "";
  button.setAttribute("aria-label", state === "idle" ? label : accessibleLabel);

  if (state === "success" || state === "error") {
    const timer = setTimeout(() => {
      logStateTimers.delete(button);
      setLogActionState(button, "idle");
    }, state === "success" ? 2000 : 3000);
    logStateTimers.set(button, timer);
  }
}

chrome.storage.local.get(["linkedinUrl", "websiteUrl", "githubUrl"]).then((saved) => {
  profileUrls.linkedinUrl = saved.linkedinUrl || "";
  profileUrls.websiteUrl = saved.websiteUrl || "";
  profileUrls.githubUrl = saved.githubUrl || "";
  copyLinkedInButton.disabled = false;
  copyWebsiteButton.disabled = false;
  copyGitHubButton.disabled = false;
}).catch((error) => {
  showMessage(error.message || "Saved profile links could not be loaded.", "error", 3000);
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local") return;
  for (const key of Object.keys(profileUrls)) {
    if (changes[key]) profileUrls[key] = changes[key].newValue || "";
  }
});

function reportPanelHeight() {
  // Every tab shares the fixed compact overlay dimensions.
}

function showSettingsMessage(text, type = "", duration = 0) {
  clearTimeout(settingsMessageTimer);
  settingsMessage.textContent = text;
  settingsMessage.className = type;
  if (duration) {
    settingsMessageTimer = setTimeout(() => {
      settingsMessage.textContent = "";
      settingsMessage.className = "";
      reportPanelHeight();
    }, duration);
  }
  reportPanelHeight();
}

function setActiveTab(activeButton, activeView) {
  closeResumeMenu();
  for (const view of [logView, settingsView, dashboardView, searchView, utilitiesView]) {
    view.hidden = view !== activeView;
  }
  for (const button of [applicationsTab, utilitiesButton, searchButton, dashboardButton, settingsButton]) {
    button.setAttribute("aria-selected", String(button === activeButton));
  }
  refreshContextButton.hidden = activeView !== logView;
  reportPanelHeight();
}

async function showSettings() {
  const saved = await chrome.storage.local.get(["linkedinUrl", "websiteUrl", "githubUrl"]);
  linkedinUrlInput.value = saved.linkedinUrl || "";
  websiteUrlInput.value = saved.websiteUrl || "";
  githubUrlInput.value = saved.githubUrl || "";
  setActiveTab(settingsButton, settingsView);
}

applicationsTab.addEventListener("click", () => setActiveTab(applicationsTab, logView));
utilitiesButton.addEventListener("click", () => setActiveTab(utilitiesButton, utilitiesView));

settingsButton.addEventListener("click", () => {
  showSettings().catch((error) => {
    setActiveTab(applicationsTab, logView);
    showMessage(error.message || "Settings could not be opened.", "error", 3000);
  });
});

async function loadDashboard() {
  const refreshButton = document.querySelector("#refresh-dashboard");
  refreshButton.disabled = true;
  refreshButton.setAttribute("aria-busy", "true");
  dashboardMessage.className = "";
  dashboardMessage.textContent = "Loading application counts...";
  try {
    const selectedMonth = dashboardMonth.value || today().slice(0, 7);
    const params = new URLSearchParams({ date: today(), month: selectedMonth });
    const response = await fetch(`${service}/api/dashboard?${params}`, {
      headers: requestHeaders
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Dashboard could not be loaded");
    totalCount.textContent = data.counts.applications_so_far;
    dayCount.textContent = data.counts.current_day;
    weekCount.textContent = data.counts.current_week;
    monthCount.textContent = data.counts.selected_month;
    monthLabel.textContent = data.month_label;
    const statuses = data.counts.statuses || {};
    statusAppliedCount.textContent = statuses.applied ?? "-";
    statusProgressCount.textContent = statuses.in_progress ?? "-";
    statusRejectedCount.textContent = statuses.rejected ?? "-";
    statusCancelledCount.textContent = statuses.cancelled ?? "-";
    dashboardMessage.textContent = "";
  } catch (error) {
    dashboardMessage.className = "dashboard-error";
    dashboardMessage.textContent = error.message || "Dashboard could not be loaded.";
  } finally {
    refreshButton.disabled = false;
    refreshButton.removeAttribute("aria-busy");
  }
  reportPanelHeight();
}

dashboardButton.addEventListener("click", () => {
  setActiveTab(dashboardButton, dashboardView);
  if (!dashboardMonth.value) dashboardMonth.value = today().slice(0, 7);
  loadDashboard();
});

document.querySelector("#refresh-dashboard").addEventListener("click", loadDashboard);
dashboardMonth.addEventListener("change", loadDashboard);

function statusClass(status) {
  return {
    "In Progress": "status-in-progress",
    Rejected: "status-rejected",
    Cancelled: "status-cancelled"
  }[status] || "";
}

function styleInsightStatus(status) {
  insightStatus.className = `status-select ${statusClass(status)}`.trim();
}

function showStatusMessage(text, className = "") {
  clearTimeout(statusMessageTimer);
  statusMessage.textContent = text;
  statusMessage.className = `status-message${className ? ` ${className}` : ""}`;
  if (text) {
    statusMessageTimer = setTimeout(() => {
      statusMessage.textContent = "";
      statusMessage.className = "status-message";
      reportPanelHeight();
    }, 3000);
  }
  reportPanelHeight();
}

function setCitizenshipStatus(target, text) {
  const signal = citizenshipSignal(text);
  target.textContent = signal.text;
  target.className = `citizenship-status ${signal.className}`;
}

function formatDisplayDate(value) {
  if (!value) return "Not recorded";
  const date = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function closeApplicationResults() {
  applicationResults.hidden = true;
  applicationSearchInput.setAttribute("aria-expanded", "false");
}

function renderApplicationResults() {
  const resultLimit = applicationStatusFilter.value ? applicationCatalog.length : 8;
  const matches = matchingApplications(applicationCatalog, applicationSearchInput.value, resultLimit, applicationStatusFilter.value);
  applicationResults.replaceChildren();
  for (const application of matches) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "search-result";
    option.setAttribute("role", "option");
    const title = document.createElement("strong");
    title.textContent = application.job_title || "Untitled role";
    const company = document.createElement("span");
    company.className = "search-result-company";
    company.textContent = `${application.company || "Unknown company"} | ${formatDisplayDate(application.application_date)}`;
    const status = document.createElement("span");
    status.className = `status-badge search-result-status ${statusClass(application.status)}`.trim();
    status.textContent = application.status || "Unknown";
    option.append(title, company, status);
    option.addEventListener("click", () => {
      applicationSearchInput.value = `${application.job_title} - ${application.company}`;
      closeApplicationResults();
      loadApplicationInsights(application.application_key);
    });
    option.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
      event.preventDefault();
      const options = [...applicationResults.querySelectorAll("button")];
      const index = options.indexOf(option);
      if (event.key === "ArrowDown") (options[index + 1] || options[0])?.focus();
      else if (index > 0) options[index - 1].focus();
      else applicationSearchInput.focus();
    });
    applicationResults.append(option);
  }
  const hasSearchFocus = document.activeElement === applicationSearchInput || document.activeElement === applicationStatusFilter;
  const shouldOpen = hasSearchFocus && matches.length > 0;
  applicationResults.hidden = !shouldOpen;
  applicationSearchInput.setAttribute("aria-expanded", String(shouldOpen));
  if (!applicationCatalog.length) searchMessage.textContent = "No applications found.";
  else if (!matches.length) searchMessage.textContent = "No matching applications.";
  else searchMessage.textContent = `${matches.length} match${matches.length === 1 ? "" : "es"}`;
  reportPanelHeight();
}

async function loadApplicationCatalog() {
  searchMessage.className = "search-message";
  searchMessage.textContent = "Loading applications...";
  applicationInsights.hidden = true;
  closeApplicationResults();
  try {
    const response = await fetch(`${service}/api/applications/search`, { headers: requestHeaders });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Applications could not be loaded");
    applicationCatalog = data.applications || [];
    renderApplicationResults();
  } catch (error) {
    applicationCatalog = [];
    searchMessage.className = "search-message search-error";
    searchMessage.textContent = error.message || "Applications could not be loaded.";
  }
  reportPanelHeight();
}

function renderInsightBreakdown(categories) {
  const rows = scoreBreakdownPresentation(categories);
  insightAxes.replaceChildren();
  for (const row of rows) {
    const container = document.createElement("div");
    container.className = `score-axis ${row.className}`;
    const label = document.createElement("dt");
    label.textContent = row.label;
    const value = document.createElement("dd");
    value.textContent = `${row.value}/100`;
    container.append(label, value);
    insightAxes.append(container);
  }
}

function closeInsightBreakdown() {
  insightBreakdown.hidden = true;
  insightScore.setAttribute("aria-expanded", "false");
}

function closeDeleteConfirmation() {
  deleteConfirmation.hidden = true;
}

function renderApplicationInsights(data) {
  const application = data.application;
  const score = data.logged_score;
  selectedInsights = data;
  closeInsightBreakdown();
  closeDeleteConfirmation();
  insightTitle.textContent = application.job_title || "Untitled role";
  insightCompany.textContent = application.company || "Unknown company";
  insightStatus.value = application.status || "Applied";
  styleInsightStatus(insightStatus.value);
  insightDate.textContent = formatDisplayDate(application.application_date);
  insightSource.textContent = application.job_source || "Not recorded";
  insightResume.textContent = application.resume_version || "Not recorded";
  insightResume.title = application.resume_version || "";
  insightSponsorship.textContent = application.requested_for_sponsorship ? "Yes" : "No";
  insightReferral.textContent = application.referral ? "Yes" : "No";
  insightLink.href = application.job_url || "";
  insightLink.hidden = !application.job_url;
  insightCompanyInput.value = application.company || "";
  insightTitleInput.value = application.job_title || "";
  insightSourceInput.value = application.job_source || "";
  insightDateInput.value = application.application_date || "";
  insightUrlInput.value = application.job_url || "";
  insightSponsorshipInput.checked = application.requested_for_sponsorship === true;
  insightReferralInput.checked = application.referral === true;
  insightDescriptionInput.value = application.job_description || "";
  insightTerms.replaceChildren();
  insightAxes.replaceChildren();
  insightScore.className = "insight-score-value";
  if (score) {
    const presentation = scorePresentation(score.score, score.band, score.fit_label);
    insightScore.textContent = presentation.text;
    insightScore.classList.add(presentation.className);
    renderInsightBreakdown(score.category_breakdown);
    insightUpdated.textContent = score.updated_at ? `Score updated ${formatDisplayDate(score.updated_at)}` : "";
    if (score.matched_terms?.length) {
      const label = document.createElement("strong");
      label.textContent = "Matched terms: ";
      insightTerms.append(label, document.createTextNode(score.matched_terms.join(", ")));
    }
  } else {
    insightScore.textContent = "Pending";
    insightUpdated.textContent = "";
  }
  insightTerms.hidden = !score?.matched_terms?.length;
  const hasBreakdown = insightAxes.childElementCount > 0 || !insightTerms.hidden;
  insightScore.disabled = !hasBreakdown;
  insightScore.title = hasBreakdown ? "Show logged score breakdown" : "";
  applicationInsights.hidden = false;
  searchMessage.textContent = "";
  reportPanelHeight();
}

insightStatus.addEventListener("change", () => styleInsightStatus(insightStatus.value));
insightDetailsEditor.addEventListener("toggle", reportPanelHeight);

insightScore.addEventListener("click", () => {
  if (insightScore.disabled) return;
  const opening = insightBreakdown.hidden;
  insightBreakdown.hidden = !opening;
  insightScore.setAttribute("aria-expanded", String(opening));
});

deleteApplicationButton.addEventListener("click", () => {
  deleteConfirmation.hidden = false;
  confirmDeleteButton.focus();
});

cancelDeleteButton.addEventListener("click", () => {
  closeDeleteConfirmation();
  deleteApplicationButton.focus();
});

saveInsightStatusButton.addEventListener("click", async () => {
  const applicationKey = selectedInsights?.application?.application_key;
  if (!applicationKey) return;
  saveInsightStatusButton.disabled = true;
  saveInsightStatusButton.textContent = "Saving";
  showStatusMessage("Saving status...");
  try {
    const response = await fetch(
      `${service}/api/applications/${encodeURIComponent(applicationKey)}/status`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify({ status: insightStatus.value })
      }
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Status could not be updated");
    selectedInsights.application.status = data.status;
    selectedInsights.application.updated_at = data.updated_at;
    const catalogEntry = applicationCatalog.find((item) => item.application_key === applicationKey);
    if (catalogEntry) catalogEntry.status = data.status;
    renderApplicationInsights(selectedInsights);
    showStatusMessage("Status updated.", "success");
  } catch (error) {
    showStatusMessage(error.message || "Status could not be updated.", "error");
  } finally {
    saveInsightStatusButton.disabled = false;
    saveInsightStatusButton.textContent = "Save";
  }
});

saveInsightDetailsButton.addEventListener("click", async () => {
  const applicationKey = selectedInsights?.application?.application_key;
  if (!applicationKey) return;
  saveInsightDetailsButton.disabled = true;
  saveInsightDetailsButton.textContent = "Saving";
  showStatusMessage("Saving details...");
  try {
    const payload = {
      company: insightCompanyInput.value.trim(),
      job_title: insightTitleInput.value.trim(),
      job_source: insightSourceInput.value.trim(),
      job_url: insightUrlInput.value.trim(),
      application_date: insightDateInput.value,
      status: insightStatus.value,
      requested_for_sponsorship: insightSponsorshipInput.checked,
      referral: insightReferralInput.checked,
      job_description: insightDescriptionInput.value.trim()
    };
    const response = await fetch(
      `${service}/api/applications/${encodeURIComponent(applicationKey)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify(payload)
      }
    );
    const data = await readServiceResponse(response, "Application details could not be updated");
    if (!response.ok) throw new Error(data.error || "Application details could not be updated");
    selectedInsights.application = data.application;
    const catalogEntry = applicationCatalog.find((item) => item.application_key === applicationKey);
    if (catalogEntry) {
      Object.assign(catalogEntry, data.application);
    }
    renderApplicationInsights(selectedInsights);
    showStatusMessage("Details updated.", "success");
  } catch (error) {
    showStatusMessage(error.message || "Application details could not be updated.", "error");
  } finally {
    saveInsightDetailsButton.disabled = false;
    saveInsightDetailsButton.textContent = "Save details";
  }
});

confirmDeleteButton.addEventListener("click", async () => {
  const applicationKey = selectedInsights?.application?.application_key;
  if (!applicationKey) return;
  confirmDeleteButton.disabled = true;
  cancelDeleteButton.disabled = true;
  confirmDeleteButton.textContent = "Deleting";
  try {
    const response = await fetch(
      `${service}/api/applications/${encodeURIComponent(applicationKey)}`,
      { method: "DELETE", headers: requestHeaders }
    );
    const data = await readServiceResponse(response, "Application could not be deleted");
    if (!response.ok) throw new Error(data.error || "Application could not be deleted");
    applicationCatalog = applicationCatalog.filter(
      (application) => application.application_key !== applicationKey
    );
    selectedInsights = undefined;
    applicationInsights.hidden = true;
    applicationSearchInput.value = "";
    closeApplicationResults();
    clearTimeout(searchMessageTimer);
    searchMessage.className = "search-message success";
    searchMessage.textContent = "Application deleted.";
    searchMessageTimer = setTimeout(() => {
      searchMessage.className = "search-message";
      searchMessage.textContent = applicationCatalog.length ? "Start typing..." : "No applications found.";
    }, 3000);
  } catch (error) {
    closeDeleteConfirmation();
    showStatusMessage(error.message || "Application could not be deleted.", "error");
  } finally {
    confirmDeleteButton.disabled = false;
    cancelDeleteButton.disabled = false;
    confirmDeleteButton.textContent = "Delete";
  }
});

async function readServiceResponse(response, fallbackMessage) {
  const body = await response.text();
  if (!body) return {};
  try {
    return JSON.parse(body);
  } catch {
    if (response.status === 405 || response.status === 501) {
      throw new Error("Delete is unavailable in the running service. Restart or redeploy it.");
    }
    if (!response.ok) throw new Error(`${fallbackMessage} (HTTP ${response.status})`);
    throw new Error("The service returned an invalid response.");
  }
}

document.addEventListener("click", (event) => {
  if (!event.target.closest(".insight-score")) closeInsightBreakdown();
  if (!event.target.closest(".insight-status")) closeDeleteConfirmation();
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  closeInsightBreakdown();
  closeDeleteConfirmation();
});

async function loadApplicationInsights(applicationKey) {
  searchMessage.className = "search-message";
  searchMessage.textContent = "Loading insights...";
  applicationInsights.hidden = true;
  try {
    const response = await fetch(
      `${service}/api/applications/insights/${encodeURIComponent(applicationKey)}`,
      { headers: requestHeaders }
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Application insights could not be loaded");
    renderApplicationInsights(data);
  } catch (error) {
    searchMessage.className = "search-message search-error";
    searchMessage.textContent = error.message || "Application insights could not be loaded.";
  }
  reportPanelHeight();
}

searchButton.addEventListener("click", () => {
  setActiveTab(searchButton, searchView);
  applicationSearchInput.value = "";
  applicationStatusFilter.value = "";
  loadApplicationCatalog().then(() => applicationSearchInput.focus());
});

applicationSearchInput.addEventListener("input", renderApplicationResults);
applicationStatusFilter.addEventListener("change", renderApplicationResults);
applicationSearchInput.addEventListener("focus", renderApplicationResults);
applicationSearchInput.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowDown") return;
  const firstResult = applicationResults.querySelector("button");
  if (!firstResult) return;
  event.preventDefault();
  firstResult.focus();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".application-search-controls")) closeApplicationResults();
});

document.querySelector("#settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const linkedinUrl = normalizeHttpUrl(linkedinUrlInput.value);
    const websiteUrl = normalizeHttpUrl(websiteUrlInput.value);
    const githubUrl = normalizeHttpUrl(githubUrlInput.value);
    if (!isLinkedInUrl(linkedinUrl)) throw new Error("LinkedIn URL must use linkedin.com");
    if (!isGitHubUrl(githubUrl)) throw new Error("GitHub URL must use github.com");
    await chrome.storage.local.set({ linkedinUrl, websiteUrl, githubUrl });
    profileUrls.linkedinUrl = linkedinUrl;
    profileUrls.websiteUrl = websiteUrl;
    profileUrls.githubUrl = githubUrl;
    linkedinUrlInput.value = linkedinUrl;
    websiteUrlInput.value = websiteUrl;
    githubUrlInput.value = githubUrl;
    showSettingsMessage("Settings saved.", "success", 3000);
  } catch (error) {
    showSettingsMessage(error.message || "Settings could not be saved.", "error", 3000);
  }
});

function today() {
  const date = new Date();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

function showMessage(text, type = "", duration = 0) {
  clearTimeout(messageTimer);
  messageText.textContent = text;
  message.className = type;
  message.hidden = !text;
  if (duration) {
    messageTimer = setTimeout(() => {
      messageText.textContent = "";
      message.className = "";
      message.hidden = true;
      reportPanelHeight();
    }, duration);
  }
  reportPanelHeight();
}

function setScoreStatus(text, className = "", loading = false) {
  scoreValue.textContent = text;
  scoreStatus.className = `score-status${className ? ` ${className}` : ""}`;
  scoreSpinner.hidden = !loading;
  reportPanelHeight();
}

function closeScoreBreakdown() {
  scorePopover.hidden = true;
  scoreStatus.setAttribute("aria-expanded", "false");
}

function clearScoreBreakdown() {
  latestScoreBreakdown = undefined;
  scoreAxes.replaceChildren();
  closeScoreBreakdown();
}

function renderScoreBreakdown(result) {
  const rows = classifierBreakdownPresentation(result);
  scoreAxes.replaceChildren();
  for (const row of rows) {
    const container = document.createElement("div");
    container.className = "score-axis";
    const label = document.createElement("dt");
    label.textContent = row.label;
    const value = document.createElement("dd");
    value.textContent = row.value;
    container.append(label, value);
    scoreAxes.append(container);
  }
}

scoreStatus.addEventListener("click", async () => {
  if (!latestScoreBreakdown) {
    await calculateScore(false);
    return;
  }
  const opening = scorePopover.hidden;
  scorePopover.hidden = !opening;
  scoreStatus.setAttribute("aria-expanded", String(opening));
});

refreshScoreButton.addEventListener("click", async () => {
  await calculateScore(true);
});

document.addEventListener("click", (event) => {
  if (!scoreControl.contains(event.target)) closeScoreBreakdown();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeScoreBreakdown();
});

function resetScore() {
  scoreRequest?.abort();
  scoreGeneration += 1;
  scoreStatus.disabled = false;
  refreshScoreButton.disabled = false;
  clearScoreBreakdown();
  setScoreStatus("View score");
}

async function calculateScore(forceRecompute = false) {
  const generation = ++scoreGeneration;
  const controller = new AbortController();
  scoreRequest = controller;
  scoreStatus.disabled = true;
  refreshScoreButton.disabled = true;
  setScoreStatus("Calculating...", "", true);
  try {
    const payload = await applicationPayload();
    payload.force_recompute = forceRecompute;
    if (generation !== scoreGeneration) return;
    if (!payload.job_description || payload.extraction_quality !== "complete") {
      clearScoreBreakdown();
      setScoreStatus("Score unavailable");
      return;
    }
    const response = await fetch(`${service}/api/applications/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...requestHeaders },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    const data = await response.json();
    if (generation !== scoreGeneration) return;
    if (!response.ok || data.state !== "completed") {
      throw new Error(data.error || "Score unavailable");
    }
    const presentation = scorePresentation(data.score, data.band, data.fit_label);
    const source = scoreSourceLabel(data.score_source);
    latestScoreBreakdown = data;
    renderScoreBreakdown(latestScoreBreakdown);
    setScoreStatus(`${presentation.text}\n${source}`, presentation.className);
  } catch (error) {
    if (error.name !== "AbortError" && generation === scoreGeneration) {
      clearScoreBreakdown();
      setScoreStatus("Score unavailable");
    }
  } finally {
    if (generation === scoreGeneration) {
      scoreStatus.disabled = false;
      refreshScoreButton.disabled = false;
    }
    if (scoreRequest === controller) scoreRequest = undefined;
  }
}

async function savedResumeVersion() {
  if (chrome.storage?.local) {
    const saved = await chrome.storage.local.get("resumeVersion");
    return saved.resumeVersion;
  }
  return localStorage.getItem("resumeVersion");
}

function rememberResumeVersion(version) {
  if (chrome.storage?.local) chrome.storage.local.set({ resumeVersion: version });
  else localStorage.setItem("resumeVersion", version);
}

function closeResumeMenu() {
  resumeMenu.hidden = true;
  resumePickerButton.setAttribute("aria-expanded", "false");
}

function renderResumePicker() {
  const selected = resumeSelect.selectedOptions[0];
  resumePickerLabel.textContent = selected?.textContent || "No resumes available";
  resumePickerButton.title = selected?.textContent || "";
  resumePickerButton.disabled = !selected;
  resumeMenu.replaceChildren();
  for (const option of resumeSelect.options) {
    const item = document.createElement("button");
    item.className = "resume-option";
    item.type = "button";
    item.setAttribute("role", "option");
    item.textContent = option.textContent;
    item.title = option.textContent;
    item.setAttribute("aria-selected", String(option.value === resumeSelect.value));
    item.addEventListener("click", () => {
      resumeSelect.value = option.value;
      resumeSelect.dispatchEvent(new Event("change"));
      closeResumeMenu();
      resumePickerButton.focus();
    });
    resumeMenu.append(item);
  }
}

resumePickerButton.addEventListener("click", () => {
  const opening = resumeMenu.hidden;
  resumeMenu.hidden = !opening;
  resumePickerButton.setAttribute("aria-expanded", String(opening));
  if (opening) {
    resumeMenu.querySelector('[aria-selected="true"]')?.focus();
  }
});

document.addEventListener("click", (event) => {
  if (!resumePicker.contains(event.target)) closeResumeMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !resumeMenu.hidden) {
    closeResumeMenu();
    resumePickerButton.focus();
  }
});

function legacyCopyText(text) {
  const input = document.createElement("textarea");
  input.value = text;
  input.setAttribute("readonly", "");
  input.style.position = "fixed";
  input.style.opacity = "0";
  document.body.append(input);
  input.focus();
  input.select();
  const copied = document.execCommand("copy");
  input.remove();
  return copied;
}

async function writeClipboard(text) {
  const modernAttempt = navigator.clipboard?.writeText
    ? navigator.clipboard.writeText(text).then(() => true).catch(() => false)
    : Promise.resolve(false);
  if (legacyCopyText(text)) return;
  if (await modernAttempt) return;
  throw new Error("Clipboard access was denied by the browser.");
}

const copyButtonTimers = new Map();

function showCopied(button) {
  clearTimeout(copyButtonTimers.get(button));
  button.querySelector(".copy-icon").hidden = true;
  button.querySelector(".check-icon").hidden = false;
  copyButtonTimers.set(button, setTimeout(() => {
    button.querySelector(".copy-icon").hidden = false;
    button.querySelector(".check-icon").hidden = true;
  }, 2000));
}

async function copyValue(value, label, button) {
  if (!value) {
    showMessage(`Set your ${label} in settings first.`, "error", 3000);
    return;
  }
  button.setAttribute("aria-busy", "true");
  try {
    await writeClipboard(value);
    showCopied(button);
  } catch (error) {
    showMessage(error.message || `Could not copy ${label}.`, "error", 3000);
  } finally {
    button.removeAttribute("aria-busy");
  }
}

function copySavedUrl(storageKey, label, button) {
  copyValue(profileUrls[storageKey], `${label} URL`, button);
}


function createApplicationPassword() {
  generatedPasswordInput.value = generatePassword(16);
  copyPasswordButton.disabled = false;
}

generatePasswordButton.addEventListener("click", () => {
  try {
    createApplicationPassword();
  } catch (error) {
    showMessage(error.message || "Could not generate a password.", "error", 3000);
  }
});

copyPasswordButton.addEventListener("click", () => {
  const password = generatedPasswordInput.value;
  if (!password) {
    showMessage("Generate a password first.", "error", 3000);
    return;
  }
  copyValue(password, "password", copyPasswordButton);
});

document.querySelector("#copy-linkedin").addEventListener("click", () => {
  copySavedUrl("linkedinUrl", "LinkedIn", copyLinkedInButton);
});

document.querySelector("#copy-website").addEventListener("click", () => {
  copySavedUrl("websiteUrl", "Website", copyWebsiteButton);
});

document.querySelector("#copy-github").addEventListener("click", () => {
  copySavedUrl("githubUrl", "GitHub", copyGitHubButton);
});

async function load() {
  closeResumeMenu();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const metadata = await getMetadata(tab.id);
  capturedMetadata = { ...metadata };
  companyInput.value = metadata.company;
  titleInput.value = metadata.job_title;
  sourceInput.value = metadata.job_source;
  descriptionEdited = false;
  descriptionInput.value = metadata.job_description || "";
  setCitizenshipStatus(citizenshipStatus, descriptionInput.value);
  dateInput.value = today();
  const response = await fetch(`${service}/api/resumes`, { headers: requestHeaders });
  const data = await response.json();
  resumeSelect.replaceChildren();
  for (const resume of data.resumes) {
    const option = document.createElement("option");
    option.value = resume.version;
    option.textContent = resume.label;
    resumeSelect.append(option);
  }
  const saved = await savedResumeVersion();
  const savedIndex = [...resumeSelect.options].findIndex((option) => option.value === saved);
  resumeSelect.selectedIndex = savedIndex >= 0 ? savedIndex : 0;
  renderResumePicker();
  resetScore();
  scheduleDuplicateCheck();
}

async function getMetadata(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "job-metadata" });
  } catch (error) {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["metadata.js", "content.js"] });
    return chrome.tabs.sendMessage(tabId, { type: "job-metadata" });
  }
}

chrome.runtime.onMessage.addListener((message) => {
  if (["active-tab-changed", "job-metadata-changed"].includes(message.type)) {
    load().catch((error) => setCaptureState("error", error.message || "Could not capture the current page."));
  }
});

refreshContextButton.addEventListener("click", async () => {
  refreshContextButton.disabled = true;
  refreshContextButton.setAttribute("aria-busy", "true");
  setCaptureState("loading");
  try {
    await load();
    setCaptureState("success");
  } catch (error) {
    setCaptureState("error", error.message || "Could not capture the current page.");
  } finally {
    refreshContextButton.disabled = false;
    refreshContextButton.removeAttribute("aria-busy");
  }
});

resumeSelect.addEventListener("change", () => {
  rememberResumeVersion(resumeSelect.value);
  renderResumePicker();
  resetScore();
  scheduleDuplicateCheck();
});

applicationFields.addEventListener("toggle", reportPanelHeight);
descriptionEditor.addEventListener("toggle", reportPanelHeight);
descriptionInput.addEventListener("input", () => {
  descriptionEdited = true;
  resetScore();
  setCitizenshipStatus(citizenshipStatus, descriptionInput.value);
  reportPanelHeight();
});

for (const input of [companyInput, titleInput, sourceInput]) {
  input.addEventListener("input", () => {
    resetScore();
    scheduleDuplicateCheck();
  });
}

function setDuplicateStatus(text, className = "") {
  duplicateStatus.textContent = text;
  duplicateStatus.className = `duplicate-status${className ? ` ${className}` : ""}`;
  reportPanelHeight();
}

function scheduleDuplicateCheck(delay = 350) {
  clearTimeout(duplicateCheckTimer);
  duplicateCheckGeneration += 1;
  const generation = duplicateCheckGeneration;
  duplicateCheckTimer = setTimeout(() => {
    checkDuplicateApplication(generation).catch((error) => {
      if (generation === duplicateCheckGeneration) {
        setDuplicateStatus(error.message || "Could not check application history", "error");
      }
    });
  }, delay);
}

async function checkDuplicateApplication(generation = duplicateCheckGeneration) {
  const hasIdentity = companyInput.value.trim() && titleInput.value.trim() && sourceInput.value.trim();
  const hasUrl = capturedMetadata.job_url || capturedMetadata.canonical_url;
  if (!hasIdentity || !hasUrl || !resumeSelect.value) {
    setDuplicateStatus("Need company + role", "incomplete");
    return;
  }
  setDuplicateStatus("Checking application history...", "checking");
  const response = await fetch(`${service}/api/applications/check`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...requestHeaders },
    body: JSON.stringify(await applicationPayload())
  });
  const data = await response.json();
  if (generation !== duplicateCheckGeneration) return;
  if (!response.ok) throw new Error(data.error || "Could not check application history");
  const status = duplicateApplicationStatus(data);
  setDuplicateStatus(status.text, status.className);
  setCitizenshipStatus(citizenshipStatus, descriptionInput.value);
}

async function applicationPayload() {
  const metadata = capturedMetadata;
  const jobDescription = descriptionInput.value.trim();
  return {
    ...metadata,
    company: companyInput.value.trim(),
    job_title: titleInput.value.trim(),
    job_source: sourceInput.value.trim(),
    application_date: dateInput.value,
    status: statusInput.value,
    requested_for_sponsorship: sponsorshipInput.checked,
    referral: referralInput.checked,
    resume_version: resumeSelect.value,
    job_description: jobDescription,
    citizenship_signal: citizenshipSignal(jobDescription).text,
    description_provenance: descriptionEdited ? "manual_supplied" : metadata.description_provenance,
    extraction_quality: jobDescription
      ? (descriptionEdited ? "complete" : metadata.extraction_quality)
      : "unavailable"
  };
}

const logWithoutScoreButton = document.querySelector("#log-without-score");
const logWithScoreButton = document.querySelector("#log-with-score");

async function logApplication(withScore, button) {
  logWithoutScoreButton.disabled = true;
  logWithScoreButton.disabled = true;
  button.setAttribute("aria-busy", "true");
  setLogActionState(button, "loading");
  try {
    const payload = await applicationPayload();
    payload.score_application = withScore;
    const response = await fetch(`${service}/api/applications`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...requestHeaders },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      const error = data.error || "Application could not be logged";
      setLogActionState(button, "error", error);
      return;
    }
    setLogActionState(button, "success");
  } catch (error) {
    setLogActionState(button, "error", error.message || "Could not connect to the local service.");
  } finally {
    logWithoutScoreButton.disabled = false;
    logWithScoreButton.disabled = false;
    button.removeAttribute("aria-busy");
  }
}

logWithoutScoreButton.addEventListener("click", () => {
  logApplication(false, logWithoutScoreButton);
});

logWithScoreButton.addEventListener("click", () => {
  logApplication(true, logWithScoreButton);
});

load().catch((error) => setCaptureState("error", error.message || "Start the local service first."));
window.addEventListener("load", reportPanelHeight);
