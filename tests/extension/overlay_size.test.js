const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "../..");
const background = fs.readFileSync(path.join(root, "extension/background.js"), "utf8");
const content = fs.readFileSync(path.join(root, "extension/content.js"), "utf8");

test("the tracker uses its compact CSS dimensions at every page zoom", () => {
  assert.doesNotMatch(background, /tabs\.getZoom|tabs\.onZoomChange/);
  assert.doesNotMatch(content, /zoom-compensation|\bzoom\s*:|scale\s*\(/);
  assert.match(content, /const TRACKER_WIDTH = 330;/);
  assert.match(content, /const TRACKER_HEIGHT = 350;/);
});

test("tab changes cannot resize the tracker", () => {
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.doesNotMatch(content, /resume-job-tracker:resize|requestedHeight/);
  assert.doesNotMatch(popup, /resume-job-tracker:resize|height-capped/);
});

test("the application message bar is hidden while idle", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /<div id="message" role="status" aria-live="polite" hidden>/);
  assert.doesNotMatch(html, /Ready to log/);
  assert.doesNotMatch(popup, /DEFAULT_MESSAGE|Ready to log/);
  assert.match(popup, /message\.hidden = true;/);
});

test("the collapse handle uses the compact chevron control", () => {
  assert.match(content, /width: 24px;\s+height: 48px;/);
  assert.match(content, /icons\/chevron-right\.svg/);
  assert.doesNotMatch(content, /collapse\.textContent = ">"/);
});

test("refresh actions use the standard shared icon", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");

  assert.match(html, /id="refresh-context"[^>]+title="Capture current page"/);
  assert.match(html, /id="refresh-context-icon" src="icons\/refresh-cw\.svg"/);
  assert.match(html, /id="refresh-score"[^>]+title="Recompute score"/);
  assert.match(html, /id="refresh-score"[\s\S]+?<img src="icons\/refresh-cw\.svg" alt="">[\s\S]+?<\/button>/);
  assert.match(html, /id="generate-password"[\s\S]+?<img src="icons\/refresh-cw\.svg" alt="">[\s\S]+?<\/button>/);
  assert.doesNotMatch(html, /<span aria-hidden="true">&#x21bb;<\/span>/);
  assert.doesNotMatch(html, /aria-label="Recompute score">&#x21bb;/);
});

test("navigation uses standard icons that inherit active tab color", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");

  assert.match(html, /\.tab-button svg \{ display: block; width: 18px; height: 18px; \}/);
  for (const id of ["applications-tab", "utilities-tab", "search", "dashboard", "settings"]) {
    const button = html.match(new RegExp(`<button class="tab-button" id="${id}"[\\s\\S]*?<\\/button>`))?.[0];
    assert.ok(button, `${id} should exist`);
    assert.match(button, /<svg[^>]+stroke="currentColor"[^>]+aria-hidden="true">/);
    assert.doesNotMatch(button, /<img /);
  }
});

test("the extension uses the neutral graphite theme", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");

  assert.match(html, /--canvas: #f8fafc;/);
  assert.match(html, /--text: #182230;/);
  assert.match(html, /--border: #d0d5dd;/);
  assert.match(html, /--subtle: #f2f4f7;/);
  assert.match(html, /--primary: #175cd3;/);
  assert.match(html, /\.surface[^}]+background: var\(--canvas\)/);
  assert.match(html, /\.metric[^}]+background: var\(--surface\)/);
});

test("the dashboard does not reserve space for a successful update message", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /#dashboard-message:empty \{ display: none; \}/);
  assert.doesNotMatch(html, /#dashboard-message \{[^}]+min-height:/);
  assert.doesNotMatch(popup, /Updated for/);
  assert.match(popup, /dashboardMessage\.textContent = "";/);
});

test("the dashboard shows a compact selected-month status breakdown", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /class="status-breakdown" aria-label="Selected month status breakdown"/);
  for (const id of ["status-applied-count", "status-progress-count", "status-rejected-count", "status-cancelled-count"]) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.match(html, /id="refresh-dashboard"[^>]+aria-label="Refresh dashboard"[\s\S]+?icons\/refresh-cw\.svg/);
  assert.match(popup, /const statuses = data\.counts\.statuses \|\| \{\};/);
  assert.match(popup, /statusAppliedCount\.textContent = statuses\.applied \?\? "-"/);
  assert.match(popup, /statusCancelledCount\.textContent = statuses\.cancelled \?\? "-"/);
});

test("the extension bundles and applies Comic Sans typography", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");

  for (const file of ["ComicNeue-Regular.otf", "ComicNeue-Bold.otf", "COMIC-NEUE-LICENSE.txt"]) {
    assert.ok(fs.existsSync(path.join(root, "extension/fonts", file)), `${file} should be bundled`);
  }
  assert.match(html, /@font-face \{ font-family: "Comic Neue";[^}]+ComicNeue-Regular\.otf/);
  assert.match(html, /font: 11px "Comic Sans MS", "Comic Neue", cursive/);
  assert.match(html, /font-family: ui-monospace, "SFMono-Regular", Consolas, monospace/);
  assert.match(html, /font-variant-numeric: tabular-nums/);
  assert.match(html, /\.search-message:empty \{ display: none; \}/);
});

test("application insights use the compact non-duplicative layout", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /class="visually-hidden" for="application-search"/);
  assert.match(html, /class="insight-item insight-resume"/);
  assert.match(html, /icons\/external-link\.svg/);
  assert.doesNotMatch(html, /class="insights-footer"/);
  assert.doesNotMatch(popup, /`Application \$\{formatDisplayDate\(application\.updated_at\)\}`/);
});

test("home actions use transient local icon feedback", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /id="refresh-context-icon" src="icons\/refresh-cw\.svg"/);
  assert.equal((html.match(/class="log-state-icon" src="icons\/refresh-cw\.svg" alt="" hidden/g) || []).length, 2);
  assert.match(html, /\.action-state-loading \{ animation: spin 700ms linear infinite; \}/);
  assert.match(html, /\.log-state-icon \{[^}]+filter: brightness\(0\) invert\(1\);/);
  assert.ok(fs.existsSync(path.join(root, "extension/icons/circle-alert.svg")));
  assert.match(popup, /setCaptureState\("loading"\)/);
  assert.match(popup, /setCaptureState\("success"\)/);
  assert.match(popup, /setCaptureState\("error", error\.message/);
  assert.match(popup, /setLogActionState\(button, "loading"\)/);
  assert.match(popup, /setLogActionState\(button, "success"\)/);
  assert.match(popup, /setLogActionState\(button, "error", error/);
  assert.doesNotMatch(popup, /showMessage\("(?:Capturing current page|Current page captured|Logging application|Application logged)/);
});

test("search details keep score expansion and deletion on demand", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.match(html, /id="insight-breakdown" role="dialog"/);
  assert.match(html, /id="delete-application"[^>]+aria-label="Delete application"/);
  assert.match(html, /id="delete-confirmation" role="dialog"/);
  assert.match(popup, /method: "DELETE"/);
  assert.match(popup, /readServiceResponse\(response, "Application could not be deleted"\)/);
  assert.match(popup, /response\.status === 405 \|\| response\.status === 501/);
});

test("the search clear control uses a pointer cursor", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");

  assert.match(html, /input::\-webkit-search-cancel-button \{ cursor: pointer; \}/);
});

test("copy utilities use stable icon buttons with check feedback", () => {
  const html = fs.readFileSync(path.join(root, "extension/popup.html"), "utf8");
  const popup = fs.readFileSync(path.join(root, "extension/popup.js"), "utf8");

  assert.equal((html.match(/class="copy-icon" src="icons\/copy\.svg"/g) || []).length, 4);
  assert.equal((html.match(/class="check-icon" src="icons\/check\.svg"[^>]+hidden/g) || []).length, 4);
  assert.match(html, /class="utility-copy-row password-row"/);
  assert.match(html, /id="generate-password"[^>]+title="Generate password"[^>]+aria-label="Generate password"/);
  assert.match(html, /id="generate-password"[\s\S]+?<img src="icons\/refresh-cw\.svg" alt="">[\s\S]+?<\/button>/);
  assert.doesNotMatch(html, /<button id="generate-password" type="button">Generate password<\/button>/);
  assert.match(popup, /function showCopied\(button\)/);
  assert.match(popup, /button\.querySelector\("\.copy-icon"\)\.hidden = true/);
  assert.match(popup, /button\.querySelector\("\.check-icon"\)\.hidden = false/);
  assert.doesNotMatch(popup, /setUtilityButtonState|utility-action-success/);
  assert.doesNotMatch(popup, /Password generated\./);
  assert.match(popup, /Could not generate a password\./);
});
