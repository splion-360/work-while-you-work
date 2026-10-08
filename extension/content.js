const OVERLAY_ID = "resume-job-tracker-overlay-host";
const TRACKER_WIDTH = 330;
const TRACKER_HEIGHT = 350;

function createOverlay() {
  const host = document.createElement("div");
  host.id = OVERLAY_ID;
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; }
    .tracker {
      position: fixed;
      z-index: 2147483647;
      top: var(--tracker-top, 12px);
      right: 16px;
      width: min(${TRACKER_WIDTH}px, calc(100vw - 32px));
      --tracker-height: min(${TRACKER_HEIGHT}px, calc(100vh - 24px));
      --tracker-top: 12px;
      height: var(--tracker-height);
      min-height: 0;
      pointer-events: none;
    }
    .scale {
      position: relative;
      width: 100%;
      height: 100%;
      pointer-events: auto;
    }
    iframe {
      width: 100%;
      height: 100%;
      border: 0;
      border-radius: 18px;
      background: #f8fafc;
      box-shadow: 0 14px 34px rgb(0 0 0 / 24%);
    }
    .collapse {
      position: absolute;
      z-index: 1;
      top: calc((100% - 48px) / 2);
      left: -24px;
      width: 24px;
      height: 48px;
      padding: 0;
      border: 1px solid #d0d5dd;
      border-right: 0;
      border-radius: 8px 0 0 8px;
      color: #475467;
      background: #ffffff;
      cursor: pointer;
      box-shadow: -5px 7px 18px rgb(0 0 0 / 12%);
    }
    .collapse img { display: block; width: 16px; height: 16px; margin: auto; }
    .collapse:hover { color: #175cd3; background: #f2f4f7; }
    .tab {
      display: none;
      position: absolute;
      top: calc((100% - 76px) / 2);
      right: -16px;
      width: 50px;
      height: 76px;
      padding: 9px;
      border: 1px solid #d0d5dd;
      border-right: 0;
      border-radius: 12px 0 0 12px;
      background: #ffffff;
      cursor: pointer;
      box-shadow: 0 8px 24px rgb(0 0 0 / 18%);
    }
    .tab img { display: block; width: 30px; height: 30px; }
    .tracker.collapsed { width: 34px; height: 76px; min-height: 0; }
    .tracker.collapsed iframe, .tracker.collapsed .collapse { display: none; }
    .tracker.collapsed .tab { display: flex; align-items: center; justify-content: center; }
  `;
  const tracker = document.createElement("div");
  tracker.className = "tracker";
  const scale = document.createElement("div");
  scale.className = "scale";
  const frame = document.createElement("iframe");
  frame.src = chrome.runtime.getURL("popup.html");
  frame.title = "Resume Job Tracker";
  frame.allow = "clipboard-write";
  const centeredTop = (height) => Math.max(12, Math.round((window.innerHeight - height) / 2));
  const resizeCollapsedTracker = () => {
    tracker.style.setProperty("--tracker-top", `${centeredTop(76)}px`);
  };
  const resizeTracker = () => {
    if (tracker.classList.contains("collapsed")) {
      resizeCollapsedTracker();
      return;
    }
    const availableHeight = Math.max(0, window.innerHeight - 24);
    const height = Math.min(TRACKER_HEIGHT, availableHeight);
    tracker.style.setProperty("--tracker-height", `${height}px`);
    tracker.style.setProperty("--tracker-top", `${centeredTop(height)}px`);
  };
  tracker.collapseTracker = () => {
    tracker.classList.add("collapsed");
    resizeCollapsedTracker();
  };
  tracker.expandTracker = () => {
    tracker.classList.remove("collapsed");
    resizeTracker();
  };
  tracker.toggleTracker = () => {
    if (tracker.classList.contains("collapsed")) {
      tracker.expandTracker();
    } else {
      tracker.collapseTracker();
    }
  };
  window.addEventListener("resize", resizeTracker);
  const collapse = document.createElement("button");
  collapse.className = "collapse";
  collapse.type = "button";
  collapse.title = "Collapse job tracker";
  collapse.setAttribute("aria-label", "Collapse job tracker");
  const collapseIcon = document.createElement("img");
  collapseIcon.src = chrome.runtime.getURL("icons/chevron-right.svg");
  collapseIcon.alt = "";
  collapse.append(collapseIcon);
  collapse.addEventListener("click", () => tracker.collapseTracker());
  const tab = document.createElement("button");
  tab.className = "tab";
  tab.type = "button";
  tab.title = "Expand job tracker";
  tab.setAttribute("aria-label", "Expand job tracker");
  const icon = document.createElement("img");
  icon.src = chrome.runtime.getURL("icons/resume-48.png");
  icon.alt = "";
  tab.append(icon);
  tab.addEventListener("click", () => tracker.expandTracker());
  scale.append(frame, collapse, tab);
  tracker.append(scale);
  shadow.append(style, tracker);
  document.documentElement.append(host);
  resizeTracker();
  return tracker;
}

function toggleOverlay() {
  const existing = document.getElementById(OVERLAY_ID);
  if (!existing) {
    createOverlay();
    return;
  }
  const tracker = existing.shadowRoot.querySelector(".tracker");
  tracker.toggleTracker();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "job-metadata") {
    sendResponse(extractMetadata(document, window.location));
    return false;
  }
  if (message.type === "toggle-job-tracker-overlay") {
    toggleOverlay();
    sendResponse({ ok: true });
  }
  return false;
});

let lastMetadata = "";
let lastUrl = window.location.href;
let changeTimer;

function notifyMetadataChange() {
  clearTimeout(changeTimer);
  changeTimer = setTimeout(() => {
    const run = () => {
      const metadata = extractMetadata(document, window.location);
      const serialized = JSON.stringify(metadata);
      if (serialized === lastMetadata) return;
      lastMetadata = serialized;
      chrome.runtime.sendMessage({ type: "job-metadata-changed" }).catch(() => {});
    };
    if ("requestIdleCallback" in window) requestIdleCallback(run, { timeout: 1000 });
    else run();
  }, 200);
}

if (adapterFor(window.location.hostname)) {
  new MutationObserver(notifyMetadataChange).observe(document.body, {
    childList: true,
    subtree: true
  });
  notifyMetadataChange();
  setInterval(() => {
    if (window.location.href === lastUrl) return;
    lastUrl = window.location.href;
    lastMetadata = "";
    notifyMetadataChange();
  }, 500);
}
