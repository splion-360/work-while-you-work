chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "toggle-job-tracker-overlay" });
  } catch (_error) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["metadata.js", "content.js"]
      });
      await chrome.tabs.sendMessage(tab.id, { type: "toggle-job-tracker-overlay" });
    } catch (_injectionError) {
      // Browser-internal pages do not permit content-script injection.
    }
  }
});

chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.runtime.sendMessage({ type: "active-tab-changed", tabId }).catch(() => {});
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    chrome.runtime.sendMessage({ type: "active-tab-changed", tabId }).catch(() => {});
  }
});
