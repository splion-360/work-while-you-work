function normalizeHttpUrl(value) {
  const trimmed = String(value || "").trim();
  if (!trimmed) return "";
  const parsed = new URL(trimmed);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("URLs must start with http:// or https://");
  }
  return parsed.toString();
}

function isLinkedInUrl(value) {
  if (!value) return true;
  const hostname = new URL(value).hostname.toLowerCase();
  return hostname === "linkedin.com" || hostname.endsWith(".linkedin.com");
}

function isGitHubUrl(value) {
  if (!value) return true;
  const hostname = new URL(value).hostname.toLowerCase();
  return hostname === "github.com" || hostname.endsWith(".github.com");
}

if (typeof module !== "undefined") module.exports = { isGitHubUrl, isLinkedInUrl, normalizeHttpUrl };
