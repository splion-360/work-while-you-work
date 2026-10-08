(function (root) {
  const SIGNALS = [
    {
      pattern: /\b(?:will not|cannot|unable to)\s+sponsor\b|\bno\s+(?:visa\s+)?sponsorship\b/i,
      text: "Sponsorship Not Available",
      className: "restricted"
    },
    {
      pattern: /\b(?:visa\s+)?sponsorship\s+(?:is\s+)?available\b|\b(?:we|employer|company)\s+(?:will|can|do)\s+sponsor\b|\bsponsors?\s+(?:work\s+)?visas?\b/i,
      text: "Sponsorship Available",
      className: "clear"
    },
    {
      pattern: /\b(?:u\.?s\.?|united states)\s+citizens?\s+or\s+(?:u\.?s\.?\s+)?(?:permanent residents?|green card holders?)\b|\b(?:citizens?\s+or\s+permanent residents?)\s+of\s+the\s+(?:u\.?s\.?|united states)\b/i,
      text: "Sponsorship Not Available",
      className: "restricted"
    },
    {
      pattern: /\b(?:u\.?s\.?|united states)\s+citizen(?:ship)?s?\b/i,
      text: "Sponsorship Not Available",
      className: "restricted"
    },
    {
      pattern: /\bgreen card\b|\bpermanent resident\b/i,
      text: "Sponsorship Not Available",
      className: "restricted"
    },
    {
      pattern: /\bu\.?s\.?\s+person(?:s)?\b/i,
      text: "Sponsorship Not Available",
      className: "restricted"
    }
  ];

  function citizenshipSignal(text) {
    const value = String(text || "");
    for (const signal of SIGNALS) {
      if (signal.pattern.test(value)) {
        return { text: signal.text, className: signal.className };
      }
    }
    return { text: "No information", className: "warning" };
  }

  root.citizenshipSignal = citizenshipSignal;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { citizenshipSignal };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
