(function (root) {
  function duplicateApplicationStatus(data) {
    if (!data.exists) return { text: "Not applied yet", className: "clear" };
    if (data.status === "Rejected") return { text: "Rejected", className: "rejected" };
    return { text: "Already applied", className: "exists" };
  }

  root.duplicateApplicationStatus = duplicateApplicationStatus;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { duplicateApplicationStatus };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
