function matchingApplications(applications, query, limit = 8, status = "") {
  const terms = String(query || "").trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const selectedStatus = String(status || "").trim();
  return applications.filter((application) => {
    if (selectedStatus && application.status !== selectedStatus) return false;
    const text = [application.company, application.job_title, application.job_source]
      .filter(Boolean)
      .join(" ")
      .toLocaleLowerCase();
    return terms.every((term) => text.includes(term));
  }).slice(0, limit);
}

if (typeof module !== "undefined") module.exports = { matchingApplications };
