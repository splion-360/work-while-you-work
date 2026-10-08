from pathlib import Path


def build_resume_catalog(resumes_dir: Path):
    resumes = []
    for path in resumes_dir.rglob("pdf/*.pdf"):
        relative = path.relative_to(resumes_dir)
        parts = relative.parts
        if len(parts) < 3:
            continue
        family = parts[1] if parts[0] == "one-column" else parts[0]
        stem = path.stem
        resumes.append(
            {
                "version": f"{family}/{stem}",
                "label": f"{family.replace('-', ' ').title()} | {stem}",
                "path": str(path),
                "modified": path.stat().st_mtime,
            }
        )
    return sorted(resumes, key=lambda item: (-item["modified"], item["version"]))
