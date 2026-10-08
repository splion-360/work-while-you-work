import hashlib
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import DatasetDict, load_dataset
from rapidfuzz import fuzz

DATASET_NAME = "med2425/resume-job-fit-merged-v1"
EXPECTED_LABELS = ("Good Fit", "Potential Fit", "No Fit")
WHITESPACE_PATTERN = re.compile(r"\s+")
MOJIBAKE_PATTERN = re.compile(r"(?:Ã.|Â.|â€|ðŸ)")
RESUME_OPENER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:summary|professional profile|professional summary|career profile)",
    re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset_rows(
    dataset_name: str = DATASET_NAME,
    *,
    revision: str = "main",
) -> tuple[Iterable[Mapping[str, Any]], list[str]]:
    dataset = load_dataset(dataset_name, revision=revision)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset).__name__}")
    split_names = list(dataset.keys())

    def rows() -> Iterable[Mapping[str, Any]]:
        for split_name, split in dataset.items():
            for row in split:
                yield {**row, "__split__": split_name}

    return rows(), split_names


def _distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _document_issues(raw_text: object, document_type: str) -> list[str]:
    if not isinstance(raw_text, str) or not raw_text:
        return []
    issues = []
    if "\ufffd" in raw_text:
        issues.append("replacement_character")
    if "\x00" in raw_text:
        issues.append("null_byte")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in raw_text
    ):
        issues.append("control_character")
    if MOJIBAKE_PATTERN.search(raw_text):
        issues.append("possible_mojibake")
    if document_type == "resume" and len(RESUME_OPENER_PATTERN.findall(raw_text)) > 1:
        issues.append("possible_concatenation")
    return issues


def find_near_duplicates(
    documents: Mapping[str, str],
    *,
    threshold: float = 97.0,
    example_limit: int = 100,
) -> dict[str, Any]:
    if not 0 < threshold <= 100:
        raise ValueError("Near-duplicate threshold must be in (0, 100]")
    entries = sorted(documents.items())
    matches: list[dict[str, str | float]] = []
    match_count = 0
    for index, (left_hash, left_text) in enumerate(entries):
        left = left_text.casefold()
        for right_hash, right_text in entries[index + 1 :]:
            right = right_text.casefold()
            maximum_ratio = 200 * min(len(left), len(right)) / (len(left) + len(right))
            if maximum_ratio < threshold:
                continue
            similarity = fuzz.ratio(left, right, score_cutoff=threshold)
            if not similarity:
                continue
            match_count += 1
            if len(matches) < example_limit:
                matches.append(
                    {
                        "left_hash": left_hash,
                        "right_hash": right_hash,
                        "similarity": round(similarity, 2),
                    }
                )
    matches.sort(key=lambda match: (-float(match["similarity"]), str(match["left_hash"])))
    return {"count": match_count, "examples": matches[:example_limit]}


def _reuse_report(counts: Counter[str], documents: Mapping[str, str]) -> dict[str, Any]:
    most_reused = []
    for item_hash, count in counts.most_common(20):
        most_reused.append(
            {
                "hash": item_hash,
                "pair_rows": count,
                "preview": documents[item_hash][:160],
            }
        )
    return {"distribution": _distribution(list(counts.values())), "most_reused": most_reused}


def _random_split_leakage(
    row_pairs: list[tuple[str, str]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    shuffled = list(row_pairs)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    validation_end = train_end + int(len(shuffled) * 0.1)
    partitions = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    train_resumes = {resume_hash for resume_hash, _ in partitions["train"]}
    train_jds = {jd_hash for _, jd_hash in partitions["train"]}
    report: dict[str, Any] = {"seed": seed, "split": "80/10/10", "rows": {}}
    for name, pairs in partitions.items():
        resumes = {resume_hash for resume_hash, _ in pairs}
        jds = {jd_hash for _, jd_hash in pairs}
        report["rows"][name] = len(pairs)
        if name != "train":
            report[name] = {
                "resume_overlap_with_train": len(resumes & train_resumes),
                "resume_overlap_rate": round(len(resumes & train_resumes) / len(resumes), 6)
                if resumes
                else 0,
                "jd_overlap_with_train": len(jds & train_jds),
                "jd_overlap_rate": round(len(jds & train_jds) / len(jds), 6) if jds else 0,
            }
    return report


def _component_report(pair_labels: Mapping[tuple[str, str], set[str]]) -> dict[str, Any]:
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for resume_hash, jd_hash in pair_labels:
        resume_node = f"resume:{resume_hash}"
        jd_node = f"jd:{jd_hash}"
        adjacency[resume_node].add(jd_node)
        adjacency[jd_node].add(resume_node)

    components = []
    remaining = set(adjacency)
    while remaining:
        pending = [remaining.pop()]
        resumes = 0
        jds = 0
        while pending:
            node = pending.pop()
            resumes += node.startswith("resume:")
            jds += node.startswith("jd:")
            unseen = adjacency[node] & remaining
            remaining.difference_update(unseen)
            pending.extend(unseen)
        components.append({"resumes": resumes, "jds": jds, "documents": resumes + jds})
    components.sort(key=lambda component: component["documents"], reverse=True)
    return {
        "count": len(components),
        "largest": components[0] if components else {"resumes": 0, "jds": 0, "documents": 0},
        "largest_five": components[:5],
    }


def _published_split_leakage(
    split_documents: Mapping[str, dict[str, set[str]]], split_rows: Mapping[str, int]
) -> dict[str, Any]:
    split_names = list(split_documents)
    if len(split_names) < 2:
        return {"available": False, "splits": split_names}
    reference_name = "train" if "train" in split_documents else split_names[0]
    reference = split_documents[reference_name]
    comparisons = {}
    for split_name in split_names:
        if split_name == reference_name:
            continue
        current = split_documents[split_name]
        comparisons[split_name] = {
            "resume_overlap_with_reference": len(current["resumes"] & reference["resumes"]),
            "resume_overlap_rate": round(
                len(current["resumes"] & reference["resumes"]) / len(current["resumes"]), 6
            )
            if current["resumes"]
            else 0,
            "jd_overlap_with_reference": len(current["jds"] & reference["jds"]),
            "jd_overlap_rate": round(
                len(current["jds"] & reference["jds"]) / len(current["jds"]), 6
            )
            if current["jds"]
            else 0,
        }
    return {
        "available": True,
        "reference": reference_name,
        "splits": {
            name: {
                "rows": split_rows[name],
                "unique_resumes": len(documents["resumes"]),
                "unique_jds": len(documents["jds"]),
            }
            for name, documents in split_documents.items()
        },
        "comparisons": comparisons,
    }


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float = 97.0,
    include_near_duplicates: bool = True,
) -> dict[str, Any]:
    total_rows = 0
    usable_rows = 0
    missing = Counter()
    labels = Counter()
    unexpected_labels = Counter()
    exact_rows: Counter[tuple[str, str, str]] = Counter()
    pair_labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    resume_counts: Counter[str] = Counter()
    jd_counts: Counter[str] = Counter()
    resumes: dict[str, str] = {}
    jds: dict[str, str] = {}
    resume_lengths: list[int] = []
    jd_lengths: list[int] = []
    issue_counts: Counter[str] = Counter()
    issue_documents: defaultdict[str, set[str]] = defaultdict(set)
    issue_examples: list[dict[str, Any]] = []
    row_pairs: list[tuple[str, str]] = []
    split_documents: defaultdict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"resumes": set(), "jds": set()}
    )
    split_rows: Counter[str] = Counter()

    for row_number, row in enumerate(rows):
        total_rows += 1
        raw_resume = row.get("resume")
        raw_jd = row.get("jd")
        resume = normalize_text(raw_resume)
        jd = normalize_text(raw_jd)
        label = normalize_text(row.get("label"))
        if not resume:
            missing["resume"] += 1
        if not jd:
            missing["jd"] += 1
        if not label:
            missing["label"] += 1
        elif label not in EXPECTED_LABELS:
            unexpected_labels[label] += 1
        labels[label or "<missing>"] += 1

        for document_type, raw_text in (("resume", raw_resume), ("jd", raw_jd)):
            for issue in _document_issues(raw_text, document_type):
                issue_key = f"{document_type}:{issue}"
                issue_counts[issue_key] += 1
                normalized_document = normalize_text(raw_text)
                if normalized_document:
                    issue_documents[issue_key].add(document_hash(normalized_document))
                if len(issue_examples) < 50:
                    issue_examples.append(
                        {"row": row_number, "document_type": document_type, "issue": issue}
                    )

        if not resume or not jd or not label:
            continue
        usable_rows += 1
        resume_hash = document_hash(resume)
        jd_hash = document_hash(jd)
        if resume_hash not in resumes:
            resumes[resume_hash] = resume
            resume_lengths.append(len(resume))
        if jd_hash not in jds:
            jds[jd_hash] = jd
            jd_lengths.append(len(jd))
        resume_counts[resume_hash] += 1
        jd_counts[jd_hash] += 1
        exact_rows[(resume_hash, jd_hash, label)] += 1
        pair_labels[(resume_hash, jd_hash)].add(label)
        row_pairs.append((resume_hash, jd_hash))
        split_name = normalize_text(row.get("__split__"))
        if split_name:
            split_rows[split_name] += 1
            split_documents[split_name]["resumes"].add(resume_hash)
            split_documents[split_name]["jds"].add(jd_hash)

    conflicts = [
        {"resume_hash": pair[0], "jd_hash": pair[1], "labels": sorted(pair_labels[pair])}
        for pair in sorted(pair_labels)
        if len(pair_labels[pair]) > 1
    ]
    label_total = sum(labels.values())
    report: dict[str, Any] = {
        "rows": {
            "total": total_rows,
            "usable": usable_rows,
            "missing": dict(sorted(missing.items())),
            "exact_duplicate_rows": sum(count - 1 for count in exact_rows.values()),
            "repeated_pair_rows": usable_rows - len(pair_labels),
        },
        "unique": {"resumes": len(resumes), "jds": len(jds), "pairs": len(pair_labels)},
        "labels": {
            "counts": dict(sorted(labels.items())),
            "percentages": {
                label: round(count * 100 / label_total, 4) for label, count in sorted(labels.items())
            }
            if label_total
            else {},
            "unexpected": dict(sorted(unexpected_labels.items())),
            "majority_class_baseline": round(max(labels.values()) / label_total, 6)
            if label_total
            else 0,
        },
        "conflicting_pairs": {"count": len(conflicts), "examples": conflicts[:100]},
        "document_lengths_characters": {
            "resumes": _distribution(resume_lengths),
            "jds": _distribution(jd_lengths),
        },
        "document_reuse": {
            "resumes": _reuse_report(resume_counts, resumes),
            "jds": _reuse_report(jd_counts, jds),
        },
        "malformed_text": {
            "row_occurrences": dict(sorted(issue_counts.items())),
            "unique_documents": {
                issue: len(document_hashes) for issue, document_hashes in sorted(issue_documents.items())
            },
            "examples": issue_examples,
        },
        "leakage": {
            "published_splits": _published_split_leakage(split_documents, split_rows),
            "naive_random_split": _random_split_leakage(row_pairs),
            "bipartite_components": _component_report(pair_labels),
        },
    }
    if include_near_duplicates:
        report["near_duplicates"] = {
            "threshold": near_duplicate_threshold,
            "resumes": find_near_duplicates(resumes, threshold=near_duplicate_threshold),
            "jds": find_near_duplicates(jds, threshold=near_duplicate_threshold),
        }
    return report
