import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from resume_jd_scoring.data import document_hash, find_near_duplicates, normalize_text

MISSING_DOMAIN = "<missing>"


@dataclass(frozen=True)
class PairRecord:
    resume_hash: str
    jd_hash: str
    label: str


@dataclass
class PairCorpus:
    pairs: list[PairRecord]
    resumes: dict[str, str]
    jds: dict[str, str]
    resume_domains: dict[str, str]
    jd_domains: dict[str, str]
    resume_domain_confidence: dict[str, float]
    jd_domain_confidence: dict[str, float]
    domain_conflicts: dict[str, dict[str, Any]]


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_pair_corpus(rows: Iterable[Mapping[str, Any]]) -> PairCorpus:
    pairs = []
    resumes: dict[str, str] = {}
    jds: dict[str, str] = {}
    resume_domain_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    jd_domain_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        resume = normalize_text(row.get("resume"))
        jd = normalize_text(row.get("jd"))
        label = normalize_text(row.get("label"))
        if not resume or not jd or not label:
            continue
        resume_hash = document_hash(resume)
        jd_hash = document_hash(jd)
        resumes.setdefault(resume_hash, resume)
        jds.setdefault(jd_hash, jd)
        resume_domain_counts[resume_hash][
            normalize_text(row.get("resume_domain")) or MISSING_DOMAIN
        ] += 1
        jd_domain_counts[jd_hash][normalize_text(row.get("jd_domain")) or MISSING_DOMAIN] += 1
        pairs.append(PairRecord(resume_hash, jd_hash, label))

    resume_domains, resume_confidence, resume_conflicts = _resolve_document_domains(
        resume_domain_counts
    )
    jd_domains, jd_confidence, jd_conflicts = _resolve_document_domains(jd_domain_counts)
    return PairCorpus(
        pairs=pairs,
        resumes=resumes,
        jds=jds,
        resume_domains=resume_domains,
        jd_domains=jd_domains,
        resume_domain_confidence=resume_confidence,
        jd_domain_confidence=jd_confidence,
        domain_conflicts={"resumes": resume_conflicts, "jds": jd_conflicts},
    )


def _resolve_document_domains(
    domain_counts: Mapping[str, Counter[str]],
) -> tuple[dict[str, str], dict[str, float], dict[str, Any]]:
    resolved = {}
    confidence_by_document = {}
    conflicts = []
    ties = 0
    confidence_values = []
    conflicting_confidence_values = []
    for item_hash, counts in sorted(domain_counts.items()):
        highest_count = max(counts.values())
        winners = sorted(domain for domain, count in counts.items() if count == highest_count)
        assigned_domain = winners[0]
        total = sum(counts.values())
        confidence = highest_count / total
        resolved[item_hash] = assigned_domain
        confidence_by_document[item_hash] = confidence
        confidence_values.append(confidence)
        if len(counts) > 1:
            ties += len(winners) > 1
            conflicting_confidence_values.append(confidence)
            conflicts.append(
                {
                    "hash": item_hash,
                    "counts": dict(sorted(counts.items())),
                    "assigned_domain": assigned_domain,
                    "assignment_confidence": round(confidence, 6),
                    "tied": len(winners) > 1,
                }
            )
    conflicts.sort(key=lambda item: (item["assignment_confidence"], item["hash"]))
    return resolved, confidence_by_document, {
        "documents": len(domain_counts),
        "conflicting_documents": len(conflicts),
        "conflict_rate": round(len(conflicts) / len(domain_counts), 6) if domain_counts else 0,
        "tied_majorities": ties,
        "assignment_confidence": {
            "minimum": round(min(confidence_values), 6) if confidence_values else 0,
            "median": round(statistics.median(confidence_values), 6) if confidence_values else 0,
            "maximum": round(max(confidence_values), 6) if confidence_values else 0,
        },
        "conflicting_assignment_confidence": {
            "minimum": round(min(conflicting_confidence_values), 6)
            if conflicting_confidence_values
            else 0,
            "median": round(statistics.median(conflicting_confidence_values), 6)
            if conflicting_confidence_values
            else 0,
            "maximum": round(max(conflicting_confidence_values), 6)
            if conflicting_confidence_values
            else 0,
        },
        "examples": conflicts[:20],
    }


def near_duplicate_groups(
    documents: Mapping[str, str], *, threshold: float = 97.0
) -> tuple[dict[str, str], dict[str, Any]]:
    union_find = UnionFind(documents)
    report = find_near_duplicates(
        documents,
        threshold=threshold,
        example_limit=max(1, len(documents) * len(documents)),
    )
    for match in report["examples"]:
        union_find.union(str(match["left_hash"]), str(match["right_hash"]))
    groups = {item_hash: union_find.find(item_hash) for item_hash in documents}
    sizes = Counter(groups.values())
    return groups, {
        "threshold": threshold,
        "similar_pairs": report["count"],
        "clusters": len(sizes),
        "non_singleton_clusters": sum(size > 1 for size in sizes.values()),
        "largest_cluster": max(sizes.values(), default=0),
    }


def stratified_fold_assignment(
    domains: Mapping[str, str],
    groups: Mapping[str, str],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    item_hashes = sorted(domains)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment: dict[str, int] = {}
    features = np.zeros((len(item_hashes), 1), dtype=np.uint8)
    labels = np.array([domains[item_hash] for item_hash in item_hashes])
    group_ids = np.array([groups[item_hash] for item_hash in item_hashes])
    for fold, (_, test_indices) in enumerate(splitter.split(features, labels, group_ids)):
        assignment.update({item_hashes[index]: fold for index in test_indices})
    if len(assignment) != len(item_hashes):
        raise RuntimeError("Fold assignment did not cover every document")
    return assignment


def _percentages(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {
        value: round(count * 100 / total, 4) for value, count in sorted(counts.items())
    } if total else {}


def _total_variation(reference: Counter[str], observed: Counter[str]) -> float:
    reference_total = sum(reference.values())
    observed_total = sum(observed.values())
    if not reference_total or not observed_total:
        return 1.0
    values = set(reference) | set(observed)
    return round(
        0.5
        * sum(
            abs(reference[value] / reference_total - observed[value] / observed_total)
            for value in values
        ),
        6,
    )


def _partition_report(
    pairs: Iterable[PairRecord],
    resume_domains: Mapping[str, str],
    jd_domains: Mapping[str, str],
    full_label_counts: Counter[str],
    full_resume_domain_counts: Counter[str],
    full_jd_domain_counts: Counter[str],
) -> dict[str, Any]:
    pair_list = list(pairs)
    labels = Counter(pair.label for pair in pair_list)
    resume_hashes = {pair.resume_hash for pair in pair_list}
    jd_hashes = {pair.jd_hash for pair in pair_list}
    unique_resume_domains = Counter(resume_domains[item_hash] for item_hash in resume_hashes)
    unique_jd_domains = Counter(jd_domains[item_hash] for item_hash in jd_hashes)
    pair_resume_domains = Counter(resume_domains[pair.resume_hash] for pair in pair_list)
    pair_jd_domains = Counter(jd_domains[pair.jd_hash] for pair in pair_list)
    return {
        "pairs": len(pair_list),
        "unique_resumes": len(resume_hashes),
        "unique_jds": len(jd_hashes),
        "labels": {"counts": dict(sorted(labels.items())), "percentages": _percentages(labels)},
        "unique_resume_domains": {
            "counts": dict(sorted(unique_resume_domains.items())),
            "percentages": _percentages(unique_resume_domains),
            "total_variation_from_full": _total_variation(
                full_resume_domain_counts, unique_resume_domains
            ),
        },
        "unique_jd_domains": {
            "counts": dict(sorted(unique_jd_domains.items())),
            "percentages": _percentages(unique_jd_domains),
            "total_variation_from_full": _total_variation(full_jd_domain_counts, unique_jd_domains),
        },
        "pair_weighted_resume_domains": {
            "counts": dict(sorted(pair_resume_domains.items())),
            "percentages": _percentages(pair_resume_domains),
        },
        "pair_weighted_jd_domains": {
            "counts": dict(sorted(pair_jd_domains.items())),
            "percentages": _percentages(pair_jd_domains),
        },
        "label_total_variation_from_full": _total_variation(full_label_counts, labels),
        "missing_resume_domains": sorted(set(full_resume_domain_counts) - set(unique_resume_domains)),
        "missing_jd_domains": sorted(set(full_jd_domain_counts) - set(unique_jd_domains)),
    }


def _base_counts(corpus: PairCorpus) -> tuple[Counter[str], Counter[str], Counter[str]]:
    return (
        Counter(pair.label for pair in corpus.pairs),
        Counter(corpus.resume_domains.values()),
        Counter(corpus.jd_domains.values()),
    )


def analyze_block_kfold(
    corpus: PairCorpus,
    resume_groups: Mapping[str, str],
    jd_groups: Mapping[str, str],
    *,
    n_splits: int,
    seed: int = 42,
) -> dict[str, Any]:
    resume_folds = stratified_fold_assignment(
        corpus.resume_domains, resume_groups, n_splits=n_splits, seed=seed
    )
    jd_folds = stratified_fold_assignment(
        corpus.jd_domains, jd_groups, n_splits=n_splits, seed=seed + 1
    )
    full_label_counts, full_resume_domains, full_jd_domains = _base_counts(corpus)
    folds = []
    for fold in range(n_splits):
        train = []
        test = []
        discarded = 0
        for pair in corpus.pairs:
            resume_held_out = resume_folds[pair.resume_hash] == fold
            jd_held_out = jd_folds[pair.jd_hash] == fold
            if resume_held_out and jd_held_out:
                test.append(pair)
            elif not resume_held_out and not jd_held_out:
                train.append(pair)
            else:
                discarded += 1
        folds.append(
            {
                "fold": fold,
                "train": _partition_report(
                    train,
                    corpus.resume_domains,
                    corpus.jd_domains,
                    full_label_counts,
                    full_resume_domains,
                    full_jd_domains,
                ),
                "test": _partition_report(
                    test,
                    corpus.resume_domains,
                    corpus.jd_domains,
                    full_label_counts,
                    full_resume_domains,
                    full_jd_domains,
                ),
                "discarded_pairs": discarded,
            }
        )
    return {"strategy": f"{n_splits}-fold", "seed": seed, "folds": folds}


def analyze_block_holdout(
    corpus: PairCorpus,
    resume_groups: Mapping[str, str],
    jd_groups: Mapping[str, str],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    resume_microfolds = stratified_fold_assignment(
        corpus.resume_domains, resume_groups, n_splits=20, seed=seed
    )
    jd_microfolds = stratified_fold_assignment(
        corpus.jd_domains, jd_groups, n_splits=20, seed=seed + 1
    )
    fold_order = list(range(20))
    random.Random(seed + 2).shuffle(fold_order)
    fold_partition = {
        **{fold: "train" for fold in fold_order[:14]},
        **{fold: "validation" for fold in fold_order[14:17]},
        **{fold: "test" for fold in fold_order[17:]},
    }
    resume_partitions = {
        item_hash: fold_partition[fold] for item_hash, fold in resume_microfolds.items()
    }
    jd_partitions = {
        item_hash: fold_partition[fold] for item_hash, fold in jd_microfolds.items()
    }
    partitions: defaultdict[str, list[PairRecord]] = defaultdict(list)
    discarded = 0
    for pair in corpus.pairs:
        resume_partition = resume_partitions[pair.resume_hash]
        jd_partition = jd_partitions[pair.jd_hash]
        if resume_partition == jd_partition:
            partitions[resume_partition].append(pair)
        else:
            discarded += 1

    full_label_counts, full_resume_domains, full_jd_domains = _base_counts(corpus)
    return {
        "strategy": "70/15/15",
        "seed": seed,
        "partitions": {
            partition: _partition_report(
                partitions[partition],
                corpus.resume_domains,
                corpus.jd_domains,
                full_label_counts,
                full_resume_domains,
                full_jd_domains,
            )
            for partition in ("train", "validation", "test")
        },
        "discarded_pairs": discarded,
    }


def analyze_split_feasibility(
    rows: Iterable[Mapping[str, Any]], *, near_duplicate_threshold: float = 97.0, seed: int = 42
) -> dict[str, Any]:
    corpus = build_pair_corpus(rows)
    resume_groups, resume_group_report = near_duplicate_groups(
        corpus.resumes, threshold=near_duplicate_threshold
    )
    jd_groups, jd_group_report = near_duplicate_groups(
        corpus.jds, threshold=near_duplicate_threshold
    )
    full_labels, full_resume_domains, full_jd_domains = _base_counts(corpus)
    return {
        "corpus": {
            "pairs": len(corpus.pairs),
            "resumes": len(corpus.resumes),
            "jds": len(corpus.jds),
            "labels": {
                "counts": dict(sorted(full_labels.items())),
                "percentages": _percentages(full_labels),
            },
            "resume_domains": {
                "counts": dict(sorted(full_resume_domains.items())),
                "percentages": _percentages(full_resume_domains),
            },
            "jd_domains": {
                "counts": dict(sorted(full_jd_domains.items())),
                "percentages": _percentages(full_jd_domains),
            },
            "domain_conflicts": corpus.domain_conflicts,
        },
        "near_duplicate_groups": {"resumes": resume_group_report, "jds": jd_group_report},
        "holdout": analyze_block_holdout(corpus, resume_groups, jd_groups, seed=seed),
        "cross_validation": {
            "3-fold": analyze_block_kfold(
                corpus, resume_groups, jd_groups, n_splits=3, seed=seed
            ),
            "5-fold": analyze_block_kfold(
                corpus, resume_groups, jd_groups, n_splits=5, seed=seed
            ),
        },
    }
