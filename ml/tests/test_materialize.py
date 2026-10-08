from resume_jd_scoring.materialize import build_manifest_rows, validate_manifest_rows
from resume_jd_scoring.splits import PairCorpus, PairRecord


def test_manifest_roles_and_validation_are_deterministic():
    corpus = PairCorpus(
        pairs=[
            PairRecord("r0", "j0", "Good Fit"),
            PairRecord("r0", "j1", "No Fit"),
            PairRecord("r1", "j0", "Potential Fit"),
            PairRecord("r1", "j1", "Good Fit"),
            PairRecord("r2", "j2", "Good Fit"),
        ],
        resumes={"r0": "Resume 0", "r1": "Resume 1", "r2": "Resume 2"},
        jds={"j0": "Job 0", "j1": "Job 1", "j2": "Job 2"},
        resume_domains={"r0": "software", "r1": "software", "r2": "finance"},
        jd_domains={"j0": "software", "j1": "software", "j2": "finance"},
        resume_domain_confidence={"r0": 1.0, "r1": 1.0, "r2": 1.0},
        jd_domain_confidence={"j0": 1.0, "j1": 1.0, "j2": 1.0},
        domain_conflicts={"resumes": {}, "jds": {}},
    )
    resume_groups = {item_hash: item_hash for item_hash in corpus.resumes}
    jd_groups = {item_hash: item_hash for item_hash in corpus.jds}
    resume_folds = {"r0": 0, "r1": 1, "r2": 2}
    jd_folds = {"j0": 0, "j1": 1, "j2": 2}

    first = build_manifest_rows(corpus, resume_groups, jd_groups, resume_folds, jd_folds)
    second = build_manifest_rows(corpus, resume_groups, jd_groups, resume_folds, jd_folds)

    assert first == second
    report = validate_manifest_rows(*first)
    assert all(report["checks"].values())
    assert report["folds"][0]["roles"] == {"discarded": 2, "test": 1, "train": 2}
