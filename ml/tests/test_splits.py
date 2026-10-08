from resume_jd_scoring.splits import (
    PairCorpus,
    PairRecord,
    UnionFind,
    analyze_block_kfold,
    build_pair_corpus,
)


def test_build_pair_corpus_reports_domain_conflicts():
    corpus = build_pair_corpus(
        [
            {
                "resume": "Resume A",
                "jd": "Job A",
                "label": "Good Fit",
                "resume_domain": "software",
                "jd_domain": "software",
            },
            {
                "resume": "Resume A",
                "jd": "Job B",
                "label": "No Fit",
                "resume_domain": "finance",
                "jd_domain": "finance",
            },
        ]
    )

    assert len(corpus.resumes) == 1
    conflicts = corpus.domain_conflicts["resumes"]
    assert conflicts["conflicting_documents"] == 1
    assert conflicts["tied_majorities"] == 1
    assert conflicts["examples"] == [
        {
            "hash": next(iter(corpus.resumes)),
            "counts": {"finance": 1, "software": 1},
            "assigned_domain": "finance",
            "assignment_confidence": 0.5,
            "tied": True,
        }
    ]


def test_union_find_keeps_transitive_near_duplicates_together():
    groups = UnionFind(["a", "b", "c"])
    groups.union("a", "b")
    groups.union("b", "c")

    assert groups.find("a") == groups.find("b") == groups.find("c")


def test_block_kfold_has_no_document_overlap_and_accounts_for_every_pair():
    resumes = {f"r{index}": f"Resume {index}" for index in range(6)}
    jds = {f"j{index}": f"Job {index}" for index in range(6)}
    corpus = PairCorpus(
        pairs=[
            PairRecord(resume_hash, jd_hash, "Good Fit" if i == j else "No Fit")
            for i, resume_hash in enumerate(resumes)
            for j, jd_hash in enumerate(jds)
        ],
        resumes=resumes,
        jds=jds,
        resume_domains={item_hash: "software" for item_hash in resumes},
        jd_domains={item_hash: "software" for item_hash in jds},
        resume_domain_confidence={item_hash: 1.0 for item_hash in resumes},
        jd_domain_confidence={item_hash: 1.0 for item_hash in jds},
        domain_conflicts={"resumes": {}, "jds": {}},
    )

    report = analyze_block_kfold(
        corpus,
        {item_hash: item_hash for item_hash in resumes},
        {item_hash: item_hash for item_hash in jds},
        n_splits=3,
    )

    for fold in report["folds"]:
        assert fold["train"]["unique_resumes"] == 4
        assert fold["train"]["unique_jds"] == 4
        assert fold["test"]["unique_resumes"] == 2
        assert fold["test"]["unique_jds"] == 2
        assert fold["train"]["pairs"] + fold["test"]["pairs"] + fold["discarded_pairs"] == 36
