from resume_jd_scoring.data import audit_rows, find_near_duplicates, normalize_text


def test_normalize_text_only_collapses_whitespace():
    assert normalize_text("  Senior\n  ML Engineer ") == "Senior ML Engineer"
    assert normalize_text(None) == ""


def test_audit_reports_duplicates_conflicts_and_missing_values():
    rows = [
        {"resume": "Resume A", "jd": "Job A", "label": "Good Fit"},
        {"resume": " Resume  A ", "jd": "Job A", "label": "Good Fit"},
        {"resume": "Resume A", "jd": "Job A", "label": "No Fit"},
        {"resume": "Resume B", "jd": "Job B", "label": "Potential Fit"},
        {"resume": "", "jd": "Job C", "label": "Good Fit"},
    ]

    report = audit_rows(rows, include_near_duplicates=False)

    assert report["rows"] == {
        "total": 5,
        "usable": 4,
        "missing": {"resume": 1},
        "exact_duplicate_rows": 1,
        "repeated_pair_rows": 2,
    }
    assert report["unique"] == {"resumes": 2, "jds": 2, "pairs": 2}
    assert report["conflicting_pairs"]["count"] == 1
    assert report["conflicting_pairs"]["examples"][0]["labels"] == ["Good Fit", "No Fit"]


def test_audit_flags_encoding_and_possible_concatenation():
    report = audit_rows(
        [
            {
                "resume": "Summary first section.\nProfessional Profile second section.\ufffd",
                "jd": "Valid job description",
                "label": "Good Fit",
            }
        ],
        include_near_duplicates=False,
    )

    assert report["malformed_text"]["row_occurrences"] == {
        "resume:possible_concatenation": 1,
        "resume:replacement_character": 1,
    }
    assert report["malformed_text"]["unique_documents"] == {
        "resume:possible_concatenation": 1,
        "resume:replacement_character": 1,
    }


def test_audit_reports_overlap_in_published_splits():
    report = audit_rows(
        [
            {"resume": "Resume A", "jd": "Job A", "label": "Good Fit", "__split__": "train"},
            {"resume": "Resume A", "jd": "Job B", "label": "No Fit", "__split__": "test"},
        ],
        include_near_duplicates=False,
    )

    assert report["leakage"]["published_splits"] == {
        "available": True,
        "reference": "train",
        "splits": {
            "train": {"rows": 1, "unique_resumes": 1, "unique_jds": 1},
            "test": {"rows": 1, "unique_resumes": 1, "unique_jds": 1},
        },
        "comparisons": {
            "test": {
                "resume_overlap_with_reference": 1,
                "resume_overlap_rate": 1.0,
                "jd_overlap_with_reference": 0,
                "jd_overlap_rate": 0.0,
            }
        },
    }


def test_near_duplicates_are_reported_separately_from_exact_duplicates():
    report = find_near_duplicates(
        {
            "first": "Machine learning engineer with Python and PyTorch experience.",
            "second": "Machine learning engineer with Python and PyTorch experience!",
            "other": "Senior accountant responsible for financial reporting.",
        },
        threshold=95.0,
    )

    assert report == {
        "count": 1,
        "examples": [{"left_hash": "first", "right_hash": "second", "similarity": 98.36}],
    }
