import tempfile
import unittest
from pathlib import Path

from service.scorer import (
    ResumeScorer,
    ScoringError,
    parse_job_criteria,
    parse_resume_evidence,
    text_chunks,
)
from service.scoring_state import ScoringStore


class FakeEmbedder:
    model = "test-embedding-v1"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in ("python", "pytorch", "machine learning")):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class ModerateEmbedder(FakeEmbedder):
    def embed(self, texts):
        self.calls += 1
        return [[0.6, 0.8] if "moderate" in text.lower() else [1.0, 0.0] for text in texts]


class ResumeScorerTests(unittest.TestCase):
    def scoring_input(self):
        return {
            "application_key": "app-1",
            "job_description": "Build machine learning systems using Python and PyTorch.",
            "job_description_hash": "job-hash",
            "resume_path": "/resumes/resume.pdf",
            "resume_content_hash": "resume-hash",
        }

    def test_scores_a_strong_match_with_category_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            embedder = FakeEmbedder()
            scorer = ResumeScorer(
                store,
                embedder,
                scorer_version="weights-v1",
                pdf_extractor=lambda _path: "Built machine learning services with Python and PyTorch.",
                file_hasher=lambda _path: "resume-hash",
            )

            result = scorer.score(self.scoring_input())

        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["band"], "Strong")
        self.assertEqual(result["matched_terms"], ["Python", "PyTorch"])
        self.assertEqual(result["category_breakdown"], {"core": 100.0})
        self.assertEqual(len(result["input_fingerprint"]), 64)

    def test_reuses_embeddings_for_unchanged_content(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            embedder = FakeEmbedder()
            scorer = ResumeScorer(
                store,
                embedder,
                scorer_version="weights-v1",
                pdf_extractor=lambda _path: "Python machine learning experience.",
                file_hasher=lambda _path: "resume-hash",
            )

            scorer.score(self.scoring_input())
            scorer.score(self.scoring_input())

        self.assertEqual(embedder.calls, 2)

    def test_scores_unrelated_content_as_weak(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scorer = ResumeScorer(
                store,
                FakeEmbedder(),
                scorer_version="weights-v1",
                pdf_extractor=lambda _path: "Python machine learning experience.",
                file_hasher=lambda _path: "resume-hash",
            )
            scoring_input = self.scoring_input() | {
                "job_description": "Manage retail sales campaigns and print advertising.",
                "job_description_hash": "unrelated-job",
            }

            result = scorer.score(scoring_input)

        self.assertEqual(result["band"], "Weak")
        self.assertLess(result["score"], 30)
        self.assertEqual(result["matched_terms"], [])

    def test_calibrates_partial_similarity_as_moderate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scorer = ResumeScorer(
                store,
                ModerateEmbedder(),
                scorer_version="weights-v1",
                pdf_extractor=lambda _path: "Production systems engineering experience.",
                file_hasher=lambda _path: "resume-hash",
            )
            scoring_input = self.scoring_input() | {
                "job_description": "Moderate overlap with production engineering work.",
                "job_description_hash": "moderate-job",
            }

            result = scorer.score(scoring_input)

        self.assertEqual(result["band"], "Moderate")
        self.assertGreaterEqual(result["score"], 55)
        self.assertLess(result["score"], 75)

    def test_chunks_never_exceed_the_embedding_input_limit(self):
        chunks = text_chunks("word " * 5000)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 1500 for chunk in chunks))

    def test_rejects_a_resume_changed_after_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scorer = ResumeScorer(
                store,
                FakeEmbedder(),
                scorer_version="weights-v1",
                pdf_extractor=lambda _path: "This must not be scored.",
                file_hasher=lambda _path: "changed-hash",
            )

            with self.assertRaisesRegex(ScoringError, "captured content hash"):
                scorer.score(self.scoring_input())

    def test_job_parser_excludes_company_context_and_classifies_sections(self):
        criteria = parse_job_criteria("""
About Gray Swan
We build secure AI products for frontier laboratories.

What You'll Do
Deploy machine learning models in production.

Who You Are
Experience training deep learning models with PyTorch.

Education
Bachelor's degree in a technical field is required.

Bonus Points If You Have
Experience with adversarial testing.
""")

        self.assertEqual(
            [(item["group"], item["text"]) for item in criteria],
            [
                ("responsibilities", "Deploy machine learning models in production."),
                ("core", "Experience training deep learning models with PyTorch."),
                ("required", "Bachelor's degree in a technical field is required."),
                ("bonus", "Experience with adversarial testing."),
            ],
        )

    def test_job_parser_uses_core_fallback_without_headings(self):
        criteria = parse_job_criteria(
            "Build production machine learning systems. Develop models using Python and PyTorch."
        )

        self.assertEqual([item["group"] for item in criteria], ["core", "core"])

    def test_job_parser_accepts_about_the_job_as_a_content_wrapper(self):
        criteria = parse_job_criteria(
            "About the job\nBuild production machine learning systems with Python."
        )

        self.assertEqual(criteria, [{
            "group": "core",
            "text": "Build production machine learning systems with Python.",
        }])

    def test_job_parser_falls_back_for_legacy_flattened_descriptions_starting_with_about(self):
        criteria = parse_job_criteria(
            "About Example AI and its products. Build production machine learning systems with Python."
        )

        self.assertEqual([item["group"] for item in criteria], ["core", "core"])

    def test_resume_parser_keeps_bullets_as_separate_evidence(self):
        evidence = parse_resume_evidence("""
EXPERIENCE
Machine Learning Engineer
- Built production ML systems with Python.
- Deployed PyTorch models on AWS.
TECHNICAL SKILLS
Python, PyTorch, AWS
""")

        self.assertEqual(
            [item["text"] for item in evidence],
            [
                "Machine Learning Engineer",
                "Built production ML systems with Python.",
                "Deployed PyTorch models on AWS.",
                "Python, PyTorch, AWS",
            ],
        )
        self.assertEqual([item["section"] for item in evidence], ["experience"] * 3 + ["skills"])

    def test_resume_parser_does_not_attach_the_next_role_to_a_finished_bullet(self):
        evidence = parse_resume_evidence("""
EXPERIENCE
- Deployed machine learning models in production.
Research Engineer
- Built research prototypes with PyTorch.
""")

        self.assertEqual(
            [item["text"] for item in evidence],
            [
                "Deployed machine learning models in production.",
                "Research Engineer",
                "Built research prototypes with PyTorch.",
            ],
        )

    def test_bonus_criteria_can_help_but_do_not_penalize_base_score(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ScoringStore(Path(directory) / "state.sqlite3")
            store.initialize()
            scorer = ResumeScorer(
                store,
                FakeEmbedder(),
                scorer_version="v2",
                pdf_extractor=lambda _path: "EXPERIENCE\n- Built machine learning systems using Python.",
                file_hasher=lambda _path: "resume-hash",
            )
            without_bonus = self.scoring_input() | {
                "job_description": "Who You Are\nBuild machine learning systems using Python.",
                "job_description_hash": "without-bonus",
            }
            with_unmatched_bonus = self.scoring_input() | {
                "job_description": (
                    "Who You Are\nBuild machine learning systems using Python.\n"
                    "Bonus Points\nManage retail print advertising."
                ),
                "job_description_hash": "with-bonus",
            }

            base = scorer.score(without_bonus)
            bonus = scorer.score(with_unmatched_bonus)

        self.assertGreaterEqual(bonus["score"], base["score"])
        self.assertIn("bonus", bonus["category_breakdown"])


if __name__ == "__main__":
    unittest.main()
