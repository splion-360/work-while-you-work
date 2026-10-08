import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from service.scoring_contract import (
    ScoringError,
    extract_pdf_text,
    file_hash,
    role_company_key,
    score_band,
    score_cache_key,
    score_input_fingerprint,
)


class ScoringContractTests(unittest.TestCase):
    def scoring_input(self):
        return {
            "application_key": "app-1",
            "resume_content_hash": "resume-hash",
            "job_description_hash": "job-hash",
            "role_company_key": "company-role",
        }

    def test_hashes_resume_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "resume.pdf"
            resume.write_bytes(b"resume contents")
            result = file_hash(resume)

        self.assertEqual(result, hashlib.sha256(b"resume contents").hexdigest())

    @patch("service.scoring_contract.subprocess.run")
    def test_extracts_normalized_pdf_text(self, run):
        run.return_value = Mock(stdout="First line\n\n Second line \n")

        result = extract_pdf_text("resume.pdf")

        self.assertEqual(result, "First line\nSecond line")
        run.assert_called_once_with(
            ["pdftotext", "resume.pdf", "-"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )

    @patch("service.scoring_contract.subprocess.run")
    def test_wraps_pdf_extraction_failures(self, run):
        run.side_effect = subprocess.CalledProcessError(1, ["pdftotext"])

        with self.assertRaisesRegex(ScoringError, "Could not extract resume text"):
            extract_pdf_text("resume.pdf")

    def test_score_identity_includes_deployment_revision(self):
        first = score_input_fingerprint(self.scoring_input(), "v1", "model", "release-1")
        second = score_input_fingerprint(self.scoring_input(), "v1", "model", "release-2")

        self.assertNotEqual(first, second)

    def test_comparison_cache_ignores_application_and_job_description_identity(self):
        first = self.scoring_input()
        second = {
            **first,
            "application_key": "app-2",
            "job_description_hash": "different-job",
        }

        self.assertEqual(
            score_cache_key(first, "v1", "model", "release"),
            score_cache_key(second, "v1", "model", "release"),
        )

    def test_role_company_identity_is_normalized(self):
        self.assertEqual(
            role_company_key(" Example AI ", "ML  Engineer"),
            role_company_key("example ai", "ml engineer"),
        )

    def test_score_bands_use_product_thresholds(self):
        self.assertEqual(score_band(54.9), "Weak")
        self.assertEqual(score_band(55), "Moderate")
        self.assertEqual(score_band(75), "Strong")


if __name__ == "__main__":
    unittest.main()
