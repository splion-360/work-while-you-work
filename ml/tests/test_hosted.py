import hashlib
import unittest

from resume_jd_scoring.hosted import HostedScoringRuntime, score_input_fingerprint


class FakeScorer:
    def score_texts(self, encoder, resume_text, jd_text, *, pooling_device, max_length):
        assert encoder == "encoder"
        assert resume_text == "resume text"
        assert jd_text == "job description"
        assert pooling_device == "cuda:0"
        assert max_length == 8192
        return {
            "score": 76.24,
            "probability": 0.7624,
            "label": "Good Fit",
            "decision_threshold": 0.394,
        }


def scoring_payload():
    scoring_input = {
        "application_key": "app-1",
        "resume_text": "resume text",
        "resume_content_hash": "resume-hash",
        "job_description": "job description",
        "job_description_hash": hashlib.sha256(b"job description").hexdigest(),
    }
    return {
        "scoring_input": scoring_input,
        "input_fingerprint": score_input_fingerprint(scoring_input, "scorer-v1", "model-v1"),
    }


def runtime():
    return HostedScoringRuntime(
        FakeScorer(),
        "encoder",
        scorer_version="scorer-v1",
        embedding_model="model-v1",
        max_length=8192,
        provenance={"hf_model_revision": "commit-1"},
    )


class HostedScoringRuntimeTests(unittest.TestCase):
    def test_returns_compatible_score_contract(self):
        result = runtime().score(scoring_payload())

        self.assertEqual(result["score"], 76.2)
        self.assertEqual(result["band"], "Strong")
        self.assertEqual(result["fit_label"], "Good Fit")
        self.assertEqual(result["input_fingerprint"], scoring_payload()["input_fingerprint"])
        self.assertGreaterEqual(result["inference_ms"], 0)
        self.assertEqual(result["hf_model_revision"], "commit-1")

    def test_rejects_local_resume_paths(self):
        payload = scoring_payload()
        payload["scoring_input"]["resume_path"] = "/resumes/private.pdf"

        with self.assertRaisesRegex(ValueError, "resume path"):
            runtime().score(payload)

    def test_rejects_changed_job_description(self):
        payload = scoring_payload()
        payload["scoring_input"]["job_description"] = "changed"

        with self.assertRaisesRegex(ValueError, "content hash"):
            runtime().score(payload)


if __name__ == "__main__":
    unittest.main()
