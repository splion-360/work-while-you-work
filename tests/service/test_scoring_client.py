import io
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from service.scoring_contract import ScoringError, score_input_fingerprint
from service.scoring_client import (
    HuggingFaceSpaceClient,
    ScoringEngineClient,
    create_scoring_client,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ScoringEngineClientTests(unittest.TestCase):
    def setUp(self):
        self.scoring_input = {
            "application_key": "app-1",
            "resume_path": "/resumes/resume.pdf",
            "resume_content_hash": hashlib.sha256(b"resume-pdf").hexdigest(),
            "job_description_hash": "jd-hash",
        }
        self.client = ScoringEngineClient(
            "http://engine",
            "scorer-v1",
            "model-v1",
            pdf_extractor=lambda _path: "Extracted resume text",
            file_hasher=lambda _path: hashlib.sha256(b"resume-pdf").hexdigest(),
        )

    def response(self, **overrides):
        fingerprint = score_input_fingerprint(self.scoring_input, "scorer-v1", "model-v1")
        return {
            "input_fingerprint": fingerprint,
            "scorer_version": "scorer-v1",
            "embedding_model": "model-v1",
            "score": 72.5,
            "band": "Moderate",
            **overrides,
        }

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_returns_result_from_engine(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps(self.response()).encode())
        result = self.client.score(self.scoring_input)
        self.assertEqual(result["score"], 72.5)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://engine/score")
        remote_input = json.loads(request.data)["scoring_input"]
        self.assertNotIn("resume_path", remote_input)
        self.assertEqual(remote_input["resume_text"], "Extracted resume text")
        self.assertEqual(remote_input["resume_content_hash"], self.scoring_input["resume_content_hash"])

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_rejects_changed_resume_before_request(self, urlopen):
        client = ScoringEngineClient(
            "http://engine",
            "scorer-v1",
            "model-v1",
            pdf_extractor=lambda _path: "Extracted resume text",
            file_hasher=lambda _path: "changed-hash",
        )

        with self.assertRaisesRegex(ScoringError, "no longer matches"):
            client.score(self.scoring_input)

        urlopen.assert_not_called()

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_rejects_mismatched_model_identity(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps(self.response(embedding_model="wrong")).encode())
        with self.assertRaisesRegex(ScoringError, "embedding model"):
            self.client.score(self.scoring_input)

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_rejects_mismatched_fingerprint(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps(self.response(input_fingerprint="wrong")).encode())
        with self.assertRaisesRegex(ScoringError, "input fingerprint"):
            self.client.score(self.scoring_input)


class HuggingFaceSpaceClientTests(unittest.TestCase):
    def setUp(self):
        self.scoring_input = {
            "application_key": "app-1",
            "resume_path": "/resumes/resume.pdf",
            "resume_content_hash": hashlib.sha256(b"resume-pdf").hexdigest(),
            "job_description_hash": "jd-hash",
        }
        self.client = HuggingFaceSpaceClient(
            "https://private-space.example",
            "secret-token",
            "scorer-v1",
            "model-v1",
            expected_model_revision="hf-revision",
            expected_model_fingerprint="bundle-fingerprint",
            expected_mlflow_model_version="2",
            pdf_extractor=lambda _path: "Extracted resume text",
            file_hasher=lambda _path: hashlib.sha256(b"resume-pdf").hexdigest(),
        )

    def response(self, **overrides):
        fingerprint = score_input_fingerprint(
            self.scoring_input, "scorer-v1", "model-v1", "hf-revision"
        )
        return {
            "input_fingerprint": fingerprint,
            "scorer_version": "scorer-v1",
            "embedding_model": "model-v1",
            "hf_model_revision": "hf-revision",
            "model_fingerprint": "bundle-fingerprint",
            "mlflow_model_version": "2",
            "score": 72.5,
            "band": "Moderate",
            **overrides,
        }

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_scores_through_private_gradio_queue(self, urlopen):
        submitted = FakeResponse(json.dumps({"event_id": "event-1"}).encode())
        completed = FakeResponse(
            f"event: complete\ndata: {json.dumps([self.response()])}\n\n".encode()
        )
        urlopen.side_effect = [submitted, completed]

        result = self.client.score(self.scoring_input)

        self.assertEqual(result["score"], 72.5)
        submit_request = urlopen.call_args_list[0].args[0]
        result_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(
            submit_request.full_url,
            "https://private-space.example/gradio_api/call/score",
        )
        self.assertEqual(
            result_request.full_url,
            "https://private-space.example/gradio_api/call/score/event-1",
        )
        self.assertEqual(submit_request.get_header("Authorization"), "Bearer secret-token")
        submitted_payload = json.loads(submit_request.data)["data"][0]["scoring_input"]
        self.assertNotIn("resume_path", submitted_payload)
        self.assertNotIn("secret-token", submit_request.data.decode())

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_rejects_mismatched_release_provenance(self, urlopen):
        urlopen.side_effect = [
            FakeResponse(json.dumps({"event_id": "event-1"}).encode()),
            FakeResponse(
                f"event: complete\ndata: {json.dumps([self.response(model_fingerprint='wrong')])}\n\n".encode()
            ),
        ]
        with self.assertRaisesRegex(ScoringError, "model_fingerprint"):
            self.client.score(self.scoring_input)

    def test_requires_token_and_release_identity(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            HuggingFaceSpaceClient(
                "https://private-space.example",
                "",
                "scorer-v1",
                "model-v1",
                expected_model_revision="revision",
                expected_model_fingerprint="fingerprint",
                expected_mlflow_model_version="2",
            )

    def test_factory_selects_hosted_client(self):
        settings = SimpleNamespace(
            scoring_backend="huggingface",
            hf_space_url="https://private-space.example",
            hf_token="secret-token",
            scorer_version="scorer-v1",
            embedding_model="model-v1",
            hf_model_revision="hf-revision",
            hf_model_fingerprint="bundle-fingerprint",
            mlflow_model_version="2",
            scoring_engine_timeout_seconds=60,
            mlflow_tracking_uri="",
            mlflow_inference_experiment="production",
        )
        self.assertIsInstance(create_scoring_client(settings), HuggingFaceSpaceClient)

    @patch("service.scoring_client.urllib.request.urlopen")
    def test_telemetry_contains_hashes_but_not_document_text(self, urlopen):
        telemetry = Mock()
        self.client.telemetry = telemetry
        urlopen.side_effect = [
            FakeResponse(json.dumps({"event_id": "event-1"}).encode()),
            FakeResponse(
                f"event: complete\ndata: {json.dumps([self.response()])}\n\n".encode()
            ),
        ]

        self.client.score(self.scoring_input)

        event = telemetry.submit.call_args.args[0]
        serialized = json.dumps(event)
        self.assertNotIn("Extracted resume text", serialized)
        self.assertIn(self.scoring_input["resume_content_hash"], serialized)
        self.assertEqual(event["tags"]["outcome"], "completed")


if __name__ == "__main__":
    unittest.main()
