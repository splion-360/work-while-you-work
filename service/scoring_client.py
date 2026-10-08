import hashlib
import json
import time
import urllib.error
import urllib.request

try:
    from service.scoring_contract import (
        ScoringError,
        extract_pdf_text,
        file_hash,
        score_input_fingerprint,
    )
except ModuleNotFoundError:
    from scoring_contract import ScoringError, extract_pdf_text, file_hash, score_input_fingerprint


class HostedScoringUnavailable(ScoringError):
    pass


class BaseScoringClient:
    def __init__(
        self,
        scorer_version,
        embedding_model,
        timeout=180,
        fingerprint_revision="",
        pdf_extractor=extract_pdf_text,
        file_hasher=file_hash,
    ):
        self.scorer_version = scorer_version
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.fingerprint_revision = fingerprint_revision
        self.pdf_extractor = pdf_extractor
        self.file_hasher = file_hasher

    def _remote_input(self, scoring_input):
        resume_path = scoring_input.get("resume_path")
        if not resume_path:
            raise ScoringError("Scoring input does not identify a resume PDF")
        try:
            current_hash = self.file_hasher(resume_path)
        except OSError as error:
            raise ScoringError(f"Could not read the captured resume PDF: {error}") from error
        if current_hash != scoring_input["resume_content_hash"]:
            raise ScoringError("Resume PDF no longer matches the captured content hash")
        resume_text = self.pdf_extractor(resume_path)
        if not resume_text.strip():
            raise ScoringError("Resume PDF produced no text")
        remote_input = dict(scoring_input)
        remote_input.pop("resume_path", None)
        remote_input["resume_text"] = resume_text
        return remote_input

    def _request_payload(self, scoring_input):
        fingerprint = score_input_fingerprint(
            scoring_input,
            self.scorer_version,
            self.embedding_model,
            self.fingerprint_revision,
        )
        return {
            "scoring_input": self._remote_input(scoring_input),
            "input_fingerprint": fingerprint,
            "deployment_revision": self.fingerprint_revision,
        }, fingerprint

    def _validate_result(self, result, expected_fingerprint):
        if result.get("input_fingerprint") != expected_fingerprint:
            raise ScoringError("Scoring engine returned a mismatched input fingerprint")
        if result.get("scorer_version") != self.scorer_version:
            raise ScoringError("Scoring engine returned a mismatched scorer version")
        if result.get("embedding_model") != self.embedding_model:
            raise ScoringError("Scoring engine returned a mismatched embedding model")
        return result


class ScoringEngineClient(BaseScoringClient):
    def __init__(self, base_url, scorer_version, embedding_model, timeout=180, **kwargs):
        super().__init__(scorer_version, embedding_model, timeout, **kwargs)
        self.base_url = base_url.rstrip("/")

    def score(self, scoring_input):
        payload, expected_fingerprint = self._request_payload(scoring_input)
        request = urllib.request.Request(
            f"{self.base_url}/score",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("error", str(error))
            except (ValueError, AttributeError):
                detail = str(error)
            raise ScoringError(f"Scoring engine rejected the request: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ScoringError(f"Scoring engine request failed: {error}") from error
        return self._validate_result(result, expected_fingerprint)


class HuggingFaceSpaceClient(BaseScoringClient):
    def __init__(
        self,
        base_url,
        token,
        scorer_version,
        embedding_model,
        *,
        expected_model_revision,
        expected_model_fingerprint,
        expected_mlflow_model_version,
        telemetry=None,
        timeout=180,
        **kwargs,
    ):
        super().__init__(
            scorer_version,
            embedding_model,
            timeout,
            fingerprint_revision=expected_model_revision,
            **kwargs,
        )
        if not token:
            raise ValueError("HF_TOKEN is required for hosted scoring")
        required = {
            "HF_MODEL_REVISION": expected_model_revision,
            "HF_MODEL_FINGERPRINT": expected_model_fingerprint,
            "MLFLOW_MODEL_VERSION": expected_mlflow_model_version,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Hosted scoring requires {', '.join(missing)}")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.expected_model_revision = expected_model_revision
        self.expected_model_fingerprint = expected_model_fingerprint
        self.expected_mlflow_model_version = str(expected_mlflow_model_version)
        self.telemetry = telemetry

    def _request(self, url, *, data=None, method="GET"):
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode().strip()
            except (OSError, UnicodeDecodeError):
                detail = str(error)
            message = f"Hugging Face Space rejected the request ({error.code}): {detail}"
            error_type = (
                HostedScoringUnavailable
                if error.code in {408, 429, 500, 502, 503, 504}
                else ScoringError
            )
            raise error_type(message) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise HostedScoringUnavailable(
                f"Hugging Face Space request failed: {error}"
            ) from error

    def _completed_result(self, response):
        event = None
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line.partition(":")[2].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = json.loads(line.partition(":")[2].strip())
            if event == "error":
                raise ScoringError(f"Hugging Face Space scoring failed: {data}")
            if event == "complete":
                if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                    raise ScoringError("Hugging Face Space returned an invalid score payload")
                return data[0]
        raise ScoringError("Hugging Face Space ended without a completed score")

    def _validate_result(self, result, expected_fingerprint):
        super()._validate_result(result, expected_fingerprint)
        checks = {
            "hf_model_revision": self.expected_model_revision,
            "model_fingerprint": self.expected_model_fingerprint,
            "mlflow_model_version": self.expected_mlflow_model_version,
        }
        for field, expected in checks.items():
            if str(result.get(field, "")) != expected:
                raise ScoringError(f"Hugging Face Space returned mismatched {field}")
        result["deployment_revision"] = self.expected_model_revision
        return result

    def score(self, scoring_input):
        started = time.perf_counter()
        payload, expected_fingerprint = self._request_payload(scoring_input)
        try:
            submit_url = f"{self.base_url}/gradio_api/call/score"
            with self._request(
                submit_url,
                data=json.dumps({"data": [payload]}).encode(),
                method="POST",
            ) as response:
                submission = json.load(response)
            event_id = submission.get("event_id")
            if not event_id:
                raise ScoringError("Hugging Face Space did not return a queue event ID")
            with self._request(f"{submit_url}/{event_id}") as response:
                result = self._completed_result(response)
            result = self._validate_result(result, expected_fingerprint)
        except Exception as error:
            self._submit_telemetry(
                scoring_input,
                expected_fingerprint,
                started,
                outcome="failed",
                status="FAILED",
                error_type=type(error).__name__,
            )
            raise
        self._submit_telemetry(
            scoring_input,
            expected_fingerprint,
            started,
            outcome="completed",
            metrics={
                "score": result["score"],
                "fit_probability": result.get("fit_probability", 0),
                "remote_inference_ms": result.get("inference_ms", 0),
            },
        )
        return result

    def _submit_telemetry(
        self,
        scoring_input,
        input_fingerprint,
        started,
        *,
        outcome,
        status="FINISHED",
        error_type="",
        metrics=None,
    ):
        if self.telemetry is None:
            return
        tags = {
            "outcome": outcome,
            "application_key_hash": hashlib.sha256(
                scoring_input["application_key"].encode()
            ).hexdigest(),
            "input_fingerprint": input_fingerprint,
            "resume_content_hash": scoring_input["resume_content_hash"],
            "job_description_hash": scoring_input["job_description_hash"],
            "scorer_version": self.scorer_version,
            "embedding_model": self.embedding_model,
            "model_revision": self.expected_model_revision,
            "model_fingerprint": self.expected_model_fingerprint,
            "mlflow_model_version": self.expected_mlflow_model_version,
        }
        if error_type:
            tags["error_type"] = error_type
        values = {"gateway_latency_ms": (time.perf_counter() - started) * 1000}
        values.update(metrics or {})
        self.telemetry.submit({"tags": tags, "metrics": values, "status": status})


def create_scoring_client(settings):
    if settings.scoring_backend == "huggingface":
        try:
            from service.mlflow_telemetry import get_mlflow_telemetry
        except ModuleNotFoundError:
            from mlflow_telemetry import get_mlflow_telemetry
        return HuggingFaceSpaceClient(
            settings.hf_space_url,
            settings.hf_token,
            settings.scorer_version,
            settings.embedding_model,
            expected_model_revision=settings.hf_model_revision,
            expected_model_fingerprint=settings.hf_model_fingerprint,
            expected_mlflow_model_version=settings.mlflow_model_version,
            telemetry=get_mlflow_telemetry(
                settings.mlflow_tracking_uri, settings.mlflow_inference_experiment
            ),
            timeout=settings.scoring_engine_timeout_seconds,
        )
    return ScoringEngineClient(
        settings.scoring_engine_url,
        settings.scorer_version,
        settings.embedding_model,
        settings.scoring_engine_timeout_seconds,
    )
