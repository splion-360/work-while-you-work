import json
import os
from pathlib import Path

import gradio as gr
import spaces
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download

from resume_jd_scoring.embeddings import model_fingerprint
from resume_jd_scoring.hosted import HostedScoringRuntime
from resume_jd_scoring.inference import BinaryResumeJDScorer


MODEL_ID = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")
BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", Path(__file__).parent / "model"))
SCORER_VERSION = os.getenv("SCORER_VERSION", "bge-m3-knrm-binary-v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-colbert")

manifest = json.loads((BUNDLE_PATH / "manifest.json").read_text(encoding="utf-8"))
deployment = json.loads((BUNDLE_PATH / "deployment.json").read_text(encoding="utf-8"))
if deployment["classifier_sha256"] != manifest["classifier"]["sha256"]:
    raise RuntimeError("deployed classifier does not match its release metadata")
if deployment["encoder_fingerprint"] != manifest["encoder"]["fingerprint"]:
    raise RuntimeError("deployed encoder does not match its release metadata")
MODEL_REVISION = deployment["bge_revision"]
model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
encoding = {
    "pipeline_version": 1,
    "model_name": "BAAI/bge-m3",
    "representation": "normalized_colbert_token_vectors",
    "max_length": int(manifest["encoder"]["max_length"]),
    "cache_dtype": "float16",
    "excluded_token": "cls",
}
fingerprint, _ = model_fingerprint(
    model_path, encoding, additional_files=("colbert_linear.pt",)
)
if fingerprint != manifest["encoder"]["fingerprint"]:
    raise RuntimeError("BGE-M3 files do not match the classifier bundle")

encoder = BGEM3FlagModel(
    str(model_path),
    use_fp16=True,
    devices=["cuda:0"],
    batch_size=1,
    passage_max_length=int(manifest["encoder"]["max_length"]),
    return_dense=False,
    return_sparse=False,
    return_colbert_vecs=True,
)
runtime = HostedScoringRuntime(
    BinaryResumeJDScorer.from_bundle(BUNDLE_PATH),
    encoder,
    scorer_version=SCORER_VERSION,
    embedding_model=EMBEDDING_MODEL,
    max_length=int(manifest["encoder"]["max_length"]),
    provenance={
        "mlflow_model_version": deployment["mlflow_model_version"],
        "mlflow_run_id": deployment["mlflow_registration_run_id"],
        "hf_model_revision": deployment["hf_release_revision"],
        "model_fingerprint": deployment["bundle_fingerprint"],
    },
)


@spaces.GPU(duration=60)
def score(payload):
    if payload.get("deployment_revision") != deployment["hf_release_revision"]:
        raise ValueError("requested model revision does not match this deployment")
    return runtime.score(payload)


with gr.Blocks() as demo:
    request = gr.JSON(label="Scoring request")
    response = gr.JSON(label="Scoring result")
    gr.Button("Score").click(score, inputs=request, outputs=response, api_name="score")


if __name__ == "__main__":
    demo.launch()
