import json
import os
from pathlib import Path

import gradio as gr
import spaces
from huggingface_hub import snapshot_download
from resume_jd_scoring.runtime import build_scoring_runtime, load_manifest

MODEL_ID = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")
BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", Path(__file__).parent / "model"))
SCORER_VERSION = os.getenv("SCORER_VERSION", "bge-m3-knrm-binary-v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-colbert")

manifest = load_manifest(BUNDLE_PATH)
deployment = json.loads((BUNDLE_PATH / "deployment.json").read_text(encoding="utf-8"))
if deployment["classifier_sha256"] != manifest["classifier"]["sha256"]:
    raise RuntimeError("deployed classifier does not match its release metadata")
if deployment["encoder_fingerprint"] != manifest["encoder"]["fingerprint"]:
    raise RuntimeError("deployed encoder does not match its release metadata")
MODEL_REVISION = deployment["bge_revision"]
model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
runtime = build_scoring_runtime(
    BUNDLE_PATH,
    model_path,
    scorer_version=SCORER_VERSION,
    embedding_model=EMBEDDING_MODEL,
    device="cuda:0",
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
