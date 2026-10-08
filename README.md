# work-while-you-work

I built this because my job-search workflow had split across browser tabs, spreadsheets, resume files, and a few ML notebooks. The project now does two things: it tracks applications from the browser and scores a selected resume against the job description before I apply.

The matching model is the part I care about most. It is not an LLM prompt or a cosine-similarity wrapper. It compares token-level BGE-M3 representations, summarizes the interaction matrix into a fixed feature vector, and runs a calibrated binary classifier trained on labeled resume/job pairs.

## How matching works

[![Application scoring and model lifecycle](assets/architecture.svg)](assets/architecture.drawio)

The rendered diagram links to the editable [draw.io source](assets/architecture.drawio).

### Why BGE-M3

[BGE-M3](https://huggingface.co/BAAI/bge-m3) is an embedding model from BAAI built for three retrieval modes: dense vectors, sparse lexical matching, and ColBERT-style multi-vector retrieval. It supports inputs up to 8,192 tokens and more than 100 languages. The three M's refer to multi-functionality, multi-linguality, and multi-granularity; the design is described in the [BGE-M3 paper](https://arxiv.org/abs/2402.03216).

This project uses only the multi-vector output. Instead of reducing a resume or job description to one vector, BGE-M3 produces a contextual vector for each token. That preserves local interactions such as a job requirement matching one specific resume phrase. The model then builds the full resume/job token-similarity matrix, applies 11 Gaussian kernels, and summarizes each kernel response with seven distribution statistics. The result is a 77-feature vector for a small linear classifier.

BGE-M3 remains frozen during training. The learned part is the `77 -> 2` classifier, followed by Platt calibration. This keeps the experiment focused on whether token-level interaction features improve resume/job matching rather than claiming to have trained a new foundation model.

The score represents the calibrated probability of a `Good Fit` label. The model bundle pins the classifier checksum, encoder fingerprint, calibration parameters, and decision threshold so the API cannot quietly load a different encoder or checkpoint.

The current production path runs the model in a private Hugging Face Space:

1. The browser extension captures the job details and selected resume.
2. The API stores a content-addressed scoring input and queues the work in SQLite.
3. A worker extracts text from the resume and sends text plus content hashes to the Space. Local file paths never leave the machine.
4. The Space verifies the requested deployment revision, scores the pair on a GPU, and returns the result with model provenance.
5. The worker validates the response fingerprint before publishing the score to Notion.

There is also a local GPU path backed by MLflow. Both deployment paths use the same runtime loader and verify the same model contract.

## MLflow

MLflow tracks the model lifecycle rather than replacing the scoring API:

- Training creates a parent run with nested runs for each fold and records parameters, epoch metrics, checkpoints, and evaluation artifacts.
- Calibration and held-out evaluation are attached to the originating runs, which keeps the selected checkpoint traceable to its split and metrics.
- The packaged classifier is registered as a versioned MLflow model. A `champion` alias identifies the bundle used by the optional local GPU engine.
- A release script downloads that registered bundle, adds immutable deployment metadata, and stages it for the private Hugging Face Space.
- Production inference events record latency, outcome, model version, and content fingerprints in a separate MLflow experiment. Resume text and job-description text are not logged.

The Hugging Face Space performs the hosted inference. MLflow provides experiment history, artifacts, model lineage, and local model promotion; it is not in the hosted request path.

## Evaluation

The checked-in evaluation record is [ml/artifacts/eval__bge_m3_row_distribution_binary.runme](ml/artifacts/eval__bge_m3_row_distribution_binary.runme). On 31,333 graph-disjoint outer-test pairs, the model reached:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.7868 |
| Accuracy at validation-selected thresholds | 0.7531 |
| Macro F1 at validation-selected thresholds | 0.6952 |
| Calibrated 10-bin ECE | 0.0320 |

These numbers describe ranking and binary fit prediction on this dataset. They do not prove that the model understands a hiring decision, and the source labels are collapsed from three classes into `No Fit` and `Good Fit`. The Runme record includes the split, calibration, and threshold-selection details.

## Repository map

```text
extension/  browser capture and application UI
service/    HTTP API, queue worker, Notion persistence, and scoring clients
ml/         dataset processing, model training, evaluation, and inference package
deploy/     Hugging Face and MLflow release scripts
```

Resume source files live in a separate repository. This repository only reads built PDFs from `../resumes` when the local stack runs.

## Local setup

Requirements:

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js for the extension tests
- Docker Compose for the application stack
- Notion and Hugging Face credentials for the live integrations

Install the ML environment and run the test suites:

```bash
make setup
make test
```

The application reads local secrets from `.env` and Docker-only overrides from `.env.docker`; both files are ignored by Git. With those configured, start the API, worker, and MLflow services:

```bash
make up
```

The API listens on `http://127.0.0.1:8765`, and MLflow listens on `http://127.0.0.1:5000`. The extension can then be loaded as an unpacked extension from `extension/`.

The local scoring engine is optional and requires an NVIDIA GPU plus a packaged model bundle. Start that profile with:

```bash
docker compose --profile local-scoring up -d --build
```
