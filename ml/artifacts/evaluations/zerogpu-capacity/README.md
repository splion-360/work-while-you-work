# ZeroGPU Capacity and Rollback Evaluation

Status: concluded on 2026-09-08.

## Decision

Use the private Hugging Face ZeroGPU Space for normal scoring. Keep MLflow model
versions and immutable Hugging Face releases as the rollback chain. Version 2
remains the MLflow `champion` and the active production deployment after this
drill.

The measured warm request latency is suitable for the extension workflow. The
free-tier capacity estimates are theoretical model-time estimates, not guaranteed
request allowances. Hugging Face's quota accounting and queue availability remain
the operational limits.

## Deployment Identity

| Field | Active value |
| --- | --- |
| MLflow model | `resume-jd-binary-scorer` |
| MLflow version | `2` |
| MLflow registration run | `a86befd84e8445d18bf3d68fc9ea29e6` |
| HF model release | `mlflow-v2-10cabd579086` |
| HF model revision | `8b062ade79600eed631cc6c48dc6f4b683fcd1c9` |
| Bundle fingerprint | `10cabd579086e89aef2ae3a8ab041f656de6f1e886e3a1c2494cfddfab416dee` |
| BGE-M3 revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Space production commit after restore | `d4a7392b00b68a8a0891a76559aa776d14ecf317` |

Every accepted benchmark response returned the expected MLflow version, HF
revision, and model fingerprint. The benchmark used generated text and retained no
resume or job-description content.

## Method

| Workload | Resume characters | JD characters | Cold samples | Warm samples |
| --- | ---: | ---: | ---: | ---: |
| Typical | 6,000 | 6,000 | 1 | 5 |
| Large | 16,000 | 20,000 | 1 | 5 |
| Maximum | 32,000 | 50,000 | 1 | 5 |

Each cold sample followed a Space restart and a transition back to `RUNNING`.
Each warm series immediately followed its cold request. `Wall` includes network,
Gradio queue, and inference time. `GPU` is the elapsed scorer runtime reported from
inside the decorated GPU function.

The five-sample p95 values are interpolated sample statistics, not stable
population estimates. In particular, the typical workload retained two warm-up
outliers rather than excluding them.

## Results

| Workload | Cold wall | Cold GPU | Warm wall p50 | Warm wall p95 | Warm GPU p50 | Warm GPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Typical | 2.853 s | 1.063 s | 0.903 s | 5.218 s | 0.062 s | 1.206 s |
| Large | 2.492 s | 1.266 s | 0.904 s | 1.028 s | 0.135 s | 0.141 s |
| Maximum | 3.507 s | 1.405 s | 1.130 s | 1.278 s | 0.356 s | 0.396 s |

The typical warm series was bimodal: two calls reported approximately one second
of scorer runtime, while three reported approximately 0.06 seconds. The p50
therefore describes the steady fast path, while p95 preserves the observed warm-up
cost. Large and maximum workloads were stable after the cold request.

## Capacity Estimate

Hugging Face currently documents 5 included ZeroGPU minutes per day for Free
accounts and 40 minutes per day for PRO accounts. PRO receives higher queue
priority and can extend usage with prepaid credits. The quota resets 24 hours after
the first use in a quota window.

Sources:

- [ZeroGPU usage tiers](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Spaces as API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)

The following estimates divide 300 or 2,400 seconds by measured warm GPU runtime.
The p50 column is the observed median-rate estimate; p95 is the conservative
estimate from this sample.

| Workload | Free at p50 | Free at p95 | PRO at p50 | PRO at p95 |
| --- | ---: | ---: | ---: | ---: |
| Typical | 4,878 | 248 | 39,024 | 1,989 |
| Large | 2,223 | 2,124 | 17,790 | 16,999 |
| Maximum | 843 | 757 | 6,749 | 6,063 |

These are upper-bound throughput estimates based on scorer-reported GPU time.
They do not include any additional effective-duration accounting Hugging Face may
apply, and they do not guarantee queue availability. The account quota display is
authoritative. Wall-clock throughput will also be lower because each request spends
time outside the GPU function.

## Benchmark

```bash
cd "$(git rev-parse --show-toplevel)"
set -a
source .env
set +a
ml/.venv/bin/python deploy/benchmark_huggingface_space.py \
  --model-revision 8b062ade79600eed631cc6c48dc6f4b683fcd1c9 \
  --model-fingerprint 10cabd579086e89aef2ae3a8ab041f656de6f1e886e3a1c2494cfddfab416dee \
  --mlflow-model-version 2 \
  --warm-runs 5 \
  --output /tmp/zerogpu_benchmark.json
```

## Rollback Drill

The rollback target was MLflow version 1:

| Field | Rollback value |
| --- | --- |
| MLflow version | `1` |
| Registration run | `650b28c4747f45cab31d279959832730` |
| HF model release | `mlflow-v1-a2b101596a36` |
| HF model revision | `e4fa8191504fa744ec6c3dff654f4b397d0bf266` |
| Bundle fingerprint | `a2b101596a36ec8bd1c62887b67c4a6e951da4ecbdc19140c5b1c989cfe58502` |
| Space rollback commit | `c85e149e05810e49470d3a5e5ffb8756c9b12f30` |

The rollback smoke returned score `86.6`, `Good Fit`, MLflow version `1`, the
version-1 registration run, and fingerprint `a2b101...`. This proves that the prior
pinned artifact was active.

Version 1 and version 2 contain the same classifier SHA and encoder fingerprint.
Their bundle fingerprints differ because version 2 adds parent-run lineage to the
manifest. This drill proves deployment rollback mechanics and provenance switching;
it does not prove recovery from a behavioral model regression.

After the rollback proof, version 2 was restored as `champion`, redeployed at Space
commit `d4a7392...`, and smoke-tested. The final smoke returned score `86.6`,
MLflow version `2`, registration run `a86bef...`, and fingerprint `10cabd...`.

```bash
cd "$(git rev-parse --show-toplevel)"
set -a
source .env
set +a
ml/.venv/bin/python deploy/set_mlflow_champion.py 1
ml/.venv/bin/python deploy/release_mlflow_champion.py
ml/.venv/bin/python deploy/publish_huggingface_space.py \
  --model-revision mlflow-v1-a2b101596a36 \
  --model-release-path releases/mlflow-v1-a2b101596a36
ml/.venv/bin/python deploy/smoke_huggingface_space.py \
  --model-revision e4fa8191504fa744ec6c3dff654f4b397d0bf266 \
  --model-fingerprint a2b101596a36ec8bd1c62887b67c4a6e951da4ecbdc19140c5b1c989cfe58502 \
  --mlflow-model-version 1
```

```bash
cd "$(git rev-parse --show-toplevel)"
set -a
source .env
set +a
ml/.venv/bin/python deploy/set_mlflow_champion.py 2
ml/.venv/bin/python deploy/publish_huggingface_space.py
ml/.venv/bin/python deploy/smoke_huggingface_space.py \
  --model-revision 8b062ade79600eed631cc6c48dc6f4b683fcd1c9 \
  --model-fingerprint 10cabd579086e89aef2ae3a8ab041f656de6f1e886e3a1c2494cfddfab416dee \
  --mlflow-model-version 2
```

## Validation

- All 18 benchmark predictions returned the pinned version-2 provenance.
- The rollback smoke returned the pinned version-1 provenance.
- The restore smoke returned the pinned version-2 provenance.
- No benchmark document text or credentials were written to the repository.
