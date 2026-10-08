FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    FlagEmbedding==1.4.2 \
    mlflow==3.16.0 \
    "pyarrow>=21.0" \
    "scikit-learn>=1.7" \
    sentence-transformers==5.1.0 \
    "transformers>=4.57.1,<4.58"

WORKDIR /app
COPY service /app/service
COPY ml/src /app/ml/src
COPY ml/runtime /app/ml/runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/ml/src:/app
ENV RESUMES_DIR=/resumes

CMD ["python", "-m", "service.scoring_engine"]
