FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.16.0

WORKDIR /app

ENV PYTHONUNBUFFERED=1

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
