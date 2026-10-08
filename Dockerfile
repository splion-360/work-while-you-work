FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY service /app/service

ENV PYTHONUNBUFFERED=1
ENV RESUMES_DIR=/resumes
EXPOSE 8765

CMD ["python", "service/server.py"]
