PYTHON ?= python3
UV ?= uv

.PHONY: setup test test-service test-extension test-ml lint up down logs

setup:
	$(UV) sync --project ml --frozen

test: test-service test-extension test-ml

test-service:
	$(PYTHON) -m unittest discover -s tests/service -p 'test_*.py'

test-extension:
	node --test extension/*.test.js tests/extension/*.test.js

test-ml:
	$(UV) run --project ml pytest -q ml/tests

lint:
	$(UV) run --project ml ruff check ml/src ml/scripts ml/tests

up:
	docker compose up -d --build job-tracker scoring-worker mlflow

down:
	docker compose down

logs:
	docker compose logs -f job-tracker scoring-worker
