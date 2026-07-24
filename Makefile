.PHONY: ready check lint format format-check test typecheck fix registry-bump docs docs-serve

# Path to a local aws-bench-datasets checkout (override with DATASETS_PATH=...)
DATASETS_PATH ?= ../aws-bench-datasets

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest --cov

check: lint format-check typecheck test

docs:
	uv run --group docs mkdocs build --strict

docs-serve:
	uv run --group docs mkdocs serve

fix:
	uv run ruff check --fix .
	uv run ruff format .

ready: fix check

registry-bump:
	uv run scripts/update_registry.py \
		--datasets-path $(DATASETS_PATH) \
		--add-unified \
		--output registry.json
