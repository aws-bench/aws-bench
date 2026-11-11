.PHONY: ready check lint format format-check test typecheck fix registry-bump docs docs-serve integration

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
# Real end-to-end acceptance tests against a dedicated test account. Never part of
# test/check/ready. Costs money and creates real AWS resources; opt-in only.
integration:
	@if [ "$$AWS_BENCH_INTEG" != "1" ] || [ -z "$$AWS_BENCH_INTEG_ACCOUNT_CONFIG" ]; then \
		echo "make integration runs REAL AWS resources in a dedicated test account."; \
		echo "Usage:"; \
		echo "  AWS_BENCH_INTEG=1 \\"; \
		echo "  AWS_BENCH_INTEG_ACCOUNT_CONFIG=/path/to/accounts.yaml \\"; \
		echo "  [AWS_BENCH_INTEG_DATASET=aws-bench-quickstart | AWS_BENCH_INTEG_DATASET_PATH=/path/to/aws-bench-datasets [AWS_BENCH_INTEG_SCENARIO=ec2-multiregion]] \\"; \
		echo "  [AWS_BENCH_INTEG_TASKS='*task-a,*task-b'] [AWS_BENCH_INTEG_HOME_LINKS=dir1,dir2] \\"; \
		echo "  make integration"; \
		echo "AWS credentials for the account config's runner role must be in the environment,"; \
		echo "and AWS_BEARER_TOKEN_BEDROCK must be set: eval \$$(uv run aws-bench env creds --eval)"; \
		echo "The env name is the account config's 'name'."; \
		echo "See docs/integration-tests.md for account setup and all variables."; \
		exit 2; \
	fi
	uv run pytest tests/integration -m integration --no-cov --timeout=0 -p no:cacheprovider -v
