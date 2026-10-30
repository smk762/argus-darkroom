UV ?= uv

.PHONY: help install lint format test build demo smoke clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install with test extras
	$(UV) venv
	$(UV) pip install -e ".[dev]"

lint:  ## Ruff check + format check
	$(UV)x ruff@0.15.16 check src/ tests/
	$(UV)x ruff@0.15.16 format --check src/ tests/

format:  ## Ruff autoformat + autofix
	$(UV)x ruff@0.15.16 format src/ tests/
	$(UV)x ruff@0.15.16 check --fix src/ tests/

test:  ## Run the test suite
	$(UV) run --no-sync pytest --tb=short -q

build:  ## Build sdist + wheel
	$(UV) build

demo:  ## Run the real server against fake backends (no GPU), on http://127.0.0.1:8106
	$(UV) run --no-sync python demo/run.py

smoke:  ## With `make demo` running: send every marker prompt and check the outcome
	$(UV) run --no-sync python demo/smoke.py

clean:  ## Remove build + cache artifacts
	rm -rf dist build .venv *.egg-info .pytest_cache .ruff_cache src/argus_darkroom/_version.py
