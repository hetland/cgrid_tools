# cgrid-tools Makefile
#
# Commands run in the `remora-grid-tools` conda environment by default.
# Override CONDA_ENV, or use RUN= to use an already active Python environment.

CONDA_ENV ?= remora-grid-tools
CONDA_RUN := conda run -n $(CONDA_ENV)
RUN := $(CONDA_RUN)

PY_SOURCES := src tests examples docs/conf.py

.PHONY: help test perf lint format check examples install-dev clean docs docs-clean docs-serve

help:
	@echo "Available targets:"
	@echo "  make test        Run the default test suite"
	@echo "  make perf        Run the opt-in performance tests (RUN_PERFORMANCE=1)"
	@echo "  make lint        Run ruff (style + bug checks)"
	@echo "  make format      Apply black + isort in-place"
	@echo "  make check       Run format check (no writes) + lint + test"
	@echo "  make examples    Execute every script under examples/"
	@echo "  make install-dev Install the package with dev extras (editable)"
	@echo "  make clean       Remove __pycache__, .pytest_cache, build artifacts"
	@echo "  make docs        Build HTML documentation with Sphinx"
	@echo "  make docs-clean  Remove docs/_build"
	@echo "  make docs-serve  Serve docs locally on port 8765"

test:
	$(RUN) pytest -q

perf:
	RUN_PERFORMANCE=1 $(RUN) pytest -q

lint:
	$(RUN) ruff check .

format:
	$(RUN) isort $(PY_SOURCES)
	$(RUN) black $(PY_SOURCES)

check:
	$(RUN) black --check $(PY_SOURCES)
	$(RUN) isort --check-only $(PY_SOURCES)
	$(RUN) ruff check .
	$(RUN) pytest -q

examples:
	@for script in examples/*.py; do \
		echo "=== Running $$script ==="; \
		$(RUN) python $$script || exit $$?; \
	done

install-dev:
	$(RUN) pip install -e ".[dev,geo]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info

docs:
	$(RUN) sphinx-build -W --keep-going -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build

docs-serve:
	$(RUN) python -m http.server --directory docs/_build/html 8765
