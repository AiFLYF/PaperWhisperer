.PHONY: help install dev test lint format check serve clean parity check-frontend

PYTHON ?= python
NODE ?= node
# vm.SourceTextModule needs this flag; without it every module fails to parse.
NODE_VM := $(NODE) --experimental-vm-modules

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

dev:  ## Install runtime + development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Run ruff checks
	$(PYTHON) -m ruff check .

format:  ## Auto-fix lintable issues
	$(PYTHON) -m ruff check --fix .

check-frontend:  ## Verify the front-end module split (needs no reference copy)
	$(NODE_VM) tools/check_js_syntax.mjs templates/static/js
	$(PYTHON) tools/check_frontend.py

check: lint test check-frontend  ## Lint + test + front-end structure

serve:  ## Run the development server
	$(PYTHON) web_app.py

parity:  ## One-off: diff the split front-end against the pre-split sources
	@echo "This compares against .parity/ originals, which are not committed."
	$(NODE_VM) .parity/check_js_syntax.mjs templates/static/js
	$(NODE_VM) .parity/check_frontend_parity.mjs .parity/orig_app.js templates/static/js
	$(NODE_VM) .parity/check_js_graph.mjs templates/static/js
	$(PYTHON) .parity/check_css_parity.py .parity/orig_style.css templates/static/css

clean:  ## Remove caches and runtime artifacts
	rm -rf .pytest_cache .ruff_cache .benchmarks
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
