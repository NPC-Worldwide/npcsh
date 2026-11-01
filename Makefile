.PHONY: help docs docs-build docs-serve install-hooks

help:
	@echo "targets:"
	@echo "  docs          build + serve docs locally on :8000"
	@echo "  docs-build    build docs into ./site (verify before push)"
	@echo "  docs-serve    serve docs locally"
	@echo "  install-hooks install pre-commit hooks"

docs-build:
	python3 -m pip install -q -r docs/requirements.txt
	mkdocs build --strict

docs-serve:
	mkdocs serve -a 127.0.0.1:8000

docs: docs-build
	mkdocs serve -a 127.0.0.1:8000

install-hooks:
	@cp scripts/pre-commit-export-skills .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "installed pre-commit hook"
