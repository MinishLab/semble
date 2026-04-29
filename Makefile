.PHONY: help install test lint typecheck fix pre-commit

help:
	@echo "Available targets:"
	@echo "  install     Install all dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run ruff linter"
	@echo "  typecheck   Run mypy"
	@echo "  fix         Auto-fix lint issues"
	@echo "  pre-commit  Run all pre-commit hooks"

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest

test-no-git:
	uv run pytest --ignore=tests/test_git.py

lint:
	uv run ruff check src/ tests/
	uv run pydoclint src/

typecheck:
	uv run mypy src/

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

pre-commit:
	uv run pre-commit run --all-files
