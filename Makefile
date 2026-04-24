.PHONY: install test lint format type-check clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=engram --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy engram/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
