.PHONY: install run lint test clean

install:
	uv sync --extra dev

run:
	uv run streamlit run app/app.py

lint:
	uv run ruff check --fix . && uv run ruff format .

test:
	uv run pytest tests/ -v

clean:
	rm -rf .venv __pycache__ .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
