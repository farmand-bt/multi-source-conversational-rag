.PHONY: install run lint clean

# On Windows (Git Bash), use Scripts/activate instead of bin/activate
VENV_ACTIVATE := venv/Scripts/activate

install:
	python -m venv venv
	. $(VENV_ACTIVATE) && pip install -e ".[dev]"

run:
	. $(VENV_ACTIVATE) && streamlit run app/app.py

lint:
	. $(VENV_ACTIVATE) && ruff check --fix . && ruff format .

clean:
	rm -rf venv __pycache__ .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
