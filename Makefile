.PHONY: install test index run

install:
	uv sync

test:
	uv run pytest

index:
	uv run python -m src.cli index

run:
	uv run streamlit run src/ui/streamlit_app.py
