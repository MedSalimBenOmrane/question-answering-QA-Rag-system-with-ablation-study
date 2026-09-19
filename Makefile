.PHONY: install test clean-data index run

install:
	uv sync

test:
	uv run pytest

clean-data:
	uv run python data_cleaning/clean_corpus.py

index:
	uv run python -m src.cli index

run:
	uv run streamlit run src/ui/streamlit_app.py
