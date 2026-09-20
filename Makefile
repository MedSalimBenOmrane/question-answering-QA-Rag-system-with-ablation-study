.PHONY: install test clean-data index run retrieval-eval generation-eval ablation

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

retrieval-eval:
	uv run python -m eval.retrieval_eval

generation-eval:
	uv run python -m eval.generation_eval

ablation:
	uv run python -m eval.run_ablation
