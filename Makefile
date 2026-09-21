.PHONY: help install test lint format clean-data index run retrieval-eval generation-eval ablation docker-build docker-up docker-down deploy-local push-ecr setup-aws clean logs health validate

help:
	@echo "RAG System - Makefile Commands"
	@echo ""
	@echo "Local Development:"
	@echo "  make install       - Install dependencies with uv"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting (Ruff)"
	@echo "  make format        - Format code (Ruff)"
	@echo ""
	@echo "Data & Index:"
	@echo "  make clean-data    - Clean corpus data"
	@echo "  make index         - Build vector index"
	@echo ""
	@echo "Run:"
	@echo "  make run           - Run Streamlit UI"
	@echo ""
	@echo "Evaluation:"
	@echo "  make retrieval-eval    - Run retrieval evaluation"
	@echo "  make generation-eval   - Run generation evaluation"
	@echo "  make ablation          - Run ablation study"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start services with docker-compose"
	@echo "  make docker-down   - Stop services"
	@echo "  make deploy-local  - Full local deployment (build + up)"
	@echo "  make logs          - View all logs"
	@echo "  make health        - Check health endpoints"
	@echo ""
	@echo "AWS:"
	@echo "  make push-ecr      - Push images to ECR"
	@echo "  make setup-aws     - Setup AWS infrastructure"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts"

install:
	uv sync

test:
	uv run pytest -v --cov=src --cov-report=term

lint:
	uv run ruff check .

format:
	uv run ruff format .

validate:
	@echo "Running validation checks..."
	make lint
	make test
	@echo "✓ All validation checks passed"

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

docker-build:
	docker-compose build --parallel

docker-up:
	docker-compose up -d
	@echo "Waiting for services..."
	@sleep 10
	@echo "Services running:"
	@docker-compose ps
	@echo ""
	@echo "Access:"
	@echo "  - Streamlit UI: http://localhost:8501"
	@echo "  - RAG API: http://localhost:8080"

docker-down:
	docker-compose down

deploy-local:
	bash scripts/deploy-local.sh

push-ecr:
	bash scripts/push-to-ecr.sh

setup-aws:
	bash scripts/setup-aws-infrastructure.sh

logs:
	docker-compose logs -f

logs-rag:
	docker-compose logs -f rag-pipeline

logs-ui:
	docker-compose logs -f streamlit-ui

health:
	@echo "Checking health endpoints..."
	@curl -f http://localhost:8080/health || echo "RAG API: UNHEALTHY"
	@curl -f http://localhost:8501/_stcore/health || echo "Streamlit: UNHEALTHY"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/ *.egg-info
	@echo "✓ Cleaned build artifacts"
