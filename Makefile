.PHONY: setup generate run lint format typecheck test check clean all

setup:
	@echo "Syncing project dependencies using uv..."
	uv sync

generate:
	@echo "Generating sample PDF documents..."
	uv run python generate_samples.py

run: generate
	@echo "Launching Streamlit UI..."
	uv run streamlit run src/doc_qa/app.py

lint:
	@echo "Running Ruff linter..."
	uv run ruff check src generate_samples.py

format:
	@echo "Running Ruff formatter..."
	uv run ruff format src generate_samples.py

typecheck:
	@echo " Running MyPy type checker..."
	MYPYPATH=src uv run mypy src generate_samples.py

test:
	@echo "Running unit tests..."
	PYTHONPATH=src uv run pytest tests/

check: format lint typecheck test
	@echo "All code quality and test checks passed!"

clean:
	@echo "Cleaning up generated files, cache, and vectorstore..."
	rm -rf chroma_db
	rm -rf sample_docs
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

all: setup check run
