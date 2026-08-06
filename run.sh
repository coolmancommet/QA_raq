#!/usr/bin/env bash
set -e

echo "Starting Document Q&A RAG System..."

# Ensure sample PDFs exist
if [ ! -f "sample_docs/company_policy.pdf" ] || [ ! -f "sample_docs/project_spec.pdf" ]; then
    echo "Generating sample PDF documents..."
    uv run generate_samples.py
fi

echo "Launching Streamlit UI..."
uv run streamlit run src/doc_qa/app.py
