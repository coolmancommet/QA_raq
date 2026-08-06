# AI-Powered Document Q&A RAG System

A lightweight, robust Retrieval-Augmented Generation (RAG) system built with **Python**, **Streamlit**, **Chroma DB**, and **HuggingFace Embeddings**, supporting both local **Ollama (Gemma)** and cloud **Google Gemini API** with automatic fallback.

---

## Key Features

1. **PDF Ingestion**: Accepts 1-3 PDF documents (with sample documents included for instant testing).
2. **Natural Language Q&A**: Ask complex questions about your documents through an intuitive Streamlit chat interface.
3. **Precise Citations**: Answers include exact source document names and page numbers, plus an expandable viewer for retrieved context chunks.
4. **Hybrid LLM Support**: Supports local Ollama (Gemma model) with seamless automatic fallback to Google Gemini API.
5. **Single-Command Setup**: Runs with a single command using `uv`.

---

## Architecture & Approach

- **Data Ingestion & Chunking**: PDFs are loaded using `PyPDFLoader` and split into overlapping chunks (1000 chars, 200 overlap) via `RecursiveCharacterTextSplitter` to preserve semantic context across chunk boundaries.
- **Embeddings & Vector Store**: Chunks are embedded locally into 384-dimensional vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed in **Chroma DB** for lightning-fast semantic similarity search.
- **Retrieval & Generation**: When a query is submitted, the top-$k$ most relevant document chunks are retrieved and passed into the LLM prompt with strict grounding and citation instructions.
- **Fallback Resilience**: The system attempts local inference via Ollama (Gemma). If Ollama is offline or unavailable, it automatically fails over to the Google Gemini API (if configured).

---

## Quick Start (Single-Command Run)

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager installed.

### Run the App
```bash
# Clone or navigate to the repository directory
cd doc-qa-rag

# Run with single command
./run.sh
```
Alternatively:
```bash
uv run streamlit run src/doc_qa/app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and configure your API keys (optional if running locally with Ollama):

```env
GEMINI_API_KEY=your_gemini_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma
```

---

## Known Limitations

1. **OCR Support**: Scanned PDFs (images without text layers) are not automatically OCR'd; text must be extractable by `pypdf`.
2. **Context Window Limits**: While top-$k$ chunking prevents context overflow, extremely dense multi-document synthesis may occasionally truncate niche details if too many chunks are retrieved.
3. **Local Resource Usage**: Running embedding models locally consumes minor CPU/RAM, and running local Ollama requires sufficient hardware (GPU/Apple Silicon recommended for local LLM inference).

---

## Future Improvements (With More Time)

1. **Hybrid Search**: Combine dense vector similarity search with sparse keyword search (BM25) using reciprocal rank fusion (RRF).
2. **Advanced Document Parsing**: Integrate OCR (`pdf2image` + Tesseract) for scanned/handwritten documents and layout-aware table extraction.
3. **Evaluation Framework**: Add RAGAS (Retrieval Augmented Generation Assessment) evaluation metrics (Faithfulness, Answer Relevance, Context Precision).
4. **Multi-turn Conversational Memory**: Maintain chat history context across multiple turns for follow-up questions.
