import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "embeddinggemma:latest"

def get_embeddings(ollama_url: str = "http://localhost:11434") -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=ollama_url)

def process_pdfs(file_paths: list[str], ollama_url: str = "http://localhost:11434") -> Chroma:
    """Load PDFs, split into chunks, and store in Chroma vector store."""
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Clean up source metadata to show clean filename
        filename = os.path.basename(path)
        for doc in docs:
            doc.metadata["source"] = filename
            # PyPDFLoader usually stores page number in page (0-indexed or 1-indexed)
            if "page" in doc.metadata:
                doc.metadata["page"] = int(doc.metadata["page"]) + 1
            else:
                doc.metadata["page"] = 1
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(all_docs)

    embeddings = get_embeddings(ollama_url=ollama_url)
    # Create or replace Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectorstore

def load_existing_vectorstore(ollama_url: str = "http://localhost:11434") -> Chroma | None:
    """Load existing vector store if available."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        embeddings = get_embeddings(ollama_url=ollama_url)
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    return None

def get_llm(provider: str, model_name: str, api_key: str | None = None, ollama_url: str = "http://localhost:11434"):
    """Initialize LLM based on provider with fallback mechanism."""
    
    if provider == "Ollama":
        try:
            return ChatOllama(model=model_name or "gemma", base_url=ollama_url, temperature=0.1)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
            
    elif provider == "Gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API key is required when using Gemini provider.")
        return ChatGoogleGenerativeAI(model=model_name or "gemini-1.5-flash", google_api_key=key, temperature=0.1)
        
    elif provider == "Auto (Ollama -> Gemini Fallback)":
        # Try Ollama first
        try:
            llm = ChatOllama(model="gemma", base_url=ollama_url, temperature=0.1)
            return llm
        except Exception as e:
            key = api_key or os.getenv("GEMINI_API_KEY")
            if key:
                return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=key, temperature=0.1)
            raise RuntimeError(f"Ollama is unavailable and no Gemini API key provided for fallback. Error: {e}")
            
    else:
        raise ValueError(f"Unknown provider: {provider}")

def answer_query(
    vectorstore: Chroma,
    query: str,
    provider: str = "Auto (Ollama -> Gemini Fallback)",
    model_name: str = "gemma",
    api_key: str | None = None,
    ollama_url: str = "http://localhost:11434",
    k: int = 4
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve relevant chunks and generate answer with citations."""
    if not vectorstore:
        return "No documents indexed yet. Please upload or load PDF documents first.", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return "I could not find any relevant information in the uploaded documents to answer your question.", []

    # Build context string with citations metadata
    context_parts = []
    sources = []
    for i, doc in enumerate(relevant_docs):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content.strip()
        context_parts.append(f"[Source {i+1}] Document: {source}, Page: {page}\nContent: {snippet}\n")
        sources.append({
            "id": i + 1,
            "source": source,
            "page": page,
            "content": snippet
        })

    context_str = "\n".join(context_parts)

    system_prompt = (
        "You are an expert AI assistant that answers questions based strictly on the provided document excerpts.\n"
        "Rules:\n"
        "1. Answer the question accurately using ONLY the provided context.\n"
        "2. Cite your sources clearly in your answer using the format [Document: <filename>, Page: <page number>].\n"
        "3. If the answer cannot be found in the context, state clearly: 'I cannot find the answer in the provided documents.'\n"
        "4. Be concise, professional, and clear."
    )

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer with citations:"

    answer = ""
    try:
        llm = get_llm(provider, model_name, api_key, ollama_url)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        if provider == "Auto (Ollama -> Gemini Fallback)":
            key = api_key or os.getenv("GEMINI_API_KEY")
            if key:
                try:
                    fallback_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=key, temperature=0.1)
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    response = fallback_llm.invoke(messages)
                    answer = f"[Note: Ollama was unavailable; answered using Gemini API fallback]\n\n{response.content}"
                except Exception as ex:
                    answer = f"Error generating answer with primary (Ollama) and fallback (Gemini) LLMs:\nPrimary error: {e}\nFallback error: {ex}"
            else:
                answer = f"Error connecting to Ollama: {e}. No Gemini API key provided for fallback."
        else:
            answer = f"Error generating answer: {e}"

    return answer, sources
