import os
import tempfile

import streamlit as st

from doc_qa.rag_engine import answer_query, load_existing_vectorstore, process_pdfs

st.set_page_config(
    page_title="AI Document Q&A RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and vectorstore
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    # Try loading existing vector store on startup
    st.session_state.vectorstore = load_existing_vectorstore()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("RAG Settings")
    
    provider = st.selectbox(
        "LLM Provider",
        ["Auto (Ollama -> Gemini Fallback)", "Ollama", "Gemini"],
        help="Select LLM provider. Auto tries local Ollama first and falls back to Gemini if API key is provided."
    )
    
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama Model", value="gemma")
    
    gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    gemini_model = st.text_input("Gemini Model", value="gemini-1.5-flash")
    
    model_name = ollama_model if "Ollama" in provider else gemini_model
    if provider == "Auto (Ollama -> Gemini Fallback)":
        model_name = ollama_model

    st.divider()
    st.subheader("Document Management")
    
    # Sample docs loading button
    if st.button("Load Sample PDFs (2 docs)", help="Quickly load pre-built sample PDFs (Company Policy & Project Spec)"):
        sample_files = [
            "sample_docs/company_policy.pdf",
            "sample_docs/project_spec.pdf"
        ]
        valid_samples = [f for f in sample_files if os.path.exists(f)]
        if valid_samples:
            with st.spinner("Processing sample PDF documents..."):
                try:
                    st.session_state.vectorstore = process_pdfs(valid_samples, ollama_url=ollama_url)
                    st.success(f"Successfully loaded {len(valid_samples)} sample documents!")
                except Exception as e:
                    st.error(f"Error processing sample documents: {e}")
        else:
            st.warning("Sample PDF files not found in sample_docs/. Please generate them.")

    uploaded_files = st.file_uploader(
        "Upload 1-3 PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload between 1 and 3 PDF files to query."
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Please upload a maximum of 3 PDF documents.")
        else:
            if st.button("Process Uploaded PDFs", type="primary"):
                with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                    temp_paths = []
                    try:
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.getvalue())
                                temp_paths.append(tmp.name)
                        
                        st.session_state.vectorstore = process_pdfs(temp_paths, ollama_url=ollama_url)
                        st.success(f"Successfully processed and indexed {len(uploaded_files)} PDF document(s)!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
                    finally:
                        for path in temp_paths:
                            if os.path.exists(path):
                                os.unlink(path)

    st.divider()
    if st.session_state.vectorstore:
        st.success("🟢 Vector Database Ready")
    else:
        st.warning("🟡 No documents loaded yet.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.markdown('<div class="main-header">Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask natural language questions about your PDF documents with precise citations and source traceability.</div>', unsafe_allow_html=True)

# Display architecture info expander
with st.expander("About this RAG Pipeline & Architecture"):
    st.markdown("""
    - **Document Ingestion**: Loads PDFs using `PyPDFLoader`, splits text into chunks using `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200).
    - **Embeddings & Vector Store**: Uses Ollama local embeddings (`nomic-embed-text`) for local embedding generation and `Chroma` for vector storage and similarity search.
    - **LLM Support**: Supports local **Ollama (Gemma model)** with automatic fallback to cloud **Google Gemini API** when configured.
    - **Citations**: Returns explicit source document filenames and page numbers for every claim, backed by source chunk inspection.
    """)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"View Retrieved Citations & Sources ({len(message['sources'])} chunks)"):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="citation-box">
                        <b>Source {src['id']}:</b> {src['source']} (Page {src['page']})<br>
                        <small>{src['content'][:350]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        if not st.session_state.vectorstore:
            error_msg = "Please upload PDF documents or load the sample PDFs in the sidebar before asking questions."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
        else:
            with st.spinner("Searching documents and generating response..."):
                answer, sources = answer_query(
                    vectorstore=st.session_state.vectorstore,
                    query=prompt,
                    provider=provider,
                    model_name=model_name,
                    api_key=gemini_api_key,
                    ollama_url=ollama_url
                )
                st.markdown(answer)
                if sources:
                    with st.expander(f"View Retrieved Citations & Sources ({len(sources)} chunks)"):
                        for src in sources:
                            st.markdown(f"""
                            <div class="citation-box">
                                <b>Source {src['id']}:</b> {src['source']} (Page {src['page']})<br>
                                <small>{src['content'][:350]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
