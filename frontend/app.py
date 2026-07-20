import os
import uuid
import requests
import streamlit as st

# Configure backend URL (uses Docker service name inside container, defaults to localhost for local testing)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LangGraph RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LangGraph RAG Assistant")
st.caption("Powered by FastAPI, ChromaDB, and LangGraph")

# -------------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# -------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything or upload a PDF to get started."}
    ]

# -------------------------------------------------------------------
# 2. SIDEBAR: FILE UPLOAD & CONTROLS
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF to Vector DB", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Processing & vectorizing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/api/upload", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully ingested `{uploaded_file.name}`!")
                    else:
                        st.error(f"Failed ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.divider()
    st.caption(f"**Session ID:** `{st.session_state.session_id}`")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.rerun()

# -------------------------------------------------------------------
# 3. DISPLAY CHAT HISTORY
# -------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------------
# 4. STREAMING CHAT INPUT HANDLER
# -------------------------------------------------------------------
if prompt := st.chat_input("Ask a question..."):
    # Display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display streaming assistant response
    with st.chat_message("assistant"):
        def generate_response_stream():
            """Generator that consumes the streaming endpoint token by token."""
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }
            try:
                with requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=120
                ) as resp:
                    if resp.status_code != 200:
                        yield f"Error: Received status code {resp.status_code}"
                        return
                    
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
            except Exception as err:
                yield f"\n[Connection Error: {str(err)}]"

        # Stream directly into UI
        full_response = st.write_stream(generate_response_stream())
        
    # Save completed assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})