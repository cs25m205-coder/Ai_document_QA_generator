import requests
import streamlit as st

API_URL = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")

st.title("AI-Powered Document Q&A")
st.caption("RAG: chunk -> embed -> FAISS retrieve -> generate")

uploaded = st.file_uploader(
    "Upload PDF, DOCX, or TXT documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Index documents", disabled=not uploaded):
    files = [
        ("files", (file.name, file.getvalue(), file.type))
        for file in uploaded
    ]

    with st.spinner("Extracting, chunking and indexing..."):
        response = requests.post(f"{API_URL}/ingest", files=files, timeout=180)

    if response.ok:
        st.success(response.json())
    else:
        st.error(response.text)

question = st.text_input("Ask a question about the indexed documents")

if st.button("Ask", disabled=not question):
    with st.spinner("Retrieving context and generating answer..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": 5},
            timeout=180,
        )

    if response.ok:
        data = response.json()
        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Retrieved sources")
        for source in data["sources"]:
            st.write(
                f"**{source['source']}** | "
                f"page={source['page']} | "
                f"chunk={source['chunk_id']} | "
                f"score={source['score']:.3f}"
            )
            with st.expander("View retrieved text"):
                st.write(source["text"])
    else:
        st.error(response.text)
