# AI-Powered Document Q&A Platform

A GitHub-ready reference implementation of the **AI-Powered Document Q&A Platform** described on my resume.

The system uses Retrieval-Augmented Generation (RAG):
1. Extract text from uploaded PDF/DOCX/TXT files.
2. Split text into token-based chunks with overlap.
3. Generate dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Normalize embeddings and index them with FAISS `IndexFlatIP`.
5. Embed a user question with the same embedding model.
6. Retrieve the top-K chunks using inner-product search (cosine similarity after L2 normalization).
7. Send the retrieved context and question to an Ollama-served LLM.
8. Expose the pipeline through FastAPI and provide a Streamlit UI.

> Important interview note: this repository is a clean reference implementation created to make the resume project reproducible. Only claim configurations you actually used during the original project. The exact model/config values below are the configuration of this GitHub implementation.

## Architecture

```text
                  +----------------------+
                  |      Streamlit       |
                  |   upload / question  |
                  +----------+-----------+
                             | HTTP
                             v
                  +----------------------+
                  |       FastAPI        |
                  |   /ingest /ask/...   |
                  +-------+-------+------+
                          |       |
                ingestion|       |query
                          |       |
                          v       v
                  +----------------------+
                  | SentenceTransformer |
                  | all-MiniLM-L6-v2     |
                  | embedding dimension  |
                  |        = 384         |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |         FAISS        |
                  |    IndexFlatIP       |
                  | normalized vectors   |
                  +----------+-----------+
                             | top-k chunks
                             v
                  +----------------------+
                  |       Ollama         |
                  | configurable LLM     |
                  +----------+-----------+
                             |
                             v
                           Answer
```

## Configuration used by this repository

| Component | Configuration |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | **384** |
| Embedding type | Dense float32 vectors |
| Normalization | L2 normalization |
| FAISS index | `IndexFlatIP` |
| Similarity | Inner product on normalized vectors = cosine similarity |
| Chunk target | **220 embedding-model tokens** |
| Chunk overlap | **40 tokens** |
| Top-K | **5** |
| Minimum retrieval score | **0.35** (configurable) |
| Generation model | Ollama model configurable by `.env`; default `llama3.2:3b` |
| Generation temperature | **0.1** |
| Max generated tokens | **512** |
| API | FastAPI |
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| DOCX extraction | python-docx |
| Index persistence | local `data/index/` |
| Metadata | JSONL in `data/index/metadata.jsonl` |

### Why 384 dimensions?

`all-MiniLM-L6-v2` produces a 384-dimensional sentence embedding. Every document chunk and every query therefore becomes a vector of shape:

```text
(384,)
```

For `N` chunks, the FAISS matrix is:

```text
(N, 384)
```

With `float32`, raw vector memory is approximately:

```text
N x 384 x 4 bytes
```

For 100,000 chunks, the raw vector data is about 146.5 MiB before FAISS/metadata overhead.

### Why IndexFlatIP?

The repository normalizes vectors before adding/searching them.

For unit-normalized vectors:

```text
A dot B = cosine_similarity(A, B)
```

`IndexFlatIP` performs an exact inner-product search. It is simple and highly explainable for a project of this size. For much larger collections, approximate nearest-neighbor indexes such as IVF/HNSW can reduce search cost at a recall/complexity trade-off.

## Chunking details

The implementation uses the embedding model's tokenizer so the chunk size is expressed in **model tokens**, not characters.

- target chunk size: 220 tokens
- overlap: 40 tokens
- effective stride: 180 tokens
- special tokens are not included in chunk windows
- chunks shorter than the target size are kept
- whitespace and empty chunks are discarded

Why not use a huge chunk? `all-MiniLM-L6-v2` has a limited model input length; keeping chunks comfortably below that limit avoids silently truncating useful text.

## RAG query flow

```text
Question
   |
   v
same embedding model
   |
   v
384-D query vector
   |
   v
L2 normalize
   |
   v
FAISS IndexFlatIP
   |
   v
top-5 chunks + similarity scores
   |
   +-- no sufficiently relevant chunks -> grounded "not found" response
   |
   v
prompt = instructions + context + question
   |
   v
Ollama LLM
   |
   v
answer
```

## Ingestion flow

```text
PDF/DOCX/TXT
    |
    v
text extraction
    |
    v
token-aware chunking
    |
    v
384-D embeddings
    |
    v
L2 normalization
    |
    v
FAISS index
    |
    v
metadata JSONL
```

Each metadata row maps a FAISS vector position to its source:

```json
{
  "source": "sample.pdf",
  "page": 3,
  "chunk_id": 17,
  "text": "..."
}
```

This is important because FAISS stores vectors, not the original text.

## API

### `GET /health`

Returns service status and index size.

### `POST /ingest`

Upload one or more `.pdf`, `.docx`, or `.txt` files.

### `POST /ask`

Request:

```json
{
  "question": "What is the main conclusion?",
  "top_k": 5
}
```

Response includes:
- generated answer
- retrieved source chunks
- similarity scores

### `GET /documents`

Lists indexed source documents.

## Local setup

### 1. Create environment

Python 3.11 is the baseline.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

Install Ollama separately, then pull the configured model:

```bash
ollama pull llama3.2:3b
```

The generation model can be changed through `.env`.

### 4. Start FastAPI

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 5. Start Streamlit

In another terminal:

```bash
streamlit run streamlit_app.py
```

## Environment variables

Copy `.env.example` to `.env`.

```text
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
TOP_K=5
SIMILARITY_THRESHOLD=0.35
CHUNK_SIZE_TOKENS=220
CHUNK_OVERLAP_TOKENS=40
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
```

## Requirements

See `requirements.txt`.

The embedding model downloads its model files on first use. This repository does **not** commit model weights to GitHub.

FAISS is CPU-based in this implementation, so a local NVIDIA GPU is not required.

## Testing

```bash
pytest -q
```

## Production extensions

These are **not claims about this implementation**:
- hybrid lexical + semantic retrieval
- reranking
- metadata filtering
- vector database
- background ingestion jobs
- document versioning
- authentication and authorization
- rate limiting
- observability/tracing
- distributed API instances
- approximate nearest-neighbor indexes
- evaluation datasets and automated RAG metrics
