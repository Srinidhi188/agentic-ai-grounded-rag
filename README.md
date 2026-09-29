# Agentic AI eBook — Grounded RAG Chatbot

A production-quality Retrieval-Augmented Generation (RAG) system engineered in Python to answer user queries strictly from the official **Agentic AI eBook** ([https://konverge.ai/pdf/Ebook-Agentic-AI.pdf](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)).

The system is designed to significantly reduce unsupported generation and prevent hallucinations by requiring verified retrieval evidence before answer synthesis and enforcing strict grounding prompt constraints.

Orchestrated using **LangGraph**, backed by **Pinecone** vector database, and exposed via **FastAPI**.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[Agentic AI PDF] --> B[PDF Loader & Extractor]
        B --> C[Conservative Text Cleaner]
        C --> D[Semantic Recursive Chunker]
        D --> E[Embedding Model]
        E --> F[(Pinecone Vector Database)]
    end

    subgraph LangGraph RAG Workflow
        Q[User Query] --> R[retrieve Node]
        R --> S{check_relevance Router}
        S -- "Top Score >= Threshold" --> T[generate_answer Node]
        S -- "Top Score < Threshold or Empty" --> U[fallback Node]
        T --> V[END]
        U --> V
    end

    V --> API[FastAPI Response: Answer + Citations + Confidence]
```

---

## LangGraph State Machine & Workflow

The orchestration layer is built with **LangGraph** (`StateGraph`) using a strongly typed state object (`RAGState`):

```python
class RAGState(TypedDict):
    user_query: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    context: str
    confidence: float
    should_answer: bool
    answer: str
    error: Optional[str]
```

### Workflow Nodes & Conditional Edges
1. **`retrieve` Node**:
   - Computes query vector embedding using the configured provider.
   - Queries Pinecone for the configured `RETRIEVAL_TOP_K` chunks.
   - Extracts chunks, page numbers, deterministic chunk IDs, and similarity scores.
   - Computes the aggregate confidence heuristic.
   - Formats candidate chunks into a structured prompt context block.
   - Evaluates evidence sufficiency against `RELEVANCE_THRESHOLD` (default: `0.65`).
2. **`check_relevance` Router (Conditional Edge)**:
   - If `should_answer` is `True` (retrieved chunks exist and top similarity score $\ge$ threshold): routes to **`generate_answer`**.
   - If `should_answer` is `False` (no documents found or score below threshold): routes directly to **`fallback`**, **completely bypassing the LLM**.
3. **`generate_answer` Node**:
   - Injects retrieved context and query into the strict grounding prompt.
   - Invokes the configured LLM provider (`temperature=0.0`).
   - Requires page citations in the format `[Page X]`.
4. **`fallback` Node**:
   - Returns a deterministic refusal response:
     > *"I couldn't find sufficient information about this in the provided Agentic AI knowledge base."*
   - Zero LLM generation occurs, completely eliminating hallucination on out-of-scope queries.

---

## Relevance Gate & Confidence Heuristic

### Relevance Gate
The relevance gate acts as an automated triage mechanism before generative inference. If the top cosine similarity score from Pinecone does not meet `RELEVANCE_THRESHOLD` (default `0.65`), the query is considered unsupported by the eBook. This prevents the LLM from attempting to fill gaps using pre-trained parametric memory.

### Confidence / Relevance Signal
The response returns a transparent confidence metric:
```json
{
  "confidence": 0.88
}
```
$$\text{confidence} = \text{round}(\max(0.0, \min(1.0, \max(\text{retrieval\_scores}))), 4)$$

> **CRITICAL NOTE ON CONFIDENCE**:
> This metric represents the **vector similarity / retrieval relevance** of the top retrieved context chunk. It is **NOT a mathematically calibrated probability of answer correctness**.

---

## Document Ingestion & Chunking Rationale

The knowledge base PDF contains **60 pages** (~78,800 characters) across 6 major sections.

- **Cleaning Strategy**: Conservative normalization that standardizes line endings, strips non-printable control characters, replaces typographic quotes, and collapses whitespace while preserving semantic paragraph boundaries (`\n\n`).
- **Chunking Strategy**: Recursive character chunking:
  - **Chunk Size**: `800 characters` (~130–160 words). Fits cohesive concepts, definitions, and architecture components without overflowing embedding attention or diluting similarity scores.
  - **Chunk Overlap**: `150 characters` (~20–25 words). Ensures boundary definitions and cross-sentence meaning are preserved across chunk splits.
  - **Deterministic IDs**: Formatted as `agentic-ai-p{page:02d}-c{chunk_index:02d}` for traceability and citation.
- **Yield**: Exactly **149 chunks** across 60 pages (average 2.5 chunks/page, 587.7 characters/chunk).

---

## Project Structure

```
RAG/
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings with validation & safe defaults
│   ├── main.py                # FastAPI app initialization, lifespan, CORS, error handling
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /health, /info, /chat route handlers
│   │   └── schemas.py         # Pydantic schemas (HealthResponse, ChatRequest, ContextChunk)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # PDF acquisition & 1-indexed page text extraction
│   │   ├── chunker.py         # Conservative cleaner & recursive semantic chunker
│   │   └── embedder.py        # Modular embeddings (OpenAI, Gemini, Local Mock)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── state.py           # TypedDict LangGraph workflow state
│   │   ├── prompts.py         # Strict anti-hallucination prompts & citations
│   │   ├── generator.py       # Modular LLM generator (OpenAI, Gemini, Mock)
│   │   └── graph.py           # LangGraph state machine & conditional routing
│   └── vectorstore/
│       ├── __init__.py
│       └── pinecone_store.py  # Pinecone index management, batch upsert & query
├── scripts/
│   ├── __init__.py
│   ├── ingest.py              # End-to-end ingestion CLI script
│   └── evaluate_retrieval.py  # Semantic retrieval evaluation script
├── tests/
│   ├── __init__.py
│   ├── test_api.py            # API health & settings validation tests
│   ├── test_retrieval.py      # PDF extraction, chunking, embedder, and vector store tests
│   └── test_rag.py            # LangGraph workflow, relevance gate, and fallback tests
├── data/                      # Cached PDF storage (gitignored)
├── docs/                      # Technical documentation
├── .env.example               # Environment variables template
├── .gitignore                 # Excludes venv, .env, PDFs, and build caches
├── requirements.txt           # Pinned dependencies for Python 3.12+
├── README.md                  # Project overview & documentation
└── LICENSE                    # MIT License
```

---

## Setup & Quickstart

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in credentials:
```powershell
cp .env.example .env
```
Key variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: `agentic-ai-rag`
- `PINECONE_NAMESPACE`: `agentic-ai-ebook`
- `EMBEDDING_PROVIDER`: `openai` or `gemini` or `local`
- `LLM_PROVIDER`: `openai` or `gemini`

### 3. Run Ingestion Pipeline
To ingest the PDF into Pinecone:
```powershell
python -m scripts.ingest
```

To test the ingestion pipeline offline without upserting:
```powershell
python -m scripts.ingest --dry-run
```

### 4. Evaluate Semantic Retrieval
To inspect retrieval performance on authentic eBook questions:
```powershell
python -m scripts.evaluate_retrieval
```

### 5. Run Automated Tests
```powershell
pytest -v tests/
```

---

## Known Limitations

1. **External Embedding API Quota**: When running in environments where commercial provider accounts (OpenAI / Gemini) have exhausted their quotas (e.g. `429 insufficient_quota` or `RESOURCE_EXHAUSTED`), unit tests and offline ingestion use the deterministic local embedder (`MockEmbedder`). Authentic live semantic evaluation requires setting active API keys.
2. **Tabular & Diagrammatic PDF Artifacts**: Certain complex flowcharts in the PDF are raster images without embedded OCR text; the system extracts and indexes all raw text pages (59 of 60 pages).
3. **Prompt Grounding vs Hard Proofs**: While the combination of the relevance gate, zero-temperature generation, and strict negative constraints drastically curtails hallucinations, prompt-based LLM generation is inherently probabilistic and should be monitored with automated evaluation benchmarks.

---

## Implementation Progress

| Phase | Milestone | Status |
|---|---|---|
| **Phase 1** | Workspace analysis, environment inspection, architecture blueprint | **Completed** |
| **Phase 2** | Project foundation, settings, API skeleton, `/health` endpoint | **Completed** |
| **Phase 3** | PDF acquisition, extraction, chunking, embeddings, Pinecone integration | **Completed** |
| **Phase 4** | LangGraph RAG workflow, relevance guardrail, grounded generation | **Completed** |
| **Phase 5** | FastAPI `/chat` endpoint integration & error handling | *Pending* |
| **Phase 6** | End-to-end question evaluation & hallucination testing | *Pending* |
| **Phase 7** | Comprehensive interview notes & architecture guide | *Pending* |
| **Phase 8** | Final code review & submission preparation | *Pending* |
