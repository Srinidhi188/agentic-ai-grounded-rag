"""
Semantic Retrieval and Grounding Evaluation Script.
Loads the curated 10-question evaluation dataset from data/evaluation_questions.json,
executes retrieval against Pinecone (or local extracted chunks in --offline mode),
and outputs detailed evidence, relevance decisions, and grounding evaluation.

Usage:
    python -m scripts.evaluate_retrieval [--top-k N] [--offline]
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any

from app.config import get_settings
from app.ingestion.embedder import get_embedder, MockEmbedder
from app.vectorstore.pinecone_store import PineconeStore, PineconeNotConfiguredError
from app.ingestion.loader import PDFLoader
from app.ingestion.chunker import TextChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")

DEFAULT_QUESTIONS_FILE = Path("data/evaluation_questions.json")


def load_evaluation_questions(filepath: Path = DEFAULT_QUESTIONS_FILE) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON file with fallback."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("questions", [])

    return [
        {
            "id": "Q1_FACTUAL",
            "category": "Direct Factual",
            "query": "What is Agentic AI according to the eBook?",
            "expected_pages": [4, 7, 8],
        },
        {
            "id": "Q7_UNSUPPORTED",
            "category": "Unsupported Out-of-Scope",
            "query": "What is the recipe for baking homemade chocolate chip cookies?",
            "expected_pages": [],
        },
    ]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def run_offline_evaluation(questions: List[Dict[str, Any]], top_k: int = 3):
    """Run deterministic retrieval evaluation on real PDF chunks using local embedder."""
    settings = get_settings()
    pdf_path = Path("data/Ebook-Agentic-AI.pdf")
    if not pdf_path.exists():
        print(f"[ERROR] Cached PDF not found at {pdf_path}. Run 'python -m scripts.ingest --dry-run' first.")
        return

    print("\n" + "=" * 80)
    print("      OFFLINE CODE-VERIFICATION EVALUATION (Deterministic Local Embedder)")
    print("=" * 80)
    print("Notice: This mode evaluates pipeline logic and chunk extraction offline.")
    print("Scores are derived deterministically and demonstrate state flow without API quota.")
    print("=" * 80 + "\n")

    loader = PDFLoader()
    pages = loader.extract_pages(pdf_path)
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_document(pages)

    embedder = MockEmbedder(dimension=1536)
    chunk_vectors = [(c, embedder.embed_query(c.text)) for c in chunks]

    for item in questions:
        qid = item.get("id")
        cat = item.get("category")
        q = item.get("query")
        exp_pages = item.get("expected_pages", [])

        q_vec = embedder.embed_query(q)
        scores = []
        for c, c_vec in chunk_vectors:
            sim = cosine_similarity(q_vec, c_vec)
            scores.append((sim, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:top_k]
        top_score = top_matches[0][0] if top_matches else 0.0
        relevance_decision = "SUFFICIENT (Route -> Generate)" if top_score >= settings.RELEVANCE_THRESHOLD else "INSUFFICIENT (Route -> Fallback)"

        print(f"[{qid}] Category: {cat}")
        print(f"Query: \"{q}\"")
        print(f"Expected Pages: {exp_pages if exp_pages else 'None (Out of Scope)'}")
        print(f"Top Score: {top_score:.4f} | Relevance Gate: {relevance_decision}")

        for rank, (score, c) in enumerate(top_matches, 1):
            preview = c.text.replace("\n", " ")[:110]
            print(f"   Rank {rank}: [Score: {score:.4f}] Chunk: {c.metadata.chunk_id} (Page {c.metadata.page})")
            print(f"           \"{preview}...\"")
        print("-" * 80)


def run_evaluation(top_k: int = 3, offline: bool = False):
    settings = get_settings()
    questions = load_evaluation_questions()

    print("=" * 80)
    print("        AGENTIC AI EBOOK — SEMANTIC RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Configured Vector Store: Pinecone ({settings.PINECONE_INDEX_NAME})")
    print(f"Configured Embedding Provider: {settings.EMBEDDING_PROVIDER} ({settings.EMBEDDING_MODEL})")
    print(f"Configured Relevance Threshold: {settings.RELEVANCE_THRESHOLD}")
    print(f"Total Evaluation Questions: {len(questions)}")
    print("=" * 80)

    if offline:
        run_offline_evaluation(questions, top_k=top_k)
        return

    # Check embedding provider
    embedder = get_embedder(settings)
    if isinstance(embedder, MockEmbedder):
        print("\n[NOTICE / LIMITATION]")
        print("Semantic retrieval evaluation requires a working embedding provider with active quota.")
        print(f"Current provider is set to or fell back to: {embedder.model_name}")
        print("To run authentic semantic evaluation, configure valid OPENAI_API_KEY or GEMINI_API_KEY in .env.")
        print("To test pipeline logic offline without API quota, run: python -m scripts.evaluate_retrieval --offline")
        print("=" * 80)
        return

    # Check Pinecone
    try:
        store = PineconeStore(settings)
        _ = store.client
    except PineconeNotConfiguredError as e:
        print("\n[NOTICE / LIMITATION]")
        print(f"Pinecone vector store is not configured: {e}")
        print("To evaluate live retrieval against Pinecone, set PINECONE_API_KEY in .env.")
        print("To test pipeline logic offline without API quota, run: python -m scripts.evaluate_retrieval --offline")
        print("=" * 80)
        return
    except Exception as e:
        print(f"\n[ERROR] Pinecone initialization failed: {e}")
        return

    # Live Pinecone Evaluation
    print(f"\nEvaluating {len(questions)} test questions against live Pinecone (top_k={top_k}):\n")
    for item in questions:
        qid = item.get("id")
        cat = item.get("category")
        q = item.get("query")
        exp_pages = item.get("expected_pages", [])

        print(f"[{qid}] Category: {cat}")
        print(f"Query: \"{q}\"")

        try:
            query_vector = embedder.embed_query(q)
            results = store.query(query_embedding=query_vector, top_k=top_k)

            if not results:
                print("  -> No results returned from Pinecone.")
            else:
                top_score = results[0]["score"]
                status = "SUFFICIENT (Route -> Generate)" if top_score >= settings.RELEVANCE_THRESHOLD else "INSUFFICIENT (Route -> Fallback)"
                print(f"  -> Top Score: {top_score:.4f} | Gate Decision: {status}")

                for rank, res in enumerate(results, 1):
                    chunk_id = res.get("chunk_id", "N/A")
                    page = res.get("page", 0)
                    score = res.get("score", 0.0)
                    preview = res.get("text", "").replace("\n", " ")[:120]
                    print(f"     Rank {rank}: [Score: {score:.4f}] Chunk: {chunk_id} (Page {page})")
                    print(f"             \"{preview}...\"")
        except Exception as err:
            print(f"  -> Live query error: {err}")

        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on Agentic AI eBook questions.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K results to retrieve per question.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline code-verification evaluation using local extracted PDF chunks and deterministic embeddings.",
    )
    args = parser.parse_args()
    run_evaluation(top_k=args.top_k, offline=args.offline)


if __name__ == "__main__":
    main()
