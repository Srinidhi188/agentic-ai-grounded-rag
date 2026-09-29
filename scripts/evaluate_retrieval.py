"""
Semantic Retrieval Evaluation Script.
Loads a curated set of questions derived from the Agentic AI eBook,
executes retrieval against the vector store, and displays the retrieved evidence.

Usage:
    python -m scripts.evaluate_retrieval [--top-k N]
"""

import argparse
import logging
import sys
from typing import List, Dict
from app.config import get_settings
from app.ingestion.embedder import get_embedder, MockEmbedder
from app.vectorstore.pinecone_store import PineconeStore, PineconeNotConfiguredError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")

# Curated test questions directly from the Agentic AI eBook sections
EVALUATION_QUESTIONS = [
    {
        "id": "q1",
        "question": "What is Agentic AI and how does it shift from reactive to proactive problem-solving?",
        "expected_section": "Chapter 1: Introduction to Agentic AI (Pages 7-10)",
    },
    {
        "id": "q2",
        "question": "How do Agentic AI and Large Language Models (LLMs) work together?",
        "expected_section": "Section 1.1 / Table Comparison (Page 10)",
    },
    {
        "id": "q3",
        "question": "What is the Anatomy of an Agentic AI System and what are its core components?",
        "expected_section": "Chapter 2: Anatomy of an Agentic AI System (Pages 12-25)",
    },
    {
        "id": "q4",
        "question": "How do Multi-Agent Systems orchestrate collaboration between specialized agents?",
        "expected_section": "Chapter 3: Multi-Agent Systems (Pages 26-38)",
    },
    {
        "id": "q5",
        "question": "What are the practical enterprise use cases of Agentic AI described in the eBook?",
        "expected_section": "Chapter 6: Practical Applications of Agentic AI (Pages 50-60)",
    },
    {
        "id": "q6_unsupported",
        "question": "What is the recipe for baking chocolate chip cookies?",
        "expected_section": "OUT OF SCOPE (Must not match or must yield low relevance)",
    },
]


def run_evaluation(top_k: int = 3):
    settings = get_settings()

    print("=" * 80)
    print("        AGENTIC AI EBOOK — SEMANTIC RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Configured Vector Store: Pinecone ({settings.PINECONE_INDEX_NAME})")
    print(f"Configured Embedding Provider: {settings.EMBEDDING_PROVIDER} ({settings.EMBEDDING_MODEL})")
    print(f"Configured Relevance Threshold: {settings.RELEVANCE_THRESHOLD}")
    print("=" * 80)

    # Check whether a real embedding provider is active
    embedder = get_embedder(settings)
    if isinstance(embedder, MockEmbedder):
        print("\n[NOTICE / LIMITATION]")
        print("Semantic retrieval evaluation requires a working embedding provider with active quota.")
        print(f"Current provider is set to or fell back to: {embedder.model_name}")
        print("To run authentic semantic evaluation, configure valid OPENAI_API_KEY or GEMINI_API_KEY")
        print("and set EMBEDDING_PROVIDER=openai or EMBEDDING_PROVIDER=gemini in .env.")
        print("=" * 80)
        return

    # Check Pinecone store
    try:
        store = PineconeStore(settings)
        # Test client connectivity
        _ = store.client
    except PineconeNotConfiguredError as e:
        print("\n[NOTICE / LIMITATION]")
        print(f"Pinecone vector store is not configured: {e}")
        print("To evaluate live retrieval against Pinecone, set PINECONE_API_KEY in .env.")
        print("=" * 80)
        return
    except Exception as e:
        print(f"\n[ERROR] Pinecone initialization failed: {e}")
        return

    # Run evaluation questions
    print(f"\nEvaluating {len(EVALUATION_QUESTIONS)} test questions (top_k={top_k}):\n")

    for item in EVALUATION_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        expected = item["expected_section"]

        print(f"[{qid}] Question: {q}")
        print(f"Expected Knowledge Area: {expected}")

        try:
            query_vector = embedder.embed_query(q)
            results = store.query(query_embedding=query_vector, top_k=top_k)

            if not results:
                print("  -> No results returned from Pinecone.")
            else:
                top_score = results[0]["score"]
                status = "RELEVANT" if top_score >= settings.RELEVANCE_THRESHOLD else "INSUFFICIENT (Will Trigger Fallback)"
                print(f"  -> Top Score: {top_score:.4f} | Status: {status}")

                for rank, res in enumerate(results, 1):
                    chunk_id = res.get("chunk_id", "N/A")
                    page = res.get("page", 0)
                    score = res.get("score", 0.0)
                    preview = res.get("text", "").replace("\n", " ")[:120]
                    print(f"     Rank {rank}: [Score: {score:.4f}] Chunk: {chunk_id} (Page {page})")
                    print(f"             \"{preview}...\"")

        except Exception as err:
            print(f"  -> Evaluation error for query: {err}")

        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval on Agentic AI questions.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K results to retrieve per question.")
    args = parser.parse_args()
    run_evaluation(top_k=args.top_k)


if __name__ == "__main__":
    main()
