"""
CLI search script for querying the Summa Theologica FAISS index.

Usage:
    python search.py "your question here"
    python search.py  # Interactive mode
"""

import sys
import json
import numpy as np
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- CONFIG ----------
EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = "out/summa.faiss"
META_PATH = "out/summa_meta.jsonl"
DEFAULT_TOP_K = 5


def embed_query(query: str) -> np.ndarray:
    """Embed a single query using OpenAI."""
    from openai import OpenAI
    client = OpenAI()
    
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v)
    return v


def load_meta() -> list[dict]:
    """Load metadata from JSONL file."""
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def search(query: str, index: faiss.Index, meta: list[dict], top_k: int = DEFAULT_TOP_K):
    """
    Search the index for chunks matching the query.
    
    Returns list of (metadata, score) tuples.
    """
    v = embed_query(query)
    
    # D = distances (similarity scores), I = indices
    D, I = index.search(v, top_k)
    
    results = []
    for rank, idx in enumerate(I[0]):
        if idx >= 0:  # FAISS returns -1 for empty results
            results.append((meta[idx], D[0][rank]))
    
    return results


def print_results(query: str, results: list[tuple[dict, float]]):
    """Pretty print search results."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    for rank, (m, score) in enumerate(results, start=1):
        print(f"\n{rank}) {m['chunk_id']}  [{m['section']}]  score={score:.4f}")
        print("-" * 60)
        
        # Truncate long texts for display
        text = m["text"]
        max_len = 800
        if len(text) > max_len:
            text = text[:max_len] + "..."
        print(text)


def interactive_mode(index: faiss.Index, meta: list[dict]):
    """Run interactive search loop."""
    print("\n" + "=" * 80)
    print("Summa Theologica Search")
    print("Enter your question (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        results = search(query, index, meta)
        print_results(query, results)


def main():
    """Main entry point."""
    import os
    
    # Check for index files
    if not os.path.exists(INDEX_PATH):
        print(f"Error: Index not found at {INDEX_PATH}")
        print("Run 'python build_index.py' first to create the index.")
        return
    
    if not os.path.exists(META_PATH):
        print(f"Error: Metadata not found at {META_PATH}")
        print("Run 'python build_index.py' first to create the index.")
        return
    
    # Load index and metadata
    print("Loading index...")
    index = faiss.read_index(INDEX_PATH)
    meta = load_meta()
    print(f"Loaded {index.ntotal} vectors")
    
    # Check if query provided as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = search(query, index, meta)
        print_results(query, results)
    else:
        interactive_mode(index, meta)


if __name__ == "__main__":
    main()

