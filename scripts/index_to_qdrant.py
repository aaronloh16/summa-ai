#!/usr/bin/env python3
"""
Index chunks to Qdrant Cloud with hybrid search (dense + sparse vectors).
"""

import json
import os
import uuid
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
)
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 20  # Smaller batch to avoid timeouts
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "summa")

# Paths
CHUNKS_FILE = Path(__file__).parent.parent / "out" / "chunks.jsonl"


def load_chunks(path: Path) -> Iterator[dict]:
    """Load chunks from JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def stable_hash(word: str) -> int:
    """Create a stable hash for a word using MD5 (consistent across runs)."""
    import hashlib
    return int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16) % 100000


def create_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Create a simple sparse vector using term frequencies.
    Uses stable MD5 hashing (consistent across runs).
    Note: This is a basic TF representation, not true BM25.
    For better quality, consider SPLADE in the future.
    """
    import re
    import math
    from collections import Counter
    
    # Tokenize and normalize
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Count term frequencies
    term_freq = Counter(words)
    
    # Create sparse vector (indices = stable hash of word, values = log TF)
    index_to_value = {}
    
    for word, count in term_freq.items():
        # Use stable MD5-based hash
        idx = stable_hash(word)
        value = math.log(1 + count)
        
        # Handle collisions by summing
        if idx in index_to_value:
            index_to_value[idx] += value
        else:
            index_to_value[idx] = value
    
    # Convert to sorted lists
    indices = sorted(index_to_value.keys())
    values = [index_to_value[i] for i in indices]
    
    return indices, values


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    # Truncate texts to avoid token limits
    MAX_CHARS = 30000
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
    
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=truncated
    )
    return [d.embedding for d in response.data]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Index chunks to Qdrant")
    parser.add_argument("--input", "-i", default=str(CHUNKS_FILE),
                        help="Input JSONL file path")
    parser.add_argument("--collection", "-c", default=COLLECTION_NAME,
                        help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true",
                        help="Recreate collection if exists")
    parser.add_argument("--dense-only", action="store_true",
                        help="Use dense vectors only (skip sparse/hybrid)")
    args = parser.parse_args()
    
    # Initialize clients
    print("Connecting to Qdrant...")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    openai_client = OpenAI()
    
    # Check/create collection
    collections = [c.name for c in qdrant.get_collections().collections]
    
    if args.collection in collections:
        if args.recreate:
            print(f"Recreating collection '{args.collection}'...")
            qdrant.delete_collection(args.collection)
        else:
            print(f"Collection '{args.collection}' exists, will append")
    
    if args.collection not in collections or args.recreate:
        print(f"Creating collection '{args.collection}'...")
        
        # Configure vectors based on mode
        vectors_config = {
            "dense": VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE
            )
        }
        
        sparse_config = None
        if not args.dense_only:
            sparse_config = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams()
                )
            }
            print("  Mode: Hybrid (dense + sparse)")
        else:
            print("  Mode: Dense only")
        
        qdrant.create_collection(
            collection_name=args.collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config
        )
    
    # Load chunks
    print(f"Loading chunks from {args.input}...")
    chunks = list(load_chunks(Path(args.input)))
    print(f"Loaded {len(chunks)} chunks")
    
    # Process in batches
    print(f"Indexing to Qdrant in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Indexing"):
        batch = chunks[i:i + BATCH_SIZE]
        
        # Extract texts for embedding
        texts = [c["text"] for c in batch]
        
        # Get dense embeddings
        dense_vectors = embed_batch(openai_client, texts)
        
        # Create points
        points = []
        for j, chunk in enumerate(batch):
            # Generate UUID from chunk ID for consistency
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
            
            # Build vector dict
            if args.dense_only:
                vector_data = {"dense": dense_vectors[j]}
            else:
                # Create sparse vector for hybrid mode
                sparse_indices, sparse_values = create_sparse_vector(chunk["text"])
                vector_data = {
                    "dense": dense_vectors[j],
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                }
            
            point = PointStruct(
                id=point_id,
                vector=vector_data,
                payload={
                    "chunk_id": chunk["id"],
                    "work": chunk["work"],
                    "work_abbrev": chunk["work_abbrev"],
                    "location": chunk["location"],
                    "section_type": chunk["section_type"],
                    "title": chunk["title"],
                    "text": chunk["text"][:5000],  # Truncate for storage
                    "source_url": chunk.get("source_url", ""),
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        qdrant.upsert(collection_name=args.collection, points=points)
    
    # Get final count
    info = qdrant.get_collection(args.collection)
    print(f"\n{'='*50}")
    print(f"Indexing complete!")
    print(f"Collection: {args.collection}")
    print(f"Total points: {info.points_count}")


if __name__ == "__main__":
    main()
