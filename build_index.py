"""
Build FAISS index from Summa Theologica XML.

This script:
1. Parses summa.xml to extract articles
2. Chunks articles into sections (objections, sed_contra, respondeo, replies)
3. Embeds chunks using OpenAI text-embedding-3-small
4. Builds a FAISS index for fast similarity search
5. Saves index and metadata to out/ directory
"""

import os
import re
import json
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- CONFIG ----------
XML_PATH = "summa.xml"
OUT_DIR = "out"
INDEX_PATH = f"{OUT_DIR}/summa.faiss"
META_PATH = f"{OUT_DIR}/summa_meta.jsonl"

# OpenAI embedding model
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64  # Reduced batch size for safety
MAX_TOKENS = 7500  # Model limit is 8192, leave buffer
MAX_CHARS = 25000  # ~6250 tokens, safe margin (avg 4 chars/token)


# ---------- HELPERS ----------
def clean_ws(s: str) -> str:
    """Clean up whitespace in text."""
    return re.sub(r"\s+", " ", s).strip()


def text_of(elem: ET.Element) -> str:
    """Extract all text from an XML element."""
    return clean_ws("".join(elem.itertext()))


def parse_articles(xml_path: str) -> list[dict]:
    """
    Parse the Summa XML and extract articles.
    
    Each article contains:
    - article_id: unique identifier (e.g., FP_Q1_A2)
    - title: article title from h4 element
    - paras: list of paragraph texts
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    articles = []
    for div4 in root.iter("div4"):
        article_id = div4.attrib.get("id", "").strip()
        if not article_id:
            continue

        h4 = div4.find("h4")
        title = text_of(h4) if h4 is not None else ""

        paras = []
        for p in div4.findall("p"):
            t = text_of(p)
            if t:
                paras.append(t)

        if title or paras:
            articles.append({
                "article_id": article_id,
                "title": title,
                "paras": paras
            })

    return articles


def split_into_sections(article: dict) -> list[dict]:
    """
    Split an article into semantic sections.
    
    Returns list of chunks with section labels:
    - objection: Individual objections
    - sed_contra: "On the contrary" section
    - respondeo: "I answer that" section (main teaching)
    - reply: Replies to objections
    """
    title = article["title"]
    full = "\n".join(article["paras"]).strip()
    
    if not full:
        return []

    # Regex patterns to identify section boundaries
    patterns = [
        ("objection", r"\bObjection\s+\d+:"),
        ("sed_contra", r"\bOn the contrary\b"),
        ("respondeo", r"\bI answer that\b"),
        ("reply", r"\bReply to Objection\s+\d+:"),
    ]

    # Find all section start positions
    hits = []
    for sec, pat in patterns:
        for m in re.finditer(pat, full):
            hits.append((m.start(), sec))
    hits.sort(key=lambda x: x[0])

    # If no section markers found, store entire article as one chunk
    if not hits:
        return [{
            "chunk_id": f"{article['article_id']}::article",
            "section": "article",
            "text": f"{title}\n\n{full}".strip()
        }]

    # Slice text by section boundaries
    slices = []
    for i, (start, sec) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(full)
        slices.append((sec, full[start:end].strip()))

    # Group slices by section type
    grouped = {}
    for sec, txt in slices:
        grouped.setdefault(sec, []).append(txt)

    # Create chunks in logical order
    ordered = ["objection", "sed_contra", "respondeo", "reply"]
    chunks = []
    
    for sec in ordered:
        if sec in grouped:
            joined = "\n\n".join(grouped[sec])
            chunks.append({
                "chunk_id": f"{article['article_id']}::{sec}",
                "section": sec,
                "text": f"{title}\n\n{joined}".strip()
            })
    
    return chunks


def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate text to fit within token limits."""
    if len(text) <= max_chars:
        return text
    # Truncate and add indicator
    return text[:max_chars - 20] + "\n\n[... truncated]"


def embed_texts_openai(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts using OpenAI's embedding API.
    
    Returns numpy array of shape (len(texts), embedding_dim).
    """
    from openai import OpenAI
    client = OpenAI()
    
    # Truncate any texts that are too long
    truncated_texts = [truncate_text(t) for t in texts]

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=truncated_texts
    )
    
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)


def main():
    """Main entry point for building the FAISS index."""
    
    # Check for XML file
    if not os.path.exists(XML_PATH):
        print(f"Error: {XML_PATH} not found!")
        print("Please place your summa.xml file in the project root.")
        return
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse XML
    print("Parsing XML...")
    articles = parse_articles(XML_PATH)
    print(f"Found {len(articles)} articles")

    # Build chunk list
    print("Splitting into sections...")
    chunks = []
    for a in articles:
        chunks.extend(split_into_sections(a))
    
    print(f"Created {len(chunks)} chunks")
    
    # Show section distribution
    section_counts = {}
    for c in chunks:
        sec = c["section"]
        section_counts[sec] = section_counts.get(sec, 0) + 1
    print("Section distribution:")
    for sec, count in sorted(section_counts.items()):
        print(f"  {sec}: {count}")

    # Check for long chunks
    long_chunks = [c for c in chunks if len(c["text"]) > MAX_CHARS]
    if long_chunks:
        print(f"\nNote: {len(long_chunks)} chunks exceed {MAX_CHARS} chars and will be truncated for embedding.")
        for c in long_chunks[:5]:  # Show first 5
            print(f"  - {c['chunk_id']}: {len(c['text'])} chars")
        if len(long_chunks) > 5:
            print(f"  ... and {len(long_chunks) - 5} more")

    # Embed in batches
    print(f"\nEmbedding chunks (batch size: {BATCH_SIZE})...")
    all_vecs = []
    all_meta = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vecs = embed_texts_openai(texts)
        all_vecs.append(vecs)

        for c in batch:
            all_meta.append({
                "chunk_id": c["chunk_id"],
                "section": c["section"],
                "text": c["text"],  # Keep full text in metadata
            })

    # Stack all vectors
    X = np.vstack(all_vecs).astype(np.float32)
    dim = X.shape[1]
    print(f"\nEmbedding dimension: {dim}")
    print(f"Total vectors: {X.shape[0]}")

    # Build FAISS index with inner product (cosine similarity after normalization)
    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(dim)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(X)
    index.add(X)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index: {INDEX_PATH}")

    # Save metadata (one JSON object per line)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved metadata: {META_PATH}")

    print("\nDone! Index is ready for search.")


if __name__ == "__main__":
    main()

