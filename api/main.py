"""
FastAPI server for Aquinas RAG search and chat.

Endpoints:
- GET /health - Health check
- POST /search - Vector similarity search via Qdrant
- POST /chat - RAG-powered chat with citations
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- CONFIG ----------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "summa")
DEFAULT_TOP_K = 5

# ---------- GLOBALS ----------
qdrant_client = None


# ---------- MODELS ----------
class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    work_filter: Optional[str] = None  # e.g., "ST", "SCG", "DV"


class SearchResult(BaseModel):
    chunk_id: str
    work: str
    work_abbrev: str
    location: str
    title: str
    text: str
    score: float
    source_url: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class ChatRequest(BaseModel):
    message: str
    top_k: int = DEFAULT_TOP_K
    work_filter: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: list[SearchResult]


# ---------- HELPERS ----------
def get_qdrant_client():
    """Get or create Qdrant client."""
    global qdrant_client
    if qdrant_client is None:
        from qdrant_client import QdrantClient
        
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set")
        
        qdrant_client = QdrantClient(url=url, api_key=api_key, timeout=60)
    
    return qdrant_client


def embed_query(query: str) -> list[float]:
    """Embed a query using OpenAI."""
    from openai import OpenAI
    client = OpenAI()
    
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    
    return resp.data[0].embedding


def search_qdrant(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    work_filter: Optional[str] = None
) -> list[SearchResult]:
    """Search Qdrant for similar chunks."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, NamedVector
    
    client = get_qdrant_client()
    query_vector = embed_query(query)
    
    # Build filter if work specified
    search_filter = None
    if work_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="work_abbrev",
                    match=MatchValue(value=work_filter)
                )
            ]
        )
    
    # Search using query_points (new API)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )
    
    search_results = []
    for hit in results.points:
        payload = hit.payload or {}
        search_results.append(SearchResult(
            chunk_id=payload.get("chunk_id", ""),
            work=payload.get("work", ""),
            work_abbrev=payload.get("work_abbrev", ""),
            location=payload.get("location", ""),
            title=payload.get("title", ""),
            text=payload.get("text", ""),
            score=hit.score,
            source_url=payload.get("source_url", "")
        ))
    
    return search_results


def generate_chat_response(question: str, sources: list[SearchResult]) -> str:
    """Generate a chat response using retrieved context."""
    from openai import OpenAI
    client = OpenAI()
    
    # Build context from sources
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[{i}] {src.work_abbrev}: {src.location}\n"
            f"Title: {src.title}\n"
            f"{src.text}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a scholarly assistant specializing in the works of Thomas Aquinas (1225-1274), the Angelic Doctor.

You have access to passages from his major works:
- **Summa Theologica (ST)**: His systematic theological masterpiece
- **Summa Contra Gentiles (SCG)**: Apologetic work for non-believers  
- **De Veritate (DV)**: Disputed questions on truth
- **De Potentia Dei (DPD)**: On the power of God
- **Quaestiones de Anima (QDA)**: Questions on the soul
- **De Ente et Essentia (DBE)**: On being and essence
- **Commentary on Metaphysics (CM)**: Commentary on Aristotle

RESPONSE GUIDELINES:

1. **Answer directly** from the provided sources. Begin with a clear thesis statement.

2. **Cite sources** using bracketed numbers [1], [2], etc. Include the work name for context:
   - "In the *Summa Theologica* [1], Aquinas argues that..."
   - "As he explains in *De Veritate* [2]..."

3. **Use Aquinas's method**: When relevant, explain his dialectical structure:
   - **Objections**: What opponents argue
   - **Sed Contra**: The authoritative counter-position
   - **Respondeo**: His definitive answer
   - **Replies**: How he addresses each objection

4. **Preserve philosophical precision**: Use technical terms (act/potency, essence/existence, form/matter) accurately.

5. **Make it accessible**: Explain complex concepts clearly for modern readers without dumbing them down.

6. **Be honest**: If the sources don't fully answer the question, say so and explain what would be needed.

7. **Format clearly**: Use **bold** for key terms, *italics* for work titles, and organized paragraphs."""

    user_prompt = f"""Answer this question using ONLY the provided source passages from Aquinas's works.

**Question:** {question}

---

**Available Sources:**

{context}

---

Provide a clear, well-structured answer with proper citations. Use markdown formatting for readability."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message.content


# ---------- LIFESPAN ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Qdrant client on startup."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(COLLECTION_NAME)
        print(f"Connected to Qdrant. Collection '{COLLECTION_NAME}' has {info.points_count} points.")
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant: {e}")
        print("Search and chat endpoints may not work.")
    yield


# ---------- APP ----------
app = FastAPI(
    title="Aquinas RAG API",
    description="Search and chat with the complete works of Thomas Aquinas using vector similarity",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", ""),  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "status": "healthy",
            "collection": COLLECTION_NAME,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/works")
async def list_works():
    """List available works and their abbreviations."""
    return {
        "works": [
            {"abbrev": "ST", "name": "Summa Theologica"},
            {"abbrev": "SCG", "name": "Summa Contra Gentiles"},
            {"abbrev": "DV", "name": "De Veritate"},
            {"abbrev": "DPD", "name": "De Potentia Dei"},
            {"abbrev": "QDA", "name": "Quaestiones de Anima"},
            {"abbrev": "DBE", "name": "On Being and Essence"},
            {"abbrev": "CM", "name": "Commentary on Metaphysics"},
        ]
    }


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search for chunks similar to the query.
    
    Optionally filter by work using work_filter (e.g., "ST", "SCG").
    Returns top-k most similar chunks with their metadata and scores.
    """
    try:
        results = search_qdrant(
            request.query, 
            request.top_k,
            request.work_filter
        )
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG-powered chat endpoint.
    
    Retrieves relevant chunks and generates a response using GPT-4o-mini.
    Optionally filter sources by work using work_filter.
    """
    try:
        # Retrieve relevant sources
        sources = search_qdrant(
            request.message, 
            request.top_k,
            request.work_filter
        )
        
        if not sources:
            return ChatResponse(
                response="I couldn't find any relevant passages in Aquinas's works for your question.",
                sources=[]
            )
        
        # Generate response with citations
        response_text = generate_chat_response(request.message, sources)
        
        return ChatResponse(response=response_text, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
