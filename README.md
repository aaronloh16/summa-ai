# Aquinas RAG

A Retrieval-Augmented Generation (RAG) application for exploring the works of Thomas Aquinas. Search and chat across 7 major works including the Summa Theologica, Summa Contra Gentiles, De Veritate, and more.

## Features

- **Semantic Search**: Find relevant passages across all of Aquinas's works using vector similarity
- **RAG Chat**: Ask questions and get cited answers from the corpus
- **Work Filtering**: Filter searches by specific works (ST, SCG, DV, etc.)
- **11,000+ Chunks**: Indexed passages from 7 major works
- **Cloud Vector DB**: Uses Qdrant Cloud for production-ready vector search

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js App   │────▶│   FastAPI       │────▶│   Qdrant Cloud  │
│   (Frontend)    │     │   (Backend)     │     │   (Vector DB)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌─────────────────┐
                        │   OpenAI API    │
                        │   (Embeddings   │
                        │    + Chat)      │
                        └─────────────────┘
```

## Works Indexed

| Abbreviation | Work |
|--------------|------|
| ST | Summa Theologica |
| SCG | Summa Contra Gentiles |
| DV | De Veritate (On Truth) |
| DPD | De Potentia Dei (On the Power of God) |
| QDA | Quaestiones de Anima (Questions on the Soul) |
| DBE | De Ente et Essentia (On Being and Essence) |
| CM | Commentary on Aristotle's Metaphysics |

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- OpenAI API key
- Qdrant Cloud account (or local Qdrant instance)

### Setup

1. **Clone and setup Python environment:**
   ```bash
   cd summa-rag
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   Required environment variables:
   ```
   OPENAI_API_KEY=sk-...
   QDRANT_URL=https://your-cluster.qdrant.cloud:6333
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_COLLECTION=summa
   ```

3. **Setup Next.js frontend:**
   ```bash
   cd web
   npm install
   ```

4. **Create `.env.local` for the frontend:**
   ```bash
   # web/.env.local
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### Running Locally

1. **Start the API server:**
   ```bash
   cd summa-rag
   source .venv/bin/activate
   uvicorn api.main:app --reload --port 8000
   ```

2. **Start the frontend (in a separate terminal):**
   ```bash
   cd summa-rag/web
   npm run dev
   ```

3. Open http://localhost:3000 in your browser.

### Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Deployment

### Deploying the Frontend to Vercel

1. **Push to GitHub** (if not already done)

2. **Import to Vercel:**
   - Go to [Vercel](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - Set the **Root Directory** to `web`

3. **Configure environment variables in Vercel:**
   ```
   NEXT_PUBLIC_API_URL=https://your-api-domain.com
   ```

4. **Deploy!**

### Deploying the API

The API can be deployed to various platforms. Options include:

#### Option A: Railway / Render / Fly.io

1. Create a new project on your chosen platform
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION`
   - `FRONTEND_URL` (your Vercel URL for CORS)

#### Option B: Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY .env .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t aquinas-rag .
docker run -p 8000:8000 --env-file .env aquinas-rag
```

#### Option C: Serverless (AWS Lambda / Google Cloud Functions)

The FastAPI app can be wrapped with Mangum for serverless deployment. See [Mangum documentation](https://mangum.io/).

## Updating the Qdrant Collection

If you need to re-index the collection (e.g., after adding new documents or fixing parsing):

### 1. Prepare Document Chunks

Parse your documents into `out/chunks.jsonl`:

```bash
source .venv/bin/activate
python scripts/parse_corpus.py
```

Each line should be a JSON object with these fields:
```json
{
  "id": "unique_chunk_id",
  "text": "The chunk text content...",
  "work": "Summa Theologica",
  "work_abbrev": "ST",
  "location": "Prima Pars, Q1, A1",
  "section_type": "respondeo",
  "title": "Whether sacred doctrine is a science?",
  "source_url": "https://..."  // optional
}
```

### 2. Re-index to Qdrant

```bash
# Recreate the collection (deletes existing data!)
python scripts/index_to_qdrant.py --recreate --dense-only

# Or add to existing collection (skips duplicates by ID)
python scripts/index_to_qdrant.py --dense-only
```

**Important flags:**
- `--recreate`: Delete and recreate the collection (fresh start)
- `--dense-only`: Use only dense vectors (recommended for simplicity)
- `--collection NAME`: Override collection name (default: from env)

### 3. Verify the Index

```bash
python -c "
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()c
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
info = client.get_collection('summa')
print(f'Points: {info.points_count}')
"
```

## Adding New Documents

1. **Add the source file** to `writings/` or `non-html/`

2. **Update `parse_corpus.py`** to handle the new document format:
   - Add a new parsing function
   - Register it in the `PARSERS` dict
   - Define the appropriate chunking strategy

3. **Run the parser** to regenerate `chunks.jsonl`:
   ```bash
   python scripts/parse_corpus.py
   ```

4. **Re-index** to Qdrant:
   ```bash
   python scripts/index_to_qdrant.py --recreate --dense-only
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with Qdrant status |
| `/works` | GET | List available works and abbreviations |
| `/search` | POST | Vector similarity search |
| `/chat` | POST | RAG chat with citations |

### Example Requests

**Search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is truth?", "top_k": 5, "work_filter": "DV"}'
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the five ways to prove God exists?", "top_k": 5}'
```

## Cost Considerations

- **OpenAI Embeddings**: ~$0.02 per 1M tokens (text-embedding-3-small)
- **OpenAI Chat**: ~$0.15 per 1M input tokens (gpt-4o-mini)
- **Qdrant Cloud**: Free tier includes 1GB storage (~100k vectors)

For the full corpus (~11k chunks), initial indexing costs approximately $0.10-0.20.

## Troubleshooting

### "Index not loaded" error
- Ensure the Qdrant collection exists and has points
- Check that `QDRANT_URL` and `QDRANT_API_KEY` are set correctly

### Slow search responses
- Qdrant free tier may have cold starts
- Consider upgrading to a paid tier for consistent performance

### CORS errors in browser
- Add your frontend URL to `FRONTEND_URL` environment variable
- Restart the API server

### Empty search results
- Verify the collection has points: `GET /health`
- Try a broader query without work filters
- Check that chunks have the expected `work_abbrev` values

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Text sources from [isidore.co](https://isidore.co/aquinas/)
- Dominican Order seal used with respect for the Order of Preachers
