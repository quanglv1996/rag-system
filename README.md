# AI Automation Platform

A **production-ready, extensible AI platform** built with Python 3.11+, FastAPI, and Clean Architecture. Integrates multiple AI providers, social platforms, and a complete RAG (Retrieval-Augmented Generation) pipeline — all configurable without code changes.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                    │
│          /api/v1/ai  /api/v1/rag  /api/v1/facebook  ...     │
└───────────────────────────┬──────────────────────────────────┘
                            │ Depends()
┌───────────────────────────▼──────────────────────────────────┐
│                     Service Layer                             │
│     AIService  RAGService  FacebookService  TelegramService  │
└──────────┬──────────────────────────────────────┬────────────┘
           │                                      │
┌──────────▼────────────┐            ┌────────────▼────────────┐
│   AI Provider (ABC)   │            │  Social Provider (ABC)  │
│  OpenAI | Google      │            │  Facebook | Telegram    │
│  Anthropic | ...      │            │  YouTube  | TikTok      │
└───────────────────────┘            └─────────────────────────┘
           │
┌──────────▼────────────┐
│  Vector DB (ABC)      │
│  ChromaDB | FAISS     │
│  Pinecone | ...       │
└───────────────────────┘
```

### Key Design Patterns

| Pattern | Usage |
|---|---|
| **Strategy** | AI/Social/Vector providers — swap without code changes |
| **Factory** | `AIProviderFactory`, `VectorProviderFactory` |
| **Repository** | `BaseRepository[T]`, `DocumentRepository` |
| **Adapter** | Each provider adapts external SDK to internal interface |
| **Dependency Injection** | FastAPI `Depends()` throughout |

---

## Folder Structure

```
project/
├── app/
│   ├── api/v1/routers/       # HTTP route handlers
│   ├── core/                 # Config, logging, security, exceptions
│   ├── database/             # SQLAlchemy engine, session, ORM models
│   ├── common/               # Enums, constants, utils, shared schemas
│   ├── interfaces/           # ABCs: AIProvider, VectorDatabase, ...
│   ├── providers/
│   │   ├── ai/               # OpenAI, Google implementations
│   │   ├── social/           # Facebook, TikTok, YouTube, Telegram
│   │   └── vector/           # FAISS, ChromaDB implementations
│   ├── rag/                  # Document loader, chunker, embedder, retriever
│   ├── services/             # Business logic layer
│   ├── repositories/         # Data access layer
│   ├── schemas/              # Pydantic request/response models
│   ├── middleware/           # Logging, rate limiting
│   ├── events/               # Startup/shutdown handlers
│   ├── prompts/              # Jinja2 prompt templates & manager
│   └── main.py               # Application factory
├── tests/                    # pytest test suite
├── migrations/               # Alembic database migrations
├── docker/                   # Dockerfile, nginx.conf
├── requirements/             # base.txt, dev.txt, prod.txt
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Docker & Docker Compose (for containerized deployment)

### Local Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd rag-system

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
make install-dev

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# 5. Run database migrations
make migrate

# 6. Start the development server
make dev
```

---

## Docker Deployment

```bash
# Build and start all services
make docker-build
make docker-up

# Check logs
make docker-logs

# Stop everything
make docker-down
```

Services started by Docker Compose:

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI application |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis cache |
| `chromadb` | 8001 | ChromaDB vector store |
| `nginx` | 80 | Reverse proxy |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all required values:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing secret (min 32 chars) |
| `DATABASE_URL` | PostgreSQL async URL |
| `REDIS_URL` | Redis connection URL |

### AI Providers

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google Generative AI key |
| `LLM_PROVIDER` | `openai` or `google` |
| `EMBEDDING_PROVIDER` | `openai` or `google` |

### Social Platforms

| Variable | Description |
|---|---|
| `FACEBOOK_APP_ID` | Facebook App ID |
| `FACEBOOK_PAGE_TOKEN` | Page access token |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API token |
| `YOUTUBE_CLIENT_ID` | Google OAuth client ID |
| `TIKTOK_CLIENT_ID` | TikTok app client key |

---

## Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Specific module
make test-rag
make test-ai
```

---

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs`

### AI

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ai/chat` | Chat completion |
| POST | `/ai/chat/stream` | Streaming chat (SSE) |
| POST | `/ai/embedding` | Generate embeddings |
| POST | `/ai/image` | Image generation |
| POST | `/ai/vision` | Image analysis |

### RAG

| Method | Endpoint | Description |
|---|---|---|
| POST | `/rag/upload` | Upload & index document |
| POST | `/rag/query` | Question answering |
| DELETE | `/rag/documents/{id}` | Delete document |

### Facebook

| Method | Endpoint | Description |
|---|---|---|
| POST | `/facebook/post` | Publish post |
| POST | `/facebook/message` | Send Messenger message |
| POST | `/facebook/comment` | Add comment |
| GET | `/facebook/webhook` | Webhook verification |
| POST | `/facebook/webhook` | Receive events |

### Telegram

| Method | Endpoint | Description |
|---|---|---|
| POST | `/telegram/send` | Send message |
| POST | `/telegram/send/media` | Send media |
| POST | `/telegram/send/keyboard` | Send inline keyboard |
| POST | `/telegram/webhook/set` | Register webhook |

### YouTube

| Method | Endpoint | Description |
|---|---|---|
| POST | `/youtube/upload` | Upload video |
| POST | `/youtube/playlists` | Create playlist |
| GET | `/youtube/videos/{id}/analytics` | Get analytics |

### TikTok

| Method | Endpoint | Description |
|---|---|---|
| POST | `/tiktok/upload` | Publish video |
| GET | `/tiktok/creator/info` | Creator profile |
| GET | `/tiktok/videos` | List videos |

---

## RAG Pipeline

```
Document File
    │
    ▼
DocumentLoader       ← PDF, DOCX, TXT, MD, HTML
    │
    ▼
TextCleaner          ← Remove noise, normalize whitespace
    │
    ▼
TextChunker          ← FIXED / RECURSIVE / SENTENCE strategy
    │
    ▼
EmbeddingService     ← OpenAI / Google embedding + Redis cache
    │
    ▼
VectorDatabase       ← ChromaDB / FAISS upsert
    │ (query time)
    ▼
RAGRetriever         ← ANN search + optional re-ranking
    │
    ▼
PromptBuilder        ← Inject context into Jinja2 template
    │
    ▼
LLM (AIProvider)     ← OpenAI / Google generation
    │
    ▼
Answer + Sources
```

---

## Adding a New AI Provider

1. Create `app/providers/ai/anthropic_provider.py`
2. Implement the `AIProvider` ABC
3. Register in `AIProviderFactory._registry`
4. Set `LLM_PROVIDER=anthropic` in `.env`

No existing code changes required.

---

## Adding a New Social Platform

1. Create `app/providers/social/instagram_provider.py`
2. Implement the `SocialProvider` ABC
3. Create `app/services/instagram_service.py`
4. Add router in `app/api/v1/routers/instagram_router.py`
5. Include in `app/api/v1/__init__.py`

---

## Code Quality

```bash
make lint          # Ruff linter
make format        # Black formatter
make type-check    # MyPy
make pre-commit-run  # All pre-commit hooks
```

---

## License

MIT License — see LICENSE file for details.
