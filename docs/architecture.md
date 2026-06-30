# AI Automation Platform — Architecture & Diagrams

## Table of Contents
1. [Sequence Diagrams](#sequence-diagrams)
2. [Class Diagrams](#class-diagrams)
3. [Component Diagram](#component-diagram)
4. [ER Diagram](#er-diagram)
5. [Deployment Diagram](#deployment-diagram)

---

## Sequence Diagrams

### RAG Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Router
    participant RAGService
    participant EmbeddingService
    participant Redis as Redis Cache
    participant VectorDB as ChromaDB/FAISS
    participant LLM as OpenAI/Google

    Client->>API: POST /api/v1/rag/query {question}
    API->>RAGService: query(question, collection)
    RAGService->>EmbeddingService: embed_single(question)
    EmbeddingService->>Redis: GET cache key
    Redis-->>EmbeddingService: cache miss
    EmbeddingService->>LLM: embedding(question)
    LLM-->>EmbeddingService: [0.1, 0.2, ...]
    EmbeddingService->>Redis: SET cache key, TTL 24h
    EmbeddingService-->>RAGService: query_embedding
    RAGService->>VectorDB: search(query_embedding, top_k=5)
    VectorDB-->>RAGService: [RetrievedChunk x 5]
    RAGService->>LLM: chat(prompt + context)
    LLM-->>RAGService: answer
    RAGService-->>API: RAGQueryResult
    API-->>Client: {answer, sources, model}
```

### Workflow Execution Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Celery as Celery Worker
    participant WorkflowEngine
    participant NodeRegistry
    participant AIService
    participant RAGService
    participant TelegramService

    Client->>API: POST /api/v1/workflow/execute
    API->>Celery: execute_workflow.delay(definition, context)
    Celery->>WorkflowEngine: execute(workflow, context)
    WorkflowEngine->>NodeRegistry: get(NodeType.RAG)
    NodeRegistry-->>WorkflowEngine: RAGNode class
    WorkflowEngine->>RAGService: query(question)
    RAGService-->>WorkflowEngine: RAGQueryResult
    WorkflowEngine->>NodeRegistry: get(NodeType.AI)
    WorkflowEngine->>AIService: chat(messages)
    AIService-->>WorkflowEngine: ChatResponse
    WorkflowEngine->>NodeRegistry: get(NodeType.SOCIAL)
    WorkflowEngine->>TelegramService: send_text(chat_id, answer)
    TelegramService-->>WorkflowEngine: sent
    WorkflowEngine-->>Celery: WorkflowContext {status: completed}
    Celery-->>API: task result
    API-->>Client: 200 {execution_id, status}
```

### OAuth2 + Auto-Refresh Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant OAuthManager
    participant Redis
    participant Platform as Facebook/Google/TikTok

    User->>API: GET /oauth/facebook/login
    API->>OAuthManager: get_facebook_login_url(redirect_uri)
    OAuthManager-->>API: auth_url
    API-->>User: redirect to auth_url

    User->>Platform: Login & Authorize
    Platform->>API: GET /oauth/facebook/callback?code=xxx
    API->>OAuthManager: exchange_facebook_code(code, redirect_uri)
    OAuthManager->>Platform: POST /oauth/access_token
    Platform-->>OAuthManager: {access_token, expires_in}
    OAuthManager->>Redis: SET token:facebook:user, TTL
    OAuthManager-->>API: OAuthToken

    Note over API,Redis: Later — token expired
    API->>OAuthManager: get_valid_token("facebook", user_id)
    OAuthManager->>Redis: GET token:facebook:user
    Redis-->>OAuthManager: OAuthToken (expired)
    OAuthManager->>Platform: refresh token request
    Platform-->>OAuthManager: new access_token
    OAuthManager->>Redis: SET new token, TTL
    OAuthManager-->>API: refreshed OAuthToken
```

### Webhook Event Processing

```mermaid
sequenceDiagram
    participant Platform as Facebook/Telegram
    participant Nginx
    participant API as Webhook Router
    participant Gateway as WebhookGateway
    participant Redis
    participant Handler as Event Handler
    participant Celery

    Platform->>Nginx: POST /api/v1/webhook/facebook
    Nginx->>API: forward request
    API->>Gateway: process_facebook(request, signature)
    Gateway->>Gateway: verify_facebook(payload, signature)
    Gateway->>Redis: SET dedup key (NX, TTL=5min)
    Redis-->>Gateway: OK (not duplicate)
    Gateway->>Gateway: parse events
    Gateway->>Handler: dispatch("message", payload)
    Handler->>Celery: send_telegram_message.delay(...)
    Handler->>Celery: execute_workflow.delay(...)
    Gateway-->>API: {status: received}
    API-->>Platform: 200 OK
```

---

## Class Diagrams

### AI Provider Hierarchy

```mermaid
classDiagram
    class AIProvider {
        <<abstract>>
        +provider_name: str
        +chat(request: ChatRequest) ChatResponse
        +stream(request: ChatRequest) AsyncGenerator
        +embedding(request: EmbeddingRequest) EmbeddingResponse
        +image_generation(request) ImageGenerationResponse
        +speech_to_text(request) SpeechToTextResponse
        +text_to_speech(request) TextToSpeechResponse
        +vision(request: VisionRequest) VisionResponse
        +get_capabilities() dict
    }

    class OpenAIProvider {
        -_client: AsyncOpenAI
        -_model: str
        -_embedding_model: str
        +provider_name = "openai"
        +chat(request) ChatResponse
        +stream(request) AsyncGenerator
        +embedding(request) EmbeddingResponse
    }

    class GoogleProvider {
        -_model_name: str
        -_embedding_model_name: str
        +provider_name = "google"
        +chat(request) ChatResponse
        +stream(request) AsyncGenerator
    }

    class AnthropicProvider {
        -_client: AsyncAnthropic
        +provider_name = "anthropic"
        +chat(request) ChatResponse
    }

    class OllamaProvider {
        -_base_url: str
        -_client: AsyncClient
        +provider_name = "ollama"
        +chat(request) ChatResponse
    }

    class AIProviderFactory {
        -_registry: dict
        +create(name: str) AIProvider
        +register(name, class_path)
    }

    AIProvider <|-- OpenAIProvider
    AIProvider <|-- GoogleProvider
    AIProvider <|-- AnthropicProvider
    AIProvider <|-- OllamaProvider
    AIProviderFactory --> AIProvider
```

### RAG Pipeline

```mermaid
classDiagram
    class RAGPipeline {
        -_loader: DocumentLoader
        -_chunker: TextChunker
        -_embedding_service: EmbeddingService
        -_vector_db: VectorDatabase
        -_retriever: RAGRetriever
        -_ai_provider: AIProvider
        +index(source, collection) IndexResult
        +query(question, collection) RAGQueryResult
        +delete_document(doc_id, chunk_ids)
    }

    class DocumentLoader {
        -_loaders: dict
        +load(path) LoadedDocument
        +load_bytes(bytes, filename) LoadedDocument
        -_load_pdf(path) LoadedDocument
        -_load_docx(path) LoadedDocument
    }

    class TextChunker {
        -_strategy: ChunkStrategy
        -_chunk_size: int
        -_chunk_overlap: int
        +chunk(text, metadata) list~TextChunk~
        -_recursive_chunk(text) list
        -_sentence_chunk(text) list
    }

    class EmbeddingService {
        -_provider: AIProvider
        -_redis: Redis
        +embed_texts(texts) list~list~float~~
        +embed_single(text) list~float~
        -_make_cache_key(text, model) str
    }

    class RAGRetriever {
        -_vector_db: VectorDatabase
        -_embedding_service: EmbeddingService
        -_top_k: int
        +retrieve(query, collection) list~RetrievedChunk~
        +retrieve_with_rerank(query) list~RetrievedChunk~
    }

    class HybridRetriever {
        -_dense_retriever: RAGRetriever
        -_bm25: BM25Retriever
        -_alpha: float
        +retrieve(query, collection) list~RetrievedChunk~
        -_rrf_merge(dense, sparse) list
    }

    RAGPipeline --> DocumentLoader
    RAGPipeline --> TextChunker
    RAGPipeline --> EmbeddingService
    RAGPipeline --> RAGRetriever
    HybridRetriever --> RAGRetriever
    EmbeddingService --> AIProvider
```

### Workflow Engine

```mermaid
classDiagram
    class WorkflowEngine {
        -_dependencies: dict
        +execute(workflow, context) WorkflowContext
        +resume(workflow, context, data) WorkflowContext
        -_execute_node(workflow, node_def, context)
        -_run_with_retry(executor, context, node_def)
    }

    class BaseNode {
        <<abstract>>
        +definition: NodeDefinition
        +execute(context, dependencies) NodeExecutionResult
        #_success(output, next_nodes) NodeExecutionResult
        #_failure(error) NodeExecutionResult
        #get_input(context, key) Any
        #set_output(context, key, value)
    }

    class AINode {
        +execute(context, deps) NodeExecutionResult
    }
    class RAGNode {
        +execute(context, deps) NodeExecutionResult
    }
    class ConditionNode {
        +execute(context, deps) NodeExecutionResult
    }
    class SocialNode {
        +execute(context, deps) NodeExecutionResult
    }
    class ApprovalNode {
        +execute(context, deps) NodeExecutionResult
    }

    class NodeRegistry {
        -_registry: dict~NodeType, type~BaseNode~~
        +register(type, class)
        +get(type) type~BaseNode~
    }

    class WorkflowLoader {
        +from_json(json_str) WorkflowDefinition
        +from_yaml(yaml_str) WorkflowDefinition
        +from_file(path) WorkflowDefinition
    }

    WorkflowEngine --> NodeRegistry
    WorkflowEngine --> WorkflowLoader
    NodeRegistry --> BaseNode
    BaseNode <|-- AINode
    BaseNode <|-- RAGNode
    BaseNode <|-- ConditionNode
    BaseNode <|-- SocialNode
    BaseNode <|-- ApprovalNode
```

---

## Component Diagram

```mermaid
graph TB
    subgraph Client Layer
        WebApp[Web App / Mobile]
        ExternalAPI[External Services]
    end

    subgraph API Gateway
        Nginx[Nginx Reverse Proxy]
        RateLimit[Rate Limiter]
        Auth[JWT / API Key Auth]
    end

    subgraph Application Layer
        FastAPI[FastAPI App]
        subgraph Routers
            AIRouter[/ai/*]
            RAGRouter[/rag/*]
            SocialRouter[/facebook, /telegram, /youtube, /tiktok]
            WorkflowRouter[/workflow/*]
        end
        subgraph Services
            AIService[AIService]
            RAGService[RAGService]
            SocialServices[Facebook/YouTube/Telegram/TikTok Services]
            WorkflowEngine[WorkflowEngine]
        end
    end

    subgraph Provider Layer
        AIProviders[OpenAI/Google/Anthropic/Ollama]
        VectorDB[ChromaDB/FAISS]
        SocialProviders[Facebook/YouTube/Telegram/TikTok APIs]
    end

    subgraph Infrastructure
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis)]
        Celery[Celery Workers]
        APScheduler[APScheduler]
    end

    WebApp --> Nginx
    ExternalAPI --> Nginx
    Nginx --> RateLimit --> Auth --> FastAPI
    FastAPI --> Routers
    Routers --> Services
    Services --> AIProviders
    Services --> VectorDB
    Services --> SocialProviders
    Services --> PostgreSQL
    Services --> Redis
    Services --> Celery
    APScheduler --> Celery
```

---

## ER Diagram

```mermaid
erDiagram
    USERS {
        uuid id PK
        string email UK
        string hashed_password
        string full_name
        boolean is_active
        boolean is_superuser
        timestamp created_at
        timestamp updated_at
        timestamp deleted_at
    }

    DOCUMENTS {
        uuid id PK
        string title
        string source
        string doc_type
        string status
        int chunk_count
        text metadata_json
        timestamp created_at
        timestamp updated_at
    }

    DOCUMENT_CHUNKS {
        uuid id PK
        uuid document_id FK
        text content
        int chunk_index
        int start_char
        int end_char
        string vector_id
        text metadata_json
        timestamp created_at
    }

    CONVERSATIONS {
        uuid id PK
        uuid user_id FK
        string title
        string provider
        string model
        timestamp created_at
        timestamp updated_at
    }

    CONVERSATION_MESSAGES {
        uuid id PK
        uuid conversation_id FK
        string role
        text content
        int token_count
        timestamp created_at
    }

    USERS ||--o{ CONVERSATIONS : "has"
    CONVERSATIONS ||--o{ CONVERSATION_MESSAGES : "contains"
    DOCUMENTS ||--o{ DOCUMENT_CHUNKS : "split into"
```

---

## Deployment Diagram

```mermaid
graph TB
    subgraph Internet
        Users[Users / Clients]
        SocialPlatforms[Facebook/Telegram/YouTube/TikTok]
    end

    subgraph Cloud / VPS
        subgraph Docker Compose
            Nginx[Nginx :80/:443]
            API[FastAPI :8000]
            Celery[Celery Workers]
            Flower[Flower :5555]
        end

        subgraph Data Layer
            PostgreSQL[(PostgreSQL :5432)]
            Redis[(Redis :6379)]
            ChromaDB[(ChromaDB :8001)]
        end

        subgraph Monitoring
            Prometheus[(Prometheus)]
            Grafana[Grafana :3000]
            Jaeger[Jaeger :16686]
        end
    end

    subgraph External AI
        OpenAI[OpenAI API]
        Google[Google GenAI API]
        Anthropic[Anthropic API]
    end

    Users --> Nginx
    SocialPlatforms --> Nginx
    Nginx --> API
    API --> PostgreSQL
    API --> Redis
    API --> ChromaDB
    API --> Celery
    API --> OpenAI
    API --> Google
    API --> Anthropic
    Celery --> Redis
    Celery --> PostgreSQL
    Celery --> ChromaDB
    Prometheus --> API
    Grafana --> Prometheus
    API --> Jaeger
```
