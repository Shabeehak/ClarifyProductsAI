# ClarifyProducts.AI - System Architecture

## System Overview

ClarifyProducts.AI is an ML-powered product review analysis platform that aggregates reviews from multiple sources (YouTube, Reddit, Google Shopping) and provides sentiment analysis, summarization, and conversational AI capabilities. The system supports both text-based and image-based product search.

## High-Level Architecture

```mermaid
graph LR
    UI["Streamlit Frontend<br/>Port 8501"]

    subgraph API["FastAPI Backend<br/>Port 8000"]
        ROUTES["API v1 Routes"]
        SERVICES["Service Layer<br/>• Recognition Service<br/>• Smart Product Service<br/>• RAG Service<br/>• LLM Service<br/>• Cache Service"]
        CACHE["Redis Cache<br/>24h TTL"]
        MLOPS["MLflow Tracking"]
    end

    subgraph MODELS["Local ML Models"]
        CLIP["CLIP ViT-B/32<br/>Image Category Recognition"]
        OCR["PaddleOCR<br/>Text Extraction"]
        BART["BART Large-CNN<br/>Review Summarization"]
        DISTILBERT["DistilBERT<br/>Sentiment Analysis"]
    end

    subgraph LLM["External LLM"]
        GEM["Gemini Multimodal<br/>Product Identification"]
    end

    subgraph SEARCH["External Review Sources"]
        SERP["SerpAPI (Retry Logic)"]
        GOOGLE["Google Shopping"]
        YT["YouTube Reviews"]
        REDDIT["Reddit Discussions"]
    end

    UI --> ROUTES
    ROUTES --> SERVICES

    SERVICES --> CACHE
    SERVICES --> CLIP
    SERVICES --> OCR
    SERVICES --> BART
    SERVICES --> DISTILBERT
    SERVICES --> GEM

    SERVICES --> SERP
    SERP --> GOOGLE
    SERP --> YT
    SERP --> REDDIT

    CLIP -.-> MLOPS
    OCR -.-> MLOPS
    BART -.-> MLOPS
    DISTILBERT -.-> MLOPS

    SERVICES --> UI

    %% ----------------------------------
    %% COLOR DEFINITIONS
    %% ----------------------------------
    classDef ui fill:#4f8ef7,stroke:#1e3a8a,color:#fff,font-weight:bold;
    classDef api fill:#7c3aed,stroke:#4c1d95,color:#fff;
    classDef services fill:#6d28d9,stroke:#4c1d95,color:#fff;
    classDef cache fill:#dc2626,stroke:#7f1d1d,color:#fff;
    classDef mlops fill:#d4d4d4,stroke:#999,color:#000;

    classDef models fill:#34d399,stroke:#065f46,color:#000;
    classDef clip fill:#6ee7b7,stroke:#065f46,color:#000;
    classDef bart fill:#f9a8d4,stroke:#be185d,color:#000;
    classDef distil fill:#fcd34d,stroke:#b45309,color:#000;
    classDef ocr fill:#86efac,stroke:#166534,color:#000;

    classDef llm fill:#4f46e5,stroke:#312e81,color:#fff;

    classDef search fill:#facc15,stroke:#b45309,color:#000;
    classDef serp fill:#fbbf24,stroke:#b45309,color:#000;

    %% ASSIGN COLORS
    class UI ui;
    class API api;
    class ROUTES api;
    class SERVICES services;
    class CACHE cache;
    class MLOPS mlops;

    class MODELS models;
    class CLIP clip;
    class BART bart;
    class DISTILBERT distil;
    class OCR ocr;

    class LLM llm;
    class GEM llm;

    class SEARCH search;
    class SERP serp;
    class GOOGLE search;
    class YT search;
    class REDDIT search;
```
## Data Flow Architecture

### Text Search User Experience Flow

```mermaid
flowchart TD
    A[User Enters Product Name] --> B[Input Processing]
    B --> C{Typo Detection}

    C -->|Typo Found| D[Show Corrected Query<br/>Ask Confirmation]
    C -->|No Typo| E[Query Normalization]
    D --> E

    E --> F[Fetch Reviews from<br/>Multiple Sources]
    F --> G[YouTube + Reddit + Twitter<br/>Google Shopping]

    G --> H[Sentiment Analysis<br/>DistilBERT]
    H --> I[Calculate Sentiment<br/>Distribution]

    I --> J[Positive: X%<br/>Neutral: Y%<br/>Negative: Z%]

    G --> K[Review Summarization<br/>BART/Gemini]
    K --> L[Generate Summary]

    G --> M[Extract Key Points]
    M --> N[Pros Extraction]
    M --> O[Cons Extraction]

    J --> P[Generate Recommendation]
    I --> P
    L --> P
    N --> P
    O --> P

    P --> Q{Overall Sentiment}
    Q -->|Positive| R[Recommended ✓<br/>Highlight Strengths]
    Q -->|Mixed| S[Consider Carefully<br/>Show Both Sides]
    Q -->|Negative| T[Not Recommended ✗<br/>Show Concerns]

    R --> U[Display Results]
    S --> U
    T --> U

    U --> V[Show in Frontend:<br/>• Summary<br/>• Sentiment Chart<br/>• Pros & Cons<br/>• Recommendation<br/>• Source Links]

    style A fill:#667eea,color:#fff
    style C fill:#ffd89b,color:#000
    style H fill:#f093fb,color:#000
    style K fill:#f093fb,color:#000
    style P fill:#4facfe,color:#fff
    style R fill:#43e97b,color:#000
    style S fill:#ffd89b,color:#000
    style T fill:#ff6b6b,color:#fff
    style V fill:#667eea,color:#fff
```

**User-Facing Features:**
- **Typo Correction:** Detects and suggests corrections for misspelled product names
- **Query Normalization:** Standardizes product names for better search results
- **Sentiment Distribution:** Visual breakdown of positive, neutral, and negative reviews
- **AI Summarization:** Concise summary of hundreds of reviews in 2-3 sentences
- **Pros & Cons Extraction:** Key strengths and weaknesses identified from reviews
- **Smart Recommendations:** Purchase advice based on overall sentiment and review analysis
- **Source Attribution:** Direct links to YouTube videos, Reddit discussions, and shopping sites

---

### Product Search Backend Flow
```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant SerpAPI
    participant YouTube
    participant Reddit
    participant Google
    participant ML_Models
    participant Gemini

    User->>Streamlit: Search Product
    Streamlit->>FastAPI: POST /smart-search

    FastAPI->>SerpAPI: Fetch Reviews
    par Parallel Data Retrieval
        SerpAPI->>YouTube: Get Video Reviews
        SerpAPI->>Reddit: Get Discussions
        SerpAPI->>Google: Get Shopping Reviews
    end

    YouTube-->>SerpAPI: Video transcripts + metadata
    Reddit-->>SerpAPI: Posts + comments
    Google-->>SerpAPI: Product reviews

    SerpAPI-->>FastAPI: Aggregated Reviews

    FastAPI->>ML_Models: Analyze Sentiment (DistilBERT)
    ML_Models-->>FastAPI: Sentiment Scores

    FastAPI->>ML_Models: Summarize (BART)

    alt BART Success
        ML_Models-->>FastAPI: Summary
    else BART Fails
        FastAPI->>Gemini: Generate Summary
        Gemini-->>FastAPI: Summary
    end

    FastAPI-->>Streamlit: Complete Results
    Streamlit-->>User: Display Analysis
```

**Key Features:**
- **Multi-Source Aggregation:** Parallel data retrieval from YouTube, Reddit, and Google Shopping
- **ML-Powered Analysis:** DistilBERT for sentiment (91% accuracy), BART for summarization (406M params)
- **Fallback Strategy:** BART → Gemini → Extractive summarization for reliability
- **Real-Time Data:** Always fresh reviews from multiple platforms

---

### RAG Chatbot Flow
```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant RAG_Service
    participant SerpAPI
    participant YouTube
    participant Reddit
    participant Twitter
    participant Gemini

    User->>Streamlit: Ask Question
    Streamlit->>RAG_Service: POST /rag/query

    Note over RAG_Service: Context Retrieval Phase

    RAG_Service->>SerpAPI: Fetch Real-time Reviews
    par Multi-Source Retrieval
        SerpAPI->>YouTube: Video reviews
        SerpAPI->>Reddit: Discussions
        SerpAPI->>Twitter: Tweets
    end

    YouTube-->>SerpAPI: Reviews with sentiment
    Reddit-->>SerpAPI: Posts + comments
    Twitter-->>SerpAPI: Tweets

    SerpAPI-->>RAG_Service: Context (Top N reviews)

    Note over RAG_Service: Generation Phase

    RAG_Service->>Gemini: Generate Response<br/>(Query + Context)
    Gemini-->>RAG_Service: AI Response

    RAG_Service-->>Streamlit: Response + Sources
    Streamlit-->>User: Display Answer
```

**Key Features:**
- **Real-Time Context Retrieval:** Fetches latest reviews from YouTube, Reddit, Twitter for current product info
- **Two-Phase Architecture:** Context Retrieval → Generation (RAG pattern)
- **Source Attribution:** Returns response with sources for transparency
- **Gemini LLM:** Uses Gemini for fast, accurate response generation based on real-time context

---

## Image Recognition Flow

```mermaid
flowchart TD
    A[User Uploads Image] --> B{Stage 1:<br/>Quality Check}

    B -->|Poor Quality| C[Return Error<br/>Ask to re-upload]
    B -->|Good Quality| D[Stage 2:<br/>CLIP Visual Analysis]

    D --> E[Get Product Category<br/>Generate Description]

    E --> F[Stage 3:<br/>PaddleOCR<br/>Text Extraction]

    F --> G{Text Extracted?}

    G -->|Yes| H[Stage 4:<br/>Gemini Multimodal Analysis]
    G -->|No| I[Fallback:<br/>Use CLIP Category]

    H --> J[Send to Gemini:<br/>- Product image<br/>- CLIP visual context<br/>- OCR extracted text]
    J --> K{Product Name<br/>Extracted?}

    K -->|Yes| L[High/Medium Confidence<br/>Product Name]
    K -->|No| I

    I --> M[Low Confidence<br/>Generic Category]

    L --> N[Search Reviews<br/>via SerpAPI]
    M --> N

    N --> O[Return Results<br/>Same as Text Search]

    C --> P[End]
    O --> P

    style A fill:#667eea,color:#fff
    style B fill:#ffd89b,color:#000
    style C fill:#ff6b6b,color:#fff
    style D fill:#f093fb,color:#000
    style F fill:#f093fb,color:#000
    style H fill:#4facfe,color:#fff
    style J fill:#4facfe,color:#fff
    style N fill:#43e97b,color:#000
    style O fill:#667eea,color:#fff
```

**Pipeline Stages:**

1. **Quality Check** → Reject poor images (blurry, too dark, too small)
2. **CLIP Analysis** → ML model (151M params) extracts visual features and product category
3. **OCR Extraction** → PaddleOCR extracts text from packaging
4. **Gemini Multimodal** → Receives image + CLIP context + OCR text for accurate product identification
5. **Fallback** → If text not found or Gemini fails, uses CLIP category
6. **Search Reviews** → Same flow as text search (SerpAPI → ML analysis)

**Hybrid ML Approach:** Combines CLIP visual feature extraction (434ms) with Gemini multimodal reasoning for accurate product identification with fallback support.

## 3-Level Summarization Fallback

```mermaid
flowchart TD
    A[Review Text Input] --> B{Try BART}
    B -->|Success| C[BART Summary<br/>10.90s avg]
    B -->|Fails/Unavailable| D{Try Gemini API}

    D -->|Success| E[Gemini Summary<br/>3.78s avg]
    D -->|Fails/No Internet| F[Extractive Summary<br/>100ms avg]

    C --> G[Return Summary]
    E --> G
    F --> G

    style A fill:#667eea,color:#fff
    style C fill:#43e97b,color:#000
    style E fill:#4facfe,color:#fff
    style F fill:#ffd89b,color:#000
    style G fill:#667eea,color:#fff
```

**Fallback Strategy:**
- **Level 1 - BART (Primary):** Best quality, offline-capable, 406M parameter transformer model
- **Level 2 - Gemini (Fallback 1):** Fast, reliable cloud-based summarization when BART unavailable
- **Level 3 - Extractive (Fallback 2):** Always works, basic quality, no external dependencies
- **Production Reliability:** Ensures summarization always succeeds regardless of environment constraints

---

## RAG Chatbot Architecture

```mermaid
graph LR
    A[User Query] --> B[RAG Service]
    B --> C[Retrieve Context]
    C --> D[SerpAPI]
    D --> E[Real-time Reviews]

    E --> F[Context + Query]
    F --> G[Gemini LLM]
    G --> H[Generate Response]

    H --> I[Return with Sources]

    style A fill:#667eea,color:#fff
    style B fill:#764ba2,color:#fff
    style D fill:#43e97b,color:#000
    style G fill:#4facfe,color:#fff
    style I fill:#667eea,color:#fff
```

**Architecture Overview:**
- **Retrieval-Augmented Generation (RAG):** Combines real-time data retrieval with LLM generation
- **Context-Aware Responses:** Gemini receives product-specific reviews as context for accurate answers
- **Source Transparency:** All responses include sources (YouTube, Reddit, Twitter) for user verification
- **No Vector Database:** Uses real-time retrieval instead of pre-indexed embeddings for always-current data

---

## MLOps Pipeline

```mermaid
flowchart TB
    A[Development] --> B[Model Benchmarking<br/>measure_model_metrics.py]
    B --> C[MLflow Registration<br/>register_models.py]

    C --> D{Metrics}
    D -->|Load Time| E[Model Performance]
    D -->|Inference Time| E
    D -->|Accuracy| E

    E --> F[MLflow Tracking Server<br/>Port 5000]

    F --> G[Experiment Comparison]
    G --> H[A/B Testing]

    H --> I{Performance Better?}
    I -->|Yes| J[Update Production]
    I -->|No| K[Keep Current]

    J --> L[Production Deployment]
    K --> L

    style A fill:#667eea,color:#fff
    style B fill:#f093fb,color:#000
    style C fill:#f093fb,color:#000
    style F fill:#ffd89b,color:#000
    style H fill:#43e97b,color:#000
    style L fill:#667eea,color:#fff
```

**MLOps Workflow:**
- **Model Benchmarking:** Measure load time, inference time, and accuracy for all ML models
- **MLflow Registration:** Register models with metadata and performance metrics in MLflow tracking server
- **Experiment Tracking:** Compare different model versions, preprocessing approaches, and hyperparameters
- **A/B Testing Framework:** Systematic comparison of model performance before production deployment
- **Data-Driven Decisions:** Update production models only when metrics demonstrate clear improvement

---

## Technology Stack Overview

```mermaid
mindmap
  root((ClarifyProducts.AI))
    Frontend
      Streamlit
      Custom CSS
      Responsive Design
    Backend
      FastAPI
      Python 3.10+
      Uvicorn
    ML Models
      CLIP Vision
        151M params
        434ms inference
      BART NLP
        406M params
        10.90s inference
      DistilBERT
        67M params
        116ms inference
      PaddleOCR
        Text extraction
        6934ms inference
    External APIs
      Gemini LLM
        Chatbot
        Summarization fallback
      SerpAPI
        YouTube
        Reddit
        Google Shopping
    MLOps
      MLflow
        Experiment tracking
        Model versioning
        A/B testing
      Loguru
        Structured logging
    Infrastructure
      Docker
      Docker Compose
```

## Production Deployment Architecture

**Current Deployment (GCP):** 

```mermaid
graph TB
    Users[Users/Clients] --> FE[Streamlit Frontend<br/>Port 8501]
    FE --> BE[FastAPI Backend<br/>Port 8000]

    BE --> Redis[Redis Cache<br/>24hr TTL]
    BE --> CLIP[CLIP Model<br/>151M params]
    BE --> BART[BART Model<br/>406M params]
    BE --> DistilBERT[DistilBERT Model<br/>67M params]
    BE --> OCR[PaddleOCR]

    BE --> Gemini[Gemini API<br/>+Retry Logic]
    BE --> Serp[SerpAPI<br/>+Retry Logic]

    Serp --> Sources[YouTube + Reddit<br/>+ Google Shopping]

    style Users fill:#667eea,color:#fff
    style FE fill:#f093fb,color:#000
    style BE fill:#43e97b,color:#000
    style Redis fill:#dc2626,color:#fff
    style CLIP fill:#ffd89b,color:#000
    style BART fill:#ffd89b,color:#000
    style DistilBERT fill:#ffd89b,color:#000
    style Gemini fill:#4facfe,color:#fff
    style Serp fill:#764ba2,color:#fff
```

**Current Production Setup:**
- **Platform:** Google Cloud Platform (GCP)
- **Instance:** e2-standard-2 (2 vCPU, 8 GB RAM, 30 GB SSD)
- **Architecture:** Single-instance deployment with Redis caching
- **Frontend:** Streamlit on http://136.114.42.68:8501
- **Backend:** FastAPI on http://136.114.42.68:8000
- **ML Models:** All running on same instance (CLIP, BART, DistilBERT, PaddleOCR)
- **Caching:** Redis with 24-hour TTL (80-90% API cost reduction)
- **Reliability:** Exponential backoff retry logic for Gemini and SerpAPI
- **Cost:** $0/month (using $300 GCP free credits)
- **Uptime:** 24/7 availability

## Future Scalability Architecture

```mermaid
graph TB
    subgraph "Cloud Infrastructure (Future)"
        LB[Load Balancer<br/>Nginx/Traefik]

        subgraph "Frontend Tier"
            FE1[Streamlit 1]
            FE2[Streamlit 2]
        end

        subgraph "Backend Tier"
            BE1[FastAPI 1<br/>ML Models]
            BE2[FastAPI 2<br/>ML Models]
        end

        subgraph "Caching & Queue"
            Redis[Redis Cluster]
            Queue[RabbitMQ/Celery]
        end

        subgraph "External Services"
            Gemini[Gemini API]
            Serp[SerpAPI]
        end

        subgraph "Monitoring"
            MLflow[MLflow Server]
            Prometheus[Prometheus + Grafana]
        end
    end

    Users[Users] --> LB
    LB --> FE1
    LB --> FE2

    FE1 --> BE1
    FE2 --> BE2

    BE1 --> Redis
    BE2 --> Redis
    BE1 --> Queue
    BE2 --> Queue

    BE1 --> Gemini
    BE1 --> Serp
    BE2 --> Gemini
    BE2 --> Serp

    BE1 -.-> MLflow
    BE2 -.-> MLflow
    BE1 -.-> Prometheus
    BE2 -.-> Prometheus

    style Users fill:#667eea,color:#fff
    style LB fill:#764ba2,color:#fff
    style FE1 fill:#f093fb,color:#000
    style FE2 fill:#f093fb,color:#000
    style BE1 fill:#43e97b,color:#000
    style BE2 fill:#43e97b,color:#000
    style Redis fill:#dc2626,color:#fff
    style MLflow fill:#ffd89b,color:#000
```

**Future Scalability Enhancements:**
- **Horizontal Scaling:** Multiple frontend and backend instances behind load balancer
- **Redis Cluster:** Distributed caching for high availability
- **Message Queue:** Async processing for long-running ML tasks
- **Monitoring:** Prometheus + Grafana for real-time metrics
- **GPU Instances:** Dedicated GPU nodes for faster ML inference
- **CDN:** Static asset delivery for frontend

---

## Key Architectural Decisions

### 1. Real-Time Data Retrieval (No Vector DB)
**Why:** Product reviews change frequently. Real-time retrieval ensures always-current data without maintaining an embedding pipeline.

**Trade-offs:**
-  Always fresh data
-  Simpler architecture
-  Lower storage costs
-  Slightly slower first request (API call latency)
-  Depends on external API availability

### 2. 3-Level Summarization Fallback
**Why:** Production reliability. BART provides best quality but may fail on resource-constrained environments.

**Fallback Order:**
1. **BART** (Primary) - Best quality, offline-capable
2. **Gemini** (Fallback 1) - Fast, reliable, requires internet
3. **Extractive** (Fallback 2) - Always works, basic quality

### 3. Multimodal Input Support
**Why:** Users may not know exact product names. Image recognition provides alternative entry point.

**Implementation:**
- PaddleOCR for text extraction from packaging
- CLIP for visual product classification
- Confidence-based fallback (text → visual → category)

### 4. Stateless Backend
**Why:** Easier to scale horizontally. Each request is independent.

**Benefits:**
- Can run multiple backend instances
- Load balancing friendly
- No session management complexity

### 5. MLflow for Experiment Tracking
**Why:** Professional ML engineering. Track model performance over time.

**Use Cases:**
- Compare model versions
- A/B testing (e.g., CLIP preprocessing variations)
- Performance monitoring
- Reproducibility

---

## Performance Characteristics

### Request Latency Breakdown

Typical product search request (text-based):

| Stage | Component | Latency | Notes |
|-------|-----------|---------|-------|
| 1. API Request | FastAPI | ~50ms | Request handling and validation |
| 2. Data Retrieval | SerpAPI | ~2000ms | Fetch reviews from multiple sources |
| 3. Sentiment Analysis | DistilBERT | ~116ms | Analyze all reviews for sentiment |
| 4. Summarization | BART | ~10900ms | Generate concise summary |
| 5. Response Formatting | FastAPI | ~50ms | Structure and return results |

**Total End-to-End Latency:** ~13 seconds

**Performance Optimizations Implemented:** 
-  **Redis Caching:** 24-hour TTL for product search and RAG responses (80-90% API cost reduction)
-  **Exponential Backoff Retry Logic:** 5 attempts for Gemini, 3 attempts for SerpAPI
-  **Async Processing:** Non-blocking I/O for external API calls
-  **Singleton Pattern:** ML models loaded once and reused

**Future Performance Opportunities:**
- **Parallel Processing:** Run sentiment analysis and summarization concurrently
- **GPU Acceleration:** Faster inference for BART and DistilBERT models
- **Model Optimization:** Quantization or distillation for faster inference

---

## Security Considerations

1. **API Keys:** Stored in `.env`, never committed
2. **Input Validation:** FastAPI request validation
3. **Rate Limiting:** Prevent API abuse (future enhancement)
4. **CORS:** Configured for frontend-backend communication
5. **Dependency Scanning:** Regular updates for vulnerabilities

---

## Scalability Considerations

**Current Production Deployment:** 
- Google Cloud Platform (GCP) e2-standard-2 instance
- 2 vCPU, 8 GB RAM, 30 GB SSD
- ML models: CLIP (151M) + BART (406M) + DistilBERT (67M)
- Redis caching layer with 24-hour TTL
- Estimated throughput: ~10 requests/minute with caching boost
- Cost: $0/month (using $300 GCP free credits)
- 24/7 uptime
- URLs: Frontend (http://136.114.42.68:8501), Backend (http://136.114.42.68:8000/docs)

**Future Scalability Path:**
- Horizontal scaling with load balancer (Nginx/Traefik)
- Multiple backend replicas for increased throughput
- Message queue (RabbitMQ/Celery) for async processing
- Dedicated ML model serving (TensorFlow Serving/TorchServe)
- CDN for static frontend assets

---

## Monitoring & Observability

**Implemented:**
- MLflow for ML model metrics and experiment tracking
- Loguru for structured application logging
- FastAPI automatic API documentation (Swagger/OpenAPI)

**Production Enhancements:**
- Prometheus + Grafana for system metrics visualization
- Sentry for error tracking and alerting
- ELK Stack (Elasticsearch, Logstash, Kibana) for log aggregation
- Custom dashboards for request latency and model performance monitoring
