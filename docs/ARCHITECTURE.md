# ClarifyProducts.AI - System Architecture

## System Overview

ClarifyProducts.AI is an ML-powered product review analysis platform that aggregates reviews from multiple sources (YouTube, Reddit, Google Shopping) and provides sentiment analysis, summarization, and conversational AI capabilities. The system supports both text-based and image-based product search.

## High-Level Architecture

```mermaid
graph TB
    UI[Streamlit Frontend<br/>Port 8501]
    API[FastAPI Backend<br/>Port 8000]

    CLIP[CLIP Vision Model<br/>151M params]
    BART[BART Summarizer<br/>406M params]
    DistilBERT[DistilBERT Sentiment<br/>67M params]
    OCR[PaddleOCR]
    Gemini[Gemini LLM]

    SerpAPI[SerpAPI]
    Sources[YouTube + Reddit + Google]
    MLflow[MLflow Tracking<br/>Port 5000]

    UI -->|HTTP| API
    API --> CLIP
    API --> BART
    API --> DistilBERT
    API --> OCR
    API --> Gemini
    API --> SerpAPI

    SerpAPI --> Sources

    CLIP -.->|Metrics| MLflow
    BART -.->|Metrics| MLflow
    DistilBERT -.->|Metrics| MLflow
    OCR -.->|Metrics| MLflow

    style UI fill:#667eea,color:#fff
    style API fill:#764ba2,color:#fff
    style CLIP fill:#f093fb,color:#000
    style BART fill:#f093fb,color:#000
    style DistilBERT fill:#f093fb,color:#000
    style Gemini fill:#4facfe,color:#fff
    style SerpAPI fill:#43e97b,color:#000
    style MLflow fill:#ffd89b,color:#000
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
    F --> G[YouTube + Reddit +<br/>Google Shopping]

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

**Why use CLIP if Gemini can see the image?**
- **CLIP provides structured ML context:** Category classification from 70+ product types, reducing Gemini's search space
- **Faster inference:** CLIP pre-processes visual features (434ms) before Gemini analysis
- **Hybrid approach:** Combines ML feature extraction (CLIP) with LLM reasoning (Gemini) for best accuracy
- **Demonstrates ML engineering:** Shows ability to integrate multiple models rather than relying on single API

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

## Deployment Architecture (Future)

```mermaid
graph TB
    subgraph "Cloud Infrastructure"
        LB[Load Balancer]

        subgraph "Frontend Instances"
            FE1[Streamlit 1]
            FE2[Streamlit 2]
        end

        subgraph "Backend Instances"
            BE1[FastAPI 1<br/>ML Models]
            BE2[FastAPI 2<br/>ML Models]
        end

        subgraph "Caching Layer"
            Redis[Redis Cache]
        end

        subgraph "External Services"
            Gemini[Gemini API]
            Serp[SerpAPI]
        end

        subgraph "Monitoring"
            MLflow[MLflow Server]
            Logs[Log Aggregation]
        end
    end

    Users[Users] --> LB
    LB --> FE1
    LB --> FE2

    FE1 --> BE1
    FE2 --> BE2

    BE1 --> Redis
    BE2 --> Redis

    BE1 --> Gemini
    BE1 --> Serp
    BE2 --> Gemini
    BE2 --> Serp

    BE1 -.-> MLflow
    BE2 -.-> MLflow
    BE1 -.-> Logs
    BE2 -.-> Logs

    style Users fill:#667eea,color:#fff
    style LB fill:#764ba2,color:#fff
    style FE1 fill:#f093fb,color:#000
    style FE2 fill:#f093fb,color:#000
    style BE1 fill:#43e97b,color:#000
    style BE2 fill:#43e97b,color:#000
    style Redis fill:#ffd89b,color:#000
    style MLflow fill:#ffd89b,color:#000
```

**Scalability Plan:**
- **Horizontal Scaling:** Multiple frontend and backend instances behind load balancer
- **Caching Layer:** Redis for frequently requested product reviews and analysis results
- **Stateless Architecture:** Each backend instance can handle any request independently
- **Monitoring & Observability:** MLflow for model metrics, centralized logging for system health
- **Current Status:** Single-instance development setup, production deployment planned

---

## Key Architectural Decisions

### 1. Real-Time Data Retrieval (No Vector DB)
**Why:** Product reviews change frequently. Real-time retrieval ensures always-current data without maintaining an embedding pipeline.

**Trade-offs:**
- ✅ Always fresh data
- ✅ Simpler architecture
- ✅ Lower storage costs
- ❌ Slightly slower first request (API call latency)
- ❌ Depends on external API availability

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

**Performance Optimization Opportunities:**
- **Caching:** Store frequently requested product reviews (Redis)
- **Parallel Processing:** Run sentiment analysis and summarization concurrently
- **GPU Acceleration:** Faster inference for BART and DistilBERT models
- **Async Processing:** Non-blocking I/O for external API calls
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

**Current Architecture:**
- Single-instance deployment suitable for demonstration and development
- Estimated throughput: ~10 requests/minute
- Low infrastructure cost

**Production Scalability Path:**
- Horizontal scaling with load balancer (Nginx/Traefik)
- Multiple backend replicas for increased throughput
- Redis caching layer for frequently requested reviews
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
