"""
API v1 Main Router
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    reviews,
    recognition,
    recommendations,
    tasks,
    sentiment,
    summarization,
    search,
    smart_search,
    rag,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    recognition.router, prefix="/recognition", tags=["Product Recognition"]
)

api_router.include_router(reviews.router, prefix="/reviews", tags=["Reviews"])

api_router.include_router(
    recommendations.router, prefix="/recommendations", tags=["Recommendations"]
)

api_router.include_router(tasks.router, prefix="/tasks", tags=["Background Tasks"])

api_router.include_router(
    sentiment.router, prefix="/sentiment", tags=["Sentiment Analysis"]
)

api_router.include_router(
    summarization.router, prefix="/summarization", tags=["AI Summarization"]
)

api_router.include_router(
    search.router, prefix="/search", tags=["Product Search (Simple)"]
)

api_router.include_router(
    smart_search.router,
    prefix="/smart-search",
    tags=["🚀 Smart Search (Scalable + Typo Tolerant)"],
)

api_router.include_router(
    rag.router, prefix="/rag", tags=["RAG (Retrieval-Augmented Generation)"]
)
