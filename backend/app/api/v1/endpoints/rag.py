"""
RAG (Retrieval-Augmented Generation) Endpoint - REAL-TIME VERSION
Handles intelligent queries using real-time review fetching + Gemini LLM
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from loguru import logger

from app.services.rag_service import get_rag_service

router = APIRouter()
rag_service = get_rag_service()


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query"""

    query: str = Field(..., min_length=3, description="User question about products")
    product_id: Optional[str] = Field(None, description="Optional product ID to filter")
    n_results: int = Field(
        10, ge=1, le=50, description="Number of context items to retrieve"
    )


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query"""

    query: str
    response: str
    sources: List[Dict]
    context_count: int
    realtime: bool = Field(
        default=True, description="Indicates data is fetched in real-time"
    )
    sources_breakdown: Optional[Dict] = Field(
        None, description="Breakdown by source (YouTube, Reddit, Twitter)"
    )


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Real-time intelligent query using RAG (Retrieval-Augmented Generation)

    **How it works:**
    1. Extracts product from your query
    2. Fetches live reviews from YouTube, Reddit, Twitter
    3. Analyzes sentiment and summarizes
    4. Uses Gemini AI to generate conversational response

    **Parameters:**
    - **query**: Natural language question about products (e.g., "best moisturizer for dry skin")
    - **product_id**: Optional filter by specific product ID
    - **n_results**: Number of reviews to fetch (1-50, default: 10)

    **Returns:**
    - AI-generated response with real-time review sources
    - Breakdown of sources (YouTube, Reddit, Twitter)
    - Sentiment analysis and consensus

    **Example:**
    ```json
    {
        "query": "is CeraVe good for sensitive skin?",
        "n_results": 5
    }
    ```
    """
    try:
        logger.info(f"Real-time RAG query: {request.query}")

        # Call RAG service (now fetches real-time data)
        result = await rag_service.rag_query(
            query=request.query,
            product_id=request.product_id,
            n_results=request.n_results,
        )

        return RAGQueryResponse(
            query=request.query,
            response=result["response"],
            sources=result.get("sources", []),
            context_count=result.get("context_count", 0),
            realtime=result.get("realtime", True),
            sources_breakdown=result.get("sources_breakdown"),
        )

    except Exception as e:
        logger.error(f"Error in real-time RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.get("/health")
async def rag_health_check():
    """
    Check RAG service health

    Returns status of Gemini API and real-time scrapers
    """
    try:
        # Check if services are initialized
        has_gemini = rag_service.llm_service.gemini_api_key is not None
        has_product_service = hasattr(rag_service, "product_service")

        return {
            "status": "healthy" if has_gemini else "degraded",
            "gemini_configured": has_gemini,
            "realtime_enabled": has_product_service,
            "mode": "real-time",
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}
