"""
Smart Product Search Endpoint - SCALABLE & PRODUCTION READY
Handles:
- Fuzzy search (typo tolerance)
- On-demand scraping (no storage needed)
- Real-time ML analysis
- Smart caching
- Works with ANY product!
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from loguru import logger

from app.services.smart_product_service import get_smart_product_service

router = APIRouter()
smart_service = get_smart_product_service()


class SmartSearchResponse(BaseModel):
    """Smart search response with Perplexity-style media and citations"""

    query: str
    normalized_query: str
    found: bool
    suggestions: List[str] = Field(default_factory=list)
    product_name: Optional[str] = None
    product_info: Optional[Dict] = None
    rating: float = 0.0
    review_count: int = 0
    sources: Dict[str, Any] = Field(default_factory=dict)
    ml_analysis: Optional[Dict] = None
    # NEW: Perplexity-style media and citations
    product_images: List[Dict] = Field(
        default_factory=list, description="Product images from various sources"
    )
    youtube_videos: List[Dict] = Field(
        default_factory=list, description="YouTube review videos with links"
    )
    reddit_discussions: List[Dict] = Field(
        default_factory=list, description="Reddit discussion links"
    )
    twitter_posts: List[Dict] = Field(
        default_factory=list, description="Twitter posts about product"
    )
    citations: List[Dict] = Field(
        default_factory=list, description="Source citations like Perplexity"
    )
    # Metadata
    from_cache: bool = False
    response_time_ms: Optional[float] = None


@router.get("/")
async def smart_search(
    q: str = Query(..., min_length=2, description="Search query (handles typos!)"),
    use_cache: bool = Query(True, description="Use cached results"),
    include_ml: bool = Query(True, description="Include ML sentiment/summary analysis"),
):
    """
    **Smart Product Search - Works with ANY product!**

    **Features:**
    - ✅ Typo tolerance ("iphne 15" → "iphone 15")
    - ✅ Spelling corrections ("samsng galxy" → "samsung galaxy")
    - ✅ On-demand scraping (no storage needed)
    - ✅ Real-time ML analysis
    - ✅ Smart caching (24h TTL)

    **Examples:**
    - `q=iphone 15` → Returns iPhone 15 reviews
    - `q=iphne 15 pro` → Auto-corrects and finds iPhone 15 Pro
    - `q=samsng s24` → Corrects to Samsung S24
    - `q=macbok pro` → Finds MacBook Pro
    - `q=sony headphones` → Finds Sony headphones

    **Parameters:**
    - **q**: Any product name (handles typos!)
    - **use_cache**: Use cached data (faster response)
    - **include_ml**: Include sentiment analysis and summary

    **Returns:**
    - Product info from multiple sources
    - Reviews with ML sentiment analysis
    - AI-generated summary (pros/cons)
    - Typo suggestions if found
    """
    try:
        import time

        start_time = time.time()

        logger.info(f"Smart search: '{q}'")

        # Use smart product service
        result = await smart_service.search_product(
            query=q, use_cache=use_cache, include_ml_analysis=include_ml
        )

        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # ms

        # Update metadata with response time
        if "metadata" in result:
            result["metadata"]["response_time_ms"] = round(response_time, 2)

        # Return the new clean structure directly (no Pydantic model)
        return result

    except Exception as e:
        logger.error(f"Error in smart search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/autocomplete")
async def autocomplete(
    q: str = Query(..., min_length=1, description="Partial query for autocomplete"),
    limit: int = Query(5, ge=1, le=20, description="Max suggestions"),
):
    """
    Get autocomplete suggestions

    **Example:**
    - `q=ipho` → Returns ["iPhone 15 Pro Max", "iPhone 15", "iPhone 14 Pro"]
    - `q=macb` → Returns ["MacBook Pro", "MacBook Air"]
    """
    try:
        suggestions = await smart_service.get_product_suggestions(q, limit=limit)

        return {"query": q, "suggestions": suggestions}

    except Exception as e:
        logger.error(f"Error in autocomplete: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_search_stats():
    """
    Get search statistics and cache info

    Returns:
    - Total searches cached
    - Cache hit rate
    - Most searched products
    """
    try:
        stats = smart_service.get_cache_stats()

        return {"cache_stats": stats, "status": "operational"}

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache():
    """Clear search cache (admin only)"""
    try:
        smart_service.cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
