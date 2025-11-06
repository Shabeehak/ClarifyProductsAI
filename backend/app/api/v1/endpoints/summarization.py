"""
AI Summarization Endpoint
Summarizes product reviews using BART or Ollama
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger

from app.ml_models.summarizer import get_summarizer

router = APIRouter()
summarizer = get_summarizer()


class SummarizationRequest(BaseModel):
    """Request schema for summarization"""

    text: str = Field(..., min_length=10, description="Text to summarize")
    max_length: int = Field(150, ge=50, le=500, description="Maximum summary length")
    min_length: int = Field(50, ge=10, le=200, description="Minimum summary length")


class StructuredSummarizationRequest(BaseModel):
    """Request for structured review summary"""

    reviews: List[str] = Field(
        ..., min_items=1, max_items=50, description="List of review texts"
    )


class SummarizationResponse(BaseModel):
    """Response schema for summarization"""

    original_text: str
    summary: str
    original_length: int
    summary_length: int


class StructuredSummaryResponse(BaseModel):
    """Structured summary with pros, cons, key points"""

    summary: str
    pros: List[str]
    cons: List[str]
    key_points: List[str]


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """
    Summarize a single text using AI

    - **text**: Text to summarize
    - **max_length**: Maximum summary length (default: 150)
    - **min_length**: Minimum summary length (default: 50)

    Returns concise AI-generated summary
    """
    try:
        logger.info(f"Summarizing text of length: {len(request.text)}")

        summary = summarizer.summarize(
            request.text, max_length=request.max_length, min_length=request.min_length
        )

        return SummarizationResponse(
            original_text=request.text,
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
        )

    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to summarize text: {str(e)}"
        )


@router.post("/summarize/reviews", response_model=StructuredSummaryResponse)
async def summarize_reviews(request: StructuredSummarizationRequest):
    """
    Generate structured summary from multiple reviews

    - **reviews**: List of review texts (max 50)

    Returns:
    - Overall summary
    - Pros (positive points)
    - Cons (negative points)
    - Key points
    """
    try:
        logger.info(f"Generating structured summary for {len(request.reviews)} reviews")

        result = summarizer.generate_structured_summary(request.reviews)

        return StructuredSummaryResponse(
            summary=result["summary"],
            pros=result["pros"],
            cons=result["cons"],
            key_points=result["key_points"],
        )

    except Exception as e:
        logger.error(f"Error during structured summarization: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate summary: {str(e)}"
        )


@router.get("/health")
async def summarization_health():
    """Check if summarization service is ready"""
    try:
        # Check if BART model is loaded
        model_status = "BART"
        fallback_available = False

        if summarizer.pipeline is None:
            try:
                summarizer.load_model()
                model_status = "BART (loaded)"
            except Exception as load_error:
                logger.warning(f"BART model not available: {load_error}")
                model_status = "BART (unavailable)"
                fallback_available = True

        return {
            "status": "healthy",
            "model": model_status,
            "gemini_fallback": fallback_available or summarizer.pipeline is None,
            "ready": True,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "ready": False}
