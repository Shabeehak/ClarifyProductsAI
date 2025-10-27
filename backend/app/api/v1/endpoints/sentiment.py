"""
Sentiment Analysis Endpoint
Analyzes sentiment of product reviews
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger

from app.ml_models.sentiment_analyzer import get_sentiment_analyzer

router = APIRouter()
sentiment_analyzer = get_sentiment_analyzer()


class SentimentRequest(BaseModel):
    """Request schema for sentiment analysis"""

    text: str = Field(..., min_length=1, description="Review text to analyze")


class SentimentBatchRequest(BaseModel):
    """Request schema for batch sentiment analysis"""

    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of review texts"
    )


class SentimentResponse(BaseModel):
    """Response schema for sentiment analysis"""

    text: str
    sentiment_label: str  # positive, negative, neutral
    sentiment_score: float
    confidence: float


class AspectSentimentRequest(BaseModel):
    """Request for aspect-based sentiment"""

    text: str
    aspect: str = Field(
        ..., description="Aspect to analyze (quality, price, service, etc.)"
    )


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of a single review text

    - **text**: Review text to analyze

    Returns sentiment label (positive/negative/neutral) and confidence score
    """
    try:
        logger.info(f"Analyzing sentiment for text: {request.text[:50]}...")

        result = sentiment_analyzer.analyze(request.text)

        return SentimentResponse(
            text=request.text,
            sentiment_label=result["sentiment_label"],
            sentiment_score=result["sentiment_score"],
            confidence=result["confidence"],
        )

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze sentiment: {str(e)}"
        )


@router.post("/analyze/batch", response_model=List[SentimentResponse])
async def analyze_sentiment_batch(request: SentimentBatchRequest):
    """
    Analyze sentiment for multiple review texts

    - **texts**: List of review texts (max 100)

    Returns list of sentiment analyses
    """
    try:
        logger.info(f"Analyzing sentiment for {len(request.texts)} reviews")

        results = sentiment_analyzer.analyze_batch(request.texts)

        responses = []
        for text, result in zip(request.texts, results):
            responses.append(
                SentimentResponse(
                    text=text,
                    sentiment_label=result["sentiment_label"],
                    sentiment_score=result["sentiment_score"],
                    confidence=result["confidence"],
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Error during batch sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze sentiment: {str(e)}"
        )


@router.post("/analyze/aspect")
async def analyze_aspect_sentiment(request: AspectSentimentRequest):
    """
    Analyze sentiment for a specific aspect (e.g., quality, price)

    - **text**: Review text
    - **aspect**: Aspect to analyze (quality, price, service, delivery, features)

    Returns sentiment for that specific aspect
    """
    try:
        logger.info(f"Analyzing aspect '{request.aspect}' sentiment")

        result = sentiment_analyzer.get_aspect_sentiment(request.text, request.aspect)

        return result

    except Exception as e:
        logger.error(f"Error during aspect sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze aspect sentiment: {str(e)}"
        )
