"""
Product Recognition Service
Handles image-based product identification using FREE CLIP model
"""
from typing import Optional
from loguru import logger
import io
from PIL import Image

from app.schemas.product import ProductRecognitionResponse
from app.ml_models.clip_recognizer import get_clip_recognizer
from app.core.config import settings


class RecognitionService:
    """Service for product image recognition using FREE open-source models"""

    def __init__(self):
        """Initialize recognition service"""
        self.clip_recognizer = get_clip_recognizer()
        logger.info("Recognition service initialized with FREE CLIP model")

    async def recognize_product(self, image_bytes: bytes) -> ProductRecognitionResponse:
        """
        Recognize product from image bytes using CLIP

        Args:
            image_bytes: Raw image bytes

        Returns:
            ProductRecognitionResponse with identification results
        """
        try:
            # Validate and load image
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Processing image: {image.size}, mode: {image.mode}")

            # Use CLIP for recognition
            result = self.clip_recognizer.recognize_from_bytes(image_bytes)

            if not result["success"]:
                logger.error(f"CLIP recognition failed: {result.get('error')}")
                raise Exception(result.get('error', 'Recognition failed'))

            # Get top prediction
            top_pred = result["top_prediction"]
            alternatives = result["predictions"][1:4]  # Next 3 predictions

            # Format response according to schema
            from app.schemas.product import ProductMatch

            return ProductRecognitionResponse(
                success=True,
                primary_match=ProductMatch(
                    product_name=top_pred["category"].title(),
                    confidence=top_pred["confidence"],
                    category=top_pred["category"]
                ),
                alternative_matches=[
                    ProductMatch(
                        product_name=alt["category"].title(),
                        confidence=alt["confidence"],
                        category=alt["category"]
                    )
                    for alt in alternatives
                ],
                message=f"Product recognized as {top_pred['category'].title()} with {top_pred['confidence']:.2%} confidence"
            )

        except Exception as e:
            logger.error(f"Error in product recognition: {str(e)}")
            # Return error response instead of raising
            return ProductRecognitionResponse(
                success=False,
                primary_match=None,
                alternative_matches=[],
                message=f"Recognition failed: {str(e)}"
            )
