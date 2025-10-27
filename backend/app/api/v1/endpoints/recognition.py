"""
Product Recognition Endpoint
Handles image upload and product identification using Complete Pipeline (CLIP + OCR + Gemini)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from loguru import logger
import tempfile
import os

from app.schemas.product import ProductRecognitionResponse, ProductMatch
from app.services.complete_recognition_service import get_complete_recognition_service

router = APIRouter()

# Lazy initialization - service will be created on first request
_recognition_service = None


def get_service():
    """Get or create recognition service instance"""
    global _recognition_service
    if _recognition_service is None:
        _recognition_service = get_complete_recognition_service()
    return _recognition_service


@router.post("/", response_model=ProductRecognitionResponse)
async def recognize_product(
    image: UploadFile = File(..., description="Product image to recognize")
):
    """
    Recognize product from uploaded image using Complete Pipeline (CLIP + OCR + Gemini)

    - **image**: Product image file (JPEG, PNG)

    Returns:
    - Product identification with confidence score
    - Recognition method used (multimodal, clip_fallback, etc.)
    - Quality assessment and detailed information
    """
    temp_file_path = None

    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image (JPEG, PNG)"
            )

        # Save uploaded file to temporary location
        image_bytes = await image.read()

        # Create temp file with proper extension
        file_ext = os.path.splitext(image.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        logger.info(f"Processing image recognition for: {image.filename}")

        # Perform complete recognition (Quality → CLIP → OCR → Gemini)
        result = get_service().recognize_product_complete(temp_file_path)

        # Convert to API response format
        response = _convert_to_api_response(result)

        logger.info(
            f"Recognition complete: {result.get('product_name')} (method: {result.get('method')})"
        )

        return response

    except Exception as e:
        logger.error(f"Error during product recognition: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process image: {str(e)}"
        )

    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.post("/batch", response_model=List[ProductRecognitionResponse])
async def recognize_products_batch(
    images: List[UploadFile] = File(..., description="Multiple product images")
):
    """
    Batch recognition for multiple product images

    - **images**: List of product image files (max 10)

    Returns:
    - List of product identifications
    """
    if len(images) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 images allowed per batch"
        )

    results = []

    for image in images:
        temp_file_path = None
        try:
            image_bytes = await image.read()

            # Save to temp file
            file_ext = os.path.splitext(image.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext
            ) as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name

            # Perform recognition
            result = get_service().recognize_product_complete(temp_file_path)
            response = _convert_to_api_response(result)
            results.append(response)

        except Exception as e:
            logger.error(f"Error processing {image.filename}: {str(e)}")
            # Add error result instead of skipping
            results.append(
                ProductRecognitionResponse(
                    success=False,
                    primary_match=None,
                    alternative_matches=[],
                    message=f"Failed to process {image.filename}: {str(e)}",
                )
            )

        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    return results


def _convert_to_api_response(result: dict) -> ProductRecognitionResponse:
    """
    Convert complete pipeline result to API response format

    Args:
        result: Result from complete_recognition_service

    Returns:
        ProductRecognitionResponse formatted for API
    """
    if not result.get("success"):
        return ProductRecognitionResponse(
            success=False,
            primary_match=None,
            alternative_matches=[],
            message=result.get("message", "Recognition failed"),
        )

    # Map confidence string to float
    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5, "unclear": 0.3}

    confidence_score = confidence_map.get(result.get("confidence", "low"), 0.5)

    # Get category from CLIP details
    category = result.get("details", {}).get("clip_category", "unknown")

    primary_match = ProductMatch(
        product_name=result.get("product_name", "Unknown"),
        confidence=confidence_score,
        category=category,
    )

    # Create informative message
    method = result.get("method", "unknown")
    message_parts = [result.get("message", "")]

    if method == "multimodal":
        message_parts.append("Identified using AI vision + text analysis")
    elif method == "clip_fallback":
        message_parts.append(
            "Generic category detected. Upload clearer image for specific product name."
        )
    elif method == "quality_check":
        message_parts.append("Image quality too poor for recognition")

    message = " | ".join(filter(None, message_parts))

    return ProductRecognitionResponse(
        success=True,
        primary_match=primary_match,
        alternative_matches=[],  # Complete pipeline returns single best match
        message=message,
    )
