"""
Enhanced Product Recognition Service
Combines CLIP + OCR + Smart Search for accurate product identification
"""
from typing import Optional, Dict, List
from loguru import logger
import io
from PIL import Image
import re

from app.schemas.product import ProductRecognitionResponse, ProductMatch
from app.ml_models.clip_recognizer import get_clip_recognizer
from app.services.smart_product_service import get_smart_product_service
from app.core.config import settings


class EnhancedRecognitionService:
    """
    Enhanced recognition service that combines:
    1. CLIP for category detection
    2. OCR for text extraction (brand, product name)
    3. Smart product search for accurate matching
    """

    def __init__(self):
        """Initialize enhanced recognition service"""
        self.clip_recognizer = get_clip_recognizer()
        self.product_service = get_smart_product_service()
        logger.info("Enhanced Recognition Service initialized (CLIP + OCR + Smart Search)")

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using simple OCR
        Falls back gracefully if OCR not available

        Args:
            image: PIL Image object

        Returns:
            Extracted text string
        """
        try:
            # Try using pytesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(image)
                logger.info(f"OCR extracted: {text[:100]}...")
                return text.strip()
            except ImportError:
                logger.debug("pytesseract not available, trying easyocr")

            # Try easyocr as fallback
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(image, detail=0)
                text = ' '.join(result)
                logger.info(f"EasyOCR extracted: {text[:100]}...")
                return text.strip()
            except ImportError:
                logger.debug("easyocr not available")

            # No OCR available - return empty
            logger.warning("No OCR library available (pytesseract/easyocr)")
            return ""

        except Exception as e:
            logger.error(f"Error during OCR: {str(e)}")
            return ""

    def parse_product_name_from_text(self, ocr_text: str, category: str) -> Optional[str]:
        """
        Extract likely product name from OCR text

        Args:
            ocr_text: Text extracted from image
            category: Product category from CLIP

        Returns:
            Best guess product name or None
        """
        if not ocr_text:
            return None

        # Clean up text
        text = ocr_text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Common brand names (you can expand this)
        known_brands = [
            # Skincare
            'cerave', 'neutrogena', 'cetaphil', 'the ordinary', 'la roche-posay',
            'olay', 'nivea', 'garnier', 'loreal', "l'oreal", 'dove', 'aveeno',
            'eucerin', 'clinique', 'estee lauder', 'lancome', 'shiseido',
            'kiehl', 'drunk elephant', 'tatcha', 'fresh', 'glossier',
            'simple', 'clean & clear', 'ponds', 'himalaya', 'biotique',
            'dot & key', 'plum', 'mamaearth', 'wow', 'lakme', 'maybelline',

            # Electronics
            'apple', 'samsung', 'sony', 'lg', 'dell', 'hp', 'lenovo',
            'asus', 'acer', 'microsoft', 'google', 'oneplus', 'xiaomi',
            'huawei', 'oppo', 'vivo', 'realme', 'nokia', 'motorola',

            # Others
            'nike', 'adidas', 'puma', 'reebok', 'colgate', 'oral-b',
            'gillette', 'head & shoulders', 'pantene', 'sunsilk', 'tresemme'
        ]

        # Find brand mentions
        text_lower = text.lower()
        detected_brands = []
        for brand in known_brands:
            if brand in text_lower:
                detected_brands.append(brand)

        # Strategy 1: If brand detected, combine with category
        if detected_brands:
            # Use the first/longest brand
            brand = max(detected_brands, key=len)

            # Try to find product variant after brand
            brand_pattern = re.compile(rf'{re.escape(brand)}\s+([a-zA-Z0-9\s\-\+&]+)', re.IGNORECASE)
            match = brand_pattern.search(text)

            if match:
                product_variant = match.group(1).strip()
                # Clean up common OCR artifacts
                product_variant = re.sub(r'\s+', ' ', product_variant)
                product_name = f"{brand.title()} {product_variant}"
                logger.info(f"Parsed product name: {product_name}")
                return product_name
            else:
                # Just brand + category
                product_name = f"{brand.title()} {category}"
                logger.info(f"Using brand + category: {product_name}")
                return product_name

        # Strategy 2: Use longest meaningful line
        if lines:
            # Filter out single characters and very short lines
            meaningful_lines = [l for l in lines if len(l) > 3]
            if meaningful_lines:
                # Use first meaningful line (usually product name)
                product_name = meaningful_lines[0]
                logger.info(f"Using first line as product: {product_name}")
                return product_name

        return None

    async def recognize_product(self, image_bytes: bytes) -> ProductRecognitionResponse:
        """
        Enhanced product recognition with OCR + Smart Search

        Args:
            image_bytes: Raw image bytes

        Returns:
            ProductRecognitionResponse with accurate product match
        """
        try:
            # Step 1: Load image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            logger.info(f"Processing image: {image.size}, mode: {image.mode}")

            # Step 2: CLIP category detection
            clip_result = self.clip_recognizer.recognize_product(image)

            if not clip_result["success"]:
                logger.error("CLIP recognition failed")
                return ProductRecognitionResponse(
                    success=False,
                    primary_match=None,
                    alternative_matches=[],
                    message="Failed to identify product category"
                )

            category = clip_result["top_prediction"]["category"]
            clip_confidence = clip_result["top_prediction"]["confidence"]

            logger.info(f"CLIP detected category: {category} ({clip_confidence:.2f})")

            # Step 3: Extract text using OCR
            ocr_text = self.extract_text_from_image(image)

            # Step 4: Parse product name from OCR
            parsed_product = self.parse_product_name_from_text(ocr_text, category)

            # Step 5: If we have a product name, search for it
            if parsed_product:
                logger.info(f"Searching for parsed product: {parsed_product}")

                try:
                    # Use smart product search to find exact match
                    search_result = await self.product_service.search_product(
                        query=parsed_product,
                        use_cache=True,
                        include_ml_analysis=False  # Just need product name
                    )

                    if search_result.get("found"):
                        # We found a real product match!
                        product_name = search_result.get("product_name", parsed_product)

                        logger.info(f"✓ Found exact product match: {product_name}")

                        return ProductRecognitionResponse(
                            success=True,
                            primary_match=ProductMatch(
                                product_name=product_name,
                                confidence=min(clip_confidence + 0.1, 0.99),  # Boost confidence
                                category=category
                            ),
                            alternative_matches=[
                                ProductMatch(
                                    product_name=alt["category"].title(),
                                    confidence=alt["confidence"],
                                    category=alt["category"]
                                )
                                for alt in clip_result["predictions"][1:4]
                            ],
                            message=f"Product identified using image analysis and text recognition"
                        )
                except Exception as e:
                    logger.warning(f"Product search failed: {e}, falling back to CLIP only")

            # Step 6: Fallback to CLIP-only result
            logger.info("Using CLIP category as fallback")

            return ProductRecognitionResponse(
                success=True,
                primary_match=ProductMatch(
                    product_name=category.title(),
                    confidence=clip_confidence,
                    category=category
                ),
                alternative_matches=[
                    ProductMatch(
                        product_name=alt["category"].title(),
                        confidence=alt["confidence"],
                        category=alt["category"]
                    )
                    for alt in clip_result["predictions"][1:4]
                ],
                message=f"Product identified as {category.title()} (generic category)"
            )

        except Exception as e:
            logger.error(f"Error in enhanced recognition: {str(e)}")
            return ProductRecognitionResponse(
                success=False,
                primary_match=None,
                alternative_matches=[],
                message=f"Recognition failed: {str(e)}"
            )


# Singleton instance
_enhanced_recognition_instance = None


def get_enhanced_recognition_service() -> EnhancedRecognitionService:
    """Get or create enhanced recognition service singleton"""
    global _enhanced_recognition_instance
    if _enhanced_recognition_instance is None:
        _enhanced_recognition_instance = EnhancedRecognitionService()
    return _enhanced_recognition_instance
