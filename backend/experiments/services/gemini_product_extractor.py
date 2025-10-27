"""
Gemini Product Extractor - Uses Google Gemini AI to extract product names from OCR text

This solves the hardcoded pattern problem by using AI to understand context and extract
product names from noisy OCR output, regardless of brand or product type.

Benefits:
- No hardcoded brand lists
- Handles unknown/new products
- Understands context (distinguishes product name from marketing)
- Free tier available (Gemini API)
"""

import os
from typing import Optional
from loguru import logger
import google.generativeai as genai


class GeminiProductExtractor:
    """
    Extract product names from OCR text using Google Gemini AI.

    Uses prompt engineering to instruct Gemini to extract only the product name,
    filtering out marketing text, slogans, and promotional content.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini product extractor.

        Args:
            api_key: Google Gemini API key. If None, uses GEMINI_API_KEY env var
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use Gemini 2.5 Flash (latest stable, fast, free tier available)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')

        logger.info("Gemini Product Extractor initialized with gemini-2.5-flash")

    def extract_product_name(self, ocr_text: str) -> Optional[str]:
        """
        Extract product name from OCR text using Gemini AI.

        Args:
            ocr_text: Raw text from OCR engine

        Returns:
            Extracted product name, or None if extraction fails

        Example:
            Input:  "BIGGER SIZE GARNIER SKINACTIVE Micellar Cleansing Water 25% OFF"
            Output: "Garnier Micellar Cleansing Water"
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text provided")
            return None

        # Craft prompt for Gemini
        prompt = f"""You are a product name extraction AI. Extract ONLY the product brand and name from the following text, which came from OCR scanning of product packaging.

RULES:
1. Extract ONLY: Brand + Product Name (e.g., "Neutrogena Hydro Boost Serum")
2. REMOVE: Marketing text (e.g., "BIGGER SIZE", "25% OFF", "NEW", "IMPROVED")
3. REMOVE: Slogans (e.g., "Good morning", "Great taste", "Clinical Formulations with Integrity")
4. REMOVE: Descriptive text (e.g., "DAILY SOURCE OF VITAMINS", "HIGH IN FIBRE")
5. REMOVE: Numbers that are not part of the product name (e.g., "40 pouches", "500g")
6. KEEP: Product descriptors that are part of the name (e.g., "Micellar Water", "Anti-Age Eraser")
7. Fix obvious OCR errors if possible (e.g., "Kelloy" → "Kellogg", "rdiary" → "Ordinary")

OCR TEXT:
{ocr_text}

EXTRACTED PRODUCT NAME (brand + product name only, nothing else):"""

        try:
            logger.debug(f"Sending to Gemini: '{ocr_text[:60]}...'")

            # Generate response
            response = self.model.generate_content(prompt)

            # Extract product name from response
            product_name = response.text.strip()

            # Basic validation
            if not product_name or len(product_name) < 2:
                logger.warning(f"Gemini returned invalid product name: '{product_name}'")
                return None

            # Remove quotes if Gemini added them
            product_name = product_name.strip('"').strip("'")

            logger.info(f"Gemini extracted: '{product_name}' from '{ocr_text[:60]}...'")
            return product_name

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None


# Singleton instance
_gemini_extractor = None


def get_gemini_extractor(api_key: Optional[str] = None) -> GeminiProductExtractor:
    """
    Get singleton Gemini extractor instance.

    Args:
        api_key: Google Gemini API key (optional)

    Returns:
        Shared Gemini extractor instance
    """
    global _gemini_extractor
    if _gemini_extractor is None:
        _gemini_extractor = GeminiProductExtractor(api_key)
    return _gemini_extractor
