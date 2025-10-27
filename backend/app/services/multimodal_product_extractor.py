"""
Multimodal Product Extractor - Combines CLIP + OCR + Gemini for accurate extraction

This solves the hallucination problem by giving Gemini visual context (CLIP)
along with text (OCR), allowing it to make informed corrections rather than guessing.

Pipeline:
1. CLIP analyzes image → Visual understanding (product type, brand, category)
2. OCR extracts text → Raw text (potentially corrupted)
3. Gemini combines both → Accurate product name (context prevents hallucination)
"""

import os
from typing import Optional, Dict, Any
from loguru import logger
import google.generativeai as genai
from PIL import Image


class MultimodalProductExtractor:
    """
    Extract product names using CLIP (vision) + OCR (text) + Gemini (reasoning).

    Addresses hallucination by providing visual context alongside OCR text.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize multimodal product extractor.

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

        # Use Gemini 2.5 Flash (latest stable, supports vision + text)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

        logger.info("Multimodal Product Extractor initialized with CLIP + OCR + Gemini")

    def extract_product_name(
        self,
        ocr_text: str,
        clip_description: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract product name using multimodal approach.

        Args:
            ocr_text: Raw text from OCR engine
            clip_description: Optional CLIP-generated description of image
            image_path: Optional path to image (Gemini can analyze directly)

        Returns:
            Extracted product name, or None if extraction fails

        Example:
            Input:
                ocr_text = "00 CONTERR M-LTSE EAR STAANTGE MAYBELLINE"
                clip_description = "makeup concealer stick, skin tone color, Maybelline brand"

            Output: "Maybelline Instant Age Eraser Concealer"
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text provided")
            return None

        # Build context-aware prompt
        prompt = self._build_prompt(ocr_text, clip_description)

        try:
            logger.debug(
                f"Sending to Gemini with CLIP context: {clip_description[:100] if clip_description else 'None'}..."
            )

            # If we have image, send it directly to Gemini for visual analysis
            if image_path:
                response = self._extract_with_image(prompt, image_path)
            else:
                response = self.model.generate_content(prompt)

            # Extract product name from response
            product_name = response.text.strip()

            # Basic validation
            if not product_name or len(product_name) < 2:
                logger.warning(
                    f"Gemini returned invalid product name: '{product_name}'"
                )
                return None

            # Remove quotes if Gemini added them
            product_name = product_name.strip('"').strip("'")

            logger.info(
                f"Multimodal extracted: '{product_name}' from OCR: '{ocr_text[:60]}...'"
            )
            return product_name

        except Exception as e:
            logger.error(f"Multimodal extraction failed: {e}")
            return None

    def _build_prompt(self, ocr_text: str, clip_description: Optional[str]) -> str:
        """
        Build context-aware prompt with CLIP description and OCR text.

        This prevents hallucination by giving Gemini visual context.
        """
        if clip_description:
            # CLIP + OCR prompt (best accuracy)
            prompt = f"""You are a product name extraction AI. You have access to:

1. VISUAL ANALYSIS (from AI vision model):
{clip_description}

2. TEXT EXTRACTED FROM IMAGE (OCR - may contain errors):
{ocr_text}

TASK:
Extract ONLY the product brand and name. Use the visual context to fix OCR errors.

RULES:
1. Combine visual understanding with OCR text
2. Fix obvious OCR mistakes (e.g., "CONTERR" → "Concealer" if visual shows concealer)
3. Remove marketing text ("BIGGER SIZE", "25% OFF", "Good morning")
4. If OCR is completely wrong and you can't determine product from visual context, output: UNCLEAR
5. Don't invent product names - only use information from visual + OCR
6. Output format: "Brand Product Name" (e.g., "Maybelline Instant Age Eraser")

EXTRACTED PRODUCT NAME:"""
        else:
            # OCR-only prompt (fallback, with anti-hallucination instructions)
            prompt = f"""You are a product name extraction AI. Extract ONLY the product brand and name from this OCR text.

OCR TEXT (may contain errors):
{ocr_text}

RULES:
1. Extract brand + product name only
2. Remove marketing text ("BIGGER SIZE", "25% OFF", slogans)
3. Fix OBVIOUS OCR errors if you're confident (e.g., "Kelloy" → "Kellogg's")
4. If text is too corrupted to understand, output: UNCLEAR
5. DO NOT guess or invent product names
6. If unsure, prefer outputting partial name over wrong name

EXTRACTED PRODUCT NAME:"""

        return prompt

    def _extract_with_image(self, prompt: str, image_path: str) -> Any:
        """
        Extract product name by sending image directly to Gemini.

        Gemini can analyze the image itself, which is even better than CLIP description.
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Send image + prompt to Gemini
            response = self.model.generate_content([prompt, image])

            return response

        except Exception as e:
            logger.error(f"Failed to extract with image: {e}")
            # Fallback to text-only
            return self.model.generate_content(prompt)

    def extract_with_confidence(
        self,
        ocr_text: str,
        clip_description: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract product name with confidence score and reasoning.

        Returns:
            {
                "product_name": str or None,
                "confidence": "high" | "medium" | "low" | "unclear",
                "reasoning": str (why this extraction)
            }
        """
        product_name = self.extract_product_name(ocr_text, clip_description, image_path)

        if not product_name or product_name == "UNCLEAR":
            return {
                "product_name": None,
                "confidence": "unclear",
                "reasoning": "OCR text too corrupted or image unclear",
            }

        # Assess confidence based on OCR quality and CLIP context
        if clip_description and len(product_name.split()) >= 2:
            confidence = "high"
            reasoning = "CLIP context + clear OCR text"
        elif clip_description:
            confidence = "medium"
            reasoning = "CLIP context available but OCR partial"
        elif len(product_name.split()) >= 2:
            confidence = "medium"
            reasoning = "Clear OCR text but no visual context"
        else:
            confidence = "low"
            reasoning = "Limited information available"

        return {
            "product_name": product_name,
            "confidence": confidence,
            "reasoning": reasoning,
        }


# Singleton instance
_multimodal_extractor = None


def get_multimodal_extractor(
    api_key: Optional[str] = None,
) -> MultimodalProductExtractor:
    """
    Get singleton multimodal extractor instance.

    Args:
        api_key: Google Gemini API key (optional, loads from settings if not provided)

    Returns:
        Shared multimodal extractor instance
    """
    global _multimodal_extractor
    if _multimodal_extractor is None:
        # Load from settings if not provided
        if api_key is None:
            from app.core.config import settings

            api_key = settings.GEMINI_API_KEY
        _multimodal_extractor = MultimodalProductExtractor(api_key)
    return _multimodal_extractor
