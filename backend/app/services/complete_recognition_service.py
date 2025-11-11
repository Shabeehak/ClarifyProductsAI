"""
Complete Product Recognition Service - CLIP + OCR + Gemini Pipeline

Integrates:
1. Image Quality Check (reject poor images early)
2. CLIP Visual Analysis (understand what product it is)
3. OCR Text Extraction (get text from image)
4. Gemini Multimodal Extraction (combine visual + text for accurate name)

This is the production-ready service that addresses all accuracy issues.
"""

from typing import Dict, Optional
from pathlib import Path
from PIL import Image
from loguru import logger

from app.ml_models.clip_recognizer import get_clip_recognizer
from app.services.multimodal_product_extractor import get_multimodal_extractor
from app.utils.image_quality import get_quality_checker
from app.utils.clip_describer import get_clip_describer


class CompleteRecognitionService:
    """
    Complete product recognition pipeline.

    Pipeline stages:
    1. Quality Check → Reject if image too poor
    2. CLIP Analysis → Get visual understanding
    3. OCR Extraction → Get text (PaddleOCR loaded on-demand)
    4. Gemini Multimodal → Combine CLIP + OCR for accurate name
    """

    def __init__(self):
        """Initialize all service components."""
        self.clip_recognizer = get_clip_recognizer()
        self.quality_checker = get_quality_checker()
        self.clip_describer = get_clip_describer()
        self.multimodal_extractor = get_multimodal_extractor()

        # OCR is loaded on-demand (heavy dependency)
        self.ocr_engine = None

        logger.info(
            "Complete Recognition Service initialized (CLIP + OCR + Gemini pipeline)"
        )

    def _load_ocr_if_needed(self):
        """Load PaddleOCR on first use (lazy loading)."""
        if self.ocr_engine is None:
            try:
                from paddleocr import PaddleOCR

                logger.info("Loading PaddleOCR engine...")
                self.ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
                logger.info("PaddleOCR loaded successfully")
            except ImportError:
                logger.error("PaddleOCR not installed. Run: pip install paddleocr")
                raise
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise

    def recognize_product_complete(
        self, image_path: str, skip_quality_check: bool = False
    ) -> Dict:
        """
        Complete product recognition pipeline.

        Args:
            image_path: Path to product image
            skip_quality_check: Skip quality check (for testing)

        Returns:
            {
                "success": bool,
                "product_name": str or None,
                "confidence": str ("high" | "medium" | "low" | "unclear"),
                "method": str (which method succeeded),
                "details": {
                    "quality": dict,
                    "clip_category": str,
                    "ocr_text": str,
                    "clip_description": str
                },
                "message": str (user-friendly message)
            }
        """
        try:
            # Stage 1: Quality Check
            if not skip_quality_check:
                logger.info(
                    f"Stage 1: Checking image quality for {Path(image_path).name}"
                )
                quality_result = self.quality_checker.check_quality(image_path)

                if not quality_result["is_acceptable"]:
                    feedback = self.quality_checker.get_quality_feedback(quality_result)
                    logger.warning(f"Image quality check failed: {feedback}")

                    return {
                        "success": False,
                        "product_name": None,
                        "confidence": "unclear",
                        "method": "quality_check",
                        "details": {"quality": quality_result},
                        "message": feedback,
                    }

                logger.info(
                    f"Quality OK (score: {quality_result['quality_score']:.2f})"
                )
            else:
                quality_result = {"quality_score": 1.0, "skipped": True}

            # Stage 2: CLIP Visual Analysis
            logger.info("Stage 2: Running CLIP visual analysis")
            image = Image.open(image_path)

            clip_result = self.clip_recognizer.recognize_from_bytes(
                open(image_path, "rb").read()
            )

            if not clip_result.get("success"):
                logger.error("CLIP recognition failed")
                return {
                    "success": False,
                    "product_name": None,
                    "confidence": "unclear",
                    "method": "clip",
                    "details": {"quality": quality_result},
                    "message": "Failed to analyze image visually",
                }

            clip_category = clip_result["top_prediction"]["category"]
            logger.info(
                f"CLIP detected: {clip_category} ({clip_result['top_prediction']['confidence']:.1%})"
            )

            # Generate CLIP description for Gemini
            clip_description = self.clip_describer.generate_description(clip_result)

            # Stage 3: OCR Text Extraction
            logger.info("Stage 3: Extracting text with OCR")
            self._load_ocr_if_needed()

            ocr_result = self.ocr_engine.ocr(image_path, cls=True)

            # Extract text from PaddleOCR result
            # PaddleOCR 2.7.0.3 format: list of lists [[[bbox], (text, confidence)], ...]
            ocr_texts = []
            if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                for line in ocr_result[0]:
                    if line and len(line) > 1 and line[1] and line[1][0].strip():
                        ocr_texts.append(line[1][0])

            ocr_text = " ".join(ocr_texts) if ocr_texts else ""

            if not ocr_text.strip():
                logger.warning("OCR extracted no text")
                # Fallback to CLIP category as product name
                return {
                    "success": True,
                    "product_name": clip_category.title(),
                    "confidence": "low",
                    "method": "clip_fallback",
                    "details": {
                        "quality": quality_result,
                        "clip_category": clip_category,
                        "clip_description": clip_description,
                        "ocr_text": "",
                    },
                    "message": f"Showing results for category: {clip_category.title()}",
                }

            logger.info(f"OCR extracted: '{ocr_text[:100]}...'")

            # Stage 4: Gemini Multimodal Extraction
            logger.info("Stage 4: Gemini multimodal extraction (CLIP + OCR)")

            extraction_result = self.multimodal_extractor.extract_with_confidence(
                ocr_text=ocr_text,
                clip_description=clip_description,
                image_path=image_path,
            )

            product_name = extraction_result["product_name"]
            confidence = extraction_result["confidence"]

            if product_name:
                logger.info(
                    f"Successfully extracted: '{product_name}' (confidence: {confidence})"
                )

                return {
                    "success": True,
                    "product_name": product_name,
                    "confidence": confidence,
                    "method": "multimodal",
                    "details": {
                        "quality": quality_result,
                        "clip_category": clip_category,
                        "clip_description": clip_description,
                        "ocr_text": ocr_text,
                    },
                    "message": f"Product identified: {product_name}",
                }
            else:
                # Fallback to CLIP if multimodal extraction failed
                logger.warning(
                    "Multimodal extraction returned unclear result, falling back to CLIP"
                )

                return {
                    "success": True,
                    "product_name": clip_category.title(),
                    "confidence": "low",
                    "method": "clip_fallback",
                    "details": {
                        "quality": quality_result,
                        "clip_category": clip_category,
                        "clip_description": clip_description,
                        "ocr_text": ocr_text,
                    },
                    "message": f"Showing results for category: {clip_category.title()}",
                }

        except Exception as e:
            logger.error(f"Complete recognition failed: {e}", exc_info=True)
            return {
                "success": False,
                "product_name": None,
                "confidence": "unclear",
                "method": "error",
                "details": {},
                "message": f"Recognition failed: {str(e)}",
            }


# Singleton instance
_complete_recognition_service = None


def get_complete_recognition_service() -> CompleteRecognitionService:
    """Get singleton complete recognition service instance."""
    global _complete_recognition_service
    if _complete_recognition_service is None:
        _complete_recognition_service = CompleteRecognitionService()
    return _complete_recognition_service
