"""
CLIP Description Generator - Converts CLIP predictions to text descriptions for Gemini

Takes CLIP recognition results and formats them as natural language descriptions
that provide visual context to Gemini for accurate product name extraction.
"""

from typing import Dict, List
from loguru import logger


class CLIPDescriber:
    """Generate text descriptions from CLIP recognition results."""

    def generate_description(self, clip_result: Dict) -> str:
        """
        Generate natural language description from CLIP recognition results.

        Args:
            clip_result: Result from CLIP recognizer with format:
                {
                    "success": bool,
                    "top_prediction": {"category": str, "confidence": float},
                    "predictions": [{"category": str, "confidence": float}, ...]
                }

        Returns:
            Natural language description suitable for Gemini context

        Example:
            Input: {"top_prediction": {"category": "moisturizer", "confidence": 0.95}}
            Output: "skincare product, moisturizer, bottle packaging"
        """
        if not clip_result.get("success"):
            logger.warning("CLIP recognition failed, returning generic description")
            return "product image, unclear category"

        top_pred = clip_result.get("top_prediction", {})
        all_preds = clip_result.get("predictions", [])

        category = top_pred.get("category", "product")
        confidence = top_pred.get("confidence", 0.0)

        # Build description parts
        parts = []

        # Add category with confidence indication
        if confidence >= 0.8:
            parts.append(f"{category}")
        elif confidence >= 0.5:
            parts.append(f"likely {category}")
        else:
            parts.append(f"possibly {category}")

        # Add alternative categories if they're plausible
        alternatives = []
        for pred in all_preds[1:4]:  # Top 3 alternatives
            if pred["confidence"] >= 0.3:  # Only if somewhat confident
                alternatives.append(pred["category"])

        if alternatives:
            parts.append(f"or {' or '.join(alternatives)}")

        # Add product type hints based on category
        category_hints = self._get_category_hints(category)
        if category_hints:
            parts.extend(category_hints)

        description = ", ".join(parts)

        logger.debug(f"Generated CLIP description: '{description}'")
        return description

    def _get_category_hints(self, category: str) -> List[str]:
        """
        Get additional descriptive hints based on product category.

        These hints help Gemini understand the product context better.
        """
        category_lower = category.lower()

        # Skincare/beauty
        if any(
            word in category_lower
            for word in ["cream", "serum", "moisturizer", "cleanser", "lotion"]
        ):
            return ["skincare product", "bottle or tube packaging"]

        if any(
            word in category_lower
            for word in ["lipstick", "mascara", "foundation", "concealer", "makeup"]
        ):
            return ["cosmetic product", "beauty item"]

        # Food/beverage
        if any(
            word in category_lower
            for word in ["cereal", "snack", "food", "beverage", "drink"]
        ):
            return ["food product", "packaged goods"]

        # Electronics
        if any(
            word in category_lower
            for word in ["phone", "laptop", "tablet", "electronics", "gadget"]
        ):
            return ["electronic device", "tech product"]

        # Pet products
        if any(word in category_lower for word in ["cat food", "dog food", "pet"]):
            return ["pet product", "animal care"]

        # Generic
        return ["consumer product"]

    def generate_detailed_description(
        self, clip_result: Dict, include_confidence: bool = False
    ) -> str:
        """
        Generate more detailed description including all predictions.

        Args:
            clip_result: CLIP recognition results
            include_confidence: Whether to include confidence scores

        Returns:
            Detailed description with all CLIP predictions
        """
        if not clip_result.get("success"):
            return "product image with unclear category"

        all_preds = clip_result.get("predictions", [])

        if not all_preds:
            return "product image"

        # Format predictions
        pred_strs = []
        for i, pred in enumerate(all_preds[:5]):  # Top 5
            category = pred["category"]
            confidence = pred["confidence"]

            if include_confidence:
                pred_strs.append(f"{category} ({confidence:.0%})")
            else:
                pred_strs.append(category)

        description = f"Product categories detected: {', '.join(pred_strs)}"

        logger.debug(f"Generated detailed CLIP description: '{description}'")
        return description


# Singleton instance
_clip_describer = None


def get_clip_describer() -> CLIPDescriber:
    """Get singleton CLIP describer instance."""
    global _clip_describer
    if _clip_describer is None:
        _clip_describer = CLIPDescriber()
    return _clip_describer
