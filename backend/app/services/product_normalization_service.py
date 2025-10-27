"""
Product Name Normalization Service
Handles typos, variations, and fuzzy matching
"""

from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import re
from loguru import logger


class ProductNormalizationService:
    """Normalize and fuzzy match product names"""

    def __init__(self):
        """Initialize normalization service"""
        # Common product variations and corrections
        self.brand_variations = {
            # Apple
            "aple": "apple",
            "appel": "apple",
            "appl": "apple",
            # Samsung
            "samsng": "samsung",
            "samung": "samsung",
            "samsug": "samsung",
            # Sony
            "sny": "sony",
            "soony": "sony",
            # Dell
            "del": "dell",
            "dele": "dell",
            # Microsoft
            "microsft": "microsoft",
            "micros": "microsoft",
            # Google
            "gogle": "google",
            "googl": "google",
            # Canon
            "canan": "canon",
            "canno": "canon",
        }

        self.product_type_variations = {
            # Phone variants
            "phone": ["phone", "smartphone", "mobile", "cellphone", "cell"],
            "iphone": ["iphone", "i phone", "i-phone", "iphne", "ifone"],
            # Laptop variants
            "laptop": ["laptop", "notebook", "portable computer", "labtop", "lptop"],
            "macbook": ["macbook", "mac book", "macbok", "mackbook"],
            # Headphones
            "headphones": [
                "headphones",
                "headphone",
                "earphones",
                "earphone",
                "hedphones",
            ],
            "airpods": ["airpods", "air pods", "airpod", "arpods"],
            # Camera
            "camera": ["camera", "cam", "camra", "cmera"],
            # Watch
            "watch": ["watch", "smartwatch", "wristwatch", "wach"],
        }

        # Common typos mapping
        self.common_typos = {
            "pro": ["pro", "proe", "por"],
            "max": ["max", "maxx", "maks"],
            "plus": ["plus", "pluss", "puls"],
            "ultra": ["ultra", "ultr", "altra"],
        }

    def normalize(self, query: str) -> str:
        """
        Normalize product query

        Args:
            query: User's search query (may have typos)

        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra spaces
        normalized = re.sub(r"\s+", " ", normalized)

        # Fix brand variations
        for typo, correct in self.brand_variations.items():
            normalized = normalized.replace(typo, correct)

        # Remove special characters except spaces and hyphens
        normalized = re.sub(r"[^a-z0-9\s\-]", "", normalized)

        logger.info(f"Normalized '{query}' to '{normalized}'")

        return normalized

    def fuzzy_match(
        self, query: str, candidates: List[str], threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches for a query

        Args:
            query: Search query
            candidates: List of possible matches
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (candidate, score) tuples, sorted by score
        """
        matches = []

        query_normalized = self.normalize(query)

        for candidate in candidates:
            candidate_normalized = self.normalize(candidate)

            # Calculate similarity
            similarity = SequenceMatcher(
                None, query_normalized, candidate_normalized
            ).ratio()

            # Also check if query is substring of candidate
            if query_normalized in candidate_normalized:
                similarity = max(similarity, 0.8)

            if similarity >= threshold:
                matches.append((candidate, similarity))

        # Sort by score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def suggest_corrections(self, query: str) -> List[str]:
        """
        Suggest corrections for common typos

        Args:
            query: Search query with potential typos

        Returns:
            List of suggested corrections
        """
        suggestions = []
        query_lower = query.lower()

        # Check brand typos
        for typo, correct in self.brand_variations.items():
            if typo in query_lower:
                suggestion = query_lower.replace(typo, correct)
                suggestions.append(suggestion)

        # Check product type variations
        for correct_type, variations in self.product_type_variations.items():
            for variant in variations:
                if variant in query_lower and variant != correct_type:
                    suggestion = query_lower.replace(variant, correct_type)
                    suggestions.append(suggestion)

        # Remove duplicates
        suggestions = list(set(suggestions))

        # Only return if different from original
        suggestions = [s for s in suggestions if s != query_lower]

        return suggestions

    def extract_product_info(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract brand, model, and type from query

        Args:
            query: Product search query

        Returns:
            Dict with brand, model, type
        """
        normalized = self.normalize(query)

        result = {
            "brand": None,
            "model": None,
            "type": None,
            "normalized_query": normalized,
        }

        # Common brands
        brands = [
            "apple",
            "samsung",
            "sony",
            "dell",
            "hp",
            "lenovo",
            "microsoft",
            "google",
            "canon",
            "nikon",
            "lg",
            "asus",
            "acer",
        ]

        for brand in brands:
            if brand in normalized:
                result["brand"] = brand
                break

        # Common product types
        types = [
            "phone",
            "laptop",
            "tablet",
            "watch",
            "headphones",
            "camera",
            "speaker",
            "monitor",
            "keyboard",
            "mouse",
            "earbuds",
        ]

        for ptype in types:
            if ptype in normalized:
                result["type"] = ptype
                break

        # Extract model (remaining words)
        words = normalized.split()
        model_parts = []
        for word in words:
            if word not in brands and word not in types:
                model_parts.append(word)

        if model_parts:
            result["model"] = " ".join(model_parts)

        return result


# Singleton instance
_normalization_service_instance = None


def get_normalization_service() -> ProductNormalizationService:
    """Get or create normalization service singleton"""
    global _normalization_service_instance
    if _normalization_service_instance is None:
        _normalization_service_instance = ProductNormalizationService()
    return _normalization_service_instance
