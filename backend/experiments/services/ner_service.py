"""
NER Service - Named Entity Recognition for Product Information Extraction

Uses GLiNER (Generalist and Lightweight Model for Named Entity Recognition)
to extract structured product information from OCR text output.

Purpose: Filter OCR text to extract only relevant product information,
removing marketing text, slogans, and promotional content.
"""

from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
from gliner import GLiNER


class NERService:
    """
    Named Entity Recognition service using GLiNER.

    Extracts product-specific entities from text:
    - BRAND: Product brand name (e.g., "Neutrogena", "Garnier")
    - PRODUCT_NAME: Full product name (e.g., "Hydro Boost Serum")
    - PRODUCT_TYPE: Category/type (e.g., "serum", "cream", "lotion")
    - SIZE: Product size (e.g., "30ml", "1 FL.OZ", "500g")
    - COLOR: Color variant (e.g., "Natural Beige", "Ivory")
    - MODEL_NUMBER: Model/SKU (e.g., "SPF 50", "No. 5")
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        """
        Initialize NER service with GLiNER model.

        Args:
            model_name: GLiNER model to use. Options:
                - "urchade/gliner_small-v2.1" (40MB, fast, good accuracy)
                - "urchade/gliner_medium-v2.1" (120MB, balanced)
                - "urchade/gliner_large-v2.1" (350MB, best accuracy)
        """
        self.model_name = model_name
        self.model = None
        self.labels = [
            "BRAND",
            "PRODUCT_NAME",
            "PRODUCT_TYPE",
            "SIZE",
            "COLOR",
            "MODEL_NUMBER"
        ]
        logger.info(f"Initializing NER Service with model: {model_name}")

    def load_model(self):
        """Load GLiNER model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            try:
                self.model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GLiNER model: {e}")
                raise

    def extract_entities(
        self,
        text: str,
        threshold: float = 0.3
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Extract product entities from text.

        Args:
            text: Input text from OCR
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            Dictionary with entity types as keys and lists of detected entities:
            {
                "BRAND": [{"text": "Neutrogena", "score": 0.95}],
                "PRODUCT_NAME": [{"text": "Hydro Boost", "score": 0.89}],
                ...
            }
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to NER service")
            return {label: [] for label in self.labels}

        self.load_model()

        try:
            # Run GLiNER prediction
            entities = self.model.predict_entities(
                text,
                self.labels,
                threshold=threshold
            )

            # Organize entities by type
            result = {label: [] for label in self.labels}
            for entity in entities:
                entity_type = entity["label"]
                entity_info = {
                    "text": entity["text"],
                    "score": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"]
                }
                result[entity_type].append(entity_info)

            # Log extraction summary
            total_entities = sum(len(v) for v in result.values())
            logger.info(f"Extracted {total_entities} entities from text: '{text[:50]}...'")
            for entity_type, entities_list in result.items():
                if entities_list:
                    logger.debug(f"  {entity_type}: {[e['text'] for e in entities_list]}")

            return result

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return {label: [] for label in self.labels}

    def extract_product_name(
        self,
        text: str,
        threshold: float = 0.3
    ) -> Optional[str]:
        """
        Extract clean product name from OCR text.

        Combines BRAND + PRODUCT_NAME entities to form complete product name,
        filtering out marketing text and promotional content.

        Args:
            text: Input text from OCR
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            Clean product name string, or None if extraction fails

        Example:
            Input:  "BIGGER SIZE GARNIER SKINACTIVE Micellar Cleansing Water 25% OFF"
            Output: "Garnier Micellar Cleansing Water"
        """
        entities = self.extract_entities(text, threshold)

        # Get brand and product name entities
        brands = entities.get("BRAND", [])
        product_names = entities.get("PRODUCT_NAME", [])

        # If we have both, combine them
        if brands and product_names:
            brand = brands[0]["text"]  # Take highest confidence brand
            product = product_names[0]["text"]  # Take highest confidence product name
            result = f"{brand} {product}".strip()
            logger.info(f"Extracted product name: '{result}' from '{text[:50]}...'")
            return result

        # If we only have brand, return it
        elif brands:
            result = brands[0]["text"]
            logger.info(f"Extracted brand only: '{result}' from '{text[:50]}...'")
            return result

        # If we only have product name, return it
        elif product_names:
            result = product_names[0]["text"]
            logger.info(f"Extracted product name only: '{result}' from '{text[:50]}...'")
            return result

        # No relevant entities found
        else:
            logger.warning(f"No product entities found in text: '{text[:50]}...'")
            return None

    def extract_all_info(
        self,
        text: str,
        threshold: float = 0.3
    ) -> Dict[str, Optional[str]]:
        """
        Extract all product information from text.

        Args:
            text: Input text from OCR
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            Dictionary with single best value for each entity type:
            {
                "brand": "Neutrogena",
                "product_name": "Hydro Boost Serum",
                "product_type": "serum",
                "size": "30ml",
                "color": "Natural",
                "model_number": None
            }
        """
        entities = self.extract_entities(text, threshold)

        result = {}
        for entity_type, entities_list in entities.items():
            # Take the highest confidence entity for each type
            if entities_list:
                result[entity_type.lower()] = entities_list[0]["text"]
            else:
                result[entity_type.lower()] = None

        return result


# Singleton instance
_ner_service = None


def get_ner_service(model_name: str = "urchade/gliner_small-v2.1") -> NERService:
    """
    Get singleton NER service instance.

    Args:
        model_name: GLiNER model to use

    Returns:
        Shared NER service instance
    """
    global _ner_service
    if _ner_service is None:
        _ner_service = NERService(model_name)
    return _ner_service
