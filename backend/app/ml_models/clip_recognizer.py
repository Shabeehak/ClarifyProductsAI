"""
CLIP-based Product Image Recognition (FREE)
Uses OpenAI's CLIP model from Hugging Face
"""

from typing import List, Dict, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from loguru import logger
import io

from app.core.config import settings


class CLIPRecognizer:
    """Product recognition using CLIP model"""

    def __init__(self):
        """Initialize CLIP model"""
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CLIP Recognizer initialized. Device: {self.device}")

    def load_model(self):
        """Load CLIP model from Hugging Face"""
        if self.model is None:
            import time
            from pathlib import Path

            try:
                logger.info(f"Loading CLIP model: {settings.CLIP_MODEL_NAME}")
                logger.info(f"Model cache directory: {settings.CLIP_MODEL_PATH}")
                logger.info(f"Target device: {self.device}")

                start_time = time.time()

                # Check if model exists in cache
                model_cache_path = Path(settings.CLIP_MODEL_PATH)
                model_exists = False
                cache_size_mb = 0

                if model_cache_path.exists():
                    # Check for model files
                    model_files = list(model_cache_path.glob("*.bin")) + list(
                        model_cache_path.glob("*.safetensors")
                    )
                    if model_files:
                        model_exists = True
                        cache_size_mb = sum(f.stat().st_size for f in model_files) / (
                            1024 * 1024
                        )
                        logger.info(f"Model found in cache ({cache_size_mb:.1f} MB)")
                    else:
                        logger.info(
                            "Model cache directory exists but no model files found"
                        )
                        logger.info(
                            "Will download from Hugging Face (approximately 600 MB)"
                        )
                else:
                    logger.info("Model not in cache, will download from Hugging Face")
                    logger.info(
                        "Expected download size: ~600 MB (may take 5-15 minutes depending on connection)"
                    )

                # Load model
                logger.info("Loading CLIP model into memory...")
                load_start = time.time()

                self.model = CLIPModel.from_pretrained(
                    settings.CLIP_MODEL_NAME, cache_dir=settings.CLIP_MODEL_PATH
                )
                model_load_time = time.time() - load_start
                logger.info(f"CLIP model loaded in {model_load_time:.2f}s")

                # Load processor
                logger.info("Loading CLIP processor...")
                processor_start = time.time()
                self.processor = CLIPProcessor.from_pretrained(
                    settings.CLIP_MODEL_NAME, cache_dir=settings.CLIP_MODEL_PATH
                )
                processor_load_time = time.time() - processor_start
                logger.info(f"CLIP processor loaded in {processor_load_time:.2f}s")

                # Move to device
                logger.info(f"Moving model to device: {self.device}")
                device_start = time.time()
                self.model.to(self.device)
                self.model.eval()
                device_time = time.time() - device_start
                logger.info(f"Model moved to {self.device} in {device_time:.2f}s")

                # Calculate model size
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )

                total_time = time.time() - start_time

                logger.info("=" * 60)
                logger.info("CLIP Model Loading Summary:")
                logger.info(f"  Model: {settings.CLIP_MODEL_NAME}")
                logger.info(f"  Device: {self.device}")
                logger.info(f"  Total parameters: {total_params:,}")
                logger.info(f"  Trainable parameters: {trainable_params:,}")
                logger.info(f"  Cache location: {settings.CLIP_MODEL_PATH}")
                if cache_size_mb > 0:
                    logger.info(f"  Cache size: {cache_size_mb:.1f} MB")
                logger.info(f"  Total load time: {total_time:.2f}s")
                logger.info(f"    - Model load: {model_load_time:.2f}s")
                logger.info(f"    - Processor load: {processor_load_time:.2f}s")
                logger.info(f"    - Device transfer: {device_time:.2f}s")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Failed to load CLIP model: {str(e)}")
                logger.error(f"Model name: {settings.CLIP_MODEL_NAME}")
                logger.error(f"Cache path: {settings.CLIP_MODEL_PATH}")
                logger.error(f"Device: {self.device}")
                from app.core.exceptions import MLModelLoadException

                raise MLModelLoadException(
                    message=f"Failed to load CLIP model: {str(e)}",
                    details={
                        "model": settings.CLIP_MODEL_NAME,
                        "cache_path": settings.CLIP_MODEL_PATH,
                        "device": self.device,
                        "error": str(e),
                    },
                ) from e

    def recognize_product(
        self, image: Image.Image, candidate_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Recognize product from image using CLIP

        Args:
            image: PIL Image object
            candidate_labels: List of product categories to match against

        Returns:
            Dictionary with recognition results
        """
        if self.model is None:
            self.load_model()

        # Default product categories if none provided
        if candidate_labels is None:
            candidate_labels = [
                # Electronics
                "smartphone",
                "laptop",
                "tablet",
                "headphones",
                "camera",
                "television",
                "smartwatch",
                "speaker",
                "gaming console",
                "keyboard",
                "mouse",
                "monitor",
                "printer",
                "router",
                "earbuds",
                "fitness tracker",
                "drone",
                "power bank",
                "phone case",
                "charger",
                "cable",
                "adapter",
                # Skincare & Beauty
                "moisturizer",
                "face cream",
                "lotion",
                "sunscreen",
                "serum",
                "cleanser",
                "face wash",
                "toner",
                "mask",
                "eye cream",
                "shampoo",
                "conditioner",
                "hair oil",
                "body wash",
                "soap",
                "perfume",
                "cologne",
                "deodorant",
                "toothpaste",
                "makeup",
                "lipstick",
                "foundation",
                "mascara",
                "nail polish",
                # Home & Kitchen
                "vacuum cleaner",
                "air purifier",
                "coffee maker",
                "blender",
                "microwave",
                "refrigerator",
                "washing machine",
                "fan",
                # Fashion
                "shoes",
                "sneakers",
                "backpack",
                "wallet",
                "sunglasses",
                "watch",
                "bag",
                "belt",
                "hat",
                "jacket",
                # Health & Fitness
                "protein powder",
                "vitamins",
                "supplements",
                "yoga mat",
                "dumbbells",
                "resistance bands",
                "water bottle",
            ]

        try:
            import time

            inference_start = time.time()

            # Prepare inputs
            preprocess_start = time.time()
            inputs = self.processor(
                text=candidate_labels, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            preprocess_time = time.time() - preprocess_start

            # Get predictions
            model_start = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            model_time = time.time() - model_start

            # Get top predictions
            postprocess_start = time.time()
            top_probs, top_indices = probs[0].topk(5)

            results = []
            for prob, idx in zip(top_probs, top_indices):
                results.append(
                    {
                        "category": candidate_labels[idx.item()],
                        "confidence": float(prob.item()),
                    }
                )
            postprocess_time = time.time() - postprocess_start

            total_time = time.time() - inference_start

            logger.info(
                f"Recognition complete. Top match: {results[0]['category']} ({results[0]['confidence']:.2f})"
            )
            logger.debug(f"CLIP inference timing:")
            logger.debug(f"  - Preprocess: {preprocess_time*1000:.1f}ms")
            logger.debug(f"  - Model inference: {model_time*1000:.1f}ms")
            logger.debug(f"  - Postprocess: {postprocess_time*1000:.1f}ms")
            logger.debug(f"  - Total: {total_time*1000:.1f}ms")

            return {
                "success": True,
                "predictions": results,
                "top_prediction": results[0],
                "inference_time_ms": total_time * 1000,
                "performance": {
                    "preprocess_ms": preprocess_time * 1000,
                    "inference_ms": model_time * 1000,
                    "postprocess_ms": postprocess_time * 1000,
                },
            }

        except Exception as e:
            logger.error(f"Error during CLIP recognition: {str(e)}")
            from app.core.exceptions import MLModelInferenceException

            raise MLModelInferenceException(
                message=f"CLIP inference failed: {str(e)}",
                details={
                    "model": "CLIP",
                    "num_labels": len(candidate_labels) if candidate_labels else 0,
                    "error": str(e),
                },
            ) from e

    def recognize_from_bytes(
        self, image_bytes: bytes, candidate_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        Recognize product from image bytes

        Args:
            image_bytes: Raw image bytes
            candidate_labels: Optional list of product categories

        Returns:
            Recognition results
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            return self.recognize_product(image, candidate_labels)

        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            from app.core.exceptions import InvalidImageException

            raise InvalidImageException(
                message=f"Failed to process image: {str(e)}",
                details={
                    "error": str(e),
                    "bytes_length": len(image_bytes) if image_bytes else 0,
                },
            ) from e


# Singleton instance
_recognizer_instance = None


def get_clip_recognizer() -> CLIPRecognizer:
    """Get or create CLIP recognizer singleton"""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = CLIPRecognizer()
    return _recognizer_instance
