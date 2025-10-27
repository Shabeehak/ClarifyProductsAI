"""
Sentiment Analysis using DistilBERT (FREE)
Pre-trained model from Hugging Face
"""

from typing import List, Dict
from transformers import pipeline
from loguru import logger

from app.core.config import settings


class SentimentAnalyzer:
    """Sentiment analysis for product reviews"""

    def __init__(self):
        """Initialize sentiment analyzer"""
        self.pipeline = None
        logger.info("Sentiment Analyzer initialized")

    def load_model(self):
        """Load sentiment analysis model"""
        if self.pipeline is None:
            import time
            from pathlib import Path

            try:
                logger.info(f"Loading sentiment model: {settings.SENTIMENT_MODEL_NAME}")
                logger.info(f"Model cache directory: {settings.SENTIMENT_MODEL_PATH}")

                start_time = time.time()

                # Check if model exists in cache
                model_cache_path = Path(settings.SENTIMENT_MODEL_PATH)
                cache_size_mb = 0

                if model_cache_path.exists():
                    # Check for model files
                    model_files = list(model_cache_path.glob("*.bin")) + list(
                        model_cache_path.glob("*.safetensors")
                    )
                    if model_files:
                        cache_size_mb = sum(f.stat().st_size for f in model_files) / (
                            1024 * 1024
                        )
                        logger.info(f"Model found in cache ({cache_size_mb:.1f} MB)")
                    else:
                        logger.info(
                            "Model cache directory exists but no model files found"
                        )
                        logger.info(
                            "Will download from Hugging Face (approximately 260 MB)"
                        )
                else:
                    logger.info("Model not in cache, will download from Hugging Face")
                    logger.info("Expected download size: ~260 MB (DistilBERT)")

                logger.info("Loading sentiment analysis pipeline...")

                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=settings.SENTIMENT_MODEL_NAME,
                    device=-1,  # CPU (use 0 for GPU)
                    model_kwargs={"cache_dir": settings.SENTIMENT_MODEL_PATH},
                )

                total_time = time.time() - start_time

                logger.info("=" * 60)
                logger.info("Sentiment Model Loading Summary:")
                logger.info(f"  Model: {settings.SENTIMENT_MODEL_NAME}")
                logger.info(f"  Device: CPU")
                logger.info(f"  Cache location: {settings.SENTIMENT_MODEL_PATH}")
                if cache_size_mb > 0:
                    logger.info(f"  Cache size: {cache_size_mb:.1f} MB")
                logger.info(f"  Total load time: {total_time:.2f}s")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Failed to load sentiment model: {str(e)}")
                logger.error(f"Model name: {settings.SENTIMENT_MODEL_NAME}")
                logger.error(f"Cache path: {settings.SENTIMENT_MODEL_PATH}")
                from app.core.exceptions import MLModelLoadException

                raise MLModelLoadException(
                    message=f"Failed to load sentiment model: {str(e)}",
                    details={
                        "model": settings.SENTIMENT_MODEL_NAME,
                        "cache_path": settings.SENTIMENT_MODEL_PATH,
                        "error": str(e),
                    },
                ) from e

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text

        Args:
            text: Review text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        if self.pipeline is None:
            self.load_model()

        try:
            import time

            start_time = time.time()
            result = self.pipeline(text[:512])[0]  # Limit to 512 tokens
            inference_time = time.time() - start_time

            # Convert to our format
            label = result["label"].lower()
            score = result["score"]

            # Map to our sentiment categories
            if label == "positive":
                sentiment_label = "positive"
            elif label == "negative":
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            logger.debug(
                f"Sentiment analysis: {sentiment_label} ({score:.3f}) in {inference_time*1000:.1f}ms"
            )

            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": score,
                "confidence": score,
                "inference_time_ms": inference_time * 1000,
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            from app.core.exceptions import MLModelInferenceException

            raise MLModelInferenceException(
                message=f"Sentiment analysis failed: {str(e)}",
                details={
                    "model": "DistilBERT",
                    "text_length": len(text) if text else 0,
                    "error": str(e),
                },
            ) from e

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts

        Args:
            texts: List of review texts

        Returns:
            List of sentiment analysis results
        """
        if self.pipeline is None:
            self.load_model()

        # Log batch processing start
        total = len(texts)
        logger.info(f"Starting batch sentiment analysis for {total} reviews")

        results = []
        for idx, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)

            # Log progress every 20%
            if total >= 5 and (idx + 1) % (total // 5) == 0:
                progress = ((idx + 1) / total) * 100
                logger.debug(f"Batch progress: {idx + 1}/{total} ({progress:.0f}%)")

        # Log batch completion with summary
        successful = sum(1 for r in results if "error" not in r)
        logger.info(
            f"Batch analysis complete: {successful}/{total} successful "
            f"({successful/total*100:.1f}%)"
        )

        return results

    def get_aspect_sentiment(self, text: str, aspect: str) -> Dict:
        """
        Get sentiment for a specific aspect (e.g., 'quality', 'price')

        Args:
            text: Review text
            aspect: Aspect to analyze (quality, price, service, etc.)

        Returns:
            Sentiment for that aspect
        """
        # Find sentences mentioning the aspect
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        relevant_sentences = []
        aspect_lower = aspect.lower()

        for sent in sentences:
            sent_lower = sent.lower()
            if aspect_lower in sent_lower:
                # Extract just the part about the aspect (simple heuristic)
                # Look for the aspect and a few words around it
                words = sent_lower.split()
                try:
                    aspect_index = words.index(aspect_lower)
                    # Take 3 words before and after
                    start = max(0, aspect_index - 3)
                    end = min(len(words), aspect_index + 4)
                    aspect_context = " ".join(sent.split()[start:end])
                    relevant_sentences.append(aspect_context)
                except ValueError:
                    # Aspect might be part of a word, take whole sentence
                    relevant_sentences.append(sent)

        if not relevant_sentences:
            return {
                "aspect": aspect,
                "sentiment_label": "neutral",
                "sentiment_score": 0.5,
                "mentions": 0,
            }

        # Analyze each relevant segment and aggregate
        sentiments = []
        for segment in relevant_sentences:
            result = self.analyze(segment)
            sentiments.append(result)

        # Aggregate sentiment scores
        if len(sentiments) == 1:
            final_result = sentiments[0]
        else:
            # Average the scores
            positive_count = sum(
                1 for s in sentiments if s["sentiment_label"] == "positive"
            )
            negative_count = sum(
                1 for s in sentiments if s["sentiment_label"] == "negative"
            )

            if positive_count > negative_count:
                final_label = "positive"
            elif negative_count > positive_count:
                final_label = "negative"
            else:
                final_label = "neutral"

            avg_score = sum(s["sentiment_score"] for s in sentiments) / len(sentiments)

            final_result = {
                "sentiment_label": final_label,
                "sentiment_score": avg_score,
            }

        return {
            "aspect": aspect,
            "sentiment_label": final_result["sentiment_label"],
            "sentiment_score": final_result["sentiment_score"],
            "mentions": len(relevant_sentences),
            "confidence": final_result["sentiment_score"],
        }


# Singleton instance
_sentiment_analyzer_instance = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create sentiment analyzer singleton"""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        _sentiment_analyzer_instance = SentimentAnalyzer()
    return _sentiment_analyzer_instance
