"""
Review Summarization using BART (FREE)
Facebook's BART model from Hugging Face
"""

from typing import List, Dict
from transformers import pipeline
from loguru import logger
import requests

from app.core.config import settings


class ReviewSummarizer:
    """AI-powered review summarization"""

    def __init__(self):
        """Initialize summarizer"""
        self.pipeline = None
        self.use_ollama = False
        logger.info("Review Summarizer initialized")

    def load_model(self):
        """Load summarization model"""
        if self.pipeline is None:
            import time
            from pathlib import Path

            try:
                start_time = time.time()

                # Try Ollama first if available
                logger.info("Checking for Ollama availability...")
                if self._check_ollama_available():
                    logger.info("Ollama is available and running")
                    logger.info(
                        f"Using Ollama for summarization at {settings.OLLAMA_API_URL}"
                    )
                    self.use_ollama = True

                    total_time = time.time() - start_time

                    logger.info("=" * 60)
                    logger.info("Summarization Model Loading Summary:")
                    logger.info(f"  Model: Ollama (LLaMA/Mistral)")
                    logger.info(f"  API URL: {settings.OLLAMA_API_URL}")
                    logger.info(f"  Setup time: {total_time:.2f}s")
                    logger.info("=" * 60)
                else:
                    logger.info("Ollama not available, using Hugging Face BART model")
                    logger.info(
                        f"Loading BART model: {settings.SUMMARIZATION_MODEL_NAME}"
                    )

                    # Check cache
                    model_cache_path = Path(settings.MODEL_CACHE_DIR) / "summarization"
                    cache_size_mb = 0

                    if model_cache_path.exists():
                        model_files = list(model_cache_path.glob("*.bin")) + list(
                            model_cache_path.glob("*.safetensors")
                        )
                        if model_files:
                            cache_size_mb = sum(
                                f.stat().st_size for f in model_files
                            ) / (1024 * 1024)
                            logger.info(
                                f"Model found in cache ({cache_size_mb:.1f} MB)"
                            )
                        else:
                            logger.info(
                                "Model cache directory exists but no model files found"
                            )
                            logger.info(
                                "Will download from Hugging Face (approximately 1.6 GB)"
                            )
                    else:
                        logger.info(
                            "Model not in cache, will download from Hugging Face"
                        )
                        logger.info("Expected download size: ~1.6 GB (BART-large-CNN)")

                    logger.info("Loading BART summarization pipeline...")

                    self.pipeline = pipeline(
                        "summarization",
                        model=settings.SUMMARIZATION_MODEL_NAME,
                        device=-1,  # CPU
                        model_kwargs={"cache_dir": model_cache_path},
                    )

                    total_time = time.time() - start_time

                    logger.info("=" * 60)
                    logger.info("Summarization Model Loading Summary:")
                    logger.info(f"  Model: {settings.SUMMARIZATION_MODEL_NAME}")
                    logger.info(f"  Device: CPU")
                    logger.info(f"  Cache location: {model_cache_path}")
                    if cache_size_mb > 0:
                        logger.info(f"  Cache size: {cache_size_mb:.1f} MB")
                    logger.info(f"  Total load time: {total_time:.2f}s")
                    logger.info("=" * 60)

            except Exception as e:
                logger.error(f"Failed to load summarization model: {str(e)}")
                logger.error(f"Model name: {settings.SUMMARIZATION_MODEL_NAME}")
                from app.core.exceptions import MLModelLoadException

                raise MLModelLoadException(
                    message=f"Failed to load summarization model: {str(e)}",
                    details={
                        "model": settings.SUMMARIZATION_MODEL_NAME,
                        "use_ollama": self.use_ollama,
                        "error": str(e),
                    },
                ) from e

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{settings.OLLAMA_API_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def summarize_with_ollama(self, text: str, max_length: int = 150) -> str:
        """Summarize using Ollama (LLaMA/Mistral)"""
        try:
            prompt = f"""Summarize the following product reviews in 2-3 concise sentences:

{text}

Summary:"""

            response = requests.post(
                f"{settings.OLLAMA_API_URL}/api/generate",
                json={
                    "model": "llama2",  # or "mistral"
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Ollama summarization failed: {str(e)}")
            raise

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize review text

        Args:
            text: Text to summarize
            max_length: Maximum summary length (in tokens, roughly)
            min_length: Minimum summary length (in tokens)

        Returns:
            Summarized text
        """
        if self.pipeline is None and not self.use_ollama:
            self.load_model()

        try:
            import time

            start_time = time.time()

            # Use Ollama if available
            if self.use_ollama:
                summary = self.summarize_with_ollama(text, max_length)
                inference_time = time.time() - start_time
                logger.debug(
                    f"Summarization (Ollama) completed in {inference_time*1000:.1f}ms"
                )
                return summary

            # Otherwise use BART
            # Clean and prepare text
            text = text.strip()

            # BART works best with 512-1024 tokens
            # Approximate: 1 token ≈ 4 characters
            max_input_chars = 4000  # ~1000 tokens

            preprocess_start = time.time()
            if len(text) > max_input_chars:
                # Take beginning and end of text for better context
                half = max_input_chars // 2
                text = text[:half] + " ... " + text[-half:]
            preprocess_time = time.time() - preprocess_start

            # Generate summary with proper parameters
            model_start = time.time()
            result = self.pipeline(
                text,
                max_length=max_length,  # Max tokens in summary
                min_length=min_length,  # Min tokens in summary
                do_sample=False,  # Deterministic
                truncation=True,  # Truncate long inputs
                clean_up_tokenization_spaces=True,
            )
            model_time = time.time() - model_start

            summary = result[0]["summary_text"].strip()

            # If summary is just extraction, create a better one
            if len(summary) > max_length * 4:  # If summary too long (chars)
                # Simple extractive summary as fallback
                sentences = text.split(".")
                summary = ". ".join(sentences[:3]) + "."

            total_time = time.time() - start_time

            logger.debug(f"Summarization timing:")
            logger.debug(f"  - Preprocess: {preprocess_time*1000:.1f}ms")
            logger.debug(f"  - Model inference: {model_time*1000:.1f}ms")
            logger.debug(f"  - Total: {total_time*1000:.1f}ms")

            return summary

        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            logger.warning("Falling back to extractive summary")
            try:
                # Better fallback: extract key sentences
                return self._extractive_summary(text, max_length)
            except Exception as fallback_error:
                from app.core.exceptions import MLModelInferenceException

                raise MLModelInferenceException(
                    message=f"Summarization failed (including fallback): {str(e)}",
                    details={
                        "model": "Ollama" if self.use_ollama else "BART",
                        "text_length": len(text) if text else 0,
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error),
                    },
                ) from e

    def _extractive_summary(self, text: str, max_length: int = 150) -> str:
        """Create extractive summary by selecting important sentences"""
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

        if not sentences:
            return text[:max_length]

        # Score sentences by importance (simple heuristic)
        scored_sentences = []
        for sent in sentences[:10]:  # First 10 sentences
            score = 0
            # Higher score for sentences with key words
            if any(
                word in sent.lower()
                for word in ["excellent", "great", "amazing", "outstanding"]
            ):
                score += 2
            if any(word in sent.lower() for word in ["however", "but", "although"]):
                score += 1.5  # Contrasting sentences are important
            if any(
                word in sent.lower()
                for word in ["quality", "performance", "battery", "camera"]
            ):
                score += 1

            scored_sentences.append((score, sent))

        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        # Build summary
        summary_parts = []
        current_length = 0

        for score, sent in scored_sentences:
            if current_length + len(sent) < max_length * 4:  # 4 chars ≈ 1 token
                summary_parts.append(sent)
                current_length += len(sent)

        summary = ". ".join(summary_parts[:3]) + "."
        return summary if len(summary) > 50 else text[:max_length] + "..."

    def generate_structured_summary(self, reviews: List[str]) -> Dict:
        """
        Generate structured summary from multiple reviews

        Args:
            reviews: List of review texts

        Returns:
            Structured summary with pros, cons, etc.
        """
        if not reviews:
            return {"summary": "", "pros": [], "cons": [], "key_points": []}

        # Combine reviews
        combined_text = "\n\n".join(reviews[:20])  # Limit to 20 reviews

        # Generate main summary
        main_summary = self.summarize(combined_text, max_length=200)

        # Extract key points (simplified approach)
        # In production, use more sophisticated NLP
        key_points = self._extract_key_points(reviews)

        return {
            "summary": main_summary,
            "pros": self._extract_pros(reviews),
            "cons": self._extract_cons(reviews),
            "key_points": key_points[:7],
        }

    def _extract_key_points(self, reviews: List[str]) -> List[str]:
        """Extract key points from reviews"""
        # Simplified extraction - can be improved with NER/topic modeling
        points = []

        # Look for common patterns
        for review in reviews[:10]:
            sentences = review.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 150:
                    # Simple filtering for relevant sentences
                    if any(
                        word in sentence.lower()
                        for word in [
                            "great",
                            "excellent",
                            "good",
                            "bad",
                            "poor",
                            "issue",
                        ]
                    ):
                        points.append(sentence)

        # Return unique points
        return list(set(points))[:7]

    def _extract_pros(self, reviews: List[str]) -> List[str]:
        """Extract positive points"""
        pros = []
        positive_keywords = [
            "great",
            "excellent",
            "love",
            "perfect",
            "amazing",
            "outstanding",
            "fantastic",
            "wonderful",
            "best",
            "awesome",
            "highly recommend",
            "impressed",
            "quality",
            "fast",
            "easy",
        ]

        for review in reviews:
            sentences = [s.strip() for s in review.split(".") if len(s.strip()) > 15]
            for sent in sentences:
                sent_lower = sent.lower()
                # Check if sentence is positive
                if any(word in sent_lower for word in positive_keywords):
                    # Avoid sentences with negative words
                    if not any(
                        neg in sent_lower
                        for neg in ["but", "however", "disappoint", "bad", "poor"]
                    ):
                        # Clean and add
                        if 20 < len(sent) < 150:  # Good sentence length
                            pros.append(sent)

        # Remove duplicates and return top 5
        unique_pros = list(dict.fromkeys(pros))  # Preserves order
        return unique_pros[:5]

    def _extract_cons(self, reviews: List[str]) -> List[str]:
        """Extract negative points"""
        cons = []
        negative_keywords = [
            "bad",
            "poor",
            "disappointing",
            "issue",
            "problem",
            "terrible",
            "awful",
            "worst",
            "hate",
            "broken",
            "not worth",
            "waste",
            "defective",
            "failed",
            "useless",
        ]

        for review in reviews:
            sentences = [s.strip() for s in review.split(".") if len(s.strip()) > 15]
            for sent in sentences:
                sent_lower = sent.lower()
                # Check if sentence is negative
                if any(word in sent_lower for word in negative_keywords):
                    # Clean and add
                    if 20 < len(sent) < 150:  # Good sentence length
                        cons.append(sent)

        # Remove duplicates and return top 5
        unique_cons = list(dict.fromkeys(cons))  # Preserves order
        return unique_cons[:5]


# Singleton instance
_summarizer_instance = None


def get_summarizer() -> ReviewSummarizer:
    """Get or create summarizer singleton"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ReviewSummarizer()
    return _summarizer_instance
