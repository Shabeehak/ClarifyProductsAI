"""
LLM Service
Support LLM provider with automatic fallback
"""

from typing import Optional, List, Dict
from enum import Enum
from loguru import logger
import requests
import os


class LLMProvider(str, Enum):
    """Supported LLM provider"""

    GEMINI = "gemini"


class LLMService:
    """
    Interface for LLM provider

    Supports:
    - Google Gemini (cloud, free tier, fast)
    """

    def __init__(
        self,
        primary_provider: LLMProvider = LLMProvider.GEMINI,
        fallback_providers: Optional[List[LLMProvider]] = None,
    ):
        """
        Initialize LLM service with provider priority

        Args:
            primary_provider: First choice LLM provider
            fallback_providers: Ordered list of fallback providers
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []

        # API keys from settings (loads from .env)
        from app.core.config import settings

        self.gemini_api_key = settings.GEMINI_API_KEY

        logger.info(f"LLM Service initialized with primary: {primary_provider}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text using configured LLM providers

        Tries primary provider first, falls back to alternatives on failure

        Args:
            prompt: User prompt/query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Optional system instruction

        Returns:
            Generated text response
        """
        # Try primary provider
        try:
            response = self._generate_with_provider(
                provider=self.primary_provider,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            if response:
                logger.info(f"Generated with {self.primary_provider}")
                return response
        except Exception as e:
            logger.warning(f"Primary provider {self.primary_provider} failed: {e}")

        # Try fallback providers
        for fallback in self.fallback_providers:
            try:
                logger.info(f"Trying fallback: {fallback}")
                response = self._generate_with_provider(
                    provider=fallback,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
                if response:
                    logger.info(f"Generated with fallback {fallback}")
                    return response
            except Exception as e:
                logger.warning(f"Fallback {fallback} failed: {e}")
                continue

        # All providers failed
        logger.error("All LLM providers failed")
        raise Exception("All LLM providers unavailable")

    def _generate_with_provider(
        self,
        provider: LLMProvider,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate with specific provider"""

        if provider == LLMProvider.GEMINI:
            return self._generate_gemini(prompt, max_tokens, temperature, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # =========================================================================
    # Provider-Specific Implementations
    # =========================================================================

    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using Google Gemini API"""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        try:
            import google.generativeai as genai

            # Configure Gemini
            genai.configure(api_key=self.gemini_api_key)

            # Use Gemini 2.0 Flash (latest, free tier, fast)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")

            # Combine system prompt with user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            return response.text.strip()

        except ImportError:
            raise Exception(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        except Exception as e:
            raise Exception(f"Gemini generation failed: {str(e)}")


# Singleton instance
_llm_service_instance = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service_instance
    if _llm_service_instance is None:
        # Default: Gemini primary (FREE, fast), Ollama fallback (local dev)
        _llm_service_instance = LLMService(
            primary_provider=LLMProvider.GEMINI, fallback_providers=[]
        )
    return _llm_service_instance
