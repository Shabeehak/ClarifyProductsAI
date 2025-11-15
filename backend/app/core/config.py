"""
Application Configuration
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
from pydantic import field_validator, Field, ValidationInfo
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "ClarifyProducts.AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    TIMEZONE: str = "UTC"

    # Redis - IMPROVED
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_TIMEOUT: int = 5
    REDIS_DB: int = 0

    # Cache
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_MAX_SIZE: int = 1000  # Max items in memory cache

    # Security (Not used in current stateless architecture - can be added later if authentication is needed)
    # SECRET_KEY and JWT settings removed as app doesn't use authentication yet

    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string
            return [origin.strip() for origin in v.split(",")]
        return v

    # API Keys - External Services
    YOUTUBE_API_KEY: str = Field(default="", env="YOUTUBE_API_KEY")
    BESTBUY_API_KEY: str = Field(default="", env="BESTBUY_API_KEY")
    SERPAPI_KEY: str = Field(default="", env="SERPAPI_KEY")  # Google Shopping API

    # LLM API Keys
    GEMINI_API_KEY: str = Field(
        default="", env="GEMINI_API_KEY"
    )  # Google Gemini (Primary LLM)

    # Open-Source ML Models
    USE_LOCAL_MODELS: bool = True
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    SUMMARIZATION_MODEL_NAME: str = "facebook/bart-large-cnn"

    # Model loading configuration
    PRELOAD_MODELS_ON_STARTUP: bool = True
    MODEL_CACHE_DIR: str = "./models"
    MODEL_DEVICE: str = "auto"  # auto, cpu, cuda

    # Reddit API
    REDDIT_CLIENT_ID: str = Field(default="", env="REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str = Field(
        default="ClarifyProducts.AI/1.0", env="REDDIT_USER_AGENT"
    )

    # Twitter/X API
    TWITTER_BEARER_TOKEN: str = Field(default="", env="TWITTER_BEARER_TOKEN")
    TWITTER_API_KEY: str = Field(default="", env="TWITTER_API_KEY")
    TWITTER_API_SECRET: str = Field(default="", env="TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN: str = Field(default="", env="TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET: str = Field(default="", env="TWITTER_ACCESS_SECRET")

    # ML Model Paths
    CLIP_MODEL_PATH: str = Field(default="./models/clip", env="CLIP_MODEL_PATH")
    SENTIMENT_MODEL_PATH: str = Field(
        default="./models/sentiment", env="SENTIMENT_MODEL_PATH"
    )
    RECOMMENDATION_MODEL_PATH: str = Field(
        default="./models/recommendation", env="RECOMMENDATION_MODEL_PATH"
    )

    @field_validator(
        "MODEL_CACHE_DIR",
        "CLIP_MODEL_PATH",
        "SENTIMENT_MODEL_PATH",
        "RECOMMENDATION_MODEL_PATH",
    )
    @classmethod
    def validate_model_paths(cls, v: str, info: ValidationInfo) -> str:
        """Validate and create model directories if they don't exist"""
        from pathlib import Path

        # Convert to Path object
        path = Path(v)

        # Make path absolute if relative
        if not path.is_absolute():
            # Relative to backend/ directory
            backend_dir = Path(__file__).parent.parent.parent
            path = backend_dir / path

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)

            # Check if writable
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()

        except PermissionError:
            raise ValueError(
                f"Model path '{v}' is not writable. "
                f"Please check permissions for: {path.absolute()}"
            )
        except Exception as e:
            raise ValueError(f"Failed to create/access model path '{v}': {str(e)}")

        return v

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Feature Flags
    ENABLE_SCRAPING: bool = True
    ENABLE_ML_MODELS: bool = True
    ENABLE_RAG: bool = True
    ENABLE_CACHING: bool = True

    # Rate Limiting - IMPROVED
    RATE_LIMIT_DEFAULT: int = 60
    RATE_LIMIT_ML_INFERENCE: int = 10
    RATE_LIMIT_SCRAPING: int = 5
    RATE_LIMIT_SEARCH: int = 30

    # Scraping Settings
    SCRAPING_DELAY_SECONDS: int = 2
    MAX_REVIEWS_PER_SOURCE: int = 100
    CACHE_TTL_HOURS: int = 24

    class Config:
        # Look for .env file in backend/ directory (parent of app/)
        env_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
        )
        case_sensitive = True
        validate_assignment = True

        @staticmethod
        def parse_env_var(field_name: str, raw_val: str):
            """Custom parser for complex types"""
            if field_name == "CORS_ORIGINS":
                if raw_val.startswith("["):
                    import json

                    return json.loads(raw_val)
                return [x.strip() for x in raw_val.split(",")]
            return raw_val


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()


# Startup validation
def validate_settings():
    """Validate critical settings on startup"""
    errors = []

    # Check Redis connection
    try:
        import redis

        r = redis.from_url(settings.REDIS_URL)
        r.ping()
    except Exception as e:
        errors.append(f"Redis connection failed: {e}")

    # Check model directories exist
    if not os.path.exists(settings.MODEL_CACHE_DIR):
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    if errors:
        raise RuntimeError(f"Configuration validation failed:\n" + "\n".join(errors))
