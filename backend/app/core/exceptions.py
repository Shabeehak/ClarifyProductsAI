"""
Custom Exception Classes for ClarifyProducts.AI

This module defines domain-specific exceptions for better error handling
and debugging throughout the application.

All exceptions inherit from ClarifyException base class for easy catching
and handling of application-specific errors.
"""

from typing import Optional, Any, Dict


class ClarifyException(Exception):
    """
    Base exception class for ClarifyProducts.AI application.

    All custom exceptions should inherit from this class to allow
    for easy catching and handling of application-specific errors.

    Attributes:
        message (str): Human-readable error message
        details (Dict[str, Any]): Additional error context
        status_code (int): HTTP status code to return
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        """
        Initialize ClarifyException.

        Args:
            message: Human-readable error message
            details: Additional error context (optional)
            status_code: HTTP status code (default: 500)
        """
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API response.

        Returns:
            Dictionary containing error information
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "status_code": self.status_code,
        }


# Product-related exceptions
class ProductNotFoundException(ClarifyException):
    """
    Raised when a product cannot be found in any source.

    Example:
        >>> raise ProductNotFoundException(
        ...     message="Product not found: iPhone 99",
        ...     details={"query": "iPhone 99", "sources_checked": ["amazon", "youtube"]}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=404)


class ProductDataInvalidException(ClarifyException):
    """
    Raised when product data is invalid or malformed.

    Example:
        >>> raise ProductDataInvalidException(
        ...     message="Missing required field: rating",
        ...     details={"product": "iPhone 15", "missing_fields": ["rating"]}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=422)


# Scraping-related exceptions
class ScraperException(ClarifyException):
    """
    Base exception for web scraping errors.

    Example:
        >>> raise ScraperException(
        ...     message="Failed to scrape Amazon",
        ...     details={"source": "amazon", "error": "Connection timeout"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=503)


class ScraperTimeoutException(ScraperException):
    """
    Raised when scraper times out.

    Example:
        >>> raise ScraperTimeoutException(
        ...     message="Scraper timed out after 30s",
        ...     details={"source": "amazon", "timeout": 30}
        ... )
    """

    pass


class ScraperRateLimitException(ScraperException):
    """
    Raised when scraper hits rate limit.

    Example:
        >>> raise ScraperRateLimitException(
        ...     message="Rate limit exceeded for Amazon",
        ...     details={"source": "amazon", "retry_after": 60}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=429)


class ScraperBlockedException(ScraperException):
    """
    Raised when scraper is blocked (403, captcha, etc.).

    Example:
        >>> raise ScraperBlockedException(
        ...     message="Blocked by Amazon (captcha required)",
        ...     details={"source": "amazon", "reason": "captcha"}
        ... )
    """

    pass


# ML Model exceptions
class MLModelException(ClarifyException):
    """
    Base exception for ML model errors.

    Example:
        >>> raise MLModelException(
        ...     message="Failed to load CLIP model",
        ...     details={"model": "CLIP", "error": "Out of memory"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=500)


class MLModelLoadException(MLModelException):
    """
    Raised when ML model fails to load.

    Example:
        >>> raise MLModelLoadException(
        ...     message="Failed to load sentiment model",
        ...     details={"model": "DistilBERT", "path": "/models/sentiment"}
        ... )
    """

    pass


class MLModelInferenceException(MLModelException):
    """
    Raised when ML model inference fails.

    Example:
        >>> raise MLModelInferenceException(
        ...     message="Inference failed for image",
        ...     details={"model": "CLIP", "input_shape": [3, 224, 224]}
        ... )
    """

    pass


class MLModelNotLoadedException(MLModelException):
    """
    Raised when trying to use a model that hasn't been loaded.

    Example:
        >>> raise MLModelNotLoadedException(
        ...     message="CLIP model not loaded. Call load_model() first.",
        ...     details={"model": "CLIP"}
        ... )
    """

    pass


# Cache-related exceptions
class CacheException(ClarifyException):
    """
    Base exception for cache operations.

    Example:
        >>> raise CacheException(
        ...     message="Cache operation failed",
        ...     details={"operation": "get", "key": "product_123"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=500)


class CacheConnectionException(CacheException):
    """
    Raised when unable to connect to cache (Redis, etc.).

    Example:
        >>> raise CacheConnectionException(
        ...     message="Failed to connect to Redis",
        ...     details={"host": "localhost", "port": 6379}
        ... )
    """

    pass


class CacheInvalidationException(CacheException):
    """
    Raised when cache invalidation fails.

    Example:
        >>> raise CacheInvalidationException(
        ...     message="Failed to invalidate cache",
        ...     details={"key": "product_123", "pattern": "product_*"}
        ... )
    """

    pass


# Database exceptions
class DatabaseException(ClarifyException):
    """
    Base exception for database operations.

    Example:
        >>> raise DatabaseException(
        ...     message="Database query failed",
        ...     details={"query": "SELECT * FROM products", "error": "Connection lost"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=500)


class DatabaseConnectionException(DatabaseException):
    """
    Raised when unable to connect to database.

    Example:
        >>> raise DatabaseConnectionException(
        ...     message="Failed to connect to PostgreSQL",
        ...     details={"host": "localhost", "database": "clarify_db"}
        ... )
    """

    pass


class DatabaseQueryException(DatabaseException):
    """
    Raised when database query fails.

    Example:
        >>> raise DatabaseQueryException(
        ...     message="Query execution failed",
        ...     details={"query": "INSERT INTO products...", "error": "Duplicate key"}
        ... )
    """

    pass


# Input validation exceptions
class ValidationException(ClarifyException):
    """
    Raised when input validation fails.

    Example:
        >>> raise ValidationException(
        ...     message="Invalid search query",
        ...     details={"query": "", "reason": "Query cannot be empty"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=400)


class InvalidImageException(ValidationException):
    """
    Raised when image is invalid or cannot be processed.

    Example:
        >>> raise InvalidImageException(
        ...     message="Invalid image format",
        ...     details={"format": "PDF", "supported": ["JPEG", "PNG", "WEBP"]}
        ... )
    """

    pass


class InvalidQueryException(ValidationException):
    """
    Raised when search query is invalid.

    Example:
        >>> raise InvalidQueryException(
        ...     message="Query too short",
        ...     details={"query": "a", "min_length": 2}
        ... )
    """

    pass


# Service exceptions
class ServiceException(ClarifyException):
    """
    Base exception for service layer errors.

    Example:
        >>> raise ServiceException(
        ...     message="Product service unavailable",
        ...     details={"service": "SmartProductService", "reason": "Initialization failed"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=503)


class ServiceUnavailableException(ServiceException):
    """
    Raised when a service is temporarily unavailable.

    Example:
        >>> raise ServiceUnavailableException(
        ...     message="Scraper service is down for maintenance",
        ...     details={"service": "RealtimeScraperService", "retry_after": 300}
        ... )
    """

    pass


class ServiceConfigurationException(ServiceException):
    """
    Raised when service is misconfigured.

    Example:
        >>> raise ServiceConfigurationException(
        ...     message="Missing API key for OpenAI",
        ...     details={"service": "RAGService", "missing_config": ["OPENAI_API_KEY"]}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=500)


# External API exceptions
class ExternalAPIException(ClarifyException):
    """
    Base exception for external API errors.

    Example:
        >>> raise ExternalAPIException(
        ...     message="YouTube API request failed",
        ...     details={"api": "YouTube Data API", "status": 403}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=502)


class ExternalAPITimeoutException(ExternalAPIException):
    """
    Raised when external API times out.

    Example:
        >>> raise ExternalAPITimeoutException(
        ...     message="Amazon API timed out",
        ...     details={"api": "Amazon Product API", "timeout": 30}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=504)


class ExternalAPIRateLimitException(ExternalAPIException):
    """
    Raised when external API rate limit is hit.

    Example:
        >>> raise ExternalAPIRateLimitException(
        ...     message="YouTube API rate limit exceeded",
        ...     details={"api": "YouTube", "limit": 10000, "retry_after": 3600}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=429)


# Authentication & Authorization exceptions
class AuthenticationException(ClarifyException):
    """
    Base exception for authentication errors.

    Raised when user credentials are missing, invalid, or authentication fails.

    Example:
        >>> raise AuthenticationException(
        ...     message="Invalid credentials",
        ...     details={"username": "user@example.com", "reason": "Password incorrect"}
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=401)


class AuthorizationException(ClarifyException):
    """
    Raised when user is authenticated but lacks permissions for the requested resource.

    Example:
        >>> raise AuthorizationException(
        ...     message="Insufficient permissions",
        ...     details={
        ...         "user_id": "123",
        ...         "required_role": "admin",
        ...         "user_role": "user"
        ...     }
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=403)


class TokenExpiredException(AuthenticationException):
    """
    Raised when JWT token has expired.

    Example:
        >>> raise TokenExpiredException(
        ...     message="Access token expired",
        ...     details={
        ...         "expired_at": "2025-10-19T10:00:00Z",
        ...         "current_time": "2025-10-19T11:00:00Z"
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str = "Token has expired",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)


class InvalidTokenException(AuthenticationException):
    """
    Raised when JWT token is invalid, malformed, or cannot be verified.

    Example:
        >>> raise InvalidTokenException(
        ...     message="Invalid JWT signature",
        ...     details={"token": "eyJ...", "reason": "Signature verification failed"}
        ... )
    """

    def __init__(
        self,
        message: str = "Invalid or malformed token",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)


class PaymentRequiredException(ClarifyException):
    """
    Raised when requested resource requires payment (HTTP 402).

    Useful for premium features, API rate limits exceeded, or subscription required.

    Example:
        >>> raise PaymentRequiredException(
        ...     message="Premium feature - subscription required",
        ...     details={
        ...         "feature": "Advanced ML Analysis",
        ...         "pricing_url": "/pricing",
        ...         "user_plan": "free"
        ...     }
        ... )
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, status_code=402)
