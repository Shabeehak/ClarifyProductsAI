"""
Retry utility with exponential backoff for handling transient API failures

Provides decorators for automatic retry logic with configurable backoff strategies.
Useful for external API calls (Gemini, SerpAPI, etc.)
"""

import time
import functools
from loguru import logger
from typing import Callable, Tuple, Type, Optional


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    exp_base: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    http_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """
    Decorator that retries a function with exponential backoff on failure.

    This is particularly useful for external API calls that may fail due to:
    - Rate limiting (HTTP 429)
    - Temporary server errors (HTTP 5xx)
    - Network timeouts
    - Transient connection issues

    Args:
        max_attempts: Maximum number of attempts (including initial call)
        initial_delay: Initial delay in seconds before first retry
        exp_base: Exponential base for calculating delay (delay = initial * exp_base^attempt)
        max_delay: Maximum delay between retries (caps exponential growth)
        exceptions: Tuple of exception types to catch and retry
        http_status_codes: HTTP status codes that should trigger a retry

    Returns:
        Decorated function that implements retry logic

    Example:
        @retry_with_backoff(max_attempts=5, initial_delay=1, exp_base=2)
        def call_external_api():
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()

    Retry Schedule (with default params):
        Attempt 1: Immediate
        Attempt 2: Wait 1s   (1 * 2^0)
        Attempt 3: Wait 2s   (1 * 2^1)
        Attempt 4: Wait 4s   (1 * 2^2)
        ...up to max_delay

    Production Use Cases:
        - Gemini API: Rate limits (15 RPM on free tier)
        - SerpAPI: Rate limits, temporary unavailability
        - HuggingFace: Model download timeouts
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    # Attempt the function call
                    result = func(*args, **kwargs)

                    # Success! Log if this wasn't the first attempt
                    if attempt > 1:
                        logger.info(
                            f"✅ {func.__name__} succeeded on attempt {attempt}/{max_attempts}"
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    # Determine if this error is retriable
                    is_retriable = _is_retriable_error(e, http_status_codes)

                    # Log the error with context
                    _log_retry_attempt(
                        func_name=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=e,
                        is_retriable=is_retriable,
                    )

                    # If not retriable or last attempt, raise the exception
                    if not is_retriable or attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {attempt} attempts. "
                            f"Error: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    # Calculate next delay with exponential backoff
                    current_delay = min(delay, max_delay)
                    logger.info(f"⏳ Retrying in {current_delay:.1f}s...")

                    time.sleep(current_delay)
                    delay *= exp_base

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def _is_retriable_error(error: Exception, http_status_codes: Tuple[int, ...]) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception that was raised
        http_status_codes: HTTP status codes that are retriable

    Returns:
        True if the error should trigger a retry, False otherwise
    """
    # Check if it's an HTTP error with a retriable status code
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code
        if status_code in http_status_codes:
            return True
        # Don't retry client errors (4xx except 429)
        if 400 <= status_code < 500 and status_code != 429:
            return False

    # Check for common retriable exceptions by type name
    error_type = type(error).__name__
    retriable_types = [
        "Timeout",
        "TimeoutError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "HTTPError",  # Generic HTTP error
        "ChunkedEncodingError",
        "ContentDecodingError",
        "TooManyRedirects",
        "RetryError",
    ]

    if any(retriable in error_type for retriable in retriable_types):
        return True

    # Check error message for common transient error patterns
    error_msg = str(error).lower()
    transient_patterns = [
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "connection error",
        "network error",
        "temporary failure",
        "service unavailable",
        "rate limit",
        "too many requests",
        "quota exceeded",
        "overloaded",
        "try again",
    ]

    if any(pattern in error_msg for pattern in transient_patterns):
        return True

    # Default: retry on unknown errors (conservative approach)
    return True


def _log_retry_attempt(
    func_name: str,
    attempt: int,
    max_attempts: int,
    error: Exception,
    is_retriable: bool,
):
    """
    Log retry attempt with appropriate context and formatting.

    Args:
        func_name: Name of the function being retried
        attempt: Current attempt number
        max_attempts: Maximum number of attempts
        error: The exception that triggered the retry
        is_retriable: Whether the error is considered retriable
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Extract HTTP status code if available
    status_code = None
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code

    # Build log message
    if status_code:
        log_msg = (
            f"🔄 {func_name} attempt {attempt}/{max_attempts} failed: "
            f"HTTP {status_code} - {error_type}"
        )
    else:
        log_msg = (
            f"🔄 {func_name} attempt {attempt}/{max_attempts} failed: "
            f"{error_type}: {error_msg[:100]}"  # Truncate long messages
        )

    if is_retriable:
        logger.warning(log_msg)
    else:
        logger.error(f"{log_msg} (not retriable)")


# Convenience decorators with pre-configured settings

def retry_api_call(func: Callable):
    """
    Decorator optimized for external API calls.

    Configuration:
    - 5 retry attempts
    - 1s initial delay
    - 2x exponential backoff
    - Max 30s delay
    - Retries on: 429, 500, 502, 503, 504

    Example:
        @retry_api_call
        def fetch_from_api():
            return requests.get("...").json()
    """
    return retry_with_backoff(
        max_attempts=5,
        initial_delay=1.0,
        exp_base=2.0,
        max_delay=30.0,
        http_status_codes=(429, 500, 502, 503, 504),
    )(func)


def retry_network_call(func: Callable):
    """
    Decorator optimized for network operations with longer delays.

    Configuration:
    - 3 retry attempts
    - 2s initial delay
    - 3x exponential backoff
    - Max 60s delay

    Example:
        @retry_network_call
        def download_large_file():
            return requests.get("...", stream=True)
    """
    return retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exp_base=3.0,
        max_delay=60.0,
    )(func)
