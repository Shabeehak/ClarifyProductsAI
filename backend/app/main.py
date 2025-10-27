"""
ClarifyProducts.AI - Main FastAPI Application

This is the main entry point for the FastAPI application.
It configures middleware, exception handlers, and routes.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from loguru import logger
import traceback

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.router import api_router
from app.core.exceptions import (
    ClarifyException,
    ProductNotFoundException,
    ScraperException,
    MLModelException,
    CacheException,
    DatabaseException,
    ValidationException,
    ServiceException,
    ExternalAPIException,
)

# Setup logging (must be done early)
setup_logging(
    log_level=getattr(settings, "LOG_LEVEL", "INFO"),
    log_dir="logs",
    app_name="clarify_products",
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Product Discovery & Review Insights Platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Custom exception handlers
@app.exception_handler(ClarifyException)
async def clarify_exception_handler(request: Request, exc: ClarifyException):
    """
    Handle all custom ClarifyException errors.

    This handler catches all application-specific exceptions and returns
    consistent JSON responses with appropriate status codes.

    Args:
        request: The FastAPI request object
        exc: The ClarifyException instance

    Returns:
        JSONResponse with error details
    """
    logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "exception_type": exc.__class__.__name__,
            "url": str(request.url),
            "method": request.method,
            "details": exc.details,
            "status_code": exc.status_code,
        },
    )

    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(ProductNotFoundException)
async def product_not_found_handler(request: Request, exc: ProductNotFoundException):
    """Handle product not found errors with specific messaging."""
    logger.warning(
        f"Product not found: {exc.message}",
        extra={"url": str(request.url), "details": exc.details},
    )

    return JSONResponse(
        status_code=404,
        content={
            "error": "ProductNotFound",
            "message": exc.message,
            "details": exc.details,
            "suggestion": "Try searching with a different product name or check spelling",
        },
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation errors with helpful feedback."""
    logger.warning(
        f"Validation error: {exc.message}",
        extra={"url": str(request.url), "details": exc.details},
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": "ValidationError",
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(MLModelException)
async def ml_model_exception_handler(request: Request, exc: MLModelException):
    """Handle ML model errors with recovery suggestions."""
    logger.error(
        f"ML Model error: {exc.message}",
        extra={"url": str(request.url), "details": exc.details},
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "MLModelError",
            "message": "Machine learning model encountered an error",
            "details": exc.details if settings.DEBUG else {},
            "suggestion": "The service is experiencing technical difficulties. Please try again later.",
        },
    )


@app.exception_handler(ScraperException)
async def scraper_exception_handler(request: Request, exc: ScraperException):
    """Handle scraper errors gracefully."""
    logger.error(
        f"Scraper error: {exc.message}",
        extra={"url": str(request.url), "details": exc.details},
    )

    return JSONResponse(
        status_code=503,
        content={
            "error": "ScraperError",
            "message": "Unable to fetch product data at this time",
            "details": exc.details if settings.DEBUG else {},
            "suggestion": "Our data sources may be temporarily unavailable. Please try again in a few moments.",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected exceptions.

    This ensures that all unhandled exceptions are logged properly
    and return a consistent error response to the client.

    Args:
        request: The FastAPI request object
        exc: The exception instance

    Returns:
        JSONResponse with generic error message
    """
    # Log the full stack trace
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "exception_type": exc.__class__.__name__,
            "url": str(request.url),
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    # Return generic error (don't expose internal details in production)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": (
                "An unexpected error occurred" if not settings.DEBUG else str(exc)
            ),
            "details": (
                {
                    "type": exc.__class__.__name__,
                    "traceback": traceback.format_exc() if settings.DEBUG else None,
                }
                if settings.DEBUG
                else {}
            ),
        },
    )


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing"""
    start_time = time.time()

    # Log request
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log response
    logger.info(
        f"Completed: {request.method} {request.url.path} "
        f"Status: {response.status_code} Time: {process_time:.3f}s"
    )

    return response


# Include API router
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # TODO: Initialize database connection pool
    # TODO: Load ML models into memory
    # TODO: Start background tasks


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info(f"Shutting down {settings.APP_NAME}")

    # TODO: Close database connections
    # TODO: Cleanup resources


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
