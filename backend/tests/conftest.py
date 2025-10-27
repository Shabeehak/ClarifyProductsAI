"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
It includes fixtures for test clients, sample data, and mock objects.
"""
import pytest
import sys
from pathlib import Path
from typing import Dict, List
from fastapi.testclient import TestClient
from PIL import Image
import io

# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.main import app


# =============================================================================
# Client Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def client():
    """
    FastAPI test client fixture (session-scoped).

    This fixture provides a TestClient instance for testing API endpoints.
    Session-scoped means it's created once per test session.

    Returns:
        TestClient: FastAPI test client

    Example:
        >>> def test_health_check(client):
        ...     response = client.get("/health")
        ...     assert response.status_code == 200
    """
    return TestClient(app)


@pytest.fixture
def async_client():
    """
    Async test client fixture for async endpoints.

    Returns:
        TestClient: FastAPI async test client
    """
    from httpx import AsyncClient
    return AsyncClient(app=app, base_url="http://test")


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_product() -> Dict:
    """
    Sample product data fixture.

    Returns:
        Dict: Sample product information

    Example:
        >>> def test_product_search(sample_product):
        ...     assert sample_product["name"] == "iPhone 15 Pro"
    """
    return {
        "name": "iPhone 15 Pro",
        "brand": "Apple",
        "category": "smartphone",
        "rating": 4.5,
        "review_count": 1234,
        "price": 999.99,
        "image_url": "https://example.com/iphone15.jpg"
    }


@pytest.fixture
def sample_reviews() -> List[Dict]:
    """
    Sample reviews fixture with mixed sentiments.

    Returns:
        List[Dict]: List of sample reviews

    Example:
        >>> def test_sentiment_analysis(sample_reviews):
        ...     assert len(sample_reviews) == 5
    """
    return [
        {
            "text": "Great phone! Excellent camera and fast performance.",
            "rating": 5,
            "sentiment": "positive"
        },
        {
            "text": "Good phone but very expensive for what you get.",
            "rating": 4,
            "sentiment": "neutral"
        },
        {
            "text": "Battery life could be better. Disappointing.",
            "rating": 3,
            "sentiment": "negative"
        },
        {
            "text": "Love the display and build quality. Highly recommend!",
            "rating": 5,
            "sentiment": "positive"
        },
        {
            "text": "Overpriced and overheating issues. Not worth it.",
            "rating": 2,
            "sentiment": "negative"
        }
    ]


@pytest.fixture
def sample_positive_reviews() -> List[str]:
    """
    Sample positive reviews for sentiment testing.

    Returns:
        List[str]: List of positive review texts
    """
    return [
        "This product is excellent! Highly recommended.",
        "Amazing quality and great value for money.",
        "Best purchase I've made this year!",
        "Outstanding features and performance.",
        "Absolutely love it! Exceeded expectations."
    ]


@pytest.fixture
def sample_negative_reviews() -> List[str]:
    """
    Sample negative reviews for sentiment testing.

    Returns:
        List[str]: List of negative review texts
    """
    return [
        "Terrible product. Broke after one week.",
        "Worst purchase ever. Complete waste of money.",
        "Poor quality and bad customer service.",
        "Don't buy this! It's a scam.",
        "Disappointed and frustrated. Not as advertised."
    ]


@pytest.fixture
def sample_image() -> Image.Image:
    """
    Generate a sample test image.

    Returns:
        Image.Image: PIL Image object for testing

    Example:
        >>> def test_image_recognition(sample_image):
        ...     assert sample_image.size == (224, 224)
    """
    # Create a simple RGB image
    image = Image.new('RGB', (224, 224), color='blue')
    return image


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """
    Convert sample image to bytes for API testing.

    Args:
        sample_image: Sample PIL Image

    Returns:
        bytes: Image as bytes

    Example:
        >>> def test_image_upload(client, sample_image_bytes):
        ...     files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        ...     response = client.post("/api/v1/recognize", files=files)
    """
    img_byte_arr = io.BytesIO()
    sample_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr.read()


# =============================================================================
# Search Query Fixtures
# =============================================================================

@pytest.fixture
def valid_search_queries() -> List[str]:
    """
    Valid search queries for testing.

    Returns:
        List[str]: List of valid search queries
    """
    return [
        "iPhone 15 Pro",
        "Samsung Galaxy S24",
        "MacBook Pro",
        "AirPods Pro",
        "Sony WH-1000XM5"
    ]


@pytest.fixture
def typo_search_queries() -> List[Dict[str, str]]:
    """
    Search queries with typos and their expected corrections.

    Returns:
        List[Dict]: List of query/correction pairs

    Example:
        >>> def test_typo_correction(typo_search_queries):
        ...     for item in typo_search_queries:
        ...         assert "query" in item and "expected" in item
    """
    return [
        {"query": "iphne 15 pro", "expected": "iphone 15 pro"},
        {"query": "samsng galaxy", "expected": "samsung galaxy"},
        {"query": "macbok pro", "expected": "macbook pro"},
        {"query": "airpds pro", "expected": "airpods pro"},
        {"query": "sny headphones", "expected": "sony headphones"}
    ]


@pytest.fixture
def invalid_search_queries() -> List[str]:
    """
    Invalid search queries for testing error handling.

    Returns:
        List[str]: List of invalid queries
    """
    return [
        "",  # Empty query
        " ",  # Whitespace only
        "a",  # Too short
        "x" * 200,  # Too long
        "!@#$%",  # Special characters only
    ]


# =============================================================================
# ML Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sentiment_analyzer():
    """
    Sentiment analyzer fixture (session-scoped).

    Returns:
        SentimentAnalyzer: Loaded sentiment analyzer

    Example:
        >>> def test_sentiment(sentiment_analyzer):
        ...     result = sentiment_analyzer.analyze("Great product!")
        ...     assert result["sentiment_label"] == "positive"
    """
    from app.ml_models.sentiment_analyzer import get_sentiment_analyzer
    analyzer = get_sentiment_analyzer()
    return analyzer


@pytest.fixture(scope="session")
def summarizer():
    """
    Summarizer fixture (session-scoped).

    Returns:
        Summarizer: Loaded summarizer model
    """
    from app.ml_models.summarizer import get_summarizer
    return get_summarizer()


@pytest.fixture(scope="session")
def clip_recognizer():
    """
    CLIP recognizer fixture (session-scoped).

    Returns:
        CLIPRecognizer: Loaded CLIP model
    """
    from app.ml_models.clip_recognizer import get_clip_recognizer
    return get_clip_recognizer()


# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
def product_normalization_service():
    """
    Product normalization service fixture.

    Returns:
        ProductNormalizationService: Service instance
    """
    from app.services.product_normalization_service import ProductNormalizationService
    return ProductNormalizationService()


@pytest.fixture
def smart_product_service():
    """
    Smart product service fixture.

    Returns:
        SmartProductService: Service instance
    """
    from app.services.smart_product_service import SmartProductService
    return SmartProductService()


@pytest.fixture
def scraper_service():
    """
    Scraper service fixture.

    Returns:
        RealtimeScraperService: Service instance
    """
    from app.services.realtime_scraper_service import RealtimeScraperService
    return RealtimeScraperService()


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def mock_scraper_response() -> Dict:
    """
    Mock scraper response for testing without actual scraping.

    Returns:
        Dict: Mock scraper response
    """
    return {
        "found": True,
        "name": "iPhone 15 Pro",
        "rating": 4.5,
        "review_count": 1234,
        "sources": {
            "amazon": {
                "rating": 4.6,
                "review_count": 500,
                "reviews": [
                    {"text": "Great phone!", "rating": 5},
                    {"text": "Good but pricey", "rating": 4}
                ]
            },
            "youtube": {
                "rating": 4.4,
                "review_count": 300,
                "reviews": [
                    {"text": "Excellent camera", "rating": 5}
                ]
            }
        },
        "reviews": [
            {"text": "Great phone!", "rating": 5},
            {"text": "Good but pricey", "rating": 4},
            {"text": "Excellent camera", "rating": 5}
        ],
        "scraped_at": "2025-10-13T12:00:00"
    }


@pytest.fixture
def mock_ml_analysis() -> Dict:
    """
    Mock ML analysis response.

    Returns:
        Dict: Mock ML analysis results
    """
    return {
        "sentiment_breakdown": {
            "positive": 15,
            "negative": 3,
            "neutral": 2,
            "positive_percent": 75.0,
            "negative_percent": 15.0,
            "neutral_percent": 10.0
        },
        "summary": "Overall, customers are very satisfied with the product's camera and performance.",
        "pros": [
            "Excellent camera quality",
            "Fast performance",
            "Great display",
            "Premium build quality"
        ],
        "cons": [
            "Expensive",
            "Battery life average",
            "Gets warm during heavy use"
        ],
        "key_points": [
            "Camera is the standout feature",
            "Performance is excellent for demanding tasks",
            "Price is a common concern"
        ],
        "analyzed_review_count": 20
    }


# =============================================================================
# Database Fixtures (for future use)
# =============================================================================

@pytest.fixture
def db_session():
    """
    Database session fixture (to be implemented).

    This will provide a test database session with rollback capability.
    """
    # TODO: Implement when database is connected
    pass


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook.

    Registers custom markers for test organization.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for API endpoints"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )
    config.addinivalue_line(
        "markers", "ml: Tests involving ML models"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "service: Service layer tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add default markers.

    This automatically marks tests based on their location.
    """
    for item in items:
        # Auto-mark tests based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark ML tests
        if "ml_models" in str(item.fspath) or "test_ml" in item.name:
            item.add_marker(pytest.mark.ml)
            item.add_marker(pytest.mark.slow)

        # Auto-mark API tests
        if "api" in str(item.fspath) or "endpoint" in item.name:
            item.add_marker(pytest.mark.api)
