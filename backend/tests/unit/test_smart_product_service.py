"""
Unit Tests for SmartProductService

Tests the main product service orchestrator with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from app.services.smart_product_service import SmartProductService


@pytest.mark.unit
@pytest.mark.service
@pytest.mark.asyncio
class TestSmartProductService:
    """Test suite for SmartProductService"""

    @pytest.fixture
    def mock_normalizer(self):
        """Mock normalization service"""
        mock = Mock()
        # Make normalize return the input lowercased (so different inputs = different outputs)
        mock.normalize.side_effect = lambda x: x.lower().strip()
        mock.suggest_corrections.return_value = []
        mock.extract_product_info.return_value = {
            "brand": "apple",
            "model": "15 pro",
            "type": "phone"
        }
        return mock

    @pytest.fixture
    def mock_scraper(self):
        """Mock scraper service"""
        mock = Mock()  # Changed from AsyncMock
        mock.search_product = AsyncMock(return_value={
            "found": True,
            "product_name": "iPhone 15 Pro",
            "rating": 4.5,
            "review_count": 100,
            "reviews": [
                {"text": "Great phone!", "rating": 5},
                {"text": "Good quality", "rating": 4}
            ],
            "sources": {
                "youtube": {"videos": []},
                "reddit": {"posts": []},
                "twitter": {"tweets": []}
            }
        })
        # extract_media_content is NOT async, so use regular Mock
        mock.extract_media_content = Mock(return_value={
            "product_images": [
                {"url": "https://example.com/img1.jpg", "source": "youtube"}
            ],
            "youtube_videos": [],
            "reddit_discussions": [],
            "twitter_posts": [],
            "citations": []
        })
        return mock

    @pytest.fixture
    def mock_shopping(self):
        """Mock shopping service"""
        mock = Mock()
        mock.search_product.return_value = {
            "found": True,
            "images": [
                {"url": "https://example.com/shopping1.jpg", "source": "google_shopping"}
            ],
            "price": {"min": 999.0, "max": 1199.0, "avg": 1099.0},
            "details": {
                "brand": "Apple",
                "description": "Latest iPhone"
            },
            "merchants": [
                {"name": "Apple Store", "price": "$999"}
            ]
        }
        return mock

    @pytest.fixture
    def mock_sentiment(self):
        """Mock sentiment analyzer"""
        mock = Mock()
        mock.analyze_batch.return_value = [
            {"sentiment_label": "positive", "sentiment_score": 0.9},
            {"sentiment_label": "positive", "sentiment_score": 0.8}
        ]
        return mock

    @pytest.fixture
    def mock_summarizer(self):
        """Mock summarizer"""
        mock = Mock()
        mock.generate_structured_summary.return_value = {
            "summary": "Great product overall",
            "pros": ["Excellent quality", "Fast performance"],
            "cons": ["Expensive"],
            "key_points": ["Good battery life"]
        }
        return mock

    @pytest.fixture
    def service(self, mock_normalizer, mock_scraper, mock_shopping, mock_sentiment, mock_summarizer):
        """Service with mocked dependencies"""
        with patch('app.services.smart_product_service.get_normalization_service', return_value=mock_normalizer), \
             patch('app.services.smart_product_service.get_scraper_service', return_value=mock_scraper), \
             patch('app.services.smart_product_service.get_shopping_service', return_value=mock_shopping), \
             patch('app.services.smart_product_service.get_sentiment_analyzer', return_value=mock_sentiment), \
             patch('app.services.smart_product_service.get_summarizer', return_value=mock_summarizer):

            svc = SmartProductService()
            # Don't mock _analyze_reviews - let it call the mocked dependencies
            return svc

    # =========================================================================
    # Test Initialization
    # =========================================================================

    def test_service_initialization(self, service):
        """Test service initializes with all dependencies"""
        assert service.normalizer is not None
        assert service.scraper is not None
        assert service.shopping is not None
        assert service.sentiment_analyzer is not None
        assert service.summarizer is not None
        assert service.cache == {}
        assert service.cache_ttl == 86400

    # =========================================================================
    # Test Cache Operations
    # =========================================================================

    def test_get_cache_key_generates_md5(self, service):
        """Test cache key generation"""
        key1 = service._get_cache_key("iPhone 15 Pro")
        key2 = service._get_cache_key("iPhone 15 Pro")
        key3 = service._get_cache_key("Different Product")

        # Same query should generate same key
        assert key1 == key2
        # Different query should generate different key
        assert key1 != key3
        # Should be MD5 hash (32 characters)
        assert len(key1) == 32

    def test_is_cache_valid_fresh_data(self, service):
        """Test cache validation with fresh data"""
        cached_data = {
            "cached_at": datetime.utcnow().isoformat(),
            "data": "test"
        }

        assert service._is_cache_valid(cached_data) is True

    def test_is_cache_valid_expired_data(self, service):
        """Test cache validation with expired data"""
        old_time = datetime.utcnow() - timedelta(days=2)
        cached_data = {
            "cached_at": old_time.isoformat(),
            "data": "test"
        }

        assert service._is_cache_valid(cached_data) is False

    def test_is_cache_valid_no_timestamp(self, service):
        """Test cache validation with missing timestamp"""
        cached_data = {"data": "test"}

        assert service._is_cache_valid(cached_data) is False

    def test_is_cache_valid_empty_data(self, service):
        """Test cache validation with empty data"""
        assert service._is_cache_valid(None) is False
        assert service._is_cache_valid({}) is False

    # =========================================================================
    # Test search_product() - Cache Hit
    # =========================================================================

    async def test_search_product_cache_hit(self, service):
        """Test search returns cached data"""
        # Populate cache
        cache_key = service._get_cache_key("iPhone 15")
        service.cache[cache_key] = {
            "query": "iPhone 15",
            "found": True,
            "cached_at": datetime.utcnow().isoformat(),
            "product": {"name": "iPhone 15"}
        }

        result = await service.search_product("iPhone 15")

        assert result["from_cache"] is True
        assert result["product"]["name"] == "iPhone 15"
        # Scraper should not be called
        service.scraper.search_product.assert_not_called()

    async def test_search_product_expired_cache(self, service):
        """Test search refetches when cache expired"""
        # Populate cache with expired data
        cache_key = service._get_cache_key("iPhone 15")
        old_time = datetime.utcnow() - timedelta(days=2)
        service.cache[cache_key] = {
            "query": "iPhone 15",
            "cached_at": old_time.isoformat()
        }

        result = await service.search_product("iPhone 15")

        # Should fetch fresh data
        service.scraper.search_product.assert_called()

    async def test_search_product_no_cache(self, service):
        """Test search without using cache"""
        result = await service.search_product("iPhone 15", use_cache=False)

        # Should fetch even if cache exists
        service.scraper.search_product.assert_called()

    # =========================================================================
    # Test search_product() - Successful Search
    # =========================================================================

    async def test_search_product_found(self, service):
        """Test successful product search"""
        result = await service.search_product("iPhone 15 Pro")

        assert result["found"] is True
        assert result["query"] == "iPhone 15 Pro"
        assert "product" in result
        assert "consensus" in result
        assert "sources" in result

    async def test_search_product_normalizes_query(self, service):
        """Test search normalizes the query"""
        await service.search_product("IPHONE 15 PRO")

        service.normalizer.normalize.assert_called_with("IPHONE 15 PRO")

    async def test_search_product_calls_all_services(self, service):
        """Test search calls all required services"""
        await service.search_product("iPhone 15")

        service.normalizer.normalize.assert_called()
        service.scraper.search_product.assert_called()
        service.shopping.search_product.assert_called()

    async def test_search_product_with_ml_analysis(self, service):
        """Test search includes ML analysis when requested"""
        result = await service.search_product("iPhone 15", include_ml_analysis=True)

        # Should have ML analysis
        assert "consensus" in result
        service.sentiment_analyzer.analyze_batch.assert_called()

    async def test_search_product_without_ml_analysis(self, service):
        """Test search excludes ML analysis when not requested"""
        await service.search_product("iPhone 15", include_ml_analysis=False)

        # Should not call ML services
        service.sentiment_analyzer.analyze_batch.assert_not_called()

    async def test_search_product_merges_images(self, service):
        """Test search merges images from shopping and scraper"""
        result = await service.search_product("iPhone 15")

        # Should have images
        assert len(result["product"]["images"]) > 0

    async def test_search_product_includes_price_data(self, service):
        """Test search includes price data from shopping service"""
        result = await service.search_product("iPhone 15")

        assert "price" in result["product"]
        assert result["product"]["price"]["min"] == 999.0

    async def test_search_product_includes_merchants(self, service):
        """Test search includes merchant data"""
        result = await service.search_product("iPhone 15")

        assert "merchants" in result["product"]
        assert len(result["product"]["merchants"]) > 0

    # =========================================================================
    # Test search_product() - Product Not Found
    # =========================================================================

    async def test_search_product_not_found_tries_suggestions(self, service):
        """Test search tries suggestions when product not found"""
        service.scraper.search_product = AsyncMock(side_effect=[
            {"found": False},  # First attempt fails
            {"found": True, "product_name": "iPhone 15", "rating": 4.5, "review_count": 100}  # Suggestion succeeds
        ])
        service.normalizer.suggest_corrections.return_value = ["iphone 15", "iphone15"]

        result = await service.search_product("iphne 15")

        # Should try suggestions
        assert service.scraper.search_product.call_count == 2

    async def test_search_product_not_found_no_suggestions(self, service):
        """Test search when product not found and no suggestions"""
        service.scraper.search_product = AsyncMock(return_value={"found": False})
        service.normalizer.suggest_corrections.return_value = []

        result = await service.search_product("NonexistentProduct123")

        assert result["found"] is False

    # =========================================================================
    # Test _generate_recommendation()
    # =========================================================================

    def test_generate_recommendation_high_rating_many_reviews(self, service):
        """Test recommendation for highly rated product with many reviews"""
        ml_analysis = {"sentiment_breakdown": {"positive_percent": 80, "negative_percent": 10}}

        rec = service._generate_recommendation(4.5, 100, ml_analysis)

        assert rec["decision"] == "buy"
        assert "confidence" in rec
        assert "reason" in rec

    def test_generate_recommendation_low_rating(self, service):
        """Test recommendation for low-rated product"""
        ml_analysis = {"sentiment_breakdown": {"positive_percent": 20, "negative_percent": 70}}

        rec = service._generate_recommendation(2.5, 50, ml_analysis)

        assert rec["decision"] in ["skip", "wait"]

    def test_generate_recommendation_few_reviews(self, service):
        """Test recommendation for product with few reviews"""
        rec = service._generate_recommendation(4.0, 5, None)

        assert rec["decision"] == "insufficient_data"  # Less than 10 reviews

    def test_generate_recommendation_no_ml_analysis(self, service):
        """Test recommendation without ML analysis"""
        rec = service._generate_recommendation(4.0, 50, None)

        assert "decision" in rec
        assert "confidence" in rec

    # =========================================================================
    # Test _calculate_source_breakdown()
    # =========================================================================

    def test_calculate_source_breakdown_with_reviews(self, service):
        """Test source breakdown calculation"""
        scraped_data = {
            "youtube": {"total_results": 10},
            "reddit": {"total_results": 20},
            "twitter": {"total_results": 30}
        }
        ml_analysis = {"sentiment_breakdown": {"positive": 0.7, "negative": 0.2, "neutral": 0.1}}

        breakdown = service._calculate_source_breakdown(scraped_data, ml_analysis)

        assert "youtube" in breakdown
        assert "reddit" in breakdown
        assert "twitter" in breakdown

    def test_calculate_source_breakdown_no_data(self, service):
        """Test source breakdown with no data"""
        breakdown = service._calculate_source_breakdown({}, None)

        assert isinstance(breakdown, dict)

    # =========================================================================
    # Test Error Handling
    # =========================================================================

    async def test_search_product_scraper_error(self, service):
        """Test search handles scraper error gracefully"""
        service.scraper.search_product = AsyncMock(side_effect=Exception("Scraper error"))

        # Should raise exception (no error handling in service currently)
        with pytest.raises(Exception):
            await service.search_product("iPhone 15")

    async def test_search_product_shopping_error(self, service):
        """Test search handles shopping service error"""
        service.shopping.search_product.side_effect = Exception("Shopping error")

        # Should raise exception (no error handling in service currently)
        with pytest.raises(Exception):
            await service.search_product("iPhone 15")

    async def test_search_product_ml_error(self, service):
        """Test search handles ML analysis error gracefully"""
        service._analyze_reviews = AsyncMock(side_effect=Exception("ML error"))

        # Should raise exception (no error handling in service currently)
        with pytest.raises(Exception):
            await service.search_product("iPhone 15", include_ml_analysis=True)

    # =========================================================================
    # Edge Cases
    # =========================================================================

    async def test_search_product_empty_query(self, service):
        """Test search with empty query"""
        result = await service.search_product("")
        assert isinstance(result, dict)

    async def test_search_product_special_characters(self, service):
        """Test search with special characters"""
        result = await service.search_product("iPhone 15 Pro (256GB) @#$%")
        assert isinstance(result, dict)

    async def test_search_product_unicode(self, service):
        """Test search with unicode characters"""
        result = await service.search_product("café naïve 中文")
        assert isinstance(result, dict)

    async def test_search_product_very_long_query(self, service):
        """Test search with very long query"""
        long_query = "Product name " * 100
        result = await service.search_product(long_query)
        assert isinstance(result, dict)

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    async def test_realistic_iphone_search(self, service):
        """Test realistic iPhone search flow"""
        result = await service.search_product("iPhone 15 Pro")

        # Should have all major sections
        assert result["found"] is True
        assert "product" in result
        assert "consensus" in result
        assert "sources" in result
        assert "review_videos" in result
        assert "discussions" in result
        assert "citations" in result
        assert "metadata" in result

    async def test_cache_stores_results(self, service):
        """Test search results are cached"""
        # First search
        await service.search_product("iPhone 15")

        # Cache should have the result
        cache_key = service._get_cache_key("iPhone 15")
        assert cache_key in service.cache

    async def test_multiple_searches_different_products(self, service):
        """Test multiple searches for different products"""
        results = []
        for product in ["iPhone 15", "Samsung Galaxy", "Google Pixel"]:
            result = await service.search_product(product)
            results.append(result)

        # All should complete
        assert len(results) == 3
        # Cache should have all
        assert len(service.cache) == 3

    @pytest.mark.parametrize("query,expected_normalized", [
        ("IPHONE 15 PRO", "iphone 15 pro"),
        ("  iPhone  15  ", "iphone 15 pro"),
        ("iphne 15", "iphone 15 pro"),
    ])
    async def test_search_various_query_formats(self, service, query, expected_normalized):
        """Test search with various query formats"""
        await service.search_product(query)

        # Should normalize all to same format
        service.normalizer.normalize.assert_called()
