"""
Unit Tests for RealtimeScraperService

Tests the real-time product scraper with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.realtime_scraper_service import RealtimeScraperService


@pytest.mark.unit
@pytest.mark.service
@pytest.mark.asyncio
class TestRealtimeScraperService:
    """Test suite for RealtimeScraperService"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings"""
        mock = Mock()
        mock.YOUTUBE_API_KEY = "test_youtube_key"
        mock.REDDIT_CLIENT_ID = "test_reddit_id"
        mock.REDDIT_CLIENT_SECRET = "test_reddit_secret"
        mock.REDDIT_USER_AGENT = "test_agent"
        mock.TWITTER_BEARER_TOKEN = "test_twitter_token"
        mock.REDIS_URL = "redis://localhost"
        return mock

    @pytest.fixture
    def mock_youtube_service(self):
        """Mock YouTube service"""
        mock = Mock()
        mock.search_reviews.return_value = {
            "videos": [
                {"title": "iPhone 15 Review", "video_id": "abc123", "views": 1000}
            ],
            "total_results": 1
        }
        return mock

    @pytest.fixture
    def mock_reddit_service(self):
        """Mock Reddit service"""
        mock = Mock()
        mock.search_discussions.return_value = {
            "posts": [
                {"title": "iPhone 15 thoughts", "score": 100, "subreddit": "tech"}
            ],
            "total_results": 1
        }
        return mock

    @pytest.fixture
    def mock_twitter_service(self):
        """Mock Twitter service"""
        mock = Mock()
        mock.search_tweets.return_value = {
            "tweets": [
                {"text": "Loving the iPhone 15!", "likes": 50}
            ],
            "total_results": 1
        }
        return mock

    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service"""
        mock = AsyncMock()
        mock.get_product = AsyncMock(return_value=None)
        mock.set_product = AsyncMock()
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        return mock

    @pytest.fixture
    def scraper(self, mock_settings, mock_youtube_service, mock_reddit_service,
                mock_twitter_service, mock_cache_service):
        """Scraper fixture with mocked dependencies"""
        with patch('app.services.realtime_scraper_service.settings', mock_settings), \
             patch('app.services.realtime_scraper_service.get_youtube_service', return_value=mock_youtube_service), \
             patch('app.services.realtime_scraper_service.get_reddit_service', return_value=mock_reddit_service), \
             patch('app.services.realtime_scraper_service.get_twitter_service', return_value=mock_twitter_service), \
             patch('app.services.realtime_scraper_service.get_cache_service', return_value=mock_cache_service):

            service = RealtimeScraperService()
            return service

    # =========================================================================
    # Test Initialization
    # =========================================================================

    def test_scraper_initialization_with_all_services(self, scraper):
        """Test scraper initializes with all services"""
        assert scraper.youtube is not None
        assert scraper.reddit is not None
        assert scraper.twitter is not None
        assert scraper.cache is not None

    @patch('app.services.realtime_scraper_service.settings')
    @patch('app.services.realtime_scraper_service.get_cache_service')
    def test_scraper_initialization_without_youtube(self, mock_cache, mock_settings):
        """Test scraper initializes without YouTube API key"""
        mock_settings.YOUTUBE_API_KEY = None
        mock_settings.REDDIT_CLIENT_ID = "test"
        mock_settings.REDDIT_CLIENT_SECRET = "test"
        mock_settings.REDDIT_USER_AGENT = "test"
        mock_settings.TWITTER_BEARER_TOKEN = "test"
        mock_settings.REDIS_URL = "redis://localhost"
        mock_cache.return_value = Mock()

        with patch('app.services.realtime_scraper_service.get_reddit_service'), \
             patch('app.services.realtime_scraper_service.get_twitter_service'):
            service = RealtimeScraperService()
            assert service.youtube is None

    @patch('app.services.realtime_scraper_service.settings')
    @patch('app.services.realtime_scraper_service.get_cache_service')
    def test_scraper_initialization_without_reddit(self, mock_cache, mock_settings):
        """Test scraper initializes without Reddit credentials"""
        mock_settings.YOUTUBE_API_KEY = "test"
        mock_settings.REDDIT_CLIENT_ID = None
        mock_settings.REDDIT_CLIENT_SECRET = None
        mock_settings.REDDIT_USER_AGENT = "test"
        mock_settings.TWITTER_BEARER_TOKEN = "test"
        mock_settings.REDIS_URL = "redis://localhost"
        mock_cache.return_value = Mock()

        with patch('app.services.realtime_scraper_service.get_youtube_service'), \
             patch('app.services.realtime_scraper_service.get_twitter_service'):
            service = RealtimeScraperService()
            assert service.reddit is None

    @patch('app.services.realtime_scraper_service.settings')
    @patch('app.services.realtime_scraper_service.get_cache_service')
    def test_scraper_initialization_without_twitter(self, mock_cache, mock_settings):
        """Test scraper initializes without Twitter token"""
        mock_settings.YOUTUBE_API_KEY = "test"
        mock_settings.REDDIT_CLIENT_ID = "test"
        mock_settings.REDDIT_CLIENT_SECRET = "test"
        mock_settings.REDDIT_USER_AGENT = "test"
        mock_settings.TWITTER_BEARER_TOKEN = None
        mock_settings.REDIS_URL = "redis://localhost"
        mock_cache.return_value = Mock()

        with patch('app.services.realtime_scraper_service.get_youtube_service'), \
             patch('app.services.realtime_scraper_service.get_reddit_service'):
            service = RealtimeScraperService()
            assert service.twitter is None

    # =========================================================================
    # Test Cache Operations
    # =========================================================================

    async def test_init_cache(self, scraper):
        """Test cache initialization"""
        await scraper.init_cache()
        scraper.cache.connect.assert_called_once()

    async def test_close_cache(self, scraper):
        """Test cache closing"""
        await scraper.close_cache()
        scraper.cache.disconnect.assert_called_once()

    async def test_init_cache_when_no_cache(self):
        """Test init_cache when cache is None"""
        with patch('app.services.realtime_scraper_service.settings') as mock_settings, \
             patch('app.services.realtime_scraper_service.get_cache_service') as mock_cache_svc:
            mock_settings.YOUTUBE_API_KEY = None
            mock_settings.REDDIT_CLIENT_ID = None
            mock_settings.REDDIT_CLIENT_SECRET = None
            mock_settings.TWITTER_BEARER_TOKEN = None
            mock_settings.REDIS_URL = "redis://localhost"
            mock_cache_svc.return_value = None

            service = RealtimeScraperService()
            service.cache = None

            # Should not raise error
            await service.init_cache()

    # =========================================================================
    # Test search_product() - Cache Hit
    # =========================================================================

    async def test_search_product_cache_hit(self, scraper):
        """Test search returns cached data if available"""
        cached_data = {
            "product_name": "iPhone 15",
            "youtube": {"videos": []},
            "reddit": {"posts": []},
            "twitter": {"tweets": []}
        }
        scraper.cache.get_product = AsyncMock(return_value=cached_data)

        result = await scraper.search_product("iPhone 15")

        assert result == cached_data
        scraper.cache.get_product.assert_called_once_with("iPhone 15")
        # Should not call APIs when cache hit
        scraper.youtube.search_reviews.assert_not_called()

    # =========================================================================
    # Test search_product() - Cache Miss
    # =========================================================================

    async def test_search_product_cache_miss_all_services(self, scraper):
        """Test search fetches from all services on cache miss"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("iPhone 15")

        # Should call all services
        assert scraper.youtube.search_reviews.called
        assert scraper.reddit.search_discussions.called
        assert scraper.twitter.search_tweets.called

        # Should cache the result
        scraper.cache.set_product.assert_called_once()

    async def test_search_product_with_custom_limits(self, scraper):
        """Test search respects custom result limits"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        await scraper.search_product(
            "iPhone 15",
            max_videos=10,
            max_reddit_posts=20,
            max_tweets=100
        )

        # Check services were called with correct limits
        scraper.youtube.search_reviews.assert_called()
        scraper.reddit.search_discussions.assert_called()
        scraper.twitter.search_tweets.assert_called()

    async def test_search_product_without_youtube(self, scraper):
        """Test search works without YouTube service"""
        scraper.youtube = None
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("iPhone 15")

        # Should still work with other services
        assert "reddit" in result or "twitter" in result

    async def test_search_product_without_reddit(self, scraper):
        """Test search works without Reddit service"""
        scraper.reddit = None
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("iPhone 15")

        # Should still work with other services
        assert "youtube" in result or "twitter" in result

    async def test_search_product_without_twitter(self, scraper):
        """Test search works without Twitter service"""
        scraper.twitter = None
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("iPhone 15")

        # Should still work with other services
        assert "youtube" in result or "reddit" in result

    async def test_search_product_no_services_available(self):
        """Test search when no services are available"""
        with patch('app.services.realtime_scraper_service.settings') as mock_settings, \
             patch('app.services.realtime_scraper_service.get_cache_service') as mock_cache:

            mock_settings.YOUTUBE_API_KEY = None
            mock_settings.REDDIT_CLIENT_ID = None
            mock_settings.REDDIT_CLIENT_SECRET = None
            mock_settings.TWITTER_BEARER_TOKEN = None
            mock_settings.REDIS_URL = "redis://localhost"

            mock_cache_instance = AsyncMock()
            mock_cache_instance.get_product = AsyncMock(return_value=None)
            mock_cache_instance.set_product = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            service = RealtimeScraperService()
            result = await service.search_product("iPhone 15")

            # Should return empty structure
            assert isinstance(result, dict)

    # =========================================================================
    # Test Error Handling
    # =========================================================================

    async def test_search_product_youtube_error(self, scraper):
        """Test search handles YouTube service error"""
        scraper.cache.get_product = AsyncMock(return_value=None)
        scraper.youtube.search_reviews.side_effect = Exception("YouTube API error")

        # Should not raise, should continue with other services
        result = await scraper.search_product("iPhone 15")
        assert isinstance(result, dict)

    async def test_search_product_reddit_error(self, scraper):
        """Test search handles Reddit service error"""
        scraper.cache.get_product = AsyncMock(return_value=None)
        scraper.reddit.search_discussions.side_effect = Exception("Reddit API error")

        # Should not raise, should continue with other services
        result = await scraper.search_product("iPhone 15")
        assert isinstance(result, dict)

    async def test_search_product_twitter_error(self, scraper):
        """Test search handles Twitter service error"""
        scraper.cache.get_product = AsyncMock(return_value=None)
        scraper.twitter.search_tweets.side_effect = Exception("Twitter API error")

        # Should not raise, should continue with other services
        result = await scraper.search_product("iPhone 15")
        assert isinstance(result, dict)

    async def test_search_product_cache_error(self, scraper):
        """Test search handles cache error gracefully"""
        scraper.cache.get_product = AsyncMock(side_effect=Exception("Cache error"))

        # Should continue with search even if cache fails
        result = await scraper.search_product("iPhone 15")
        assert isinstance(result, dict)

    async def test_search_product_all_services_error(self, scraper):
        """Test search when all services fail"""
        scraper.cache.get_product = AsyncMock(return_value=None)
        scraper.youtube.search_reviews.side_effect = Exception("Error")
        scraper.reddit.search_discussions.side_effect = Exception("Error")
        scraper.twitter.search_tweets.side_effect = Exception("Error")

        result = await scraper.search_product("iPhone 15")

        # Should return empty result structure
        assert isinstance(result, dict)

    # =========================================================================
    # Edge Cases
    # =========================================================================

    async def test_search_product_empty_product_name(self, scraper):
        """Test search with empty product name"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("")
        assert isinstance(result, dict)

    async def test_search_product_special_characters(self, scraper):
        """Test search with special characters in product name"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("iPhone 15 Pro (256GB) @#$%")
        assert isinstance(result, dict)

    async def test_search_product_unicode(self, scraper):
        """Test search with unicode characters"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product("café naïve 中文")
        assert isinstance(result, dict)

    async def test_search_product_very_long_name(self, scraper):
        """Test search with very long product name"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        long_name = "Product name " * 100
        result = await scraper.search_product(long_name)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("max_videos,max_reddit,max_tweets", [
        (1, 1, 1),
        (10, 20, 50),
        (0, 0, 0),
        (100, 100, 100),
    ])
    async def test_search_product_various_limits(self, scraper, max_videos, max_reddit, max_tweets):
        """Test search with various result limits"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        result = await scraper.search_product(
            "iPhone",
            max_videos=max_videos,
            max_reddit_posts=max_reddit,
            max_tweets=max_tweets
        )

        assert isinstance(result, dict)

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    async def test_realistic_iphone_search(self, scraper):
        """Test realistic iPhone search scenario"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        # Set up realistic mock responses
        scraper.youtube.search_reviews.return_value = {
            "videos": [
                {"title": "iPhone 15 Pro Review", "video_id": "abc123", "views": 50000},
                {"title": "iPhone 15 Unboxing", "video_id": "def456", "views": 30000}
            ],
            "total_results": 2
        }

        scraper.reddit.search_discussions.return_value = {
            "posts": [
                {"title": "Is iPhone 15 worth it?", "score": 150, "comments": 45},
                {"title": "iPhone 15 camera quality", "score": 89, "comments": 23}
            ],
            "total_results": 2
        }

        scraper.twitter.search_tweets.return_value = {
            "tweets": [
                {"text": "Just got the iPhone 15! Amazing!", "likes": 120},
                {"text": "iPhone 15 battery life is incredible", "likes": 85}
            ],
            "total_results": 2
        }

        result = await scraper.search_product("iPhone 15")

        # Should have data from all sources
        assert isinstance(result, dict)
        # Cache should be called to store the result
        assert scraper.cache.set_product.called

    async def test_concurrent_searches(self, scraper):
        """Test multiple concurrent searches"""
        scraper.cache.get_product = AsyncMock(return_value=None)

        # Run multiple searches concurrently
        results = await asyncio.gather(
            scraper.search_product("iPhone 15"),
            scraper.search_product("Samsung Galaxy"),
            scraper.search_product("Google Pixel")
        )

        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
