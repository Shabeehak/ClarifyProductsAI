"""
Unit Tests for GoogleShoppingService

Tests the Google Shopping product search functionality via SerpAPI.
"""
import pytest
from unittest.mock import Mock, patch
from app.services.google_shopping_service import GoogleShoppingService, get_shopping_service


@pytest.mark.unit
@pytest.mark.service
class TestGoogleShoppingService:
    """Test suite for GoogleShoppingService"""

    @pytest.fixture
    def service_with_key(self):
        """Service with API key"""
        return GoogleShoppingService(api_key="test_api_key_123")

    @pytest.fixture
    def service_without_key(self):
        """Service without API key"""
        with patch('app.services.google_shopping_service.settings') as mock_settings:
            mock_settings.SERPAPI_KEY = None
            return GoogleShoppingService(api_key=None)

    @pytest.fixture
    def mock_shopping_results(self):
        """Mock shopping results from SerpAPI"""
        return [
            {
                "title": "iPhone 15 Pro Max 256GB",
                "price": "$1,199.00",
                "source": "Apple Store",
                "link": "https://apple.com/iphone",
                "thumbnail": "https://example.com/image1.jpg",
                "rating": 4.8,
                "reviews": 1500,
                "delivery": "Free delivery",
                "snippet": "Latest iPhone with A17 Pro chip"
            },
            {
                "title": "iPhone 15 Pro",
                "price": "$999.00",
                "source": "Best Buy",
                "link": "https://bestbuy.com/iphone",
                "thumbnail": "https://example.com/image2.jpg",
                "rating": 4.7,
                "reviews": 800
            }
        ]

    # =========================================================================
    # Test Initialization
    # =========================================================================

    def test_service_initialization_with_key(self, service_with_key):
        """Test service initializes with API key"""
        assert service_with_key.api_key == "test_api_key_123"
        assert service_with_key.base_url == "https://serpapi.com/search"

    def test_service_initialization_without_key(self, service_without_key):
        """Test service initializes without API key"""
        assert service_without_key.api_key is None

    @patch('app.services.google_shopping_service.settings')
    def test_service_uses_settings_key(self, mock_settings):
        """Test service uses API key from settings if not provided"""
        mock_settings.SERPAPI_KEY = "settings_api_key"
        service = GoogleShoppingService()
        assert service.api_key == "settings_api_key"

    def test_singleton_instance(self):
        """Test get_shopping_service returns singleton"""
        service1 = get_shopping_service("test_key")
        service2 = get_shopping_service("test_key")
        assert service1 is service2

    # =========================================================================
    # Test search_product() - Success Cases
    # =========================================================================

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_success(self, mock_get, service_with_key, mock_shopping_results):
        """Test successful product search"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"shopping_results": mock_shopping_results}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone 15 Pro")

        assert result["found"] is True
        assert result["product_name"] == "iPhone 15 Pro"
        assert len(result["images"]) > 0
        assert len(result["merchants"]) > 0
        assert "price" in result

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_constructs_correct_params(self, mock_get, service_with_key):
        """Test search constructs correct API parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_get.return_value = mock_response

        service_with_key.search_product("Test Product", max_results=10)

        # Check API was called with correct parameters
        call_args = mock_get.call_args
        params = call_args[1]['params']
        assert params['engine'] == 'google_shopping'
        assert params['q'] == 'Test Product'
        assert params['api_key'] == 'test_api_key_123'
        assert params['num'] == 10
        assert params['gl'] == 'us'
        assert params['hl'] == 'en'

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_with_max_results(self, mock_get, service_with_key, mock_shopping_results):
        """Test search respects max_results parameter"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": mock_shopping_results}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone", max_results=1)

        assert result["found"] is True

    # =========================================================================
    # Test search_product() - No API Key
    # =========================================================================

    def test_search_product_no_api_key(self, service_without_key):
        """Test search without API key returns empty result"""
        result = service_without_key.search_product("iPhone")

        assert result["found"] is False
        assert result["product_name"] == "iPhone"
        assert result["images"] == []
        assert result["merchants"] == []

    # =========================================================================
    # Test search_product() - Error Cases
    # =========================================================================

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_no_results(self, mock_get, service_with_key):
        """Test search with no results"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("NonexistentProduct123")

        assert result["found"] is False
        assert result["product_name"] == "NonexistentProduct123"

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_timeout(self, mock_get, service_with_key):
        """Test search handles timeout"""
        import requests
        mock_get.side_effect = requests.Timeout()

        result = service_with_key.search_product("iPhone")

        assert result["found"] is False

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_request_error(self, mock_get, service_with_key):
        """Test search handles request error"""
        import requests
        mock_get.side_effect = requests.RequestException("API Error")

        result = service_with_key.search_product("iPhone")

        assert result["found"] is False

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_http_error(self, mock_get, service_with_key):
        """Test search handles HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone")

        assert result["found"] is False

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_invalid_json(self, mock_get, service_with_key):
        """Test search handles invalid JSON response"""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone")

        assert result["found"] is False

    # =========================================================================
    # Test _process_shopping_results()
    # =========================================================================

    def test_process_shopping_results_complete(self, service_with_key, mock_shopping_results):
        """Test processing complete shopping results"""
        result = service_with_key._process_shopping_results("iPhone 15", mock_shopping_results)

        assert result["product_name"] == "iPhone 15"
        assert result["found"] is True
        assert len(result["images"]) == 2
        assert len(result["merchants"]) == 2
        assert "price" in result
        assert "details" in result

    def test_process_shopping_results_empty(self, service_with_key):
        """Test processing empty results"""
        result = service_with_key._process_shopping_results("Test", [])

        assert result["found"] is True  # Still processes
        assert result["images"] == []
        assert result["merchants"] == []

    def test_process_shopping_results_extracts_images(self, service_with_key, mock_shopping_results):
        """Test image extraction from results"""
        result = service_with_key._process_shopping_results("iPhone", mock_shopping_results)

        images = result["images"]
        assert len(images) > 0
        assert images[0]["url"] == "https://example.com/image1.jpg"
        assert images[0]["source"] == "google_shopping"
        assert "alt_text" in images[0]
        assert "merchant" in images[0]

    def test_process_shopping_results_limits_images(self, service_with_key):
        """Test image extraction limits to 5"""
        many_results = [
            {"thumbnail": f"https://example.com/image{i}.jpg", "title": f"Product {i}"}
            for i in range(10)
        ]

        result = service_with_key._process_shopping_results("Test", many_results)

        assert len(result["images"]) <= 5

    def test_process_shopping_results_extracts_merchants(self, service_with_key, mock_shopping_results):
        """Test merchant extraction from results"""
        result = service_with_key._process_shopping_results("iPhone", mock_shopping_results)

        merchants = result["merchants"]
        assert len(merchants) == 2
        assert merchants[0]["name"] == "Apple Store"
        assert merchants[0]["price"] == "$1,199.00"
        assert merchants[0]["link"] == "https://apple.com/iphone"
        assert merchants[0]["rating"] == 4.8
        assert merchants[0]["reviews"] == 1500

    def test_process_shopping_results_extracts_details(self, service_with_key, mock_shopping_results):
        """Test product details extraction"""
        result = service_with_key._process_shopping_results("iPhone", mock_shopping_results)

        details = result["details"]
        assert details["title"] == "iPhone 15 Pro Max 256GB"
        assert details["description"] == "Latest iPhone with A17 Pro chip"
        assert details["source"] == "Google Shopping"
        assert "brand" in details

    # =========================================================================
    # Test _extract_price_info()
    # =========================================================================

    def test_extract_price_info_with_prices(self, service_with_key, mock_shopping_results):
        """Test price extraction with valid prices"""
        result = service_with_key._extract_price_info(mock_shopping_results)

        assert result["min"] == 999.0
        assert result["max"] == 1199.0
        assert result["avg"] == pytest.approx(1099.0)
        assert result["currency"] == "USD"
        assert "$999.00 - $1199.00" in result["formatted"]

    def test_extract_price_info_no_prices(self, service_with_key):
        """Test price extraction with no prices"""
        results = [{"title": "Product", "price": ""}]
        result = service_with_key._extract_price_info(results)

        assert result["min"] is None
        assert result["max"] is None
        assert result["avg"] is None
        assert result["formatted"] == "Price not available"

    def test_extract_price_info_single_price(self, service_with_key):
        """Test price extraction with single price"""
        results = [{"price": "$599.99"}]
        result = service_with_key._extract_price_info(results)

        assert result["min"] == 599.99
        assert result["max"] == 599.99
        assert result["avg"] == 599.99

    def test_extract_price_info_handles_commas(self, service_with_key):
        """Test price extraction handles comma separators"""
        results = [{"price": "$1,999.00"}]
        result = service_with_key._extract_price_info(results)

        assert result["min"] == 1999.0

    def test_extract_price_info_invalid_price(self, service_with_key):
        """Test price extraction handles invalid price format"""
        results = [{"price": "Not a price"}, {"price": "$100.00"}]
        result = service_with_key._extract_price_info(results)

        # Should only count valid price
        assert result["min"] == 100.0
        assert result["max"] == 100.0

    def test_extract_price_info_empty_results(self, service_with_key):
        """Test price extraction with empty results"""
        result = service_with_key._extract_price_info([])

        assert result["min"] is None
        assert result["formatted"] == "Price not available"

    # =========================================================================
    # Test _extract_brand()
    # =========================================================================

    def test_extract_brand_first_word(self, service_with_key):
        """Test brand extraction from first word"""
        result = service_with_key._extract_brand("Apple iPhone 15 Pro")
        assert result == "Apple"

    def test_extract_brand_removes_articles(self, service_with_key):
        """Test brand extraction skips articles"""
        result = service_with_key._extract_brand("The Samsung Galaxy S24")
        assert result == "Samsung"

    def test_extract_brand_single_word(self, service_with_key):
        """Test brand extraction with single word title"""
        result = service_with_key._extract_brand("iPhone")
        assert result == "iPhone"

    def test_extract_brand_empty_title(self, service_with_key):
        """Test brand extraction with empty title"""
        result = service_with_key._extract_brand("")
        assert result == ""

    @pytest.mark.parametrize("title,expected", [
        ("Apple iPhone 15", "Apple"),
        ("Samsung Galaxy", "Samsung"),
        ("The Sony WH-1000XM5", "Sony"),
        ("A Microsoft Surface", "Microsoft"),
        ("Google Pixel 8", "Google"),
    ])
    def test_extract_brand_various_titles(self, service_with_key, title, expected):
        """Test brand extraction with various titles"""
        result = service_with_key._extract_brand(title)
        assert result == expected

    # =========================================================================
    # Test _empty_result()
    # =========================================================================

    def test_empty_result_structure(self, service_with_key):
        """Test empty result has correct structure"""
        result = service_with_key._empty_result("Test Product")

        assert result["product_name"] == "Test Product"
        assert result["found"] is False
        assert result["images"] == []
        assert result["merchants"] == []
        assert result["total_results"] == 0
        assert "price" in result
        assert result["price"]["formatted"] == "Price not available"
        assert result["source"] == "google_shopping"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_special_characters(self, mock_get, service_with_key):
        """Test search with special characters in query"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone 15 Pro Max (256GB) @#$%")

        assert result["product_name"] == "iPhone 15 Pro Max (256GB) @#$%"

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_unicode(self, mock_get, service_with_key):
        """Test search with unicode characters"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("café naïve 中文")

        assert result["product_name"] == "café naïve 中文"

    def test_process_results_missing_optional_fields(self, service_with_key):
        """Test processing results with missing optional fields"""
        minimal_result = [
            {
                "title": "Product",
                "price": "$100",
                # Missing: thumbnail, rating, reviews, delivery, etc.
            }
        ]

        result = service_with_key._process_shopping_results("Product", minimal_result)

        # Should handle gracefully
        assert result["found"] is True
        assert len(result["merchants"]) == 1
        assert result["merchants"][0]["rating"] is None

    def test_process_results_no_thumbnail(self, service_with_key):
        """Test processing results without thumbnails"""
        results = [{"title": "Product", "price": "$100"}]  # No thumbnail

        result = service_with_key._process_shopping_results("Product", results)

        assert result["images"] == []

    @patch('app.services.google_shopping_service.requests.get')
    def test_search_product_very_long_query(self, mock_get, service_with_key):
        """Test search with very long query"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_get.return_value = mock_response

        long_query = "Product name " * 100
        result = service_with_key.search_product(long_query)

        assert result["product_name"] == long_query

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    @patch('app.services.google_shopping_service.requests.get')
    def test_realistic_iphone_search(self, mock_get, service_with_key):
        """Test realistic iPhone search scenario"""
        realistic_results = [
            {
                "title": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
                "price": "$1,199.00",
                "source": "Apple Store",
                "link": "https://www.apple.com/shop/buy-iphone/iphone-15-pro",
                "thumbnail": "https://store.storeimages.cdn-apple.com/iphone15.jpg",
                "rating": 4.8,
                "reviews": 2453,
                "delivery": "Free delivery",
                "snippet": "iPhone 15 Pro Max. Forged in titanium with A17 Pro chip."
            }
        ]

        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": realistic_results}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("iPhone 15 Pro Max")

        assert result["found"] is True
        assert "iPhone" in result["details"]["title"]
        assert result["price"]["min"] == 1199.0
        assert len(result["images"]) > 0
        assert result["merchants"][0]["name"] == "Apple Store"

    @patch('app.services.google_shopping_service.requests.get')
    def test_price_comparison_scenario(self, mock_get, service_with_key):
        """Test price comparison across merchants"""
        price_variations = [
            {"title": "Product A", "price": "$99.99", "source": "Store 1"},
            {"title": "Product A", "price": "$109.99", "source": "Store 2"},
            {"title": "Product A", "price": "$89.99", "source": "Store 3"},
        ]

        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": price_variations}
        mock_get.return_value = mock_response

        result = service_with_key.search_product("Product A")

        assert result["price"]["min"] == 89.99
        assert result["price"]["max"] == 109.99
        assert len(result["merchants"]) == 3
