"""
Google Shopping Service via SerpAPI
Fetches product details, images, prices, and merchant listings
"""

from typing import Dict, List, Optional
from loguru import logger
import requests
from app.core.config import settings


class GoogleShoppingService:
    """Google Shopping product information via SerpAPI"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Shopping service

        Args:
            api_key: SerpAPI key (get from https://serpapi.com)
        """
        self.api_key = api_key or settings.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"

        if self.api_key:
            logger.info("Google Shopping service initialized with SerpAPI")
        else:
            logger.warning(
                "No SerpAPI key configured - Google Shopping will be disabled"
            )

    def search_product(self, product_name: str, max_results: int = 5) -> Dict:
        """
        Search for product on Google Shopping

        Args:
            product_name: Product to search for
            max_results: Maximum number of results

        Returns:
            Dict with product details, images, prices, merchants

        Example:
            result = service.search_product("iPhone 15 Pro")
            {
                "product_name": "iPhone 15 Pro",
                "images": [...],
                "price": {...},
                "details": {...},
                "merchants": [...],
                "found": True
            }
        """
        if not self.api_key:
            logger.warning("SerpAPI key not configured")
            return self._empty_result(product_name)

        try:
            logger.info(f"Searching Google Shopping for: {product_name}")

            params = {
                "engine": "google_shopping",
                "q": product_name,
                "api_key": self.api_key,
                "num": max_results,
                "gl": "us",  # Country: US
                "hl": "en",  # Language: English
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract shopping results
            shopping_results = data.get("shopping_results", [])

            if not shopping_results:
                logger.warning(f"No shopping results found for: {product_name}")
                return self._empty_result(product_name)

            # Process results
            product_data = self._process_shopping_results(
                product_name, shopping_results
            )

            logger.info(
                f"Found {len(shopping_results)} shopping results for '{product_name}'"
            )

            return product_data

        except requests.Timeout:
            logger.error(f"SerpAPI request timeout for: {product_name}")
            return self._empty_result(product_name)
        except requests.RequestException as e:
            logger.error(f"SerpAPI request error: {e}")
            return self._empty_result(product_name)
        except Exception as e:
            logger.error(f"Error searching Google Shopping: {e}")
            return self._empty_result(product_name)

    def _process_shopping_results(self, product_name: str, results: List[Dict]) -> Dict:
        """Process shopping results into structured format"""

        # Get first result for main product details
        first_result = results[0] if results else {}

        # Extract images from all results
        images = []
        for result in results[:5]:
            if result.get("thumbnail"):
                images.append(
                    {
                        "url": result["thumbnail"],
                        "source": "google_shopping",
                        "alt_text": result.get("title", product_name),
                        "merchant": result.get("source", "Unknown"),
                    }
                )

        # Extract price information
        price_info = self._extract_price_info(results)

        # Extract merchants
        merchants = []
        for result in results[:10]:
            merchant = {
                "name": result.get("source", "Unknown"),
                "price": result.get("price", "N/A"),
                "link": result.get("link", ""),
                "rating": result.get("rating"),
                "reviews": result.get("reviews"),
                "delivery": result.get("delivery", ""),
            }
            merchants.append(merchant)

        # Product details from first result
        product_details = {
            "title": first_result.get("title", product_name),
            "brand": self._extract_brand(first_result.get("title", "")),
            "description": first_result.get("snippet", ""),
            "category": first_result.get("category", ""),
            "source": "Google Shopping",
        }

        return {
            "product_name": product_name,
            "found": True,
            "images": images,
            "price": price_info,
            "details": product_details,
            "merchants": merchants,
            "total_results": len(results),
            "source": "google_shopping",
        }

    def _extract_price_info(self, results: List[Dict]) -> Dict:
        """Extract price information from results"""
        prices = []

        for result in results:
            price_str = result.get("price", "")
            if price_str:
                # Try to extract numeric price
                try:
                    # Remove currency symbols and commas
                    price_clean = price_str.replace("$", "").replace(",", "")
                    price_float = float(price_clean)
                    prices.append(price_float)
                except (ValueError, AttributeError):
                    pass

        if not prices:
            return {
                "min": None,
                "max": None,
                "avg": None,
                "currency": "USD",
                "formatted": "Price not available",
            }

        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)

        return {
            "min": min_price,
            "max": max_price,
            "avg": avg_price,
            "currency": "USD",
            "formatted": f"${min_price:.2f} - ${max_price:.2f}",
        }

    def _extract_brand(self, title: str) -> str:
        """Extract brand from product title"""
        # Common brand patterns (first word usually)
        words = title.split()
        if words:
            potential_brand = words[0]
            # Remove common prefixes
            if potential_brand.lower() in ["the", "a", "an"]:
                return words[1] if len(words) > 1 else ""
            return potential_brand
        return ""

    def _empty_result(self, product_name: str) -> Dict:
        """Return empty result structure"""
        return {
            "product_name": product_name,
            "found": False,
            "images": [],
            "price": {
                "min": None,
                "max": None,
                "avg": None,
                "currency": "USD",
                "formatted": "Price not available",
            },
            "details": {},
            "merchants": [],
            "total_results": 0,
            "source": "google_shopping",
        }


# Singleton instance
_shopping_service_instance: Optional[GoogleShoppingService] = None


def get_shopping_service(api_key: Optional[str] = None) -> GoogleShoppingService:
    """
    Get or create Google Shopping service singleton

    Args:
        api_key: Optional SerpAPI key override

    Returns:
        GoogleShoppingService instance
    """
    global _shopping_service_instance
    if _shopping_service_instance is None:
        _shopping_service_instance = GoogleShoppingService(api_key)
    return _shopping_service_instance
