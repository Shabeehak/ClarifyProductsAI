"""
Redis Cache Service for Product Data
Caches API responses and ML results with 24-hour TTL
"""

import redis.asyncio as redis
import json
from typing import Optional, Dict, Any, List
from loguru import logger
from datetime import datetime


class CacheService:
    """Redis cache for product data and ML results"""

    def __init__(self, redis_url: str):
        """
        Initialize cache service

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.TTL_24H = 86400  # 24 hours in seconds
        self.TTL_1H = 3600  # 1 hour in seconds
        logger.info("Cache service initialized")

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
            )
            # Test connection
            await self.redis.ping()
            logger.info("[OK] Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Cache will be disabled, using no-cache mode")
            self.redis = None

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("[OK] Redis cache disconnected")

    def _normalize_key(self, text: str) -> str:
        """
        Normalize text for use as Redis key

        Args:
            text: Text to normalize

        Returns:
            Normalized key (lowercase, hyphens instead of spaces)
        """
        return text.lower().strip().replace(" ", "-").replace("_", "-")

    async def get_product(self, product_name: str) -> Optional[Dict]:
        """
        Get cached product data

        Args:
            product_name: Product name

        Returns:
            Cached product data or None if not found

        Example:
            data = await cache.get_product("iPhone 15")
            if data:
                print(f"Cache HIT: {data['name']}")
        """
        if not self.redis:
            return None

        key = f"product:{self._normalize_key(product_name)}"

        try:
            data = await self.redis.get(key)
            if data:
                logger.info(f"Cache HIT: product '{product_name}'")
                parsed = json.loads(data)
                # Add cache metadata
                parsed["_cached"] = True
                parsed["_cache_key"] = key
                return parsed
            else:
                logger.info(f"Cache MISS: product '{product_name}'")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Cache JSON decode error for '{product_name}': {e}")
            # Delete corrupted cache entry
            await self.redis.delete(key)
            return None
        except Exception as e:
            logger.error(f"Cache get error for '{product_name}': {e}")
            return None

    async def set_product(
        self, product_name: str, data: Dict, ttl: Optional[int] = None
    ):
        """
        Cache product data

        Args:
            product_name: Product name
            data: Product data dictionary
            ttl: Time-to-live in seconds (default: 24 hours)

        Example:
            await cache.set_product("iPhone 15", product_data)
        """
        if not self.redis:
            return

        key = f"product:{self._normalize_key(product_name)}"
        ttl = ttl or self.TTL_24H

        try:
            # Add cache metadata
            data["_cached_at"] = datetime.utcnow().isoformat()
            data["_ttl"] = ttl

            await self.redis.setex(key, ttl, json.dumps(data))
            logger.info(f"Cached product '{product_name}' (TTL: {ttl/3600:.1f}h)")
        except Exception as e:
            logger.error(f"Cache set error for '{product_name}': {e}")

    async def get_ml_result(self, product_name: str, model: str) -> Optional[Dict]:
        """
        Get cached ML model result

        Args:
            product_name: Product name
            model: Model name (e.g., "sentiment", "summary", "clip")

        Returns:
            Cached ML result or None

        Example:
            sentiment = await cache.get_ml_result("iPhone 15", "sentiment")
            if sentiment:
                print(f"Sentiment: {sentiment['label']}")
        """
        if not self.redis:
            return None

        key = f"ml:{model}:{self._normalize_key(product_name)}"

        try:
            data = await self.redis.get(key)
            if data:
                logger.info(f"Cache HIT: ML {model} for '{product_name}'")
                return json.loads(data)
            else:
                logger.debug(f"Cache MISS: ML {model} for '{product_name}'")
                return None
        except Exception as e:
            logger.error(f"Cache ML get error: {e}")
            return None

    async def set_ml_result(
        self, product_name: str, model: str, result: Dict, ttl: Optional[int] = None
    ):
        """
        Cache ML model result

        Args:
            product_name: Product name
            model: Model name
            result: ML model output
            ttl: Time-to-live in seconds (default: 24 hours)

        Example:
            await cache.set_ml_result(
                "iPhone 15",
                "sentiment",
                {"label": "positive", "score": 0.95}
            )
        """
        if not self.redis:
            return

        key = f"ml:{model}:{self._normalize_key(product_name)}"
        ttl = ttl or self.TTL_24H

        try:
            # Add timestamp
            result["_computed_at"] = datetime.utcnow().isoformat()

            await self.redis.setex(key, ttl, json.dumps(result))
            logger.info(
                f"Cached ML {model} for '{product_name}' (TTL: {ttl/3600:.1f}h)"
            )
        except Exception as e:
            logger.error(f"Cache ML set error: {e}")

    async def get_reviews(self, product_name: str) -> Optional[List[Dict]]:
        """Get cached reviews for a product"""
        if not self.redis:
            return None

        key = f"reviews:{self._normalize_key(product_name)}"

        try:
            data = await self.redis.get(key)
            if data:
                logger.info(f"Cache HIT: reviews for '{product_name}'")
                return json.loads(data)
            else:
                return None
        except Exception as e:
            logger.error(f"Cache reviews get error: {e}")
            return None

    async def set_reviews(
        self, product_name: str, reviews: List[Dict], ttl: Optional[int] = None
    ):
        """Cache product reviews"""
        if not self.redis:
            return

        key = f"reviews:{self._normalize_key(product_name)}"
        ttl = ttl or self.TTL_24H

        try:
            await self.redis.setex(key, ttl, json.dumps(reviews))
            logger.info(f"Cached {len(reviews)} reviews for '{product_name}'")
        except Exception as e:
            logger.error(f"Cache reviews set error: {e}")

    async def invalidate_product(self, product_name: str):
        """
        Invalidate all cached data for a product

        Args:
            product_name: Product name

        Use case: User reports stale data, force refresh
        """
        if not self.redis:
            return

        normalized = self._normalize_key(product_name)
        keys_to_delete = [
            f"product:{normalized}",
            f"reviews:{normalized}",
            f"ml:sentiment:{normalized}",
            f"ml:summary:{normalized}",
            f"ml:clip:{normalized}",
        ]

        try:
            deleted = await self.redis.delete(*keys_to_delete)
            logger.info(
                f"Invalidated cache for '{product_name}' ({deleted} keys deleted)"
            )
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

    async def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        if not self.redis:
            return {"enabled": False}

        try:
            info = await self.redis.info("stats")
            keyspace = await self.redis.info("keyspace")

            # Count keys by type
            product_keys = 0
            ml_keys = 0
            review_keys = 0

            async for key in self.redis.scan_iter(match="product:*"):
                product_keys += 1
            async for key in self.redis.scan_iter(match="ml:*"):
                ml_keys += 1
            async for key in self.redis.scan_iter(match="reviews:*"):
                review_keys += 1

            return {
                "enabled": True,
                "total_keys": product_keys + ml_keys + review_keys,
                "product_keys": product_keys,
                "ml_keys": ml_keys,
                "review_keys": review_keys,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
                ),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)

    async def clear_all(self):
        """
        Clear all cached data (use with caution!)

        Warning: This deletes ALL keys in Redis
        """
        if not self.redis:
            return

        try:
            await self.redis.flushdb()
            logger.warning("Cleared all cache data")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Singleton instance
_cache_service_instance: Optional[CacheService] = None


def get_cache_service(redis_url: str) -> CacheService:
    """
    Get or create cache service singleton

    Args:
        redis_url: Redis connection URL

    Returns:
        CacheService instance
    """
    global _cache_service_instance
    if _cache_service_instance is None:
        _cache_service_instance = CacheService(redis_url)
    return _cache_service_instance
