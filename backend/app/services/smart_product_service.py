"""
Smart Product Service
Main controller for scalable product search with caching
"""

from typing import Dict, Optional, List
from loguru import logger
import hashlib
from datetime import datetime, timedelta
import json

from app.services.product_normalization_service import get_normalization_service
from app.services.realtime_scraper_service import get_scraper_service
from app.services.google_shopping_service import get_shopping_service
from app.ml_models.sentiment_analyzer import get_sentiment_analyzer
from app.ml_models.summarizer import get_summarizer


class SmartProductService:
    """
    Scalable product service with fuzzy search and caching
    Handles millions of products without storing them all!
    """

    def __init__(self):
        """Initialize smart product service"""
        self.normalizer = get_normalization_service()
        self.scraper = get_scraper_service()
        self.shopping = get_shopping_service()  # NEW: Google Shopping
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.summarizer = get_summarizer()

        # In-memory cache (will be replaced with Redis later)
        self.cache = {}
        self.cache_ttl = 86400  # 24 hours

        logger.info("Smart Product Service initialized")

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query"""
        normalized = self.normalizer.normalize(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid"""
        if not cached_data:
            return False

        cached_at = datetime.fromisoformat(cached_data.get("cached_at", "2000-01-01"))
        age = (datetime.utcnow() - cached_at).total_seconds()

        return age < self.cache_ttl

    async def search_product(
        self, query: str, use_cache: bool = True, include_ml_analysis: bool = True
    ) -> Dict:
        """
        Smart product search with typo tolerance and caching

        Args:
            query: User's search query (can have typos!)
            use_cache: Whether to use cached results
            include_ml_analysis: Whether to include ML sentiment/summary

        Returns:
            Complete product information with reviews and analysis
        """
        # Step 1: Normalize query
        normalized_query = self.normalizer.normalize(query)
        logger.info(f"Searching for: '{query}' (normalized: '{normalized_query}')")

        # Step 2: Check cache
        cache_key = self._get_cache_key(query)

        if use_cache and cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if self._is_cache_valid(cached_data):
                logger.info(f"Cache HIT for '{query}'")
                cached_data["from_cache"] = True
                return cached_data

        logger.info(f"Cache MISS for '{query}' - fetching fresh data")

        # Step 3: Check for typo corrections
        suggestions = self.normalizer.suggest_corrections(query)
        product_info = self.normalizer.extract_product_info(query)

        # Step 4a: Fetch Google Shopping data (product images, price, merchants)
        shopping_data = self.shopping.search_product(normalized_query, max_results=5)

        # Step 4b: Scrape product reviews in real-time
        scraped_data = await self.scraper.search_product(normalized_query)

        if not scraped_data.get("found"):
            # Try suggestions if original query failed
            if suggestions:
                logger.info(f"Trying suggestions: {suggestions}")
                for suggestion in suggestions[:2]:  # Try top 2 suggestions
                    scraped_data = await self.scraper.search_product(suggestion)
                    if scraped_data.get("found"):
                        normalized_query = suggestion
                        break

        # Step 5: Process reviews with ML (if requested)
        ml_analysis = None
        if include_ml_analysis and scraped_data.get("reviews"):
            ml_analysis = await self._analyze_reviews(scraped_data["reviews"])

        # Step 6: Extract media content (images, videos, citations)
        media_content = {}
        if scraped_data.get("found"):
            media_content = self.scraper.extract_media_content(scraped_data)

        # Step 6b: Merge Google Shopping images (prioritize official product images)
        if shopping_data.get("found") and shopping_data.get("images"):
            # Add shopping images at the front
            all_images = shopping_data["images"] + media_content.get(
                "product_images", []
            )
            media_content["product_images"] = all_images[:10]  # Keep top 10

        # Step 7: Calculate per-source breakdowns
        source_breakdown = self._calculate_source_breakdown(scraped_data, ml_analysis)

        # Step 8: Generate recommendation
        recommendation = self._generate_recommendation(
            scraped_data.get("rating", 0),
            scraped_data.get("review_count", 0),
            ml_analysis,
        )

        # Step 9: Build enhanced response (Clean, frontend-friendly)
        # DO NOT ADD OLD FIELDS - only use new nested structure
        result = {
            # Basic query info
            "query": query,
            "found": scraped_data.get("found", False),
            # Product details section (nested structure)
            "product": {
                "name": scraped_data.get("product_name", normalized_query),
                "images": media_content.get("product_images", [])[:5],  # Top 5 images
                "overall_rating": scraped_data.get("rating", 0),
                "total_reviews": scraped_data.get("review_count", 0),
                "recommendation": recommendation,
                # NEW: Google Shopping data
                "price": shopping_data.get("price", {}),
                "brand": shopping_data.get("details", {}).get("brand", ""),
                "description": shopping_data.get("details", {}).get("description", ""),
                "merchants": shopping_data.get("merchants", [])[:5],  # Top 5 merchants
            },
            # Review consensus (AI-powered)
            "consensus": {
                "summary": (
                    ml_analysis.get("summary", "")
                    if ml_analysis
                    else "No reviews available for analysis."
                ),
                "sentiment": (
                    ml_analysis.get("sentiment_breakdown", {}) if ml_analysis else {}
                ),
                "pros": ml_analysis.get("pros", [])[:5] if ml_analysis else [],
                "cons": ml_analysis.get("cons", [])[:5] if ml_analysis else [],
                "key_points": (
                    ml_analysis.get("key_points", [])[:5] if ml_analysis else []
                ),
            },
            # Source breakdown
            "sources": source_breakdown,
            # YouTube review videos
            "review_videos": media_content.get("youtube_videos", [])[
                :5
            ],  # Top 5 videos
            # Community discussions
            "discussions": {
                "reddit": media_content.get("reddit_discussions", [])[:5],
                "twitter": media_content.get("twitter_posts", [])[:3],
            },
            # Citations (Perplexity-style)
            "citations": media_content.get("citations", []),
            # Metadata
            "metadata": {
                "from_cache": False,
                "scraped_at": scraped_data.get("fetched_at"),
                "response_time_ms": 0,  # Will be set by endpoint
            },
        }

        logger.info(f"Built result with keys: {list(result.keys())}")
        logger.info(f"Product section keys: {list(result.get('product', {}).keys())}")

        # Step 8: Cache the result
        self.cache[cache_key] = result

        return result

    def _calculate_source_breakdown(
        self, scraped_data: Dict, ml_analysis: Optional[Dict]
    ) -> Dict:
        """Calculate per-source statistics"""
        sources = scraped_data.get("sources", {})

        youtube_data = sources.get("youtube", {})
        reddit_data = sources.get("reddit", {})
        twitter_data = sources.get("twitter", {})

        return {
            "youtube": {
                "video_count": len(youtube_data.get("videos", [])),
                "total_views": sum(
                    v.get("view_count", 0) for v in youtube_data.get("videos", [])
                ),
                "avg_rating": 0.0,  # Will be calculated from engagement
                "has_data": len(youtube_data.get("videos", [])) > 0,
            },
            "reddit": {
                "post_count": len(reddit_data.get("posts", [])),
                "comment_count": reddit_data.get("total_comments", 0),
                "total_engagement": sum(
                    p.get("score", 0) for p in reddit_data.get("posts", [])
                ),
                "avg_rating": 0.0,
                "has_data": len(reddit_data.get("posts", [])) > 0,
            },
            "twitter": {
                "tweet_count": len(twitter_data.get("tweets", [])),
                "total_likes": sum(
                    t.get("like_count", 0) for t in twitter_data.get("tweets", [])
                ),
                "total_retweets": sum(
                    t.get("retweet_count", 0) for t in twitter_data.get("tweets", [])
                ),
                "avg_rating": 0.0,
                "has_data": len(twitter_data.get("tweets", [])) > 0,
            },
        }

    def _generate_recommendation(
        self, rating: float, review_count: int, ml_analysis: Optional[Dict]
    ) -> Dict:
        """Generate buy/wait/skip recommendation"""

        # Default recommendation
        decision = "insufficient_data"
        confidence = 0.0
        reason = "Not enough reviews to make a recommendation."

        if review_count >= 10:  # Minimum reviews threshold
            # Get sentiment if available
            positive_percent = 0
            if ml_analysis and ml_analysis.get("sentiment_breakdown"):
                sentiment = ml_analysis["sentiment_breakdown"]
                positive_percent = sentiment.get("positive_percent", 0)

            # Decision logic
            if rating >= 4.0 and positive_percent >= 70:
                decision = "buy"
                confidence = min(
                    95, 60 + (rating - 4) * 10 + (positive_percent - 70) / 3
                )
                reason = f"Highly rated ({rating}/5.0) with {positive_percent:.0f}% positive sentiment"

            elif rating >= 3.5 and positive_percent >= 60:
                decision = "consider"
                confidence = min(
                    85, 50 + (rating - 3.5) * 10 + (positive_percent - 60) / 4
                )
                reason = f"Good rating ({rating}/5.0) with mixed reviews"

            elif rating >= 3.0:
                decision = "wait"
                confidence = 70
                reason = f"Average rating ({rating}/5.0). Wait for more reviews or price drops"

            else:
                decision = "skip"
                confidence = 80
                reason = f"Low rating ({rating}/5.0) with negative sentiment"

        return {
            "decision": decision,  # buy, consider, wait, skip, insufficient_data
            "confidence": round(confidence, 1),
            "reason": reason,
            "badge": self._get_recommendation_badge(decision),
        }

    def _get_recommendation_badge(self, decision: str) -> str:
        """Get emoji/badge for recommendation"""
        badges = {
            "buy": "Recommended",
            "consider": "Worth Considering",
            "wait": "Wait for Updates",
            "skip": "Not Recommended",
            "insufficient_data": "Need More Data",
        }
        return badges.get(decision, "Unknown")

    async def _analyze_reviews(self, reviews: List[Dict]) -> Dict:
        """
        Analyze reviews with ML models

        Args:
            reviews: List of review texts

        Returns:
            ML analysis results
        """
        if not reviews:
            return None

        try:
            # Extract review texts
            review_texts = [r.get("text", "") for r in reviews if r.get("text")]

            if not review_texts:
                return None

            # Run sentiment analysis
            sentiments = self.sentiment_analyzer.analyze_batch(
                review_texts[:20]
            )  # Limit to 20

            # Aggregate sentiment
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for sentiment in sentiments:
                label = sentiment.get("sentiment_label", "neutral")
                sentiment_counts[label] += 1

            total = sum(sentiment_counts.values())
            sentiment_breakdown = {
                "positive": sentiment_counts["positive"],
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"],
                "positive_percent": (
                    (sentiment_counts["positive"] / total * 100) if total > 0 else 0
                ),
                "negative_percent": (
                    (sentiment_counts["negative"] / total * 100) if total > 0 else 0
                ),
                "neutral_percent": (
                    (sentiment_counts["neutral"] / total * 100) if total > 0 else 0
                ),
            }

            # Generate summary
            summary_result = self.summarizer.generate_structured_summary(
                review_texts[:15]
            )

            return {
                "sentiment_breakdown": sentiment_breakdown,
                "summary": summary_result.get("summary", ""),
                "pros": summary_result.get("pros", []),
                "cons": summary_result.get("cons", []),
                "key_points": summary_result.get("key_points", []),
                "analyzed_review_count": len(review_texts),
            }

        except Exception as e:
            logger.error(f"Error during ML analysis: {str(e)}")
            return None

    async def get_product_suggestions(
        self, partial_query: str, limit: int = 5
    ) -> List[str]:
        """
        Get product suggestions for autocomplete

        Args:
            partial_query: Partial search query
            limit: Maximum suggestions

        Returns:
            List of suggested product names
        """
        # TODO: Implement autocomplete based on popular searches
        # For now, return normalized query
        normalized = self.normalizer.normalize(partial_query)

        suggestions = []

        # Add common completions
        if "iphone" in normalized or "phone" in normalized:
            suggestions.extend(["iPhone 15 Pro Max", "iPhone 15", "iPhone 14 Pro"])
        elif "macbook" in normalized or "laptop" in normalized:
            suggestions.extend(["MacBook Pro", "MacBook Air", "Dell XPS 15"])
        elif "airpods" in normalized or "headphone" in normalized:
            suggestions.extend(["AirPods Pro", "Sony WH-1000XM5", "Bose QuietComfort"])

        return suggestions[:limit]

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        valid_cache_count = 0
        for cached_data in self.cache.values():
            if self._is_cache_valid(cached_data):
                valid_cache_count += 1

        return {
            "total_cached": len(self.cache),
            "valid_cached": valid_cache_count,
            "cache_ttl_hours": self.cache_ttl / 3600,
        }


# Singleton instance
_smart_product_service_instance = None


def get_smart_product_service() -> SmartProductService:
    """Get or create smart product service singleton"""
    global _smart_product_service_instance
    if _smart_product_service_instance is None:
        _smart_product_service_instance = SmartProductService()
    return _smart_product_service_instance
