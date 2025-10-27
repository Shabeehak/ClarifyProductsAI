"""
Real-Time Product Scraper Service - PRODUCTION VERSION
Fetches product data from YouTube, Reddit, and Twitter (NO DATABASE STORAGE)
All data cached in Redis with 24-hour TTL
"""

from typing import List, Dict, Optional
import asyncio
from loguru import logger
from datetime import datetime

from app.core.config import settings
from app.services.youtube_service import get_youtube_service
from app.services.reddit_service import get_reddit_service
from app.services.twitter_service import get_twitter_service
from app.services.cache_service import get_cache_service


class RealtimeScraperService:
    """Real-time multi-source product data scraper"""

    def __init__(self):
        """Initialize scraper service with all APIs"""
        # Initialize services
        self.youtube = None
        self.reddit = None
        self.twitter = None
        self.cache = None

        # YouTube service
        if settings.YOUTUBE_API_KEY:
            self.youtube = get_youtube_service(settings.YOUTUBE_API_KEY)
            logger.info("[OK] YouTube service initialized")
        else:
            logger.warning("YouTube API key not configured")

        # Reddit service
        if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
            self.reddit = get_reddit_service(
                settings.REDDIT_CLIENT_ID,
                settings.REDDIT_CLIENT_SECRET,
                settings.REDDIT_USER_AGENT,
            )
            logger.info("[OK] Reddit service initialized")
        else:
            logger.warning("Reddit API credentials not configured")

        # Twitter service
        if settings.TWITTER_BEARER_TOKEN:
            self.twitter = get_twitter_service(
                bearer_token=settings.TWITTER_BEARER_TOKEN
            )
            logger.info("[OK] Twitter service initialized")
        else:
            logger.warning("Twitter Bearer Token not configured")

        # Cache service
        self.cache = get_cache_service(settings.REDIS_URL)
        logger.info("Real-time scraper service initialized")

    async def init_cache(self):
        """Initialize cache connection"""
        if self.cache:
            await self.cache.connect()

    async def close_cache(self):
        """Close cache connection"""
        if self.cache:
            await self.cache.disconnect()

    async def search_product(
        self,
        product_name: str,
        max_videos: int = 5,
        max_reddit_posts: int = 10,
        max_tweets: int = 50,
    ) -> Dict:
        """
        Search for product across all sources (YouTube, Reddit, Twitter)

        Args:
            product_name: Product name to search
            max_videos: Maximum YouTube videos
            max_reddit_posts: Maximum Reddit discussions
            max_tweets: Maximum tweets

        Returns:
            Comprehensive product data from all sources

        Example:
            result = await scraper.search_product("iPhone 15")
            print(f"Videos: {len(result['youtube']['videos'])}")
            print(f"Reddit posts: {len(result['reddit']['posts'])}")
            print(f"Tweets: {len(result['twitter']['tweets'])}")
        """
        logger.info(f"=== Searching for product: {product_name} ===")

        # Check cache first
        cached_data = await self.cache.get_product(product_name)
        if cached_data:
            logger.info(f"[CACHE HIT] Returning cached data for '{product_name}'")
            return cached_data

        logger.info(f"[CACHE MISS] Fetching fresh data for '{product_name}'")

        # Fetch from all sources in parallel
        tasks = []

        if self.youtube:
            tasks.append(self._fetch_youtube(product_name, max_videos))
        else:
            tasks.append(self._return_empty("youtube"))

        if self.reddit:
            tasks.append(self._fetch_reddit(product_name, max_reddit_posts))
        else:
            tasks.append(self._return_empty("reddit"))

        if self.twitter:
            tasks.append(self._fetch_twitter(product_name, max_tweets))
        else:
            tasks.append(self._return_empty("twitter"))

        # Execute all fetches in parallel
        youtube_data, reddit_data, twitter_data = await asyncio.gather(*tasks)

        # Aggregate results
        product_data = {
            "product_name": product_name,
            "found": any(
                [
                    youtube_data.get("videos"),
                    reddit_data.get("posts"),
                    twitter_data.get("tweets"),
                ]
            ),
            "sources": {
                "youtube": youtube_data,
                "reddit": reddit_data,
                "twitter": twitter_data,
            },
            "fetched_at": datetime.utcnow().isoformat(),
            "cache_ttl_hours": 24,
        }

        # Calculate summary statistics
        product_data["summary"] = {
            "total_youtube_videos": len(youtube_data.get("videos", [])),
            "videos_with_transcripts": youtube_data.get("videos_with_transcripts", 0),
            "total_reddit_posts": len(reddit_data.get("posts", [])),
            "total_reddit_comments": reddit_data.get("total_comments", 0),
            "total_tweets": len(twitter_data.get("tweets", [])),
            "twitter_avg_engagement": twitter_data.get("avg_engagement", 0),
        }

        # Calculate rating and review count
        total_reviews = 0
        total_rating = 0
        rating_count = 0

        # Count YouTube videos as reviews
        youtube_videos = len(youtube_data.get("videos", []))
        total_reviews += youtube_videos

        # Count Reddit posts and comments as reviews
        reddit_posts = len(reddit_data.get("posts", []))
        reddit_comments = reddit_data.get("total_comments", 0)
        total_reviews += reddit_posts + reddit_comments

        # Count tweets as reviews
        tweets = len(twitter_data.get("tweets", []))
        total_reviews += tweets

        # Calculate average rating from engagement scores
        # YouTube engagement (views/likes ratio)
        for video in youtube_data.get("videos", []):
            engagement = video.get("engagement_score", 0)
            if engagement > 0:
                # Convert engagement to 1-5 scale (engagement_score is typically 0-10)
                rating = min(5.0, max(1.0, engagement / 2))
                total_rating += rating
                rating_count += 1

        # Reddit engagement (score/upvote_ratio)
        for post in reddit_data.get("posts", []):
            score = post.get("score", 0)
            upvote_ratio = post.get("upvote_ratio", 0.5)
            if score > 0:
                # Convert to 1-5 scale based on score and ratio
                rating = 1.0 + (upvote_ratio * 4)  # 0.0-1.0 ratio -> 1.0-5.0
                total_rating += rating
                rating_count += 1

        # Twitter engagement (likes/retweets)
        for tweet in twitter_data.get("tweets", []):
            likes = tweet.get("like_count", 0)
            if likes > 0:
                # Simple engagement to rating conversion
                rating = min(5.0, 3.0 + (likes / 100))  # Base 3, up to 5
                total_rating += rating
                rating_count += 1

        # Calculate final average rating
        average_rating = (total_rating / rating_count) if rating_count > 0 else 0.0

        # Add to product data
        product_data["rating"] = round(average_rating, 2)
        product_data["review_count"] = total_reviews

        # Cache for 24 hours
        await self.cache.set_product(product_name, product_data, ttl=86400)

        logger.info(f"=== Product search complete: {product_name} ===")
        logger.info(
            f"  YouTube: {product_data['summary']['total_youtube_videos']} videos"
        )
        logger.info(f"  Reddit: {product_data['summary']['total_reddit_posts']} posts")
        logger.info(f"  Twitter: {product_data['summary']['total_tweets']} tweets")
        logger.info(
            f"  Rating: {product_data['rating']}/5.0 from {rating_count} sources"
        )
        logger.info(f"  Total Reviews: {product_data['review_count']}")

        return product_data

    async def _fetch_youtube(self, product_name: str, max_videos: int) -> Dict:
        """Fetch YouTube video reviews"""
        try:
            logger.info(f"Fetching YouTube data for '{product_name}'...")
            result = self.youtube.get_product_reviews(
                product_name, max_videos=max_videos
            )
            logger.info(f"[OK] YouTube: {len(result.get('videos', []))} videos fetched")
            return result
        except Exception as e:
            logger.error(f"YouTube fetch error: {e}")
            return {
                "videos": [],
                "total_videos": 0,
                "videos_with_transcripts": 0,
                "source": "youtube",
                "error": str(e),
            }

    async def _fetch_reddit(self, product_name: str, max_posts: int) -> Dict:
        """Fetch Reddit discussions"""
        try:
            logger.info(f"Fetching Reddit data for '{product_name}'...")
            # Run in thread pool (PRAW is synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.reddit.get_product_discussions_with_comments,
                product_name,
                max_posts,
                20,  # max comments per post
            )
            logger.info(f"[OK] Reddit: {len(result.get('posts', []))} posts fetched")
            return result
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return {
                "posts": [],
                "total_posts": 0,
                "total_comments": 0,
                "source": "reddit",
                "error": str(e),
            }

    async def _fetch_twitter(self, product_name: str, max_tweets: int) -> Dict:
        """Fetch Twitter mentions"""
        try:
            logger.info(f"Fetching Twitter data for '{product_name}'...")
            # Run in thread pool (tweepy might block)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.twitter.get_product_sentiment, product_name, max_tweets
            )
            logger.info(f"[OK] Twitter: {len(result.get('tweets', []))} tweets fetched")
            return result
        except Exception as e:
            logger.error(f"Twitter fetch error: {e}")
            return {
                "tweets": [],
                "total_tweets": 0,
                "avg_engagement": 0,
                "source": "twitter",
                "error": str(e),
            }

    async def _return_empty(self, source: str) -> Dict:
        """Return empty result for disabled source"""
        return {
            "source": source,
            "error": f"{source} API not configured",
            "videos": [] if source == "youtube" else None,
            "posts": [] if source == "reddit" else None,
            "tweets": [] if source == "twitter" else None,
        }

    def extract_all_reviews(self, product_data: Dict) -> List[str]:
        """
        Extract all review texts from all sources for ML analysis

        Args:
            product_data: Dictionary from search_product()

        Returns:
            List of review/opinion texts

        Example:
            data = await scraper.search_product("iPhone 15")
            reviews = scraper.extract_all_reviews(data)
            # Pass to sentiment analyzer
        """
        reviews = []

        # Extract YouTube transcripts
        youtube_data = product_data.get("sources", {}).get("youtube", {})
        for video in youtube_data.get("videos", []):
            if video.get("transcript"):
                reviews.append(
                    f"YouTube Review: {video['transcript'][:2000]}"
                )  # Limit to 2000 chars

        # Extract Reddit posts and comments
        reddit_data = product_data.get("sources", {}).get("reddit", {})
        for post in reddit_data.get("posts", []):
            if post.get("title"):
                reviews.append(f"Reddit: {post['title']}")
            if post.get("selftext"):
                reviews.append(f"Reddit: {post['selftext'][:500]}")

            for comment in post.get("comments", []):
                if comment.get("body"):
                    reviews.append(f"Reddit Comment: {comment['body'][:300]}")

        # Extract tweets
        twitter_data = product_data.get("sources", {}).get("twitter", {})
        for tweet in twitter_data.get("tweets", []):
            if tweet.get("text"):
                reviews.append(f"Tweet: {tweet['text']}")

        logger.info(f"Extracted {len(reviews)} total reviews for ML analysis")

        return reviews

    def extract_media_content(self, product_data: Dict) -> Dict:
        """
        Extract media content (images, video links, citations) from search results
        Similar to how Perplexity shows sources and images

        Args:
            product_data: Result from search_product()

        Returns:
            Dict with product_images, youtube_videos, citations

        Example:
            data = await scraper.search_product("iPhone 15")
            media = scraper.extract_media_content(data)
            # Returns: {"product_images": [...], "youtube_videos": [...], "citations": [...]}
        """
        media_content = {
            "product_images": [],
            "youtube_videos": [],
            "reddit_discussions": [],
            "twitter_posts": [],
            "citations": [],
        }

        citation_index = 1

        # Extract YouTube video data
        youtube_data = product_data.get("sources", {}).get("youtube", {})
        for video in youtube_data.get("videos", []):
            video_info = {
                "video_id": video.get("video_id"),
                "title": video.get("title"),
                "url": f"https://www.youtube.com/watch?v={video.get('video_id')}",
                "thumbnail": video.get("thumbnail"),
                "channel": video.get("channel"),
                "views": video.get("view_count", 0),
                "duration_seconds": video.get("duration_seconds", 0),
                "published_at": video.get("published_at"),
                "has_transcript": video.get("has_transcript", False),
                "citation_number": citation_index,
            }
            media_content["youtube_videos"].append(video_info)

            # Use YouTube thumbnails as product images
            if video.get("thumbnail"):
                media_content["product_images"].append(
                    {
                        "url": video.get("thumbnail"),
                        "source": "youtube",
                        "source_title": video.get("title"),
                        "alt_text": f"{video.get('title')} - YouTube Review",
                    }
                )

            # Add to citations
            media_content["citations"].append(
                {
                    "number": citation_index,
                    "type": "youtube",
                    "title": video.get("title"),
                    "url": f"https://www.youtube.com/watch?v={video.get('video_id')}",
                    "source": video.get("channel"),
                    "thumbnail": video.get("thumbnail"),
                }
            )
            citation_index += 1

        # Extract Reddit discussion links
        reddit_data = product_data.get("sources", {}).get("reddit", {})
        for post in reddit_data.get("posts", []):
            discussion_info = {
                "title": post.get("title"),
                "url": post.get("url"),
                "subreddit": post.get("subreddit"),
                "score": post.get("score", 0),
                "comment_count": len(post.get("comments", [])),
                "created_at": post.get("created_at"),
                "citation_number": citation_index,
            }
            media_content["reddit_discussions"].append(discussion_info)

            # Add to citations
            media_content["citations"].append(
                {
                    "number": citation_index,
                    "type": "reddit",
                    "title": post.get("title"),
                    "url": post.get("url"),
                    "source": f"r/{post.get('subreddit')}",
                    "score": post.get("score", 0),
                }
            )
            citation_index += 1

        # Extract Twitter posts
        twitter_data = product_data.get("sources", {}).get("twitter", {})
        for tweet in twitter_data.get("tweets", []):
            tweet_info = {
                "text": tweet.get("text"),
                "username": tweet.get("username"),
                "url": f"https://twitter.com/{tweet.get('username')}/status/{tweet.get('id')}",
                "likes": tweet.get("like_count", 0),
                "retweets": tweet.get("retweet_count", 0),
                "created_at": tweet.get("created_at"),
                "citation_number": citation_index,
            }
            media_content["twitter_posts"].append(tweet_info)

            # Add to citations (only for high-engagement tweets)
            if tweet.get("like_count", 0) > 10:
                media_content["citations"].append(
                    {
                        "number": citation_index,
                        "type": "twitter",
                        "title": f"Tweet from @{tweet.get('username')}",
                        "url": f"https://twitter.com/{tweet.get('username')}/status/{tweet.get('id')}",
                        "source": f"@{tweet.get('username')}",
                        "engagement": tweet.get("like_count", 0)
                        + tweet.get("retweet_count", 0),
                    }
                )
                citation_index += 1

        logger.info(
            f"Extracted media: {len(media_content['product_images'])} images, "
            f"{len(media_content['youtube_videos'])} videos, "
            f"{len(media_content['citations'])} citations"
        )

        return media_content

    async def get_product_details(
        self, product_name: str, sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Get detailed product information from specific sources

        Args:
            product_name: Product name
            sources: List of sources (default: all)
                    Options: ["youtube", "reddit", "twitter"]

        Returns:
            Product data from requested sources
        """
        if sources is None:
            sources = ["youtube", "reddit", "twitter"]

        # Determine max results based on sources
        max_videos = 5 if "youtube" in sources else 0
        max_posts = 10 if "reddit" in sources else 0
        max_tweets = 50 if "twitter" in sources else 0

        return await self.search_product(
            product_name,
            max_videos=max_videos,
            max_reddit_posts=max_posts,
            max_tweets=max_tweets,
        )

    async def invalidate_cache(self, product_name: str):
        """
        Force refresh product data

        Args:
            product_name: Product name to invalidate

        Use case: User reports stale data
        """
        if self.cache:
            await self.cache.invalidate_product(product_name)
            logger.info(f"Cache invalidated for '{product_name}'")


# Singleton instance
_scraper_service_instance: Optional[RealtimeScraperService] = None


def get_scraper_service() -> RealtimeScraperService:
    """
    Get or create realtime scraper service singleton

    Returns:
        RealtimeScraperService instance
    """
    global _scraper_service_instance
    if _scraper_service_instance is None:
        _scraper_service_instance = RealtimeScraperService()
    return _scraper_service_instance
