"""
Twitter/X Sentiment Service
Analyzes product mentions and sentiment on Twitter/X
Uses tweepy (Twitter API v2)
"""

from typing import List, Dict, Optional
from loguru import logger
import tweepy
from datetime import datetime, timedelta


class TwitterService:
    """Twitter/X product sentiment analyzer"""

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
    ):
        """
        Initialize Twitter service

        Args:
            bearer_token: Twitter API v2 Bearer Token (for read-only)
            api_key: Twitter API key (for OAuth 1.0a)
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_secret: Twitter access secret

        Note: For basic search, bearer_token is sufficient
        """
        self.client = None

        # Try Bearer Token first (simpler, read-only)
        if bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=bearer_token)
                logger.info("Twitter API initialized (Bearer Token)")
                return
            except Exception as e:
                logger.error(f"Twitter Bearer Token init failed: {e}")

        # Fallback to OAuth 1.0a
        if api_key and api_secret and access_token and access_secret:
            try:
                self.client = tweepy.Client(
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_secret,
                )
                logger.info("Twitter API initialized (OAuth 1.0a)")
                return
            except Exception as e:
                logger.error(f"Twitter OAuth init failed: {e}")

        logger.warning("Twitter API not configured")

    def search_product_mentions(
        self, product_name: str, max_tweets: int = 100, days_back: int = 7
    ) -> List[Dict]:
        """
        Search for product mentions on Twitter

        Args:
            product_name: Product name to search
            max_tweets: Maximum number of tweets (free tier: 100/month)
            days_back: Number of days to look back

        Returns:
            List of tweet dictionaries

        Example:
            tweets = service.search_product_mentions("iPhone 15", max_tweets=50)
            for tweet in tweets:
                print(f"@{tweet['username']}: {tweet['text']}")
        """
        if not self.client:
            logger.error("Twitter API not initialized")
            return []

        try:
            logger.info(f"Searching Twitter for: {product_name}")

            # Build search query
            # Exclude retweets for original content only
            query = f'"{product_name}" -is:retweet lang:en'

            # Calculate start time (days back)
            start_time = datetime.utcnow() - timedelta(days=days_back)

            # Search tweets
            tweets_response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),  # API limit: 100
                tweet_fields=["created_at", "public_metrics", "author_id", "lang"],
                user_fields=["username", "verified", "public_metrics"],
                expansions=["author_id"],
                start_time=start_time,
            )

            if not tweets_response.data:
                logger.warning(f"No tweets found for '{product_name}'")
                return []

            # Extract user info
            users_dict = {}
            if tweets_response.includes and "users" in tweets_response.includes:
                for user in tweets_response.includes["users"]:
                    users_dict[user.id] = {
                        "username": user.username,
                        "verified": (
                            user.verified if hasattr(user, "verified") else False
                        ),
                        "followers": (
                            user.public_metrics["followers_count"]
                            if hasattr(user, "public_metrics")
                            else 0
                        ),
                    }

            # Process tweets
            tweets = []
            for tweet in tweets_response.data:
                user_info = users_dict.get(tweet.author_id, {})

                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": (
                        tweet.created_at.isoformat() if tweet.created_at else None
                    ),
                    "author_id": tweet.author_id,
                    "username": user_info.get("username", "unknown"),
                    "verified": user_info.get("verified", False),
                    "followers": user_info.get("followers", 0),
                    "retweet_count": (
                        tweet.public_metrics["retweet_count"]
                        if hasattr(tweet, "public_metrics")
                        else 0
                    ),
                    "reply_count": (
                        tweet.public_metrics["reply_count"]
                        if hasattr(tweet, "public_metrics")
                        else 0
                    ),
                    "like_count": (
                        tweet.public_metrics["like_count"]
                        if hasattr(tweet, "public_metrics")
                        else 0
                    ),
                    "url": f"https://twitter.com/{user_info.get('username', 'i')}/status/{tweet.id}",
                }

                # Calculate engagement score
                engagement = (
                    tweet_data["like_count"]
                    + tweet_data["retweet_count"] * 2
                    + tweet_data["reply_count"]
                )
                tweet_data["engagement_score"] = engagement

                tweets.append(tweet_data)

            # Sort by engagement
            tweets.sort(key=lambda x: x["engagement_score"], reverse=True)

            logger.info(f"Found {len(tweets)} tweets for '{product_name}'")

            return tweets

        except tweepy.TooManyRequests:
            logger.error("Twitter API rate limit exceeded")
            return []
        except tweepy.Forbidden:
            logger.error("Twitter API access forbidden - check credentials")
            return []
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
            return []

    def get_product_sentiment(self, product_name: str, max_tweets: int = 100) -> Dict:
        """
        Get product sentiment from Twitter mentions

        Args:
            product_name: Product name
            max_tweets: Maximum tweets to analyze

        Returns:
            Dictionary with tweets and aggregate sentiment

        Example:
            result = service.get_product_sentiment("iPhone 15")
            print(f"Total tweets: {result['total_tweets']}")
            print(f"Avg engagement: {result['avg_engagement']}")
        """
        tweets = self.search_product_mentions(product_name, max_tweets=max_tweets)

        if not tweets:
            return {
                "product_name": product_name,
                "tweets": [],
                "total_tweets": 0,
                "avg_engagement": 0,
                "top_influencers": [],
                "source": "twitter",
            }

        # Calculate statistics
        total_engagement = sum(t["engagement_score"] for t in tweets)
        avg_engagement = total_engagement / len(tweets) if tweets else 0

        # Find top influencers (verified or high followers)
        influencers = [
            {
                "username": t["username"],
                "followers": t["followers"],
                "verified": t["verified"],
                "tweet": t["text"][:100],
            }
            for t in tweets
            if t["verified"] or t["followers"] > 10000
        ]
        influencers = sorted(influencers, key=lambda x: x["followers"], reverse=True)[
            :5
        ]

        return {
            "product_name": product_name,
            "tweets": tweets,
            "total_tweets": len(tweets),
            "avg_engagement": round(avg_engagement, 2),
            "total_likes": sum(t["like_count"] for t in tweets),
            "total_retweets": sum(t["retweet_count"] for t in tweets),
            "top_influencers": influencers,
            "source": "twitter",
        }

    def extract_tweet_texts(self, sentiment_data: Dict) -> List[str]:
        """
        Extract tweet texts for sentiment analysis

        Args:
            sentiment_data: Dictionary from get_product_sentiment()

        Returns:
            List of tweet texts

        Example:
            data = service.get_product_sentiment("iPhone 15")
            texts = service.extract_tweet_texts(data)
            # Pass to sentiment analyzer
        """
        tweets = sentiment_data.get("tweets", [])
        return [tweet["text"] for tweet in tweets if tweet.get("text")]


# Singleton instance
_twitter_service_instance: Optional[TwitterService] = None


def get_twitter_service(
    bearer_token: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    access_token: Optional[str] = None,
    access_secret: Optional[str] = None,
) -> TwitterService:
    """
    Get or create Twitter service singleton

    Args:
        bearer_token: Twitter API v2 Bearer Token
        api_key: Twitter API key
        api_secret: Twitter API secret
        access_token: Twitter access token
        access_secret: Twitter access secret

    Returns:
        TwitterService instance
    """
    global _twitter_service_instance
    if _twitter_service_instance is None:
        _twitter_service_instance = TwitterService(
            bearer_token=bearer_token,
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            access_secret=access_secret,
        )
    return _twitter_service_instance
