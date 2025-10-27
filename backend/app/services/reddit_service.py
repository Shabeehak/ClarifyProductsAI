"""
Reddit Discussion Service
Scrapes product discussions from relevant subreddits
Uses PRAW (Python Reddit API Wrapper)
"""

from typing import List, Dict, Optional
from loguru import logger
import praw
from datetime import datetime, timedelta


class RedditService:
    """Reddit product discussion scraper"""

    # Relevant subreddits for product discussions
    PRODUCT_SUBREDDITS = [
        "gadgets",
        "technology",
        "reviews",
        "BuyItForLife",
        "ProductPorn",
        "shutupandtakemymoney",
        "buyitforlife",
        "androidapps",
        "iphone",
        "android",
        "headphones",
        "buildapc",
        "mechanicalkeyboards",
    ]

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit service

        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: User agent string (e.g., "ClarifyProducts.AI/1.0")
        """
        self.reddit = None

        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                # Test connection
                self.reddit.user.me()
                logger.info("Reddit API initialized (read-only mode)")
            except Exception as e:
                logger.error(f"Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit API credentials not configured")

    def search_product_discussions(
        self,
        product_name: str,
        max_posts: int = 20,
        time_filter: str = "month",  # hour, day, week, month, year, all
    ) -> List[Dict]:
        """
        Search for product discussions across subreddits

        Args:
            product_name: Product name to search
            max_posts: Maximum number of posts to fetch
            time_filter: Time filter (month, week, year, all)

        Returns:
            List of discussion posts

        Example:
            posts = service.search_product_discussions("iPhone 15", max_posts=10)
            for post in posts:
                print(f"{post['title']} - {post['score']} upvotes")
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []

        all_posts = []

        try:
            logger.info(f"Searching Reddit for: {product_name}")

            # Search across all relevant subreddits
            for subreddit_name in self.PRODUCT_SUBREDDITS:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Search within subreddit
                    search_results = subreddit.search(
                        product_name,
                        time_filter=time_filter,
                        limit=max_posts // len(self.PRODUCT_SUBREDDITS) + 1,
                    )

                    for submission in search_results:
                        post = {
                            "id": submission.id,
                            "title": submission.title,
                            "subreddit": submission.subreddit.display_name,
                            "author": (
                                str(submission.author)
                                if submission.author
                                else "[deleted]"
                            ),
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "created_utc": datetime.fromtimestamp(
                                submission.created_utc
                            ).isoformat(),
                            "url": f"https://reddit.com{submission.permalink}",
                            "selftext": (
                                submission.selftext[:1000]
                                if submission.selftext
                                else ""
                            ),  # Limit to 1000 chars
                            "is_self": submission.is_self,
                            "link_flair_text": submission.link_flair_text,
                        }

                        all_posts.append(post)

                except Exception as e:
                    logger.warning(f"Error searching r/{subreddit_name}: {e}")
                    continue

            # Sort by score (upvotes) descending
            all_posts.sort(key=lambda x: x["score"], reverse=True)

            # Limit to max_posts
            all_posts = all_posts[:max_posts]

            logger.info(f"Found {len(all_posts)} Reddit posts for '{product_name}'")

            return all_posts

        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []

    def get_post_comments(
        self, post_id: str, max_comments: int = 50, min_score: int = 1
    ) -> List[Dict]:
        """
        Get comments from a Reddit post

        Args:
            post_id: Reddit post ID
            max_comments: Maximum number of comments to fetch
            min_score: Minimum comment score (filter low-quality comments)

        Returns:
            List of comment dictionaries

        Example:
            comments = service.get_post_comments("abc123", max_comments=20)
            for comment in comments:
                print(f"{comment['author']}: {comment['body'][:100]}")
        """
        if not self.reddit:
            return []

        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(
                limit=0
            )  # Remove "More comments" placeholders

            comments = []

            for comment in submission.comments.list()[:max_comments]:
                # Filter by score
                if comment.score < min_score:
                    continue

                comment_data = {
                    "id": comment.id,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "body": comment.body[:500],  # Limit to 500 chars
                    "score": comment.score,
                    "created_utc": datetime.fromtimestamp(
                        comment.created_utc
                    ).isoformat(),
                    "is_submitter": comment.is_submitter,
                    "parent_id": comment.parent_id,
                }

                comments.append(comment_data)

            # Sort by score
            comments.sort(key=lambda x: x["score"], reverse=True)

            return comments[:max_comments]

        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            return []

    def get_product_discussions_with_comments(
        self, product_name: str, max_posts: int = 10, max_comments_per_post: int = 20
    ) -> Dict:
        """
        Get product discussions with top comments

        Args:
            product_name: Product name
            max_posts: Maximum number of posts
            max_comments_per_post: Maximum comments per post

        Returns:
            Dictionary with posts and their comments

        Example:
            result = service.get_product_discussions_with_comments("iPhone 15")
            print(f"Found {len(result['posts'])} discussions")
            for post in result['posts']:
                print(f"{post['title']}: {len(post['comments'])} comments")
        """
        # Get posts
        posts = self.search_product_discussions(product_name, max_posts=max_posts)

        if not posts:
            return {
                "product_name": product_name,
                "posts": [],
                "total_posts": 0,
                "total_comments": 0,
                "source": "reddit",
            }

        total_comments = 0

        # Get comments for each post
        for post in posts:
            comments = self.get_post_comments(
                post["id"], max_comments=max_comments_per_post
            )

            post["comments"] = comments
            post["comment_count_fetched"] = len(comments)
            total_comments += len(comments)

        logger.info(f"Fetched {total_comments} total comments from {len(posts)} posts")

        return {
            "product_name": product_name,
            "posts": posts,
            "total_posts": len(posts),
            "total_comments": total_comments,
            "source": "reddit",
        }

    def extract_user_opinions(self, discussions: Dict) -> List[str]:
        """
        Extract all user opinions from posts and comments

        Args:
            discussions: Dictionary from get_product_discussions_with_comments()

        Returns:
            List of opinion texts (for sentiment analysis)

        Example:
            discussions = service.get_product_discussions_with_comments("iPhone 15")
            opinions = service.extract_user_opinions(discussions)
            print(f"Extracted {len(opinions)} opinions")
        """
        opinions = []

        for post in discussions.get("posts", []):
            # Add post title and text
            if post.get("title"):
                opinions.append(post["title"])

            if post.get("selftext"):
                opinions.append(post["selftext"])

            # Add comments
            for comment in post.get("comments", []):
                if comment.get("body"):
                    opinions.append(comment["body"])

        logger.info(f"Extracted {len(opinions)} opinions from Reddit discussions")

        return opinions

    def get_trending_products(
        self, subreddit: str = "gadgets", time_filter: str = "week", limit: int = 10
    ) -> List[Dict]:
        """
        Get trending product discussions from a subreddit

        Args:
            subreddit: Subreddit name (default: 'gadgets')
            time_filter: Time filter (day, week, month)
            limit: Number of posts

        Returns:
            List of trending posts
        """
        if not self.reddit:
            return []

        try:
            sub = self.reddit.subreddit(subreddit)

            posts = []
            for submission in sub.top(time_filter=time_filter, limit=limit):
                post = {
                    "title": submission.title,
                    "subreddit": submission.subreddit.display_name,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "url": f"https://reddit.com{submission.permalink}",
                    "created_utc": datetime.fromtimestamp(
                        submission.created_utc
                    ).isoformat(),
                }
                posts.append(post)

            return posts

        except Exception as e:
            logger.error(f"Error fetching trending posts: {e}")
            return []


# Singleton instance
_reddit_service_instance: Optional[RedditService] = None


def get_reddit_service(
    client_id: str, client_secret: str, user_agent: str
) -> RedditService:
    """
    Get or create Reddit service singleton

    Args:
        client_id: Reddit app client ID
        client_secret: Reddit app client secret
        user_agent: User agent string

    Returns:
        RedditService instance
    """
    global _reddit_service_instance
    if _reddit_service_instance is None:
        _reddit_service_instance = RedditService(client_id, client_secret, user_agent)
    return _reddit_service_instance
