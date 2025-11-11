"""
RAG (Retrieval-Augmented Generation) Service
Orchestrates real-time review fetching and generation with LLM
"""

from typing import List, Dict, Optional
from loguru import logger
import requests
import hashlib
import json

from app.services.smart_product_service import get_smart_product_service
from app.services.llm_service import get_llm_service
from app.core.config import settings
from app.db.redis_client import get_redis_client


class RAGService:
    """Service for RAG pipeline: Retrieve → Augment → Generate"""

    def __init__(self):
        """Initialize RAG service"""
        self.product_service = get_smart_product_service()
        self.llm_service = get_llm_service()
        self.redis_client = get_redis_client()
        self.cache_ttl = settings.CACHE_TTL_SECONDS  # 24 hours
        logger.info("RAG service initialized with LLM support and Redis caching")

    # ADD NEW METHOD:
    async def retrieve_context_realtime(
        self, query: str, n_results: int = 10
    ) -> List[Dict]:
        """
        Retrieve real-time reviews for product in query

        Args:
            query: User query (e.g., "best moisturizer for dry skin")
            n_results: Max reviews to return

        Returns:
        List of reviews from YouTube, Reddit, Twitter
        """
        try:
            logger.info(f"Fetching real-time data for: {query}")

            # Use smart product service to get real-time data
            product_data = await self.product_service.search_product(
                query=query,
                use_cache=True,  # Use cache for performance
                include_ml_analysis=True,  # Include sentiment analysis
            )

            if not product_data.get("found"):
                logger.warning(f"No product data found for: {query}")
                return []

            # Extract reviews from all sources
            reviews = []

            # YouTube reviews (from review_videos)
            youtube_videos = product_data.get("review_videos", [])
            if isinstance(youtube_videos, list):
                for video in youtube_videos[:n_results]:
                    if not isinstance(video, dict):
                        continue
                    # Use transcript if available, otherwise use title + description
                    text = video.get("transcript", "")
                    if not text:
                        text = (
                            f"{video.get('title', '')}. {video.get('description', '')}"
                        )

                    if text.strip():
                        reviews.append(
                            {
                                "id": f"yt_{video.get('video_id', '')}",
                                "text": text,
                                "metadata": {
                                    "source": "YouTube",
                                    "channel": video.get("channel", "Unknown"),
                                    "views": video.get("views", 0),
                                    "sentiment": video.get("sentiment", "neutral"),
                                    "rating": video.get("sentiment_score", 0) * 5,
                                },
                            }
                        )

            # Reddit reviews (from discussions)
            reddit_posts = product_data.get("discussions", [])
            if isinstance(reddit_posts, list):
                for post in reddit_posts[:n_results]:
                    if not isinstance(post, dict):
                        continue
                    # Combine title, body, and top comments
                    text_parts = []
                    if post.get("title"):
                        text_parts.append(post["title"])
                    if post.get("text"):
                        text_parts.append(post["text"])

                    # Add top comments
                    comments = post.get("comments", [])
                    if isinstance(comments, list):
                        for comment in comments[:3]:
                            if isinstance(comment, dict) and comment.get("text"):
                                text_parts.append(comment["text"])

                    text = " ".join(text_parts)
                    if text.strip():
                        reviews.append(
                            {
                                "id": f"reddit_{post.get('id', '')}",
                                "text": text,
                                "metadata": {
                                    "source": "Reddit",
                                    "subreddit": post.get("subreddit", ""),
                                    "score": post.get("score", 0),
                                    "sentiment": post.get("sentiment", "neutral"),
                                    "rating": post.get("sentiment_score", 3.0),
                                },
                            }
                        )

            # Twitter reviews (from sources.twitter if exists)
            sources = product_data.get("sources", {})
            twitter_data = sources.get("twitter", {})
            twitter_tweets = twitter_data.get("tweets", [])
            if isinstance(twitter_tweets, list):
                for tweet in twitter_tweets[:n_results]:
                    if not isinstance(tweet, dict):
                        continue
                    if tweet.get("text"):
                        reviews.append(
                            {
                                "id": f"twitter_{tweet.get('id', '')}",
                                "text": tweet.get("text", ""),
                                "metadata": {
                                    "source": "Twitter",
                                    "likes": tweet.get("likes", 0),
                                    "sentiment": tweet.get("sentiment", "neutral"),
                                    "rating": tweet.get("sentiment_score", 3.0),
                                },
                            }
                        )

            logger.info(f"Retrieved {len(reviews)} real-time reviews from product_data")
            logger.debug(f"Review IDs: {[r.get('id', 'no-id') for r in reviews[:5]]}")

            # Ensure reviews is a list before slicing
            if not isinstance(reviews, list):
                logger.error(f"Reviews is not a list, got type: {type(reviews)}")
                return []

            return reviews[:n_results] if len(reviews) > n_results else reviews

        except Exception as e:
            logger.error(f"Error fetching real-time context: {str(e)}", exc_info=True)
            return []

    def generate_response(
        self, query: str, context: List[Dict], model: str = "llama2"
    ) -> str:
        """
        Generate response using LLM with retrieved context

        Args:
            query: User query
            context: Retrieved reviews/context
            model: LLM model to use (deprecated, uses multi-LLM service)

        Returns:
            Generated response
        """
        try:
            # Build context string from retrieved reviews
            context_text = self._format_context(context)

            # Create prompt with system instruction
            system_prompt = """You are an expert product review analyst and shopping assistant.
Your role is to provide comprehensive, detailed, and helpful answers based on real user reviews.
Always be thorough, balanced, and cite specific review details."""

            user_prompt = f"""Context (Real User Reviews):
{context_text}

User Question: {query}

Instructions:
1. Provide a DETAILED and COMPREHENSIVE answer based on the reviews above
2. Always cite review sources (e.g., "According to Review 1...", "Reviewers mention...")
3. Cover multiple aspects: features, pros, cons, user experiences, value for money
4. If comparing products, provide detailed side-by-side analysis
5. Mention specific details from reviews (ratings, experiences, recommendations)
6. Present both positive and negative feedback fairly
7. Conclude with a clear recommendation or summary
8. Write 3-5 paragraphs for thorough coverage
9. Be conversational, engaging, and helpful

Detailed Answer:"""

            logger.info(
                f"Generating response with {len(context)} context items using multi-LLM..."
            )

            # Use multi-LLM service (auto fallback)
            response = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.7,
            )

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.warning("All LLM providers failed, using extractive fallback")
            return self._extractive_fallback(query, context)

    def _generate_cache_key(self, query: str, n_results: int) -> str:
        """Generate cache key from query and parameters"""
        # Normalize query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        # Create hash for cache key
        cache_data = f"{normalized}:{n_results}"
        hash_key = hashlib.md5(cache_data.encode()).hexdigest()
        return f"rag:query:{hash_key}"

    async def rag_query(
        self,
        query: str,
        product_id: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict:
        """
        Complete RAG pipeline: Retrieve + Generate (with Redis caching)

        Args:
            query: User query
            product_id: Optional product filter
            n_results: Number of contexts to retrieve

        Returns:
            Dict with response and sources
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, n_results)
            cached_result = self.redis_client.get(cache_key)

            if cached_result:
                logger.info(f"✅ Cache HIT for RAG query: {query[:50]}...")
                try:
                    return json.loads(cached_result)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode cached RAG result, regenerating")

            logger.info(f"❌ Cache MISS for RAG query: {query[:50]}...")

            # Step 1: Retrieve real-time context
            context = await self.retrieve_context_realtime(query, n_results)

            if not context:
                result = {
                    "response": "I couldn't find recent reviews for this product. Could you provide more details or try a different product name?",
                    "sources": [],
                    "context_count": 0,
                    "realtime": True,
                    "cached": False,
                }
                # Don't cache "not found" results
                return result

            # Step 2: Generate response with Gemini
            response = self.generate_response(query, context)

            # Step 3: Build result
            result = {
                "response": response,
                "sources": [
                    {
                        "id": c["id"],
                        "text": c["text"][:200] + "...",
                        "metadata": c["metadata"],
                    }
                    for c in context[:5]
                ],
                "context_count": len(context),
                "realtime": True,
                "cached": False,
                "sources_breakdown": {
                    "youtube": len(
                        [c for c in context if c["metadata"]["source"] == "YouTube"]
                    ),
                    "reddit": len(
                        [c for c in context if c["metadata"]["source"] == "Reddit"]
                    ),
                    "twitter": len(
                        [c for c in context if c["metadata"]["source"] == "Twitter"]
                    ),
                },
            }

            # Cache the successful result for 24 hours
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(result)
                )
                logger.info(f"💾 Cached RAG result for: {query[:50]}...")
            except Exception as cache_error:
                logger.warning(f"Failed to cache RAG result: {cache_error}")

            return result

        except Exception as e:
            logger.error(f"Error in real-time RAG query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "context_count": 0,
                "realtime": False,
                "cached": False,
            }

    def _format_context(self, context: List[Dict]) -> str:
        """Format retrieved context for prompt"""
        formatted = []
        for i, item in enumerate(context, 1):
            text = item["text"]
            metadata = item.get("metadata", {})
            rating = metadata.get("rating", "N/A")
            source = metadata.get("source", "Unknown")

            formatted.append(
                f"Review {i} (Rating: {rating}, Source: {source}):\n{text}\n"
            )

        return "\n".join(formatted)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""You are a helpful product review assistant. Answer the user's question based on the provided product reviews.

Context (Product Reviews):
{context}

User Question: {query}

Instructions:
1. Answer based ONLY on the provided reviews
2. Be concise and specific
3. Mention positive and negative aspects if relevant
4. If reviews are mixed, present both sides
5. Cite the review numbers when making claims

Answer:"""

        return prompt

    def _extractive_fallback(self, query: str, context: List[Dict]) -> str:
        """Fallback extractive summary when LLM unavailable"""
        if not context:
            return (
                "I couldn't find specific reviews for this product in my database. "
                "This might be because:\n"
                "1. The product is not yet in our review database\n"
                "2. Try searching with a different product name or brand\n"
                "3. You can try the main search feature to add this product\n\n"
                "Would you like to search for a similar or related product?"
            )

        # Simple extractive approach
        summary_parts = []

        # Count sentiments
        positive = sum(
            1 for c in context if c.get("metadata", {}).get("sentiment") == "positive"
        )
        negative = sum(
            1 for c in context if c.get("metadata", {}).get("sentiment") == "negative"
        )
        neutral = len(context) - positive - negative

        # Create a comprehensive summary
        summary_parts.append(
            f"Based on {len(context)} relevant reviews:\n"
            f"• {positive} positive reviews ({positive/len(context)*100:.0f}%)\n"
            f"• {negative} negative reviews ({negative/len(context)*100:.0f}%)\n"
            f"• {neutral} neutral reviews ({neutral/len(context)*100:.0f}%)"
        )

        # Determine overall sentiment
        if positive > len(context) * 0.6:
            summary_parts.append("\nOverall: Mostly positive feedback")
        elif negative > len(context) * 0.4:
            summary_parts.append("\nOverall: Mixed to negative feedback")
        else:
            summary_parts.append("\nOverall: Mixed opinions")

        # Add top relevant review snippets
        summary_parts.append("\nKey points from reviews:")
        for i, item in enumerate(context[:3], 1):
            text = item["text"][:150]
            rating = item.get("metadata", {}).get("rating", "N/A")
            summary_parts.append(f"\n{i}. ({rating}/5) {text}...")

        summary_parts.append(
            "\n\nNote: This is a basic summary as the AI service is temporarily unavailable. "
            "Please try again or use the product search feature."
        )

        return "\n".join(summary_parts)


# Singleton instance
_rag_service_instance = None


def get_rag_service() -> RAGService:
    """Get or create RAG service singleton"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
