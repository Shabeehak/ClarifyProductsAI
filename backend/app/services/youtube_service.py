"""
YouTube Video Review Service
Extracts video transcripts and metadata for product reviews
Uses YouTube Data API v3 + youtube-transcript-api
"""

from typing import List, Dict, Optional
from loguru import logger
from googleapiclient.discovery import build
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
import re


class YouTubeService:
    """YouTube video review extractor"""

    def __init__(self, api_key: str):
        """
        Initialize YouTube service

        Args:
            api_key: YouTube Data API v3 key from Google Cloud Console
        """
        self.api_key = api_key
        self.youtube = None
        if api_key:
            self.youtube = build("youtube", "v3", developerKey=api_key)
            logger.info("YouTube API initialized")
        else:
            logger.warning("YouTube API key not configured")

    def search_product_reviews(
        self,
        product_name: str,
        max_results: int = 10,
        min_duration: int = 120,  # Min 2 minutes
    ) -> List[Dict]:
        """
        Search for product review videos

        Args:
            product_name: Product name to search
            max_results: Maximum number of videos (default: 10)
            min_duration: Minimum video duration in seconds (default: 120)

        Returns:
            List of video dictionaries with metadata

        Example:
            videos = service.search_product_reviews("iPhone 15")
            for video in videos:
                print(f"{video['title']} - {video['channel']}")
        """
        if not self.youtube:
            logger.error("YouTube API not initialized")
            return []

        # Build search query
        search_query = f"{product_name} review"

        try:
            logger.info(f"Searching YouTube for: {search_query}")

            # Search for videos
            search_response = (
                self.youtube.search()
                .list(
                    q=search_query,
                    part="id,snippet",
                    maxResults=max_results * 2,  # Get more, filter later
                    type="video",
                    videoDefinition="high",  # HD videos only
                    relevanceLanguage="en",
                    order="relevance",  # Most relevant first
                )
                .execute()
            )

            video_ids = []
            videos = []

            # Extract video IDs
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                video_ids.append(video_id)

                # Store basic info
                videos.append(
                    {
                        "video_id": video_id,
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "channel": item["snippet"]["channelTitle"],
                        "channel_id": item["snippet"]["channelId"],
                        "published_at": item["snippet"]["publishedAt"],
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                    }
                )

            if not video_ids:
                logger.warning(f"No videos found for '{product_name}'")
                return []

            # Get detailed video statistics
            videos_response = (
                self.youtube.videos()
                .list(part="statistics,contentDetails", id=",".join(video_ids))
                .execute()
            )

            # Merge statistics with video info
            for i, video in enumerate(videos):
                if i < len(videos_response["items"]):
                    stats = videos_response["items"][i]["statistics"]
                    content = videos_response["items"][i]["contentDetails"]

                    # Add statistics
                    video["view_count"] = int(stats.get("viewCount", 0))
                    video["like_count"] = int(stats.get("likeCount", 0))
                    video["comment_count"] = int(stats.get("commentCount", 0))

                    # Parse duration
                    duration = self._parse_duration(content.get("duration", "PT0S"))
                    video["duration_seconds"] = duration

                    # Calculate engagement score
                    views = video["view_count"]
                    likes = video["like_count"]
                    video["engagement_score"] = (
                        (likes / views * 100) if views > 0 else 0
                    )

            # Filter by duration and sort by engagement
            filtered_videos = [
                v for v in videos if v.get("duration_seconds", 0) >= min_duration
            ]

            # Sort by engagement score (higher is better)
            filtered_videos.sort(
                key=lambda x: x.get("engagement_score", 0), reverse=True
            )

            logger.info(
                f"Found {len(filtered_videos)} quality review videos for '{product_name}'"
            )

            return filtered_videos[:max_results]

        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return []

    def get_video_transcript(self, video_id: str) -> Optional[str]:
        """
        Extract transcript/captions from YouTube video

        Args:
            video_id: YouTube video ID

        Returns:
            Full transcript text or None if not available

        Example:
            transcript = service.get_video_transcript("dQw4w9WgXcQ")
            print(transcript[:200])  # First 200 characters
        """
        try:
            logger.info(f"Fetching transcript for video: {video_id}")

            # Get transcript (auto-generated or manual)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Combine all text segments
            full_transcript = " ".join([item["text"] for item in transcript_list])

            # Clean up transcript
            full_transcript = self._clean_transcript(full_transcript)

            logger.info(f"Transcript extracted: {len(full_transcript)} characters")

            return full_transcript

        except TranscriptsDisabled:
            logger.warning(f"Transcripts disabled for video: {video_id}")
            return None
        except NoTranscriptFound:
            logger.warning(f"No transcript available for video: {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id}: {e}")
            return None

    def get_product_reviews(self, product_name: str, max_videos: int = 5) -> Dict:
        """
        Get product reviews with transcripts

        Args:
            product_name: Product name to search
            max_videos: Maximum number of videos to analyze

        Returns:
            Dictionary with videos and transcripts

        Example:
            result = service.get_product_reviews("iPhone 15", max_videos=3)
            print(f"Found {len(result['videos'])} videos")
            for video in result['videos']:
                if video['transcript']:
                    print(f"Transcript: {video['transcript'][:100]}...")
        """
        # Search for videos
        videos = self.search_product_reviews(product_name, max_results=max_videos)

        if not videos:
            return {
                "product_name": product_name,
                "videos": [],
                "total_videos": 0,
                "videos_with_transcripts": 0,
            }

        # Extract transcripts for each video
        videos_with_transcripts = 0

        for video in videos:
            transcript = self.get_video_transcript(video["video_id"])

            if transcript:
                video["transcript"] = transcript
                video["transcript_length"] = len(transcript)
                video["has_transcript"] = True
                videos_with_transcripts += 1
            else:
                video["transcript"] = None
                video["transcript_length"] = 0
                video["has_transcript"] = False

            # Generate video URL
            video["url"] = f"https://www.youtube.com/watch?v={video['video_id']}"

        logger.info(
            f"Product '{product_name}': {videos_with_transcripts}/{len(videos)} videos have transcripts"
        )

        return {
            "product_name": product_name,
            "videos": videos,
            "total_videos": len(videos),
            "videos_with_transcripts": videos_with_transcripts,
            "source": "youtube",
        }

    def _parse_duration(self, duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds

        Args:
            duration: ISO 8601 duration (e.g., "PT4M13S")

        Returns:
            Duration in seconds
        """
        # Parse format: PT1H2M3S or PT4M13S or PT30S
        hours = 0
        minutes = 0
        seconds = 0

        # Extract hours
        h_match = re.search(r"(\d+)H", duration)
        if h_match:
            hours = int(h_match.group(1))

        # Extract minutes
        m_match = re.search(r"(\d+)M", duration)
        if m_match:
            minutes = int(m_match.group(1))

        # Extract seconds
        s_match = re.search(r"(\d+)S", duration)
        if s_match:
            seconds = int(s_match.group(1))

        return hours * 3600 + minutes * 60 + seconds

    def _clean_transcript(self, transcript: str) -> str:
        """
        Clean up transcript text

        Args:
            transcript: Raw transcript text

        Returns:
            Cleaned transcript
        """
        # Remove multiple spaces
        transcript = re.sub(r"\s+", " ", transcript)

        # Remove [Music], [Applause], etc.
        transcript = re.sub(r"\[.*?\]", "", transcript)

        # Remove timestamps if any
        transcript = re.sub(r"\d{1,2}:\d{2}", "", transcript)

        # Strip whitespace
        transcript = transcript.strip()

        return transcript


# Singleton instance
_youtube_service_instance: Optional[YouTubeService] = None


def get_youtube_service(api_key: str) -> YouTubeService:
    """
    Get or create YouTube service singleton

    Args:
        api_key: YouTube Data API v3 key

    Returns:
        YouTubeService instance
    """
    global _youtube_service_instance
    if _youtube_service_instance is None:
        _youtube_service_instance = YouTubeService(api_key)
    return _youtube_service_instance
