"""
Unit Tests for ReviewSummarizer

Tests the review summarization functionality using BART/Ollama models.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.ml_models.summarizer import ReviewSummarizer, get_summarizer
from app.core.exceptions import MLModelLoadException, MLModelInferenceException


@pytest.mark.unit
@pytest.mark.ml
class TestReviewSummarizer:
    """Test suite for ReviewSummarizer"""

    @pytest.fixture
    def summarizer(self):
        """Summarizer fixture"""
        return ReviewSummarizer()

    @pytest.fixture
    def mock_pipeline(self):
        """Mock transformers pipeline"""
        mock = Mock()
        mock.return_value = [{"summary_text": "This is a test summary."}]
        return mock

    # =========================================================================
    # Test Initialization
    # =========================================================================

    def test_summarizer_initialization(self, summarizer):
        """Test summarizer initializes without loading model"""
        assert summarizer.pipeline is None
        assert summarizer.use_ollama is False

    def test_singleton_instance(self):
        """Test get_summarizer returns singleton"""
        summarizer1 = get_summarizer()
        summarizer2 = get_summarizer()
        assert summarizer1 is summarizer2

    # =========================================================================
    # Test _check_ollama_available()
    # =========================================================================

    @patch('app.ml_models.summarizer.requests.get')
    @patch('app.ml_models.summarizer.settings')
    def test_check_ollama_available_success(self, mock_settings, mock_get, summarizer):
        """Test Ollama availability check when available"""
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = summarizer._check_ollama_available()

        assert result is True
        mock_get.assert_called_once()

    @patch('app.ml_models.summarizer.requests.get')
    @patch('app.ml_models.summarizer.settings')
    def test_check_ollama_available_failure(self, mock_settings, mock_get, summarizer):
        """Test Ollama availability check when not available"""
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"
        mock_get.side_effect = Exception("Connection refused")

        result = summarizer._check_ollama_available()

        assert result is False

    @patch('app.ml_models.summarizer.requests.get')
    @patch('app.ml_models.summarizer.settings')
    def test_check_ollama_available_non_200(self, mock_settings, mock_get, summarizer):
        """Test Ollama availability check with non-200 response"""
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = summarizer._check_ollama_available()

        assert result is False

    # =========================================================================
    # Test load_model()
    # =========================================================================

    @patch('app.ml_models.summarizer.pipeline')
    @patch('app.ml_models.summarizer.settings')
    @patch.object(ReviewSummarizer, '_check_ollama_available')
    def test_load_model_with_ollama(self, mock_check, mock_settings, mock_pipeline_func, summarizer):
        """Test model loading when Ollama is available"""
        mock_check.return_value = True
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"

        summarizer.load_model()

        assert summarizer.use_ollama is True
        mock_pipeline_func.assert_not_called()  # Should not load BART

    @patch('app.ml_models.summarizer.pipeline')
    @patch('app.ml_models.summarizer.settings')
    @patch.object(ReviewSummarizer, '_check_ollama_available')
    def test_load_model_without_ollama(self, mock_check, mock_settings, mock_pipeline_func, summarizer):
        """Test model loading when Ollama is not available"""
        mock_check.return_value = False
        mock_settings.SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"
        mock_settings.MODEL_CACHE_DIR = "/tmp/models"

        mock_pipe = Mock()
        mock_pipeline_func.return_value = mock_pipe

        summarizer.load_model()

        assert summarizer.use_ollama is False
        assert summarizer.pipeline is not None
        mock_pipeline_func.assert_called_once()

    @patch('app.ml_models.summarizer.pipeline')
    @patch('app.ml_models.summarizer.settings')
    @patch.object(ReviewSummarizer, '_check_ollama_available')
    def test_load_model_failure(self, mock_check, mock_settings, mock_pipeline_func, summarizer):
        """Test model loading failure raises exception"""
        mock_check.return_value = False
        mock_settings.SUMMARIZATION_MODEL_NAME = "invalid-model"
        mock_settings.MODEL_CACHE_DIR = "/tmp/models"
        mock_pipeline_func.side_effect = Exception("Model not found")

        with pytest.raises(MLModelLoadException) as exc_info:
            summarizer.load_model()

        assert "Failed to load summarization model" in str(exc_info.value)

    # =========================================================================
    # Test summarize_with_ollama()
    # =========================================================================

    @patch('app.ml_models.summarizer.requests.post')
    @patch('app.ml_models.summarizer.settings')
    def test_summarize_with_ollama_success(self, mock_settings, mock_post, summarizer):
        """Test successful Ollama summarization"""
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Summary from Ollama"}
        mock_post.return_value = mock_response

        result = summarizer.summarize_with_ollama("Test review text")

        assert result == "Summary from Ollama"
        mock_post.assert_called_once()

    @patch('app.ml_models.summarizer.requests.post')
    @patch('app.ml_models.summarizer.settings')
    def test_summarize_with_ollama_failure(self, mock_settings, mock_post, summarizer):
        """Test Ollama summarization failure"""
        mock_settings.OLLAMA_API_URL = "http://localhost:11434"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            summarizer.summarize_with_ollama("Test review text")

        assert "Ollama API error" in str(exc_info.value)

    # =========================================================================
    # Test summarize()
    # =========================================================================

    def test_summarize_with_bart(self, summarizer, mock_pipeline):
        """Test summarization using BART model"""
        summarizer.pipeline = mock_pipeline
        summarizer.use_ollama = False

        result = summarizer.summarize("This is a long review text that needs to be summarized.")

        assert result == "This is a test summary."
        mock_pipeline.assert_called_once()

    @patch.object(ReviewSummarizer, 'summarize_with_ollama')
    def test_summarize_with_ollama_mode(self, mock_ollama, summarizer):
        """Test summarization using Ollama"""
        summarizer.use_ollama = True
        mock_ollama.return_value = "Ollama summary"

        result = summarizer.summarize("Test review text")

        assert result == "Ollama summary"
        mock_ollama.assert_called_once()

    def test_summarize_loads_model_if_needed(self, summarizer):
        """Test summarize loads model if not loaded"""
        with patch.object(summarizer, 'load_model') as mock_load:
            mock_pipe = Mock()
            mock_pipe.return_value = [{"summary_text": "Summary"}]

            def set_pipeline():
                summarizer.pipeline = mock_pipe
            mock_load.side_effect = set_pipeline

            result = summarizer.summarize("Test text")

            mock_load.assert_called_once()
            assert result == "Summary"

    def test_summarize_truncates_long_text(self, summarizer, mock_pipeline):
        """Test summarization truncates very long text"""
        summarizer.pipeline = mock_pipeline
        summarizer.use_ollama = False

        long_text = "word " * 2000  # Very long text
        result = summarizer.summarize(long_text)

        # Should still produce a summary
        assert isinstance(result, str)
        mock_pipeline.assert_called()

    def test_summarize_with_custom_lengths(self, summarizer, mock_pipeline):
        """Test summarization with custom max/min lengths"""
        summarizer.pipeline = mock_pipeline
        summarizer.use_ollama = False

        result = summarizer.summarize("Test text", max_length=100, min_length=30)

        # Check that pipeline was called with correct parameters
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['max_length'] == 100
        assert call_kwargs['min_length'] == 30

    def test_summarize_empty_text(self, summarizer, mock_pipeline):
        """Test summarizing empty text"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"summary_text": ""}]

        result = summarizer.summarize("")

        assert isinstance(result, str)

    @patch.object(ReviewSummarizer, '_extractive_summary')
    def test_summarize_fallback_on_error(self, mock_extractive, summarizer, mock_pipeline):
        """Test summarization falls back to extractive on error"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.side_effect = Exception("Model error")
        mock_extractive.return_value = "Extractive summary"

        result = summarizer.summarize("Test text")

        assert result == "Extractive summary"
        mock_extractive.assert_called_once()

    @patch.object(ReviewSummarizer, '_extractive_summary')
    def test_summarize_raises_on_fallback_failure(self, mock_extractive, summarizer, mock_pipeline):
        """Test summarization raises exception if fallback also fails"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.side_effect = Exception("Model error")
        mock_extractive.side_effect = Exception("Fallback error")

        with pytest.raises(MLModelInferenceException) as exc_info:
            summarizer.summarize("Test text")

        assert "Summarization failed" in str(exc_info.value)

    # =========================================================================
    # Test _extractive_summary()
    # =========================================================================

    def test_extractive_summary_basic(self, summarizer):
        """Test basic extractive summarization"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = summarizer._extractive_summary(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_extractive_summary_with_keywords(self, summarizer):
        """Test extractive summary prioritizes sentences with keywords"""
        text = "This product is excellent and amazing. This is okay. The quality is outstanding."
        result = summarizer._extractive_summary(text)

        # Should prefer sentences with positive keywords
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extractive_summary_short_text(self, summarizer):
        """Test extractive summary with very short text"""
        text = "Short."
        result = summarizer._extractive_summary(text)

        assert isinstance(result, str)

    def test_extractive_summary_no_periods(self, summarizer):
        """Test extractive summary with text having no periods"""
        text = "This is a long text without any periods"
        result = summarizer._extractive_summary(text)

        assert isinstance(result, str)

    # =========================================================================
    # Test generate_structured_summary()
    # =========================================================================

    def test_structured_summary_empty_reviews(self, summarizer):
        """Test structured summary with empty reviews"""
        result = summarizer.generate_structured_summary([])

        assert result["summary"] == ""
        assert result["pros"] == []
        assert result["cons"] == []
        assert result["key_points"] == []

    @patch.object(ReviewSummarizer, 'summarize')
    def test_structured_summary_with_reviews(self, mock_summarize, summarizer):
        """Test structured summary generation"""
        mock_summarize.return_value = "Overall summary"
        reviews = [
            "This product is excellent and great quality.",
            "The product is terrible and poor quality.",
            "Amazing performance and fast delivery."
        ]

        result = summarizer.generate_structured_summary(reviews)

        assert result["summary"] == "Overall summary"
        assert isinstance(result["pros"], list)
        assert isinstance(result["cons"], list)
        assert isinstance(result["key_points"], list)

    @patch.object(ReviewSummarizer, 'summarize')
    def test_structured_summary_limits_key_points(self, mock_summarize, summarizer):
        """Test structured summary limits key points to 7"""
        mock_summarize.return_value = "Summary"
        reviews = ["Great product. " * 20]

        result = summarizer.generate_structured_summary(reviews)

        assert len(result["key_points"]) <= 7

    # =========================================================================
    # Test _extract_key_points()
    # =========================================================================

    def test_extract_key_points_finds_relevant_sentences(self, summarizer):
        """Test key point extraction finds relevant sentences"""
        reviews = [
            "The product is great and excellent.",
            "This is a filler sentence without keywords.",
            "The quality is poor and bad."
        ]

        result = summarizer._extract_key_points(reviews)

        assert isinstance(result, list)
        assert len(result) <= 7

    def test_extract_key_points_empty_reviews(self, summarizer):
        """Test key point extraction with empty reviews"""
        result = summarizer._extract_key_points([])

        assert result == []

    def test_extract_key_points_unique(self, summarizer):
        """Test key points are unique"""
        reviews = [
            "The product is great.",
            "The product is great.",
            "The product is great."
        ]

        result = summarizer._extract_key_points(reviews)

        # Should deduplicate
        assert len(result) <= 1

    # =========================================================================
    # Test _extract_pros()
    # =========================================================================

    def test_extract_pros_finds_positive_sentences(self, summarizer):
        """Test pros extraction finds positive sentences"""
        reviews = [
            "This product is excellent and amazing!",
            "The quality is great and outstanding.",
            "This is terrible."  # Should be excluded
        ]

        result = summarizer._extract_pros(reviews)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should not include negative sentence
        assert not any("terrible" in pro.lower() for pro in result)

    def test_extract_pros_excludes_mixed_sentiment(self, summarizer):
        """Test pros extraction excludes mixed sentiment sentences"""
        reviews = [
            "The quality is great but the price is high.",
            "This product is excellent and amazing."
        ]

        result = summarizer._extract_pros(reviews)

        # Should prefer purely positive sentences (without "but")
        # The second sentence should be included
        assert len(result) > 0
        assert any("excellent" in pro for pro in result)

    def test_extract_pros_limits_to_five(self, summarizer):
        """Test pros extraction limits to 5 items"""
        reviews = [
            "Great product. Excellent quality. Amazing performance. "
            "Fantastic design. Wonderful experience. Perfect buy. "
            "Outstanding value. Impressive features."
        ]

        result = summarizer._extract_pros(reviews)

        assert len(result) <= 5

    def test_extract_pros_filters_length(self, summarizer):
        """Test pros extraction filters sentences by length"""
        reviews = [
            "Great.",  # Too short
            "This product is excellent and has great quality.",  # Good length
            "x" * 200  # Too long
        ]

        result = summarizer._extract_pros(reviews)

        # Should only include good length sentences
        assert all(20 < len(pro) < 150 for pro in result)

    def test_extract_pros_empty_reviews(self, summarizer):
        """Test pros extraction with empty reviews"""
        result = summarizer._extract_pros([])

        assert result == []

    # =========================================================================
    # Test _extract_cons()
    # =========================================================================

    def test_extract_cons_finds_negative_sentences(self, summarizer):
        """Test cons extraction finds negative sentences"""
        reviews = [
            "This product is terrible and awful.",
            "The quality is poor and disappointing.",
            "This is excellent."  # Should be excluded
        ]

        result = summarizer._extract_cons(reviews)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should not include positive sentence
        assert not any("excellent" in con.lower() for con in result)

    def test_extract_cons_limits_to_five(self, summarizer):
        """Test cons extraction limits to 5 items"""
        reviews = [
            "Bad quality. Poor design. Terrible performance. "
            "Awful experience. Disappointing product. Broken item. "
            "Useless features. Waste of money."
        ]

        result = summarizer._extract_cons(reviews)

        assert len(result) <= 5

    def test_extract_cons_filters_length(self, summarizer):
        """Test cons extraction filters sentences by length"""
        reviews = [
            "Bad.",  # Too short
            "This product is terrible and has poor quality.",  # Good length
            "x" * 200  # Too long
        ]

        result = summarizer._extract_cons(reviews)

        # Should only include good length sentences
        assert all(20 < len(con) < 150 for con in result)

    def test_extract_cons_empty_reviews(self, summarizer):
        """Test cons extraction with empty reviews"""
        result = summarizer._extract_cons([])

        assert result == []

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_summarize_special_characters(self, summarizer, mock_pipeline):
        """Test summarizing text with special characters"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"summary_text": "Summary"}]

        result = summarizer.summarize("Review with special chars!!! @#$%")

        assert isinstance(result, str)

    def test_summarize_unicode_text(self, summarizer, mock_pipeline):
        """Test summarizing unicode text"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"summary_text": "Summary"}]

        result = summarizer.summarize("Unicode text: café, naïve, 中文")

        assert isinstance(result, str)

    def test_structured_summary_large_dataset(self, summarizer):
        """Test structured summary with many reviews"""
        with patch.object(summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Summary"
            reviews = [f"Review {i}" for i in range(100)]

            result = summarizer.generate_structured_summary(reviews)

            # Should handle large dataset
            assert isinstance(result, dict)
            assert "summary" in result

    @pytest.mark.parametrize("text,max_len,min_len", [
        ("Short review", 100, 30),
        ("Medium length review text", 150, 50),
        ("Very long review " * 50, 200, 80),
    ])
    def test_summarize_various_lengths(self, summarizer, mock_pipeline, text, max_len, min_len):
        """Test summarization with various text lengths"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"summary_text": "Summary"}]

        result = summarizer.summarize(text, max_length=max_len, min_length=min_len)

        assert isinstance(result, str)

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    def test_realistic_review_summarization(self, summarizer, mock_pipeline):
        """Test summarizing realistic product review"""
        summarizer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"summary_text": "Great product with excellent quality."}]

        review = """
        I purchased this product two weeks ago and I'm very impressed with the quality.
        The build feels solid and premium. The performance is excellent and exceeds my expectations.
        Battery life is outstanding, lasting all day on a single charge.
        The only minor issue is the price, which is a bit high, but the quality justifies it.
        Overall, I highly recommend this product.
        """

        result = summarizer.summarize(review)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_realistic_structured_summary(self, summarizer):
        """Test generating structured summary from realistic reviews"""
        with patch.object(summarizer, 'summarize') as mock_summarize:
            mock_summarize.return_value = "Overall positive feedback with some concerns."

            reviews = [
                "Excellent product! Great quality and fast shipping.",
                "The performance is amazing. Very impressed.",
                "Poor customer service. Took weeks to respond.",
                "Quality is outstanding. Worth the price."
            ]

            result = summarizer.generate_structured_summary(reviews)

            assert result["summary"] == "Overall positive feedback with some concerns."
            assert len(result["pros"]) > 0
            assert len(result["cons"]) > 0
