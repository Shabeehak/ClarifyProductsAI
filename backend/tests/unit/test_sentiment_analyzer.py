"""
Unit Tests for SentimentAnalyzer

Tests the sentiment analysis functionality using DistilBERT model.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.ml_models.sentiment_analyzer import SentimentAnalyzer, get_sentiment_analyzer
from app.core.exceptions import MLModelLoadException, MLModelInferenceException


@pytest.mark.unit
@pytest.mark.ml
class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Analyzer fixture"""
        return SentimentAnalyzer()

    @pytest.fixture
    def mock_pipeline(self):
        """Mock transformers pipeline"""
        mock = Mock()
        # Default positive sentiment response
        mock.return_value = [{"label": "POSITIVE", "score": 0.95}]
        return mock

    # =========================================================================
    # Test Initialization
    # =========================================================================

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes without loading model"""
        assert analyzer.pipeline is None

    def test_singleton_instance(self):
        """Test get_sentiment_analyzer returns singleton"""
        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()
        assert analyzer1 is analyzer2

    # =========================================================================
    # Test load_model()
    # =========================================================================

    @patch('app.ml_models.sentiment_analyzer.pipeline')
    @patch('app.ml_models.sentiment_analyzer.settings')
    def test_load_model_success(self, mock_settings, mock_pipeline_func, analyzer):
        """Test successful model loading"""
        mock_settings.SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
        mock_settings.SENTIMENT_MODEL_PATH = "/tmp/models"

        mock_pipe = Mock()
        mock_pipeline_func.return_value = mock_pipe

        analyzer.load_model()

        assert analyzer.pipeline is not None
        mock_pipeline_func.assert_called_once()

    @patch('app.ml_models.sentiment_analyzer.pipeline')
    @patch('app.ml_models.sentiment_analyzer.settings')
    def test_load_model_already_loaded(self, mock_settings, mock_pipeline_func, analyzer):
        """Test loading model when already loaded doesn't reload"""
        analyzer.pipeline = Mock()
        existing_pipeline = analyzer.pipeline

        analyzer.load_model()

        # Should not reload
        assert analyzer.pipeline is existing_pipeline
        mock_pipeline_func.assert_not_called()

    @patch('app.ml_models.sentiment_analyzer.pipeline')
    @patch('app.ml_models.sentiment_analyzer.settings')
    def test_load_model_failure(self, mock_settings, mock_pipeline_func, analyzer):
        """Test model loading failure raises exception"""
        mock_settings.SENTIMENT_MODEL_NAME = "invalid-model"
        mock_settings.SENTIMENT_MODEL_PATH = "/tmp/models"
        mock_pipeline_func.side_effect = Exception("Model not found")

        with pytest.raises(MLModelLoadException) as exc_info:
            analyzer.load_model()

        assert "Failed to load sentiment model" in str(exc_info.value)

    # =========================================================================
    # Test analyze()
    # =========================================================================

    def test_analyze_positive_sentiment(self, analyzer, mock_pipeline):
        """Test analyzing positive text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.95}]

        result = analyzer.analyze("This product is amazing!")

        assert result["sentiment_label"] == "positive"
        assert result["sentiment_score"] == 0.95
        assert result["confidence"] == 0.95
        assert "inference_time_ms" in result

    def test_analyze_negative_sentiment(self, analyzer, mock_pipeline):
        """Test analyzing negative text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "NEGATIVE", "score": 0.88}]

        result = analyzer.analyze("This product is terrible!")

        assert result["sentiment_label"] == "negative"
        assert result["sentiment_score"] == 0.88
        assert result["confidence"] == 0.88

    def test_analyze_neutral_sentiment(self, analyzer, mock_pipeline):
        """Test analyzing neutral text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "NEUTRAL", "score": 0.60}]

        result = analyzer.analyze("This product exists.")

        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.60

    def test_analyze_loads_model_if_not_loaded(self, analyzer):
        """Test analyze loads model if not already loaded"""
        with patch.object(analyzer, 'load_model') as mock_load:
            mock_pipe = Mock()
            mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.9}]
            analyzer.pipeline = None

            # Manually set pipeline after load_model is called
            def set_pipeline():
                analyzer.pipeline = mock_pipe
            mock_load.side_effect = set_pipeline

            result = analyzer.analyze("Great product")

            mock_load.assert_called_once()
            assert result["sentiment_label"] == "positive"

    def test_analyze_truncates_long_text(self, analyzer, mock_pipeline):
        """Test analyze truncates text to 512 tokens"""
        analyzer.pipeline = mock_pipeline
        long_text = "word " * 1000  # Very long text

        analyzer.analyze(long_text)

        # Check that pipeline was called with truncated text
        called_text = mock_pipeline.call_args[0][0]
        assert len(called_text) <= 512

    def test_analyze_empty_text(self, analyzer, mock_pipeline):
        """Test analyzing empty text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "NEUTRAL", "score": 0.5}]

        result = analyzer.analyze("")

        assert "sentiment_label" in result

    def test_analyze_inference_failure(self, analyzer, mock_pipeline):
        """Test analyze raises exception on inference failure"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.side_effect = Exception("Inference error")

        with pytest.raises(MLModelInferenceException) as exc_info:
            analyzer.analyze("Test text")

        assert "Sentiment analysis failed" in str(exc_info.value)

    # =========================================================================
    # Test analyze_batch()
    # =========================================================================

    def test_analyze_batch_multiple_texts(self, analyzer, mock_pipeline):
        """Test batch analysis of multiple texts"""
        analyzer.pipeline = mock_pipeline
        texts = [
            "Great product!",
            "Terrible quality",
            "It's okay"
        ]

        # Return different sentiments for each text
        mock_pipeline.side_effect = [
            [{"label": "POSITIVE", "score": 0.9}],
            [{"label": "NEGATIVE", "score": 0.85}],
            [{"label": "NEUTRAL", "score": 0.6}]
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert results[0]["sentiment_label"] == "positive"
        assert results[1]["sentiment_label"] == "negative"
        assert results[2]["sentiment_label"] == "neutral"

    def test_analyze_batch_empty_list(self, analyzer):
        """Test batch analysis with empty list"""
        results = analyzer.analyze_batch([])

        assert results == []

    def test_analyze_batch_single_text(self, analyzer, mock_pipeline):
        """Test batch analysis with single text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        results = analyzer.analyze_batch(["Great!"])

        assert len(results) == 1
        assert results[0]["sentiment_label"] == "positive"

    def test_analyze_batch_loads_model(self, analyzer):
        """Test batch analysis loads model if needed"""
        with patch.object(analyzer, 'load_model') as mock_load:
            mock_pipe = Mock()
            mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.9}]

            def set_pipeline():
                analyzer.pipeline = mock_pipe
            mock_load.side_effect = set_pipeline

            analyzer.analyze_batch(["Test"])

            mock_load.assert_called_once()

    # =========================================================================
    # Test get_aspect_sentiment()
    # =========================================================================

    def test_get_aspect_sentiment_found(self, analyzer, mock_pipeline):
        """Test aspect sentiment when aspect is mentioned"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        text = "The quality is excellent. The price is reasonable."
        result = analyzer.get_aspect_sentiment(text, "quality")

        assert result["aspect"] == "quality"
        assert result["sentiment_label"] == "positive"
        assert result["mentions"] > 0

    def test_get_aspect_sentiment_not_found(self, analyzer):
        """Test aspect sentiment when aspect not mentioned"""
        text = "This is a product review."
        result = analyzer.get_aspect_sentiment(text, "delivery")

        assert result["aspect"] == "delivery"
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.5
        assert result["mentions"] == 0

    def test_get_aspect_sentiment_multiple_mentions(self, analyzer, mock_pipeline):
        """Test aspect sentiment with multiple mentions"""
        analyzer.pipeline = mock_pipeline
        # Return different sentiments for multiple mentions
        mock_pipeline.side_effect = [
            [{"label": "POSITIVE", "score": 0.9}],
            [{"label": "NEGATIVE", "score": 0.8}]
        ]

        text = "The quality is great. But the quality control is poor."
        result = analyzer.get_aspect_sentiment(text, "quality")

        assert result["aspect"] == "quality"
        assert result["mentions"] == 2

    def test_get_aspect_sentiment_aggregation(self, analyzer, mock_pipeline):
        """Test aspect sentiment aggregates multiple mentions correctly"""
        analyzer.pipeline = mock_pipeline
        # More positive than negative
        mock_pipeline.side_effect = [
            [{"label": "POSITIVE", "score": 0.9}],
            [{"label": "POSITIVE", "score": 0.85}],
            [{"label": "NEGATIVE", "score": 0.7}]
        ]

        text = "Price is good. Price is fair. Price is high."
        result = analyzer.get_aspect_sentiment(text, "price")

        assert result["sentiment_label"] == "positive"  # Majority wins
        assert result["mentions"] == 3

    def test_get_aspect_sentiment_case_insensitive(self, analyzer, mock_pipeline):
        """Test aspect sentiment is case insensitive"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        text = "The QUALITY is excellent."
        result = analyzer.get_aspect_sentiment(text, "quality")

        assert result["mentions"] > 0
        assert result["sentiment_label"] == "positive"

    def test_get_aspect_sentiment_context_extraction(self, analyzer, mock_pipeline):
        """Test aspect sentiment extracts context around aspect"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        text = "The build quality is really good and durable."
        result = analyzer.get_aspect_sentiment(text, "quality")

        # Should have found the aspect
        assert result["mentions"] > 0
        mock_pipeline.assert_called()

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_analyze_special_characters(self, analyzer, mock_pipeline):
        """Test analyzing text with special characters"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        result = analyzer.analyze("Great!!! 😊👍 #awesome")

        assert "sentiment_label" in result

    def test_analyze_unicode_text(self, analyzer, mock_pipeline):
        """Test analyzing unicode text"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        result = analyzer.analyze("Très bien! 很好!")

        assert "sentiment_label" in result

    def test_aspect_sentiment_no_sentences(self, analyzer):
        """Test aspect sentiment with text having no periods"""
        text = "quality"
        result = analyzer.get_aspect_sentiment(text, "quality")

        assert result["mentions"] > 0

    def test_aspect_sentiment_empty_text(self, analyzer):
        """Test aspect sentiment with empty text"""
        result = analyzer.get_aspect_sentiment("", "quality")

        assert result["sentiment_label"] == "neutral"
        assert result["mentions"] == 0

    @pytest.mark.parametrize("text,expected_label", [
        ("This is excellent!", "positive"),
        ("This is terrible!", "negative"),
        ("This is okay.", "neutral"),
    ])
    def test_analyze_various_sentiments(self, analyzer, mock_pipeline, text, expected_label):
        """Test analyzing various sentiment texts"""
        analyzer.pipeline = mock_pipeline

        # Map expected label to mock response
        label_map = {
            "positive": "POSITIVE",
            "negative": "NEGATIVE",
            "neutral": "NEUTRAL"
        }
        mock_pipeline.return_value = [{"label": label_map[expected_label], "score": 0.9}]

        result = analyzer.analyze(text)

        assert result["sentiment_label"] == expected_label

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_analyze_includes_timing(self, analyzer, mock_pipeline):
        """Test analyze includes inference timing"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        result = analyzer.analyze("Test")

        assert "inference_time_ms" in result
        assert isinstance(result["inference_time_ms"], (int, float))
        assert result["inference_time_ms"] >= 0

    def test_batch_large_dataset(self, analyzer, mock_pipeline):
        """Test batch processing large number of texts"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        texts = [f"Review {i}" for i in range(100)]
        results = analyzer.analyze_batch(texts)

        assert len(results) == 100
        # All should have processed successfully
        assert all("sentiment_label" in r for r in results)

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    def test_analyze_realistic_positive_review(self, analyzer, mock_pipeline):
        """Test analyzing realistic positive review"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.92}]

        review = "I absolutely love this product! The quality is outstanding and it exceeded my expectations."
        result = analyzer.analyze(review)

        assert result["sentiment_label"] == "positive"
        assert result["sentiment_score"] > 0.8

    def test_analyze_realistic_negative_review(self, analyzer, mock_pipeline):
        """Test analyzing realistic negative review"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "NEGATIVE", "score": 0.88}]

        review = "Very disappointed with this purchase. Poor quality and broke after one day."
        result = analyzer.analyze(review)

        assert result["sentiment_label"] == "negative"
        assert result["sentiment_score"] > 0.8

    def test_analyze_mixed_sentiment_review(self, analyzer, mock_pipeline):
        """Test analyzing review with mixed sentiment"""
        analyzer.pipeline = mock_pipeline
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.65}]

        review = "The product works well but the price is too high."
        result = analyzer.analyze(review)

        # Should still return a sentiment
        assert result["sentiment_label"] in ["positive", "negative", "neutral"]
