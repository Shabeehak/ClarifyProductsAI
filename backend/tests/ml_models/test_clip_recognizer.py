"""
ML Model Tests for CLIP Image Recognizer

Tests the CLIP model for product recognition functionality.
"""
import pytest
from PIL import Image
from io import BytesIO
import requests

from app.ml_models.clip_recognizer import get_clip_recognizer


@pytest.mark.ml
@pytest.mark.clip
class TestCLIPRecognizer:
    """Test suite for CLIP Image Recognition Model"""

    @pytest.fixture(scope="class")
    def recognizer(self):
        """Get CLIP recognizer instance (shared across tests)"""
        recognizer = get_clip_recognizer()
        recognizer.load_model()
        return recognizer

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image"""
        image = Image.new('RGB', (224, 224), color='blue')
        return image

    @pytest.fixture
    def real_product_image(self):
        """Download a real product image"""
        try:
            # Download a smartphone image from Unsplash
            image_url = "https://images.unsplash.com/photo-1592286927505-2b1f7c3be98e?w=400"
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception:
            pass

        # Fallback to test image
        return Image.new('RGB', (224, 224), color='blue')

    # =========================================================================
    # Model Loading Tests
    # =========================================================================

    def test_model_loads_successfully(self, recognizer):
        """Test CLIP model loads without errors"""
        assert recognizer is not None
        assert hasattr(recognizer, 'model')
        assert hasattr(recognizer, 'processor')

    def test_singleton_pattern(self):
        """Test recognizer uses singleton pattern"""
        recognizer1 = get_clip_recognizer()
        recognizer2 = get_clip_recognizer()

        # Should return the same instance
        assert recognizer1 is recognizer2

    # =========================================================================
    # Recognition Tests
    # =========================================================================

    def test_recognize_sample_image(self, recognizer, sample_image):
        """Test recognition with a simple test image"""
        result = recognizer.recognize_product(sample_image)

        assert result is not None
        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_recognize_real_product_image(self, recognizer, real_product_image):
        """Test recognition with a real product image"""
        result = recognizer.recognize_product(real_product_image)

        assert result["success"] is True
        assert "top_prediction" in result
        assert "predictions" in result

    def test_recognition_result_structure(self, recognizer, sample_image):
        """Test recognition result has correct structure"""
        result = recognizer.recognize_product(sample_image)

        assert "success" in result
        assert result["success"] is True

        # Should have top prediction
        assert "top_prediction" in result
        top_pred = result["top_prediction"]
        assert "category" in top_pred
        assert "confidence" in top_pred

        # Should have list of predictions
        assert "predictions" in result
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) > 0

    def test_prediction_confidence_values(self, recognizer, sample_image):
        """Test confidence values are in valid range"""
        result = recognizer.recognize_product(sample_image)

        top_pred = result["top_prediction"]
        assert 0 <= top_pred["confidence"] <= 1

        # All predictions should have valid confidence
        for pred in result["predictions"]:
            assert "confidence" in pred
            assert 0 <= pred["confidence"] <= 1

    def test_predictions_sorted_by_confidence(self, recognizer, sample_image):
        """Test predictions are sorted by confidence (descending)"""
        result = recognizer.recognize_product(sample_image)

        predictions = result["predictions"]
        confidences = [p["confidence"] for p in predictions]

        # Should be in descending order
        assert confidences == sorted(confidences, reverse=True)

    def test_top_prediction_matches_first_prediction(self, recognizer, sample_image):
        """Test top prediction is the same as first in predictions list"""
        result = recognizer.recognize_product(sample_image)

        top_pred = result["top_prediction"]
        first_pred = result["predictions"][0]

        assert top_pred["category"] == first_pred["category"]
        assert top_pred["confidence"] == first_pred["confidence"]

    # =========================================================================
    # Image Format Tests
    # =========================================================================

    @pytest.mark.parametrize("image_format", ['RGB', 'RGBA', 'L'])
    def test_recognize_various_image_modes(self, recognizer, image_format):
        """Test recognition with various PIL image modes"""
        if image_format == 'L':
            # Grayscale
            image = Image.new('L', (224, 224), color=128)
        else:
            image = Image.new(image_format, (224, 224), color='red')

        result = recognizer.recognize_product(image)

        assert result["success"] is True
        assert "top_prediction" in result

    @pytest.mark.parametrize("size", [
        (100, 100),
        (224, 224),
        (512, 512),
        (1024, 768),
    ])
    def test_recognize_various_image_sizes(self, recognizer, size):
        """Test recognition with various image sizes"""
        image = Image.new('RGB', size, color='green')

        result = recognizer.recognize_product(image)

        assert result["success"] is True
        assert "top_prediction" in result

    @pytest.mark.parametrize("aspect_ratio", [
        (224, 224),   # Square
        (300, 200),   # Landscape
        (200, 300),   # Portrait
    ])
    def test_recognize_various_aspect_ratios(self, recognizer, aspect_ratio):
        """Test recognition with various aspect ratios"""
        image = Image.new('RGB', aspect_ratio, color='purple')

        result = recognizer.recognize_product(image)

        assert result["success"] is True

    # =========================================================================
    # Error Handling
    # =========================================================================

    def test_recognize_with_none_image(self, recognizer):
        """Test recognition with None image"""
        result = recognizer.recognize_product(None)

        # Should handle gracefully
        assert isinstance(result, dict)
        if not result.get("success"):
            assert "error" in result

    def test_recognize_with_invalid_image(self, recognizer):
        """Test recognition with invalid image object"""
        result = recognizer.recognize_product("not an image")

        # Should handle gracefully
        assert isinstance(result, dict)
        if not result.get("success"):
            assert "error" in result

    def test_recognize_with_corrupted_image(self, recognizer):
        """Test recognition with corrupted image data"""
        try:
            # Try to create image from invalid data
            corrupted_image = Image.open(BytesIO(b"corrupted data"))
            result = recognizer.recognize_product(corrupted_image)

            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception:
            # Expected to fail during image creation
            pass

    # =========================================================================
    # Category Tests
    # =========================================================================

    def test_prediction_has_categories(self, recognizer, sample_image):
        """Test predictions include category information"""
        result = recognizer.recognize_product(sample_image)

        for pred in result["predictions"]:
            assert "category" in pred
            assert isinstance(pred["category"], str)
            assert len(pred["category"]) > 0

    def test_multiple_category_predictions(self, recognizer, sample_image):
        """Test model returns multiple category predictions"""
        result = recognizer.recognize_product(sample_image)

        predictions = result["predictions"]

        # Should have multiple predictions (at least 3)
        assert len(predictions) >= 3

        # Categories should be unique
        categories = [p["category"] for p in predictions]
        # Most should be unique (some overlap is ok)
        assert len(set(categories)) >= 2

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_recognition_performance(self, recognizer, sample_image):
        """Test recognition completes in reasonable time"""
        import time

        start_time = time.time()
        result = recognizer.recognize_product(sample_image)
        elapsed_time = time.time() - start_time

        # Should complete within 5 seconds after model is loaded
        assert elapsed_time < 5
        assert result["success"] is True

    def test_batch_recognition_consistency(self, recognizer, sample_image):
        """Test recognition returns consistent results"""
        result1 = recognizer.recognize_product(sample_image)
        result2 = recognizer.recognize_product(sample_image)

        # Should return same top prediction for same image
        assert result1["top_prediction"]["category"] == result2["top_prediction"]["category"]

    # =========================================================================
    # Integration-like Tests
    # =========================================================================

    def test_full_recognition_workflow(self, recognizer):
        """Test complete recognition workflow"""
        # Download real image
        try:
            image_url = "https://images.unsplash.com/photo-1592286927505-2b1f7c3be98e?w=400"
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content))
        except Exception:
            image = Image.new('RGB', (224, 224), color='blue')

        # Perform recognition
        result = recognizer.recognize_product(image)

        # Verify complete result
        assert result["success"] is True
        assert "top_prediction" in result
        assert "predictions" in result

        top_pred = result["top_prediction"]
        assert "category" in top_pred
        assert "confidence" in top_pred
        assert 0 <= top_pred["confidence"] <= 1

        # Should have multiple alternatives
        assert len(result["predictions"]) >= 3

        # Verify all predictions have required fields
        for pred in result["predictions"]:
            assert "category" in pred
            assert "confidence" in pred
            assert isinstance(pred["category"], str)
            assert isinstance(pred["confidence"], float)
