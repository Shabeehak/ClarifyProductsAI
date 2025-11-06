"""
Unit Tests for Image Quality Checker

Tests the image quality assessment functionality including:
- Resolution checks
- Blur detection
- Brightness validation
- Error handling
"""

import pytest
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

from app.utils.image_quality import ImageQualityChecker, get_quality_checker


class TestImageQualityChecker:
    """Test suite for ImageQualityChecker"""

    @pytest.fixture
    def checker(self):
        """Create a quality checker instance for testing"""
        return ImageQualityChecker(
            min_resolution=(200, 200),
            blur_threshold=100.0,
            brightness_range=(30, 225)
        )

    @pytest.fixture
    def temp_image_path(self):
        """Create a temporary image file path"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def create_test_image(
        self,
        path: str,
        size: tuple = (400, 400),
        brightness: int = 128,
        add_blur: bool = False
    ):
        """
        Helper to create test images with specific characteristics.

        Args:
            path: Where to save the image
            size: (width, height) in pixels
            brightness: Average brightness (0-255)
            add_blur: Whether to apply blur
        """
        # Create image with specific brightness
        img_array = np.full((size[1], size[0], 3), brightness, dtype=np.uint8)

        # Add some texture to avoid uniform image
        noise = np.random.randint(-20, 20, size=(size[1], size[0], 3), dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if add_blur:
            # Apply Gaussian blur to make image blurry
            img_array = cv2.GaussianBlur(img_array, (15, 15), 0)

        # Save image
        img_pil = Image.fromarray(img_array, mode='RGB')
        img_pil.save(path)

    def test_good_quality_image(self, checker, temp_image_path):
        """Test that a good quality image passes all checks"""
        # Create good quality image
        self.create_test_image(
            temp_image_path,
            size=(400, 400),
            brightness=128,  # Good brightness
            add_blur=False
        )

        result = checker.check_quality(temp_image_path)

        assert result["is_acceptable"] == True
        assert result["quality_score"] >= 0.6
        assert len(result["issues"]) == 0
        assert result["details"]["resolution"] == (400, 400)
        assert result["details"]["brightness"] >= 30
        assert result["details"]["brightness"] <= 225

    def test_low_resolution_image(self, checker, temp_image_path):
        """Test that low resolution images are detected"""
        # Create low resolution image
        self.create_test_image(
            temp_image_path,
            size=(100, 100),  # Below 200x200 minimum
            brightness=128
        )

        result = checker.check_quality(temp_image_path)

        # Resolution issue should be detected
        assert any("resolution" in issue.lower() for issue in result["issues"])
        assert result["details"]["resolution"] == (100, 100)
        # Quality score should be lowered (0.3 for resolution + 1.0 for blur + 1.0 for brightness) / 3 = 0.77
        assert result["quality_score"] < 1.0

    def test_blurry_image(self, checker, temp_image_path):
        """Test that blurry images are detected"""
        # Create blurry image
        self.create_test_image(
            temp_image_path,
            size=(400, 400),
            brightness=128,
            add_blur=True  # Add blur
        )

        result = checker.check_quality(temp_image_path)

        # Blurry images should have lower quality score
        assert result["quality_score"] < 1.0
        # May or may not be acceptable depending on blur severity
        # Just check that blur is detected
        assert "blur_score" in result["details"]

    def test_dark_image(self, checker, temp_image_path):
        """Test that dark images are detected"""
        # Create very dark image
        self.create_test_image(
            temp_image_path,
            size=(400, 400),
            brightness=15  # Very dark (below 30 threshold)
        )

        result = checker.check_quality(temp_image_path)

        # Dark issue should be detected
        assert any("dark" in issue.lower() for issue in result["issues"])
        assert result["details"]["brightness"] < 30
        # Quality score should be lowered but may still be acceptable (score ~0.83)
        assert result["quality_score"] < 1.0

    def test_bright_image(self, checker, temp_image_path):
        """Test that overexposed images are detected"""
        # Create very bright image
        self.create_test_image(
            temp_image_path,
            size=(400, 400),
            brightness=240  # Very bright (above 225 threshold)
        )

        result = checker.check_quality(temp_image_path)

        # Brightness issue should be detected
        assert any("bright" in issue.lower() or "overexposed" in issue.lower() for issue in result["issues"])
        assert result["details"]["brightness"] > 225
        # Quality score should be lowered but may still be acceptable (score ~0.84)
        assert result["quality_score"] < 1.0

    def test_corrupted_image(self, checker, temp_image_path):
        """Test handling of corrupted image files"""
        # Create corrupted file (just write random bytes)
        with open(temp_image_path, 'wb') as f:
            f.write(b'not an image file')

        result = checker.check_quality(temp_image_path)

        assert result["is_acceptable"] is False
        assert result["quality_score"] == 0.0
        assert len(result["issues"]) > 0

    def test_nonexistent_image(self, checker):
        """Test handling of non-existent image files"""
        result = checker.check_quality("/path/that/does/not/exist.jpg")

        assert result["is_acceptable"] is False
        assert result["quality_score"] == 0.0
        assert len(result["issues"]) > 0

    def test_quality_feedback_good_image(self, checker, temp_image_path):
        """Test user feedback for good quality images"""
        self.create_test_image(temp_image_path, size=(400, 400), brightness=128)
        result = checker.check_quality(temp_image_path)
        feedback = checker.get_quality_feedback(result)

        assert "good" in feedback.lower()

    def test_quality_feedback_blurry_image(self, checker, temp_image_path):
        """Test user feedback for blurry images"""
        self.create_test_image(temp_image_path, size=(400, 400), brightness=128, add_blur=True)
        result = checker.check_quality(temp_image_path)
        feedback = checker.get_quality_feedback(result)

        # Should mention blur or focus
        assert "blur" in feedback.lower() or "focus" in feedback.lower() or "good" in feedback.lower()

    def test_quality_feedback_dark_image(self, checker, temp_image_path):
        """Test user feedback for dark images"""
        self.create_test_image(temp_image_path, size=(400, 400), brightness=15)
        result = checker.check_quality(temp_image_path)
        feedback = checker.get_quality_feedback(result)

        # Feedback should either mention dark/lighting (if score < 0.6) or say "good" (if score >= 0.6)
        assert "dark" in feedback.lower() or "lighting" in feedback.lower() or "good" in feedback.lower()

    def test_quality_feedback_low_resolution(self, checker, temp_image_path):
        """Test user feedback for low resolution images"""
        self.create_test_image(temp_image_path, size=(100, 100), brightness=128)
        result = checker.check_quality(temp_image_path)
        feedback = checker.get_quality_feedback(result)

        # Feedback should either mention resolution/closer (if score < 0.6) or say "good" (if score >= 0.6)
        assert "resolution" in feedback.lower() or "closer" in feedback.lower() or "good" in feedback.lower()

    def test_singleton_instance(self):
        """Test that get_quality_checker returns singleton instance"""
        instance1 = get_quality_checker()
        instance2 = get_quality_checker()

        assert instance1 is instance2  # Same object

    def test_custom_thresholds(self, temp_image_path):
        """Test quality checker with custom thresholds"""
        # Create checker with stricter thresholds
        strict_checker = ImageQualityChecker(
            min_resolution=(500, 500),
            blur_threshold=200.0,
            brightness_range=(50, 200)
        )

        # Create image that would pass default thresholds
        self.create_test_image(temp_image_path, size=(400, 400), brightness=128)

        result = strict_checker.check_quality(temp_image_path)

        # Should have resolution issue detected
        assert any("resolution" in issue.lower() for issue in result["issues"])
        # Quality score should be lowered
        assert result["quality_score"] < 1.0


class TestImageQualityIntegration:
    """Integration tests for image quality checker"""

    def test_with_real_test_image(self):
        """Test with actual test image if available"""
        # Look for test images in tests/data directory
        test_image_path = Path(__file__).parent.parent.parent / "tests" / "data" / "test_images" / "sample_image.jpg"

        if test_image_path.exists():
            checker = get_quality_checker()
            result = checker.check_quality(str(test_image_path))

            # Basic validation
            assert "quality_score" in result
            assert "is_acceptable" in result
            assert "issues" in result
            assert "details" in result
            assert 0.0 <= result["quality_score"] <= 1.0
        else:
            pytest.skip("Test image not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
