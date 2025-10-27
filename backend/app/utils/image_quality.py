"""
Image Quality Checker - Validates image quality before OCR/recognition

Checks for:
- Blur detection (Laplacian variance)
- Brightness/darkness (histogram analysis)
- Resolution (minimum dimensions)
- File corruption
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
from loguru import logger


class ImageQualityChecker:
    """Check image quality for product recognition tasks."""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (200, 200),
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (30, 225),
    ):
        """
        Initialize quality checker with thresholds.

        Args:
            min_resolution: Minimum (width, height) in pixels
            blur_threshold: Laplacian variance threshold (higher = sharper)
            brightness_range: Acceptable (min, max) average brightness [0-255]
        """
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range

    def check_quality(self, image_path: str) -> Dict:
        """
        Check image quality and return assessment.

        Args:
            image_path: Path to image file

        Returns:
            {
                "quality_score": float (0-1),
                "is_acceptable": bool,
                "issues": List[str],
                "details": {
                    "resolution": tuple,
                    "blur_score": float,
                    "brightness": float
                }
            }
        """
        try:
            # Load image
            img_pil = Image.open(image_path)
            img_cv = cv2.imread(image_path)

            if img_cv is None:
                return {
                    "quality_score": 0.0,
                    "is_acceptable": False,
                    "issues": ["Failed to load image - file may be corrupted"],
                    "details": {},
                }

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            issues = []
            scores = []

            # Check 1: Resolution
            width, height = img_pil.size
            resolution_ok = (
                width >= self.min_resolution[0] and height >= self.min_resolution[1]
            )

            if not resolution_ok:
                issues.append(
                    f"Resolution too low: {width}x{height} (minimum: {self.min_resolution[0]}x{self.min_resolution[1]})"
                )
                scores.append(0.3)  # Low score for poor resolution
            else:
                scores.append(1.0)  # Good resolution

            # Check 2: Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < self.blur_threshold:
                issues.append(
                    f"Image is blurry (blur score: {laplacian_var:.1f}, threshold: {self.blur_threshold})"
                )
                blur_score = min(
                    laplacian_var / self.blur_threshold, 1.0
                )  # Normalize to 0-1
                scores.append(blur_score)
            else:
                scores.append(1.0)  # Sharp image

            # Check 3: Brightness
            avg_brightness = np.mean(gray)

            if avg_brightness < self.brightness_range[0]:
                issues.append(f"Image is too dark (brightness: {avg_brightness:.1f})")
                brightness_score = avg_brightness / self.brightness_range[0]
                scores.append(brightness_score)
            elif avg_brightness > self.brightness_range[1]:
                issues.append(
                    f"Image is too bright/overexposed (brightness: {avg_brightness:.1f})"
                )
                brightness_score = 1.0 - (
                    (avg_brightness - self.brightness_range[1])
                    / (255 - self.brightness_range[1])
                )
                scores.append(max(brightness_score, 0.3))
            else:
                scores.append(1.0)  # Good brightness

            # Calculate overall quality score
            quality_score = np.mean(scores)

            # Image is acceptable if score >= 0.6
            is_acceptable = quality_score >= 0.6

            result = {
                "quality_score": quality_score,
                "is_acceptable": is_acceptable,
                "issues": issues,
                "details": {
                    "resolution": (width, height),
                    "blur_score": laplacian_var,
                    "brightness": avg_brightness,
                },
            }

            if is_acceptable:
                logger.info(
                    f"Image quality OK: score={quality_score:.2f}, resolution={width}x{height}"
                )
            else:
                logger.warning(
                    f"Image quality LOW: score={quality_score:.2f}, issues={len(issues)}"
                )

            return result

        except Exception as e:
            logger.error(f"Error checking image quality: {e}")
            return {
                "quality_score": 0.0,
                "is_acceptable": False,
                "issues": [f"Error analyzing image: {str(e)}"],
                "details": {},
            }

    def get_quality_feedback(self, quality_result: Dict) -> str:
        """
        Generate user-friendly feedback message.

        Args:
            quality_result: Result from check_quality()

        Returns:
            User-friendly message string
        """
        if quality_result["is_acceptable"]:
            return "Image quality is good"

        if not quality_result["issues"]:
            return "Image quality could not be assessed"

        # Prioritize issues for user feedback
        issues = quality_result["issues"]

        if any("blurry" in issue.lower() for issue in issues):
            return (
                "Image is too blurry. Please hold camera steady and focus on product."
            )

        if any("dark" in issue.lower() for issue in issues):
            return "Image is too dark. Please use better lighting."

        if any("bright" in issue.lower() for issue in issues):
            return "Image is overexposed. Please reduce lighting or avoid glare."

        if any("resolution" in issue.lower() for issue in issues):
            return "Image resolution is too low. Please move closer to the product."

        # Generic message
        return "Image quality is low. Please retake with better lighting and focus."


# Singleton instance
_quality_checker = None


def get_quality_checker() -> ImageQualityChecker:
    """Get singleton image quality checker instance."""
    global _quality_checker
    if _quality_checker is None:
        _quality_checker = ImageQualityChecker()
    return _quality_checker
