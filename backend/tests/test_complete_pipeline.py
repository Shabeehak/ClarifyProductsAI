"""
Complete Pipeline Test - Tests CLIP + OCR + Gemini on Real Images

Tests the full production pipeline:
1. Image Quality Check
2. CLIP Visual Analysis
3. OCR Text Extraction
4. Gemini Multimodal Extraction

Run from backend/tests: python test_complete_pipeline.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.complete_recognition_service import get_complete_recognition_service

# Setup loguru logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
    encoding="utf-8"
)


def calculate_match_ratio(expected: str, extracted: str) -> float:
    """Calculate word match ratio."""
    expected_words = set(expected.lower().split())
    extracted_words = set(extracted.lower().split())

    if not expected_words:
        return 0.0

    common_words = expected_words & extracted_words
    return len(common_words) / len(expected_words)


def test_complete_pipeline():
    """Test complete pipeline on actual dataset."""
    logger.info("="*80)
    logger.info("Complete Pipeline Test - CLIP + OCR + Gemini")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Load dataset
    tests_dir = Path(__file__).parent
    dataset_dir = tests_dir / "data" / "ocr_test_dataset"
    labels_file = dataset_dir / "labels.json"

    if not labels_file.exists():
        logger.error(f"labels.json not found at {labels_file}")
        print(f"\nERROR: labels.json not found. Please run from backend/tests directory.")
        return 0.0

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    images_to_test = labels_data["images"]

    # Filter out TODO entries and select subset for testing
    valid_images = [
        img for img in images_to_test
        if not img["expected_text"].startswith("[TODO")
    ]

    # Test on a subset (first 10 images to save API quota)
    test_images = valid_images[:10]

    logger.info(f"\nFound {len(valid_images)} valid labeled images")
    logger.info(f"Testing on {len(test_images)} images (subset to save API quota)")
    logger.info("="*80)

    # Initialize service
    logger.info("Initializing Complete Recognition Service...")
    try:
        service = get_complete_recognition_service()
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("1. PaddleOCR installed: pip install paddleocr")
        print("2. OpenCV installed: pip install opencv-python")
        print("3. GEMINI_API_KEY in .env file")
        return 0.0

    results = []
    correct_count = 0

    for i, img_data in enumerate(test_images, 1):
        image_path = dataset_dir / img_data["image_path"]
        expected = img_data["expected_text"]
        quality = img_data.get("quality", "unknown")

        logger.info(f"\n[{i}/{len(test_images)}] {img_data['image_path']}")
        logger.info(f"Expected:  {expected}")
        logger.info(f"Quality:   {quality}")

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            results.append({
                "image": img_data["image_path"],
                "expected": expected,
                "extracted": None,
                "status": "FILE_NOT_FOUND",
                "match_ratio": 0.0
            })
            continue

        # Run complete pipeline
        result = service.recognize_product_complete(str(image_path))

        extracted = result.get("product_name")
        confidence = result.get("confidence")
        method = result.get("method")

        logger.info(f"Extracted: {extracted}")
        logger.info(f"Method:    {method}")
        logger.info(f"Confidence: {confidence}")

        if extracted:
            match_ratio = calculate_match_ratio(expected, extracted)

            if match_ratio >= 0.7:
                logger.info(f"CORRECT (match: {match_ratio:.1%})")
                correct_count += 1
                status = "CORRECT"
            else:
                logger.warning(f"WRONG (match: {match_ratio:.1%})")
                status = "WRONG"

            results.append({
                "image": img_data["image_path"],
                "expected": expected,
                "extracted": extracted,
                "status": status,
                "match_ratio": match_ratio,
                "confidence": confidence,
                "method": method,
                "quality": quality
            })
        else:
            logger.error(f"FAILED: {result.get('message')}")
            results.append({
                "image": img_data["image_path"],
                "expected": expected,
                "extracted": None,
                "status": "FAILED",
                "match_ratio": 0.0,
                "confidence": "unclear",
                "method": method,
                "quality": quality,
                "error": result.get("message")
            })

    # Calculate statistics
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0.0

    logger.info(f"\nTotal images tested: {total}")
    logger.info(f"Correct extractions: {correct_count}")
    logger.info(f"Overall accuracy:    {accuracy:.1f}%")

    # Breakdown by quality
    by_quality = {}
    for result in results:
        q = result.get("quality", "unknown")
        if q not in by_quality:
            by_quality[q] = {"correct": 0, "total": 0}
        by_quality[q]["total"] += 1
        if result["status"] == "CORRECT":
            by_quality[q]["correct"] += 1

    logger.info("\nAccuracy by Image Quality:")
    for quality, stats in sorted(by_quality.items()):
        q_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        logger.info(f"  {quality:10s}: {stats['correct']}/{stats['total']} ({q_accuracy:.1f}%)")

    # Show detailed results
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS")
    logger.info("="*80)

    for result in results:
        status_text = "CORRECT" if result["status"] == "CORRECT" else result["status"]
        logger.info(f"\n{status_text}: {result['image']}")
        logger.info(f"  Expected:   {result['expected']}")
        logger.info(f"  Extracted:  {result.get('extracted', 'None')}")
        if result.get("match_ratio") is not None:
            logger.info(f"  Match:      {result['match_ratio']:.1%}")
        logger.info(f"  Method:     {result.get('method', 'N/A')}")
        logger.info(f"  Confidence: {result.get('confidence', 'N/A')}")

    logger.info("\n" + "="*80)
    logger.info("Complete Pipeline Test Finished")
    logger.info("="*80)

    return accuracy


def main():
    """Run complete pipeline test."""
    try:
        accuracy = test_complete_pipeline()

        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Complete Pipeline Accuracy: {accuracy:.1f}%")
        print()
        print("Comparison to previous approaches:")
        print(f"  Raw OCR:              48.0% (baseline)")
        print(f"  GLiNER NER:            0.0% (failed)")
        print(f"  Text Cleaning:        25.0% (limited)")
        print(f"  Gemini OCR-only:      12.5% (hallucination)")
        print(f"  Complete Pipeline:    {accuracy:.1f}% (CLIP + OCR + Gemini)")

        if accuracy >= 70:
            print("\nSUCCESS: Target accuracy achieved!")
            print("The multimodal approach (CLIP + OCR + Gemini) works!")
        elif accuracy >= 60:
            print("\nGOOD: Significant improvement over baseline!")
            print("Consider adding image quality rejection to reach 70%+")
        else:
            print("\nNEEDS ANALYSIS: Check logs for common failure patterns")

        print(f"\nDetailed log: {log_file}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
