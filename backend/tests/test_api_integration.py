"""
Test API Integration with Complete Recognition Pipeline
Tests the FastAPI endpoint with the new complete pipeline
"""
import requests
from pathlib import Path
from loguru import logger

# API configuration
API_URL = "http://localhost:8000/api/v1/recognition/"
TEST_IMAGE_DIR = Path(__file__).parent / "labeled_test_images"


def test_single_image_upload():
    """Test single image upload to API"""
    logger.info("Testing single image upload...")

    # Use a high-quality test image
    test_image = TEST_IMAGE_DIR / "high_quality" / "cetaphile_gentle_skin_cleanser.jpg"

    if not test_image.exists():
        logger.error(f"Test image not found: {test_image}")
        return

    # Upload image
    with open(test_image, "rb") as f:
        files = {"image": (test_image.name, f, "image/jpeg")}
        response = requests.post(API_URL, files=files)

    # Check response
    if response.status_code == 200:
        result = response.json()
        logger.info(f"✅ SUCCESS: {response.status_code}")
        logger.info(f"   Product: {result.get('primary_match', {}).get('product_name')}")
        logger.info(f"   Confidence: {result.get('primary_match', {}).get('confidence')}")
        logger.info(f"   Category: {result.get('primary_match', {}).get('category')}")
        logger.info(f"   Message: {result.get('message')}")
        return True
    else:
        logger.error(f"❌ FAILED: {response.status_code}")
        logger.error(f"   Error: {response.text}")
        return False


def test_batch_upload():
    """Test batch image upload"""
    logger.info("\nTesting batch upload (3 images)...")

    test_images = [
        TEST_IMAGE_DIR / "high_quality" / "cetaphile_gentle_skin_cleanser.jpg",
        TEST_IMAGE_DIR / "high_quality" / "aveeno_daily_moisturizing_body_lotion.jpg",
        TEST_IMAGE_DIR / "high_quality" / "neutrogena_hydro_boost_hydrating_lotion.jpg",
    ]

    # Filter existing images
    existing_images = [img for img in test_images if img.exists()]

    if not existing_images:
        logger.error("No test images found for batch upload")
        return False

    # Upload images
    files = [
        ("images", (img.name, open(img, "rb"), "image/jpeg"))
        for img in existing_images
    ]

    response = requests.post(API_URL + "batch", files=files)

    # Close file handles
    for _, file_tuple in files:
        file_tuple[1].close()

    # Check response
    if response.status_code == 200:
        results = response.json()
        logger.info(f"✅ SUCCESS: {response.status_code}")
        logger.info(f"   Processed {len(results)} images:")

        for i, result in enumerate(results, 1):
            product_name = result.get('primary_match', {}).get('product_name', 'Unknown')
            confidence = result.get('primary_match', {}).get('confidence', 0)
            logger.info(f"   [{i}] {product_name} (confidence: {confidence:.2f})")

        return True
    else:
        logger.error(f"❌ FAILED: {response.status_code}")
        logger.error(f"   Error: {response.text}")
        return False


def test_invalid_file():
    """Test error handling with invalid file"""
    logger.info("\nTesting invalid file upload...")

    # Try to upload a text file
    files = {"image": ("test.txt", b"Not an image", "text/plain")}
    response = requests.post(API_URL, files=files)

    if response.status_code == 400:
        logger.info(f"✅ Correctly rejected invalid file: {response.status_code}")
        return True
    else:
        logger.warning(f"⚠️  Expected 400, got {response.status_code}")
        return False


def main():
    """Run all API tests"""
    logger.info("="* 80)
    logger.info("API Integration Tests - Complete Recognition Pipeline")
    logger.info("="* 80)

    logger.info(f"\nAPI Endpoint: {API_URL}")
    logger.info(f"Test Images: {TEST_IMAGE_DIR}")

    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code != 200:
            logger.error("❌ FastAPI server is not running!")
            logger.info("\nPlease start the server first:")
            logger.info("  cd backend")
            logger.info("  uvicorn app.main:app --reload")
            return
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to FastAPI server!")
        logger.info("\nPlease start the server first:")
        logger.info("  cd backend")
        logger.info("  uvicorn app.main:app --reload")
        return

    # Run tests
    results = []
    results.append(("Single Image Upload", test_single_image_upload()))
    results.append(("Batch Upload", test_batch_upload()))
    results.append(("Invalid File Handling", test_invalid_file()))

    # Summary
    logger.info("\n" + "="* 80)
    logger.info("TEST SUMMARY")
    logger.info("="* 80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:.<50} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    logger.info(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        logger.info("\n🎉 All API integration tests passed!")
    else:
        logger.warning("\n⚠️  Some tests failed. Please check the logs.")


if __name__ == "__main__":
    main()
