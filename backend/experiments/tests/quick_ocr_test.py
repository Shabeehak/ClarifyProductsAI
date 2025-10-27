"""
Quick OCR Test - Tests PaddleOCR and EasyOCR with sample_image.jpg
Run from backend/tests directory: python quick_ocr_test.py
"""
import sys
import os
from datetime import datetime
from pathlib import Path
from loguru import logger

# Setup loguru logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"quick_ocr_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure loguru
logger.remove()  # Remove default handler
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

def test_paddle():
    """Test PaddleOCR"""
    logger.info("="*60)
    logger.info("Testing PaddleOCR")
    logger.info("="*60)

    print("\n" + "="*60)
    print("Testing PaddleOCR...")
    print("="*60)

    try:
        from paddleocr import PaddleOCR
        logger.info("PaddleOCR imported successfully")

        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        logger.info("PaddleOCR model initialized")

        # Use sample_image.jpg
        logger.info("Running PaddleOCR on sample_image.jpg")
        result = ocr.predict("sample_image.jpg")

        if result and len(result) > 0:
            # PaddleOCR 3.3.0+ returns dict with 'rec_texts' and 'rec_scores'
            result_dict = result[0]
            if 'rec_texts' in result_dict and result_dict['rec_texts']:
                texts = result_dict['rec_texts']
                scores = result_dict['rec_scores']

                logger.info(f"PaddleOCR SUCCESS - Extracted {len(texts)} text regions")
                for text, score in zip(texts, scores):
                    logger.info(f"  Extracted: '{text}' (confidence: {score:.2f})")

                print("SUCCESS: PaddleOCR is working!")
                print("\nExtracted text:")
                for text, score in zip(texts, scores):
                    print(f"  - {text} (conf: {score:.2f})")
                return True
            else:
                logger.warning("PaddleOCR: No text extracted")
                print("FAILED: No text extracted")
                return False
        else:
            logger.warning("PaddleOCR: No results returned")
            print("FAILED: No results")
            return False

    except Exception as e:
        logger.error(f"PaddleOCR FAILED: {str(e)}", exc_info=True)
        print(f"FAILED: {e}")
        return False


def test_easy():
    """Test EasyOCR"""
    print("\n" + "="*60)
    print("Testing EasyOCR...")
    print("="*60)

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        # Use sample_image.jpg
        result = reader.readtext("sample_image.jpg")

        if result:
            print("SUCCESS: EasyOCR is working!")
            print("\nExtracted text:")
            for detection in result:
                bbox, text, confidence = detection
                print(f"  - {text} (conf: {confidence:.2f})")
            return True
        else:
            print("FAILED: No results")
            return False

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("="*60)
    print("Quick OCR Test")
    print("="*60)

    # Check if sample_image.jpg exists
    if not os.path.exists("sample_image.jpg"):
        print("\nERROR: sample_image.jpg not found!")
        print("Please run this from backend/tests directory")
        print("Or place sample_image.jpg in current directory")
        return

    print("\nFound: sample_image.jpg")

    # Test both
    paddle_ok = test_paddle()
    easy_ok = test_easy()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PaddleOCR: {'WORKING' if paddle_ok else 'FAILED'}")
    print(f"EasyOCR: {'WORKING' if easy_ok else 'FAILED'}")

    if paddle_ok and easy_ok:
        print("\nSUCCESS: Both OCR engines working!")
        print("You can now collect test images and run comprehensive comparison.")
    elif paddle_ok or easy_ok:
        print(f"\nPARTIAL: {1 if paddle_ok else 0 + 1 if easy_ok else 0}/2 engines working")
        print("You can proceed with the working engine(s)")
    else:
        print("\nFAILED: No OCR engines working")


if __name__ == "__main__":
    main()
