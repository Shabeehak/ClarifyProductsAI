"""
Comprehensive OCR Engine Comparison Test
Compares multiple OCR engines on product image dataset

Usage:
    python backend/tests/ocr_tests/test_ocr_comparison.py

Requirements:
    - Test dataset in backend/tests/data/ocr_test_dataset/
    - labels.json with ground truth annotations
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
from loguru import logger

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# Setup loguru logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"ocr_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


class OCRComparison:
    """Compare multiple OCR engines on product images"""

    def __init__(self, dataset_path: str = None):
        # Default to data/ocr_test_dataset relative to tests directory
        if dataset_path is None:
            tests_dir = Path(__file__).parent.parent
            self.dataset_path = tests_dir / "data" / "ocr_test_dataset"
        else:
            self.dataset_path = Path(dataset_path)

        logger.info(f"Initializing OCR Comparison with dataset: {self.dataset_path}")
        self.labels = self.load_labels()
        self.results = {
            "PaddleOCR": {"correct": 0, "total": 0, "times": [], "confidences": [], "errors": []},
            "EasyOCR": {"correct": 0, "total": 0, "times": [], "confidences": [], "errors": []},
            "Tesseract": {"correct": 0, "total": 0, "times": [], "confidences": [], "errors": []}
        }
        logger.info(f"Found {len(self.labels.get('images', []))} images in labels.json")

    def load_labels(self) -> Dict:
        """Load ground truth labels"""
        labels_path = self.dataset_path / "labels.json"

        if not labels_path.exists():
            logger.warning(f"labels.json not found at {labels_path}")
            logger.warning("Creating empty labels file. Please add your ground truth data.")
            return {"images": []}

        logger.info(f"Loading labels from {labels_path}")
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            logger.info(f"Loaded {len(labels.get('images', []))} image entries")
            return labels

    def test_paddle_ocr(self, image_path: str) -> Dict[str, Any]:
        """Test PaddleOCR on an image"""
        logger.debug(f"Testing PaddleOCR on {image_path}")
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 3.3.0+ uses predict() instead of ocr()
            ocr = PaddleOCR(use_textline_orientation=True, lang='en')

            start_time = time.time()
            result = ocr.predict(image_path)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.debug(f"PaddleOCR processing time: {elapsed_time:.0f}ms")

            if result and len(result) > 0:
                # PaddleOCR 3.3.0+ returns dict with 'rec_texts' and 'rec_scores'
                result_dict = result[0]
                if 'rec_texts' in result_dict and result_dict['rec_texts']:
                    texts = result_dict['rec_texts']
                    confidences = result_dict['rec_scores']

                    extracted_text = " ".join(texts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                else:
                    extracted_text = ""
                    avg_confidence = 0.0

                return {
                    "text": extracted_text,
                    "confidence": avg_confidence,
                    "time_ms": elapsed_time,
                    "success": True
                }
            else:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "time_ms": elapsed_time,
                    "success": False,
                    "error": "No text detected"
                }

        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "time_ms": 0,
                "success": False,
                "error": str(e)
            }

    def test_easy_ocr(self, image_path: str) -> Dict[str, Any]:
        """Test EasyOCR on an image"""
        try:
            import easyocr

            reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            start_time = time.time()
            result = reader.readtext(image_path)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            if result:
                # Combine all detected text
                texts = [detection[1] for detection in result]
                confidences = [detection[2] for detection in result]

                extracted_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                return {
                    "text": extracted_text,
                    "confidence": avg_confidence,
                    "time_ms": elapsed_time,
                    "success": True
                }
            else:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "time_ms": elapsed_time,
                    "success": False,
                    "error": "No text detected"
                }

        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "time_ms": 0,
                "success": False,
                "error": str(e)
            }

    def test_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Test Tesseract on an image"""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)

            start_time = time.time()
            extracted_text = pytesseract.image_to_string(img).strip()
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            return {
                "text": extracted_text,
                "confidence": 0.0,  # Tesseract doesn't provide confidence easily
                "time_ms": elapsed_time,
                "success": bool(extracted_text)
            }

        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "time_ms": 0,
                "success": False,
                "error": str(e)
            }

    def calculate_accuracy(self, extracted: str, expected: str) -> bool:
        """
        Simple accuracy check - considers it correct if expected text
        is found within extracted text (case-insensitive, partial match)
        """
        extracted_lower = extracted.lower()
        expected_lower = expected.lower()

        # Check if key words from expected are in extracted
        expected_words = expected_lower.split()
        matched_words = sum(1 for word in expected_words if word in extracted_lower)

        # Consider correct if at least 70% of expected words are found
        accuracy_threshold = 0.7
        return (matched_words / len(expected_words)) >= accuracy_threshold if expected_words else False

    def run_comparison(self):
        """Run comparison on all images in dataset"""
        logger.info("="*80)
        logger.info("Starting Comprehensive OCR Comparison")
        logger.info("="*80)

        print("\n" + "="*80)
        print("Comprehensive OCR Engine Comparison")
        print("="*80)

        if not self.labels.get("images"):
            error_msg = "No test images found in labels.json"
            logger.error(error_msg)
            print(f"\n✗ {error_msg}")
            print("✗ Please add test images and ground truth to:")
            print(f"   {self.dataset_path / 'labels.json'}")
            return

        total_images = len(self.labels["images"])
        logger.info(f"Testing {total_images} images")
        print(f"\nTesting {total_images} images...")
        print("-"*80)

        for idx, image_data in enumerate(self.labels["images"], 1):
            image_path = self.dataset_path / image_data["image_path"]
            expected_text = image_data["expected_text"]
            category = image_data.get("category", "unknown")

            logger.info(f"[{idx}/{total_images}] Testing: {image_path.name}, Category: {category}, Expected: {expected_text}")

            print(f"\n[{idx}/{total_images}] Testing: {image_path.name}")
            print(f"Category: {category}")
            print(f"Expected: {expected_text}")

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                print(f"✗ Image not found: {image_path}")
                continue

            # Test PaddleOCR
            print("\n  PaddleOCR...", end=" ")
            paddle_result = self.test_paddle_ocr(str(image_path))
            if paddle_result["success"]:
                is_correct = self.calculate_accuracy(paddle_result["text"], expected_text)
                self.results["PaddleOCR"]["total"] += 1
                self.results["PaddleOCR"]["times"].append(paddle_result["time_ms"])
                self.results["PaddleOCR"]["confidences"].append(paddle_result["confidence"])

                if is_correct:
                    self.results["PaddleOCR"]["correct"] += 1
                    logger.info(f"PaddleOCR: CORRECT ({paddle_result['time_ms']:.0f}ms, conf: {paddle_result['confidence']:.2f})")
                    print(f"✓ CORRECT ({paddle_result['time_ms']:.0f}ms, conf: {paddle_result['confidence']:.2f})")
                else:
                    logger.info(f"PaddleOCR: WRONG - Got: {paddle_result['text'][:100]}")
                    print(f"✗ WRONG ({paddle_result['time_ms']:.0f}ms)")
                    print(f"    Got: {paddle_result['text'][:50]}...")
            else:
                logger.error(f"PaddleOCR: FAILED - {paddle_result.get('error', 'Unknown error')}")
                print(f"✗ FAILED: {paddle_result.get('error', 'Unknown error')}")
                self.results["PaddleOCR"]["errors"].append(str(image_path.name))

            # Test EasyOCR
            print("  EasyOCR...", end=" ")
            easy_result = self.test_easy_ocr(str(image_path))
            if easy_result["success"]:
                is_correct = self.calculate_accuracy(easy_result["text"], expected_text)
                self.results["EasyOCR"]["total"] += 1
                self.results["EasyOCR"]["times"].append(easy_result["time_ms"])
                self.results["EasyOCR"]["confidences"].append(easy_result["confidence"])

                if is_correct:
                    self.results["EasyOCR"]["correct"] += 1
                    logger.info(f"EasyOCR: CORRECT ({easy_result['time_ms']:.0f}ms, conf: {easy_result['confidence']:.2f})")
                    print(f"✓ CORRECT ({easy_result['time_ms']:.0f}ms, conf: {easy_result['confidence']:.2f})")
                else:
                    logger.info(f"EasyOCR: WRONG - Got: {easy_result['text'][:100]}")
                    print(f"✗ WRONG ({easy_result['time_ms']:.0f}ms)")
                    print(f"    Got: {easy_result['text'][:50]}...")
            else:
                print(f"✗ FAILED: {easy_result.get('error', 'Unknown error')}")
                self.results["EasyOCR"]["errors"].append(str(image_path.name))

            # Test Tesseract
            print("  Tesseract...", end=" ")
            tess_result = self.test_tesseract(str(image_path))
            if tess_result["success"]:
                is_correct = self.calculate_accuracy(tess_result["text"], expected_text)
                self.results["Tesseract"]["total"] += 1
                self.results["Tesseract"]["times"].append(tess_result["time_ms"])

                if is_correct:
                    self.results["Tesseract"]["correct"] += 1
                    print(f"✓ CORRECT ({tess_result['time_ms']:.0f}ms)")
                else:
                    print(f"✗ WRONG ({tess_result['time_ms']:.0f}ms)")
                    print(f"    Got: {tess_result['text'][:50]}...")
            else:
                print(f"✗ FAILED: {tess_result.get('error', 'Unknown error')}")
                self.results["Tesseract"]["errors"].append(str(image_path.name))

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        print(f"\n{'Engine':<15} {'Accuracy':<12} {'Avg Time':<12} {'Avg Conf':<12} {'Status'}")
        print("-"*80)

        for engine_name, data in self.results.items():
            if data["total"] > 0:
                accuracy = (data["correct"] / data["total"]) * 100
                avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
                avg_conf = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0

                status = "✓ PASS" if accuracy >= 85 else "⚠ NEEDS IMPROVEMENT"

                print(f"{engine_name:<15} {accuracy:>6.1f}%      {avg_time:>6.0f}ms      {avg_conf:>6.2f}       {status}")
            else:
                print(f"{engine_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'✗ NO DATA'}")

        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        # Find best engine
        best_engine = None
        best_accuracy = 0

        for engine_name, data in self.results.items():
            if data["total"] > 0:
                accuracy = (data["correct"] / data["total"]) * 100
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_engine = engine_name

        if best_engine:
            print(f"\n✓ Best performing engine: {best_engine} ({best_accuracy:.1f}% accuracy)")
            print(f"✓ Recommended for primary OCR in cascade")

            # Suggest cascade order
            engines_by_accuracy = sorted(
                [(name, (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0)
                 for name, data in self.results.items()],
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n✓ Suggested cascade order:")
            for idx, (engine, acc) in enumerate(engines_by_accuracy, 1):
                if acc > 0:
                    print(f"   {idx}. {engine} ({acc:.1f}% accuracy)")

        print("\n" + "="*80)

    def save_results(self):
        """Save results to JSON file"""
        output_path = self.dataset_path / "ocr_comparison_results.json"

        # Calculate summary statistics
        summary = {}
        for engine_name, data in self.results.items():
            if data["total"] > 0:
                summary[engine_name] = {
                    "accuracy": (data["correct"] / data["total"]) * 100,
                    "total_tests": data["total"],
                    "correct": data["correct"],
                    "avg_time_ms": sum(data["times"]) / len(data["times"]) if data["times"] else 0,
                    "avg_confidence": sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0,
                    "errors": data["errors"]
                }
                logger.info(f"{engine_name} Results: {summary[engine_name]['accuracy']:.1f}% accuracy, {summary[engine_name]['avg_time_ms']:.0f}ms avg time")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Run the OCR comparison"""
    logger.info("="*80)
    logger.info("OCR Comparison Test Started")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    comparison = OCRComparison()
    comparison.run_comparison()

    logger.info("OCR Comparison Test Completed")
    logger.info("="*80)

    print("\n" + "="*80)
    print("Test Summary:")
    print("="*80)
    print("1. Results saved to: ocr_comparison_results.json")
    print("2. Detailed log saved to: " + str(log_file.name))
    print("3. Review accuracy metrics to select optimal OCR engine")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
