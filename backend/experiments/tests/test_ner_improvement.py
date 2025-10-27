"""
NER Improvement Test - Measures accuracy improvement with GLiNER NER

Tests OCR + NER pipeline to show improvement from 48% to target 85%+ accuracy.
Run from backend/tests directory: python test_ner_improvement.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ner_service import get_ner_service

# Setup loguru logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"ner_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


def test_ner_extraction():
    """Test NER extraction on sample OCR outputs from log file."""
    logger.info("="*80)
    logger.info("NER Improvement Test Started")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Initialize NER service
    logger.info("Initializing NER service with GLiNER")
    ner_service = get_ner_service()
    ner_service.load_model()

    # Test cases from actual OCR failures in log file
    test_cases = [
        {
            "expected": "Garnier Micellar Cleansing Water",
            "ocr_output": "BIGGER SIZE GARNIER SKINACTIVE Micellar Cleansing Water",
            "source": "Line 35: garnier_micellar_cleansing_water.jpg"
        },
        {
            "expected": "Anua Heartleaf Quercetinol Pore Deep Cleansing Foam",
            "ocr_output": "Anua HEARTLEAF + BHA HEARTLEAF QUERCETINO ORE YANTEEN",
            "source": "Line 17: ANUA_Heartleaf_Quercetinol_Pore_Deep_Cleansing_Foam.jpg"
        },
        {
            "expected": "Kellogg's Rice Krispies Multigrain Shapes cereal",
            "ocr_output": "Kelloy Rice: KrisPies MN DAILY SOURCE OF VITAMINS AND IRON SHAPES HIGH IN FIBRE SNAD 25 0/o OFF GREA",
            "source": "Line 68: kellogs_rice_krispies.jpg"
        },
        {
            "expected": "Purina Gourmet Perle Chef's Collection 40 pouches",
            "ocr_output": "3N  GOURMET PURINA  POUCHES 40",
            "source": "Line 59: cat_food_gourmet_perle_collection.jpg"
        },
        {
            "expected": "Brüggen Corn Flakes",
            "ocr_output": "Brigge CORN FLAKES Good morning",
            "source": "Line 65: corn_flakes_briggen.jpeg"
        },
        {
            "expected": "Maybelline Instant Anti-Age Eraser Eye Concealer For Face",
            "ocr_output": "00 CONTERR M-LTSE EAR STAANTGE MAYBELLINE",
            "source": "Line 38: Maybelline Instant Anti-Age Eraser Eye Concealer for Face.jpg"
        },
        {
            "expected": "Perfect Fit Adult 1+ Complete Dry Cat Food",
            "ocr_output": "WITH HIGH ITYINGREDIENTS\"  S PERFECT FIT SUPP WHOLE BODY HEALTH ADULT1+ total Chicken 5",
            "source": "Line 83: perfect_fit_cat_1kg_food.jpg"
        },
        {
            "expected": "Ordinary Natural Moisturizing",
            "ocr_output": "Clinical Formulations with Integrity, Formulations Cliniques Empreintes d'integrité. The rdiary. Nat",
            "source": "Line 50: Ordinary_natural_moisturizing.jpg"
        }
    ]

    logger.info(f"\nTesting NER extraction on {len(test_cases)} OCR failure cases")
    logger.info("="*80)

    correct_before = 0  # All these were WRONG in original test
    correct_after = 0
    results = []

    for i, test_case in enumerate(test_cases, 1):
        expected = test_case["expected"]
        ocr_output = test_case["ocr_output"]
        source = test_case["source"]

        logger.info(f"\n[{i}/{len(test_cases)}] {source}")
        logger.info(f"Expected: {expected}")
        logger.info(f"OCR Output: {ocr_output}")

        # Extract product name using NER
        extracted = ner_service.extract_product_name(ocr_output, threshold=0.3)

        if extracted:
            logger.info(f"NER Extracted: {extracted}")

            # Check if NER extraction matches expected
            # Use same matching logic as original test (70% word match)
            expected_words = set(expected.lower().split())
            extracted_words = set(extracted.lower().split())

            if expected_words and extracted_words:
                common_words = expected_words & extracted_words
                match_ratio = len(common_words) / len(expected_words)

                if match_ratio >= 0.7:
                    logger.info(f"✓ CORRECT (match ratio: {match_ratio:.2%})")
                    correct_after += 1
                    results.append({
                        "source": source,
                        "expected": expected,
                        "ocr": ocr_output,
                        "extracted": extracted,
                        "status": "CORRECT",
                        "match_ratio": match_ratio
                    })
                else:
                    logger.warning(f"✗ WRONG (match ratio: {match_ratio:.2%})")
                    results.append({
                        "source": source,
                        "expected": expected,
                        "ocr": ocr_output,
                        "extracted": extracted,
                        "status": "WRONG",
                        "match_ratio": match_ratio
                    })
            else:
                logger.warning("✗ WRONG (empty word sets)")
                results.append({
                    "source": source,
                    "expected": expected,
                    "ocr": ocr_output,
                    "extracted": extracted,
                    "status": "WRONG",
                    "match_ratio": 0.0
                })
        else:
            logger.error("✗ FAILED (NER extraction returned None)")
            results.append({
                "source": source,
                "expected": expected,
                "ocr": ocr_output,
                "extracted": None,
                "status": "FAILED",
                "match_ratio": 0.0
            })

    # Calculate improvement
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    before_accuracy = (correct_before / len(test_cases)) * 100
    after_accuracy = (correct_after / len(test_cases)) * 100
    improvement = after_accuracy - before_accuracy

    logger.info(f"\nBefore NER (Raw OCR):  {correct_before}/{len(test_cases)} correct ({before_accuracy:.1f}%)")
    logger.info(f"After NER (OCR + NER): {correct_after}/{len(test_cases)} correct ({after_accuracy:.1f}%)")
    logger.info(f"Improvement: +{improvement:.1f}%")

    # Show which cases were fixed
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS")
    logger.info("="*80)

    for result in results:
        status_symbol = "✓" if result["status"] == "CORRECT" else "✗"
        logger.info(f"\n{status_symbol} {result['source']}")
        logger.info(f"  Expected:  {result['expected']}")
        logger.info(f"  OCR:       {result['ocr'][:60]}...")
        logger.info(f"  Extracted: {result['extracted']}")
        if result['status'] in ['CORRECT', 'WRONG']:
            logger.info(f"  Match:     {result['match_ratio']:.1%}")

    logger.info("\n" + "="*80)
    logger.info("NER Improvement Test Completed")
    logger.info("="*80)

    return after_accuracy


def main():
    """Run NER improvement test."""
    try:
        accuracy = test_ner_extraction()

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"NER + OCR Accuracy: {accuracy:.1f}%")

        if accuracy >= 85:
            print("\n✓ SUCCESS: Target accuracy (85%+) achieved!")
        elif accuracy >= 70:
            print("\n⚠ PARTIAL: Good improvement, but below target (85%)")
        else:
            print("\n✗ NEEDS WORK: Accuracy still below 70%")

        print(f"\nLog file saved: {log_file}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
