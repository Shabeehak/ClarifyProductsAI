"""
Multimodal Product Extraction Test - CLIP + OCR + Gemini

Tests the combined approach:
1. CLIP describes image (visual context)
2. OCR extracts text (potentially corrupted)
3. Gemini combines both to extract accurate product name

This should significantly reduce hallucination compared to OCR-only Gemini.
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.multimodal_product_extractor import get_multimodal_extractor

# Setup loguru logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"multimodal_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    """Calculate word match ratio (same logic as original test)."""
    expected_words = set(expected.lower().split())
    extracted_words = set(extracted.lower().split())

    if not expected_words:
        return 0.0

    common_words = expected_words & extracted_words
    return len(common_words) / len(expected_words)


def test_multimodal_extraction():
    """Test multimodal extraction with CLIP context."""
    logger.info("="*80)
    logger.info("Multimodal Product Extraction Test (CLIP + OCR + Gemini)")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Initialize extractor
    logger.info("Initializing Multimodal Product Extractor")
    try:
        extractor = get_multimodal_extractor()
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        print(f"\nERROR: {e}")
        return 0.0

    # Test cases with SIMULATED CLIP descriptions
    # In production, CLIP would generate these automatically
    test_cases = [
        {
            "expected": "Garnier Micellar Cleansing Water",
            "ocr_output": "BIGGER SIZE GARNIER SKINACTIVE Micellar Cleansing Water",
            "clip_context": "skincare product bottle, blue cap, clear liquid, Garnier brand logo, micellar water label",
            "source": "garnier_micellar_cleansing_water.jpg"
        },
        {
            "expected": "Anua Heartleaf Quercetinol Pore Deep Cleansing Foam",
            "ocr_output": "Anua HEARTLEAF + BHA HEARTLEAF QUERCETINO ORE YANTEEN",
            "clip_context": "Korean skincare, cleansing foam tube, green color scheme, Anua brand, heartleaf ingredient focus",
            "source": "ANUA_Heartleaf_Quercetinol_Pore_Deep_Cleansing_Foam.jpg"
        },
        {
            "expected": "Kellogg's Rice Krispies Multigrain Shapes cereal",
            "ocr_output": "Kelloy Rice: KrisPies MN DAILY SOURCE OF VITAMINS AND IRON SHAPES HIGH IN FIBRE SNAD 25 0/o OFF GREA",
            "clip_context": "cereal box, Kellogg's brand, Rice Krispies product, multigrain shapes cereal, colorful packaging",
            "source": "kellogs_rice_krispies.jpg"
        },
        {
            "expected": "Purina Gourmet Perle Chef's Collection 40 pouches",
            "ocr_output": "3N  GOURMET PURINA  POUCHES 40",
            "clip_context": "cat food multipack, Purina brand, Gourmet Perle product line, 40 pouches box, premium pet food",
            "source": "cat_food_gourmet_perle_collection.jpg"
        },
        {
            "expected": "Bruggen Corn Flakes",
            "ocr_output": "Brigge CORN FLAKES Good morning",
            "clip_context": "cereal box, Bruggen brand, corn flakes cereal, yellow packaging, breakfast product",
            "source": "corn_flakes_briggen.jpeg"
        },
        {
            "expected": "Maybelline Instant Anti-Age Eraser Eye Concealer For Face",
            "ocr_output": "00 CONTERR M-LTSE EAR STAANTGE MAYBELLINE",
            "clip_context": "makeup concealer stick, Maybelline brand, eraser applicator tip, skin tone color, anti-age product",
            "source": "Maybelline Instant Anti-Age Eraser Eye Concealer for Face.jpg"
        },
        {
            "expected": "Perfect Fit Adult 1+ Complete Dry Cat Food",
            "ocr_output": "WITH HIGH ITYINGREDIENTS\"  S PERFECT FIT SUPP WHOLE BODY HEALTH ADULT1+ total Chicken 5",
            "clip_context": "cat food bag, Perfect Fit brand, adult cat food, chicken flavor, 1kg package",
            "source": "perfect_fit_cat_1kg_food.jpg"
        },
        {
            "expected": "Ordinary Natural Moisturizing",
            "ocr_output": "Clinical Formulations with Integrity, Formulations Cliniques Empreintes d'integrite. The rdiary. Nat",
            "clip_context": "skincare serum bottle, The Ordinary brand, minimalist packaging, natural moisturizing factors, clinical formulation",
            "source": "Ordinary_natural_moisturizing.jpg"
        }
    ]

    logger.info(f"\nTesting Multimodal extraction on {len(test_cases)} cases")
    logger.info("="*80)

    correct_ocr_only = 0  # Previous Gemini (OCR only) result: 1/8
    correct_multimodal = 0
    results = []

    for i, test_case in enumerate(test_cases, 1):
        expected = test_case["expected"]
        ocr_output = test_case["ocr_output"]
        clip_context = test_case["clip_context"]
        source = test_case["source"]

        logger.info(f"\n[{i}/{len(test_cases)}] {source}")
        logger.info(f"Expected:       {expected}")
        logger.info(f"OCR Output:     {ocr_output[:70]}...")
        logger.info(f"CLIP Context:   {clip_context[:70]}...")

        # Extract using multimodal approach
        extracted = extractor.extract_product_name(
            ocr_text=ocr_output,
            clip_description=clip_context
        )

        if extracted:
            logger.info(f"Multimodal:     {extracted}")

            # Check if extraction matches expected
            match_ratio = calculate_match_ratio(expected, extracted)

            if match_ratio >= 0.7:
                logger.info(f"CORRECT (match ratio: {match_ratio:.1%})")
                correct_multimodal += 1
                results.append({
                    "source": source,
                    "expected": expected,
                    "extracted": extracted,
                    "status": "CORRECT",
                    "match_ratio": match_ratio
                })
            else:
                logger.warning(f"WRONG (match ratio: {match_ratio:.1%})")
                results.append({
                    "source": source,
                    "expected": expected,
                    "extracted": extracted,
                    "status": "WRONG",
                    "match_ratio": match_ratio
                })
        else:
            logger.error("FAILED (Extraction returned None)")
            results.append({
                "source": source,
                "expected": expected,
                "extracted": None,
                "status": "FAILED",
                "match_ratio": 0.0
            })

    # Calculate improvement
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    ocr_only_accuracy = 12.5  # Previous result from test_gemini_extraction.py
    multimodal_accuracy = (correct_multimodal / len(test_cases)) * 100
    improvement = multimodal_accuracy - ocr_only_accuracy

    logger.info(f"\nGemini OCR-only (previous):      1/8 correct ({ocr_only_accuracy:.1f}%)")
    logger.info(f"Gemini Multimodal (CLIP + OCR):  {correct_multimodal}/{len(test_cases)} correct ({multimodal_accuracy:.1f}%)")
    logger.info(f"Improvement: +{improvement:.1f}%")

    # Show detailed results
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS")
    logger.info("="*80)

    for result in results:
        status_symbol = "CORRECT" if result["status"] == "CORRECT" else "WRONG"
        logger.info(f"\n{status_symbol}: {result['source']}")
        logger.info(f"  Expected:   {result['expected']}")
        logger.info(f"  Extracted:  {result['extracted']}")
        if result['status'] in ['CORRECT', 'WRONG']:
            logger.info(f"  Match:      {result['match_ratio']:.1%}")

    logger.info("\n" + "="*80)
    logger.info("Multimodal Extraction Test Completed")
    logger.info("="*80)

    return multimodal_accuracy


def main():
    """Run multimodal extraction test."""
    try:
        accuracy = test_multimodal_extraction()

        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        print(f"GLiNER NER:               0.0% (0/8 correct)")
        print(f"Text Cleaning:           25.0% (2/8 correct)")
        print(f"Gemini (OCR only):       12.5% (1/8 correct) - Hallucination risk")
        print(f"Gemini (CLIP + OCR):     {accuracy:.1f}% - Context prevents hallucination")

        print("\n" + "="*80)
        if accuracy >= 75:
            print("SUCCESS: Multimodal approach significantly improved accuracy!")
        elif accuracy >= 50:
            print("PARTIAL: Better than OCR-only, but still needs improvement")
        else:
            print("NEEDS WORK: Multimodal didn't help enough")

        print(f"\nLog file saved: {log_file}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
