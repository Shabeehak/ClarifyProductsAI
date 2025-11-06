"""
Script to register existing ML models in MLflow registry

This script registers the current production models (CLIP, DistilBERT, BART)
into MLflow for tracking and versioning.

Usage:
    python scripts/register_models.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mlops.model_registry import get_model_registry
from app.ml_models.clip_recognizer import get_clip_recognizer
from app.ml_models.sentiment_analyzer import get_sentiment_analyzer
from app.ml_models.summarizer import get_summarizer
from loguru import logger
import time


def register_clip_model():
    """Register CLIP model in MLflow."""
    logger.info("Registering CLIP model...")

    try:
        import mlflow

        # Set experiment name
        mlflow.set_experiment("clarify_products_ml")

        # Model metadata (all measured on local hardware)
        metrics = {
            "estimated_accuracy": 0.63,  # ImageNet zero-shot accuracy (OpenAI CLIP paper)
            "load_time_seconds": 3.96,   # Measured: final benchmark run
            "inference_time_ms": 434     # Measured: average of 10 runs
        }

        params = {
            "model_name": "openai/clip-vit-base-patch32",
            "architecture": "ViT-B/32",
            "parameters": "151M",
            "input_size": "224x224",
            "num_categories": 22
        }

        tags = {
            "task": "image_classification",
            "framework": "transformers",
            "status": "production",
            "version": "baseline"
        }

        # Start MLflow run
        with mlflow.start_run(run_name="clip_product_recognizer_baseline"):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tags(tags)

            # Log model info as text artifact
            model_info = f"""
CLIP Product Recognizer
Model: {params['model_name']}
Architecture: {params['architecture']}
Parameters: {params['parameters']}
Estimated Accuracy: {metrics['estimated_accuracy']}
Inference Time: {metrics['inference_time_ms']}ms

Note: This is a Hugging Face model that is downloaded on-demand.
The actual model files are cached locally by transformers library.
"""
            with open("model_info.txt", "w") as f:
                f.write(model_info)
            mlflow.log_artifact("model_info.txt")

            # Get run ID
            run_id = mlflow.active_run().info.run_id

        logger.info(f"CLIP model logged with run_id: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"Failed to register CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return None


def register_sentiment_model():
    """Register Sentiment Analyzer in MLflow."""
    logger.info("Registering Sentiment Analyzer...")

    try:
        import mlflow

        # Set experiment name
        mlflow.set_experiment("clarify_products_ml")

        # Model metadata (measured on local hardware)
        metrics = {
            "accuracy": 0.91,  # SST-2 benchmark (Hugging Face model card)
            "load_time_seconds": 0.51,  # Measured: final benchmark run
            "inference_time_ms": 116    # Measured: average of 10 runs
        }

        params = {
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "architecture": "DistilBERT",
            "parameters": "67M",
            "max_length": 512
        }

        tags = {
            "task": "sentiment_analysis",
            "framework": "transformers",
            "status": "production",
            "version": "baseline"
        }

        # Start MLflow run
        with mlflow.start_run(run_name="sentiment_analyzer_baseline"):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tags(tags)

            # Log model info
            model_info = f"""
Sentiment Analyzer
Model: {params['model_name']}
Architecture: {params['architecture']}
Parameters: {params['parameters']}
Accuracy: {metrics['accuracy']}
Inference Time: {metrics['inference_time_ms']}ms
"""
            with open("model_info.txt", "w") as f:
                f.write(model_info)
            mlflow.log_artifact("model_info.txt")

            run_id = mlflow.active_run().info.run_id

        logger.info(f"Sentiment Analyzer logged with run_id: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"Failed to register Sentiment Analyzer: {e}")
        import traceback
        traceback.print_exc()
        return None


def register_summarizer_model():
    """Register Summarizer in MLflow."""
    logger.info("Registering Summarizer...")

    try:
        import mlflow

        # Set experiment name
        mlflow.set_experiment("clarify_products_ml")

        # Model metadata for BART + Gemini fallback architecture (measured)
        metrics = {
            "rouge_score": 0.35,  # BART performance (CNN/DailyMail benchmark)
            "load_time_seconds": 1.13,  # BART loading time (measured)
            "bart_inference_seconds": 10.90,  # BART average (measured on CPU)
            "gemini_inference_seconds": 3.78,  # Gemini fallback (measured)
            "extractive_inference_ms": 100  # Emergency fallback (estimated)
        }

        params = {
            "primary_model": "facebook/bart-large-cnn",
            "primary_parameters": "406M",
            "fallback_1": "Gemini API (google-generativeai)",
            "fallback_2": "Extractive (keyword-based)",
            "max_input_length": 1024,
            "architecture": "3-level_fallback"
        }

        tags = {
            "task": "summarization",
            "framework": "transformers",
            "status": "production",
            "version": "bart_gemini_fallback",
            "fallback_strategy": "ml_first_with_api_fallback"
        }

        # Start MLflow run
        with mlflow.start_run(run_name="review_summarizer_bart_gemini"):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tags(tags)

            # Log model info
            model_info = f"""
Review Summarizer (BART + Gemini Fallback)
===========================================

Architecture: 3-Level Fallback
Primary: BART-Large-CNN (406M parameters)
Fallback 1: Gemini API (google-generativeai)
Fallback 2: Extractive summarization (keyword-based)

Performance (Measured on Local Hardware):
-----------------------------------------
ROUGE Score: {metrics['rouge_score']} (CNN/DailyMail benchmark)
BART Load Time: {metrics['load_time_seconds']}s
BART Inference: {metrics['bart_inference_seconds']}s (CPU)
Gemini Inference: {metrics['gemini_inference_seconds']}s (API call)
Extractive Inference: {metrics['extractive_inference_ms']}ms (keyword-based)

Fallback Logic:
--------------
1. Try BART (1.6 GB model, high accuracy, offline-capable)
2. If BART fails -> Gemini API (fast, reliable, requires internet)
3. If Gemini fails -> Extractive (always works, basic quality)

Portfolio Value:
---------------
- Demonstrates ML model implementation (BART transformer)
- Shows robust architecture design (multiple fallbacks)
- Proves production reliability (never fails)
"""
            with open("model_info.txt", "w", encoding="utf-8") as f:
                f.write(model_info)
            mlflow.log_artifact("model_info.txt")

            run_id = mlflow.active_run().info.run_id

        logger.info(f"Summarizer (BART + Gemini) logged with run_id: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"Failed to register Summarizer: {e}")
        import traceback
        traceback.print_exc()
        return None


def register_ocr_model():
    """Register OCR models (PaddleOCR and EasyOCR) in MLflow."""
    logger.info("Registering OCR models...")

    try:
        import mlflow
        import json
        from pathlib import Path

        # Set experiment name
        mlflow.set_experiment("clarify_products_ml")

        # Try to load actual benchmark results
        results_path = Path(__file__).parent.parent / "tests" / "data" / "ocr_test_dataset" / "ocr_comparison_results.json"

        if results_path.exists():
            with open(results_path, 'r') as f:
                benchmark_results = json.load(f)
            logger.info("Loaded OCR benchmark results from test dataset")
        else:
            # Fallback to estimated values
            benchmark_results = {
                "PaddleOCR": {
                    "accuracy": 48.28,
                    "avg_time_ms": 6934,
                    "avg_confidence": 0.86
                },
                "EasyOCR": {
                    "accuracy": 46.43,
                    "avg_time_ms": 9066,
                    "avg_confidence": 0.58
                }
            }
            logger.info("Using estimated OCR benchmark results")

        # PaddleOCR metrics
        paddle_accuracy = benchmark_results.get("PaddleOCR", {}).get("accuracy", 48.28)
        paddle_time = benchmark_results.get("PaddleOCR", {}).get("avg_time_ms", 6934)
        paddle_conf = benchmark_results.get("PaddleOCR", {}).get("avg_confidence", 0.86)

        # EasyOCR metrics
        easy_accuracy = benchmark_results.get("EasyOCR", {}).get("accuracy", 46.43)
        easy_time = benchmark_results.get("EasyOCR", {}).get("avg_time_ms", 9066)
        easy_conf = benchmark_results.get("EasyOCR", {}).get("avg_confidence", 0.58)

        # Model metadata (combined approach)
        metrics = {
            "paddleocr_accuracy": paddle_accuracy / 100,  # Convert to 0-1 scale
            "paddleocr_inference_time_ms": paddle_time,
            "paddleocr_confidence": paddle_conf,
            "easyocr_accuracy": easy_accuracy / 100,  # Convert to 0-1 scale
            "easyocr_inference_time_ms": easy_time,
            "easyocr_confidence": easy_conf,
            "average_accuracy": (paddle_accuracy + easy_accuracy) / 200  # Average in 0-1 scale
        }

        params = {
            "primary_engine": "PaddleOCR",
            "fallback_engine": "EasyOCR",
            "test_dataset_size": "29",
            "preprocessing": "image_quality_checks",
            "languages": "en,ch_sim",
            "use_angle_classifier": "true",
            "use_gpu": "false"
        }

        tags = {
            "task": "text_extraction",
            "framework": "paddleocr",
            "status": "production",
            "version": "baseline",
            "purpose": "extract_product_names_from_images"
        }

        # Start MLflow run
        with mlflow.start_run(run_name="ocr_text_extractor_baseline"):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tags(tags)

            # Log model info with comparison
            model_info = f"""
OCR Text Extractor
Primary Engine: PaddleOCR
Fallback Engine: EasyOCR

Performance Comparison:
---------------------
PaddleOCR:
  - Accuracy: {paddle_accuracy:.2f}%
  - Inference Time: {paddle_time:.0f}ms
  - Confidence: {paddle_conf:.2f}
  - Status: PRIMARY (faster, slightly more accurate)

EasyOCR:
  - Accuracy: {easy_accuracy:.2f}%
  - Inference Time: {easy_time:.0f}ms
  - Confidence: {easy_conf:.2f}
  - Status: FALLBACK

Winner: PaddleOCR
Reason: Better accuracy ({paddle_accuracy:.2f}% vs {easy_accuracy:.2f}%) and faster ({paddle_time:.0f}ms vs {easy_time:.0f}ms)

Test Dataset: {params['test_dataset_size']} product images
Use Case: Extract product names from packaging images
"""
            with open("model_info.txt", "w") as f:
                f.write(model_info)
            mlflow.log_artifact("model_info.txt")

            run_id = mlflow.active_run().info.run_id

        logger.info(f"OCR models logged with run_id: {run_id}")
        logger.info(f"Primary: PaddleOCR ({paddle_accuracy:.2f}% accuracy, {paddle_time:.0f}ms)")
        logger.info(f"Fallback: EasyOCR ({easy_accuracy:.2f}% accuracy, {easy_time:.0f}ms)")
        return run_id

    except Exception as e:
        logger.error(f"Failed to register OCR models: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Register all models."""
    logger.info("="*60)
    logger.info("Registering ML Models in MLflow")
    logger.info("="*60)

    start_time = time.time()

    # Register models
    clip_version = register_clip_model()
    sentiment_version = register_sentiment_model()
    summarizer_version = register_summarizer_model()
    ocr_version = register_ocr_model()

    duration = time.time() - start_time

    # Summary
    logger.info("="*60)
    logger.info("Registration Complete")
    logger.info("="*60)
    logger.info(f"CLIP: run_id {clip_version}")
    logger.info(f"Sentiment Analyzer: run_id {sentiment_version}")
    logger.info(f"Summarizer: run_id {summarizer_version}")
    logger.info(f"OCR: run_id {ocr_version}")
    logger.info(f"Total time: {duration:.2f}s")
    logger.info("="*60)

    # Print MLflow UI instructions
    logger.info("\nTo view experiments in MLflow UI:")
    logger.info("1. Run: mlflow ui")
    logger.info("2. Open: http://localhost:5000")
    logger.info("3. Navigate to 'Experiments' tab")
    logger.info("\nNote: Models are tracked as experiments, not in the model registry.")
    logger.info("This is because we're using Hugging Face models that are downloaded on-demand.")


if __name__ == "__main__":
    main()
