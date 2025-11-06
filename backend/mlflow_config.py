"""
MLflow Configuration for ClarifyProducts.AI

Centralized configuration for ML experiment tracking and model registry.
"""
import os
from pathlib import Path

# MLflow tracking configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "clarify_products_ml")

# Model registry configuration
MODEL_REGISTRY_PATH = Path("./mlruns")
MODEL_REGISTRY_PATH.mkdir(parents=True, exist_ok=True)

# Model names in registry
MODEL_NAMES = {
    "clip": "clip_product_recognizer",
    "sentiment": "sentiment_analyzer",
    "summarizer": "review_summarizer",
    "ocr": "ocr_text_extractor"
}

# Performance thresholds for quality gates
QUALITY_GATES = {
    "clip_product_recognizer": {
        "min_accuracy": 0.60,
        "max_inference_time_ms": 300
    },
    "sentiment_analyzer": {
        "min_accuracy": 0.85,
        "max_inference_time_ms": 100
    },
    "review_summarizer": {
        "min_rouge_score": 0.30,
        "max_inference_time_ms": 5000
    },
    "ocr_text_extractor": {
        "min_accuracy": 0.45,
        "max_inference_time_ms": 10000
    }
}

# A/B testing configuration
AB_TEST_TRAFFIC_SPLIT = {
    "champion": 0.9,  # 90% traffic to production model
    "challenger": 0.1  # 10% traffic to new model
}

#  Monitoring configuration
ENABLE_MODEL_MONITORING = os.getenv("ENABLE_MODEL_MONITORING", "true").lower() == "true"
MONITORING_SAMPLE_RATE = float(os.getenv("MONITORING_SAMPLE_RATE", "0.1"))  # Log 10% of predictions

# Feature flags
ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
ENABLE_MODEL_CACHING = os.getenv("ENABLE_MODEL_CACHING", "true").lower() == "true"
