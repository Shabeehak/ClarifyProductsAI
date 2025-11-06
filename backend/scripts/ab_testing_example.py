"""
A/B Testing Example - How to compare model versions

This script demonstrates how to set up and run A/B tests for ML models.

Usage:
    python scripts/ab_testing_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mlops.ab_testing import ABTestManager, ABVariant
from loguru import logger


def example_1_create_experiment():
    """Example 1: Create an A/B test to compare CLIP model versions."""
    logger.info("="*60)
    logger.info("Example 1: Creating A/B Test for CLIP Model")
    logger.info("="*60)

    ab_test = ABTestManager()

    # Define variants
    variants = [
        ABVariant(
            variant_id="champion",
            model_name="clip_product_recognizer",
            model_version="v1.0_baseline",
            traffic_percentage=0.9,  # 90% of traffic
            description="Current production model (CLIP ViT-B/32)"
        ),
        ABVariant(
            variant_id="challenger",
            model_name="clip_product_recognizer",
            model_version="v2.0_with_preprocessing",
            traffic_percentage=0.1,  # 10% of traffic
            description="New model with image preprocessing"
        )
    ]

    # Create experiment
    experiment = ab_test.create_experiment(
        experiment_id="clip_v1_vs_v2_preprocessing",
        model_name="clip_product_recognizer",
        variants=variants,
        min_sample_size=500,  # Need 500 samples per variant
        duration_days=7  # Run for 7 days
    )

    logger.info(f"Created experiment: {experiment.experiment_id}")
    logger.info(f"Variants: {len(variants)}")
    logger.info(f"Duration: 7 days")
    logger.info(f"Champion gets: 90% of traffic")
    logger.info(f"Challenger gets: 10% of traffic")


def example_2_get_variant_for_user():
    """Example 2: Get the assigned variant for a user."""
    logger.info("")
    logger.info("="*60)
    logger.info("Example 2: Getting Variant Assignment for Users")
    logger.info("="*60)

    ab_test = ABTestManager()

    # Simulate 10 different users
    for i in range(1, 11):
        user_id = f"user{i:03d}"
        variant = ab_test.get_variant("clip_v1_vs_v2_preprocessing", user_id)

        if variant:
            logger.info(
                f"{user_id} → {variant.variant_id} "
                f"(using {variant.model_version})"
            )


def example_3_log_results():
    """Example 3: Simulate logging A/B test results."""
    logger.info("")
    logger.info("="*60)
    logger.info("Example 3: Logging A/B Test Results")
    logger.info("="*60)

    ab_test = ABTestManager()

    # Simulate results for 100 users
    import random
    random.seed(42)

    for i in range(1, 101):
        user_id = f"user{i:03d}"
        variant = ab_test.get_variant("clip_v1_vs_v2_preprocessing", user_id)

        if not variant:
            continue

        # Simulate metrics
        # Challenger is slightly better on average
        if variant.variant_id == "champion":
            accuracy = random.uniform(0.60, 0.66)  # 60-66%
            latency_ms = random.uniform(220, 280)  # 220-280ms
        else:  # challenger
            accuracy = random.uniform(0.63, 0.69)  # 63-69% (better)
            latency_ms = random.uniform(240, 300)  # 240-300ms (slightly slower)

        # Log result
        ab_test.log_result(
            experiment_id="clip_v1_vs_v2_preprocessing",
            variant_id=variant.variant_id,
            user_id=user_id,
            metrics={
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "confidence": random.uniform(0.7, 0.95)
            },
            metadata={
                "product_category": random.choice(["electronics", "clothing", "food"])
            }
        )

    logger.info("Logged 100 simulated results")


def example_4_view_stats():
    """Example 4: View experiment statistics."""
    logger.info("")
    logger.info("="*60)
    logger.info("Example 4: Viewing Experiment Statistics")
    logger.info("="*60)

    ab_test = ABTestManager()
    stats = ab_test.get_experiment_stats("clip_v1_vs_v2_preprocessing")

    for variant_id, variant_stats in stats.items():
        logger.info(f"\n{variant_id.upper()}:")
        logger.info(f"  Sample size: {variant_stats['sample_size']}")

        for metric_name, metric_stats in variant_stats["metrics"].items():
            logger.info(
                f"  {metric_name}: "
                f"mean={metric_stats['mean']:.4f}, "
                f"min={metric_stats['min']:.4f}, "
                f"max={metric_stats['max']:.4f}"
            )


def example_5_analyze_and_decide():
    """Example 5: Analyze experiment and get recommendation."""
    logger.info("")
    logger.info("="*60)
    logger.info("Example 5: Analyzing Experiment & Getting Recommendation")
    logger.info("="*60)

    ab_test = ABTestManager()
    analysis = ab_test.analyze_experiment(
        "clip_v1_vs_v2_preprocessing",
        primary_metric="accuracy"
    )

    if "error" in analysis:
        logger.error(f"Analysis error: {analysis['error']}")
        return

    logger.info(f"\nExperiment: {analysis['experiment_id']}")
    logger.info(f"Primary metric: {analysis['primary_metric']}")
    logger.info(f"Minimum samples met: {analysis['min_samples_met']}")

    if analysis['champion_score']:
        logger.info(f"\nChampion (current): {analysis['champion_score']:.4f}")
    if analysis['challenger_score']:
        logger.info(f"Challenger (new): {analysis['challenger_score']:.4f}")

    if analysis['improvement_percentage']:
        logger.info(
            f"Improvement: {analysis['improvement_percentage']:+.2f}%"
        )

    logger.info(f"\nRECOMMENDATION: {analysis['recommendation']}")

    if analysis['winner']:
        logger.info(f"Winner: {analysis['winner']}")


def example_6_sentiment_analyzer_test():
    """Example 6: A/B test for sentiment analyzer."""
    logger.info("")
    logger.info("="*60)
    logger.info("Example 6: A/B Test for Sentiment Analyzer")
    logger.info("="*60)

    ab_test = ABTestManager()

    # Compare DistilBERT vs RoBERTa
    variants = [
        ABVariant(
            variant_id="champion",
            model_name="sentiment_analyzer",
            model_version="distilbert-base-uncased-finetuned-sst-2-english",
            traffic_percentage=0.8,
            description="Current DistilBERT model (fast, 91% accuracy)"
        ),
        ABVariant(
            variant_id="challenger",
            model_name="sentiment_analyzer",
            model_version="roberta-base-go_emotions",
            traffic_percentage=0.2,
            description="RoBERTa model (slower, potentially more accurate)"
        )
    ]

    experiment = ab_test.create_experiment(
        experiment_id="sentiment_distilbert_vs_roberta",
        model_name="sentiment_analyzer",
        variants=variants,
        min_sample_size=1000,
        duration_days=14  # Run for 2 weeks
    )

    logger.info(f"Created experiment: {experiment.experiment_id}")
    logger.info(f"Testing: DistilBERT (80%) vs RoBERTa (20%)")
    logger.info(f"Goal: Determine if RoBERTa's accuracy improvement justifies slower speed")


def main():
    """Run all examples."""
    logger.info("A/B Testing Examples for ClarifyProducts.AI")
    logger.info("")

    # Run examples
    example_1_create_experiment()
    example_2_get_variant_for_user()
    example_3_log_results()
    example_4_view_stats()
    example_5_analyze_and_decide()
    example_6_sentiment_analyzer_test()

    logger.info("")
    logger.info("="*60)
    logger.info("Examples Complete!")
    logger.info("="*60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Check ./ab_experiments/ for saved experiment data")
    logger.info("2. View results in MLflow UI at http://localhost:5000")
    logger.info("3. Integrate A/B testing into your API endpoints")


if __name__ == "__main__":
    main()
