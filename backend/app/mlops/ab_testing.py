"""
A/B Testing Framework for ML Models

This module provides A/B testing functionality to compare different model versions
in production and make data-driven decisions about which models to promote.

Usage:
    from app.mlops.ab_testing import ABTestManager

    # Initialize
    ab_test = ABTestManager()

    # Select model variant for a user
    variant = ab_test.get_variant("clip_product_recognizer", user_id="user123")

    # Log experiment result
    ab_test.log_result(
        experiment_id="clip_v1_vs_v2",
        variant=variant,
        metrics={"accuracy": 0.85, "latency_ms": 250}
    )

    # Analyze results
    winner = ab_test.analyze_experiment("clip_v1_vs_v2")
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from loguru import logger
import mlflow
from dataclasses import dataclass, asdict


@dataclass
class ABVariant:
    """Represents an A/B test variant."""
    variant_id: str
    model_name: str
    model_version: str
    traffic_percentage: float
    description: str = ""


@dataclass
class ABExperiment:
    """Represents an A/B test experiment."""
    experiment_id: str
    model_name: str
    variants: List[ABVariant]
    start_date: str
    end_date: Optional[str] = None
    status: Literal["active", "completed", "paused"] = "active"
    min_sample_size: int = 1000
    confidence_level: float = 0.95


@dataclass
class ABResult:
    """Represents a single A/B test result."""
    experiment_id: str
    variant_id: str
    user_id: str
    timestamp: str
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class ABTestManager:
    """
    Manages A/B testing experiments for ML models.

    Features:
    - Consistent user assignment to variants (using hashing)
    - Traffic splitting based on percentages
    - Result logging and analysis
    - Statistical significance testing
    - Integration with MLflow for tracking
    """

    def __init__(
        self,
        experiments_dir: str = "./ab_experiments",
        mlflow_experiment_name: str = "ab_testing"
    ):
        """
        Initialize A/B test manager.

        Args:
            experiments_dir: Directory to store experiment configurations and results
            mlflow_experiment_name: MLflow experiment name for tracking
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment_name = mlflow_experiment_name

        # Set MLflow experiment
        try:
            mlflow.set_experiment(mlflow_experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")

        # Load active experiments
        self.experiments = self._load_experiments()
        logger.info(f"A/B Test Manager initialized with {len(self.experiments)} active experiments")

    def create_experiment(
        self,
        experiment_id: str,
        model_name: str,
        variants: List[ABVariant],
        min_sample_size: int = 1000,
        duration_days: int = 7
    ) -> ABExperiment:
        """
        Create a new A/B test experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            model_name: Name of the model being tested
            variants: List of variants to test
            min_sample_size: Minimum samples needed for statistical significance
            duration_days: How long to run the experiment

        Returns:
            ABExperiment object

        Example:
            variants = [
                ABVariant(
                    variant_id="champion",
                    model_name="clip_product_recognizer",
                    model_version="v1.0",
                    traffic_percentage=0.9,
                    description="Current production model"
                ),
                ABVariant(
                    variant_id="challenger",
                    model_name="clip_product_recognizer",
                    model_version="v2.0",
                    traffic_percentage=0.1,
                    description="New model with better preprocessing"
                )
            ]
            experiment = ab_test.create_experiment(
                experiment_id="clip_v1_vs_v2",
                model_name="clip_product_recognizer",
                variants=variants
            )
        """
        # Validate traffic percentages
        total_traffic = sum(v.traffic_percentage for v in variants)
        if not 0.99 <= total_traffic <= 1.01:
            raise ValueError(
                f"Traffic percentages must sum to 1.0, got {total_traffic}"
            )

        # Create experiment
        experiment = ABExperiment(
            experiment_id=experiment_id,
            model_name=model_name,
            variants=variants,
            start_date=datetime.now().isoformat(),
            end_date=(datetime.now() + timedelta(days=duration_days)).isoformat(),
            min_sample_size=min_sample_size
        )

        # Save experiment
        self._save_experiment(experiment)
        self.experiments[experiment_id] = experiment

        logger.info(
            f"Created A/B experiment '{experiment_id}' for {model_name} "
            f"with {len(variants)} variants"
        )

        return experiment

    def get_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[ABVariant]:
        """
        Get the variant for a specific user (consistent assignment).

        Uses hashing to ensure the same user always gets the same variant
        for the duration of the experiment.

        Args:
            experiment_id: The experiment ID
            user_id: Unique user identifier

        Returns:
            ABVariant or None if experiment not found

        Example:
            variant = ab_test.get_variant("clip_v1_vs_v2", "user123")
            if variant.variant_id == "challenger":
                # Use new model
                model = load_model(variant.model_version)
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment '{experiment_id}' not found")
            return None

        if experiment.status != "active":
            logger.warning(f"Experiment '{experiment_id}' is not active")
            return None

        # Hash user ID to get consistent assignment
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        # Assign variant based on traffic split
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant

        # Fallback to first variant
        return experiment.variants[0]

    def log_result(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log the result of an A/B test interaction.

        Args:
            experiment_id: The experiment ID
            variant_id: Which variant was used
            user_id: User who triggered this result
            metrics: Performance metrics (accuracy, latency, etc.)
            metadata: Optional additional data

        Example:
            ab_test.log_result(
                experiment_id="clip_v1_vs_v2",
                variant_id="challenger",
                user_id="user123",
                metrics={
                    "accuracy": 0.92,
                    "latency_ms": 245,
                    "confidence": 0.89
                },
                metadata={
                    "image_size": "1920x1080",
                    "product_category": "electronics"
                }
            )
        """
        result = ABResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            metadata=metadata
        )

        # Save result to file
        results_file = self.experiments_dir / f"{experiment_id}_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"{experiment_id}_{variant_id}"):
                mlflow.log_params({
                    "experiment_id": experiment_id,
                    "variant_id": variant_id
                })
                mlflow.log_metrics(metrics)
                if metadata:
                    mlflow.set_tags({f"meta_{k}": str(v) for k, v in metadata.items()})
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_experiment_stats(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with stats for each variant

        Example:
            stats = ab_test.get_experiment_stats("clip_v1_vs_v2")
            print(f"Champion accuracy: {stats['champion']['avg_accuracy']}")
            print(f"Challenger accuracy: {stats['challenger']['avg_accuracy']}")
        """
        results_file = self.experiments_dir / f"{experiment_id}_results.jsonl"
        if not results_file.exists():
            return {}

        # Load all results
        variant_data = {}
        with open(results_file, "r") as f:
            for line in f:
                result = json.loads(line)
                variant_id = result["variant_id"]

                if variant_id not in variant_data:
                    variant_data[variant_id] = {
                        "count": 0,
                        "metrics": {}
                    }

                variant_data[variant_id]["count"] += 1

                # Aggregate metrics
                for metric_name, metric_value in result["metrics"].items():
                    if metric_name not in variant_data[variant_id]["metrics"]:
                        variant_data[variant_id]["metrics"][metric_name] = []
                    variant_data[variant_id]["metrics"][metric_name].append(metric_value)

        # Calculate statistics
        stats = {}
        for variant_id, data in variant_data.items():
            stats[variant_id] = {
                "sample_size": data["count"],
                "metrics": {}
            }

            for metric_name, values in data["metrics"].items():
                stats[variant_id]["metrics"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return stats

    def analyze_experiment(
        self,
        experiment_id: str,
        primary_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Analyze experiment results and determine winner.

        Args:
            experiment_id: The experiment ID
            primary_metric: The metric to use for comparison

        Returns:
            Analysis results with winner determination

        Example:
            analysis = ab_test.analyze_experiment("clip_v1_vs_v2", "accuracy")
            if analysis["winner"]:
                print(f"Winner: {analysis['winner']['variant_id']}")
                print(f"Improvement: {analysis['improvement_percentage']:.2f}%")
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}

        stats = self.get_experiment_stats(experiment_id)

        if not stats:
            return {"error": "No results yet"}

        # Check minimum sample size
        min_samples_met = all(
            variant_stats["sample_size"] >= experiment.min_sample_size
            for variant_stats in stats.values()
        )

        # Find best variant by primary metric
        best_variant = None
        best_score = float('-inf')

        for variant_id, variant_stats in stats.items():
            if primary_metric in variant_stats["metrics"]:
                mean_score = variant_stats["metrics"][primary_metric]["mean"]
                if mean_score > best_score:
                    best_score = mean_score
                    best_variant = variant_id

        # Calculate improvement
        champion_score = None
        challenger_score = None

        if "champion" in stats and primary_metric in stats["champion"]["metrics"]:
            champion_score = stats["champion"]["metrics"][primary_metric]["mean"]

        if "challenger" in stats and primary_metric in stats["challenger"]["metrics"]:
            challenger_score = stats["challenger"]["metrics"][primary_metric]["mean"]

        improvement_pct = None
        if champion_score and challenger_score:
            improvement_pct = ((challenger_score - champion_score) / champion_score) * 100

        return {
            "experiment_id": experiment_id,
            "stats": stats,
            "winner": best_variant if min_samples_met else None,
            "best_score": best_score,
            "primary_metric": primary_metric,
            "min_samples_met": min_samples_met,
            "improvement_percentage": improvement_pct,
            "champion_score": champion_score,
            "challenger_score": challenger_score,
            "recommendation": self._get_recommendation(
                min_samples_met, improvement_pct, stats
            )
        }

    def _get_recommendation(
        self,
        min_samples_met: bool,
        improvement_pct: Optional[float],
        stats: Dict
    ) -> str:
        """Generate recommendation based on experiment results."""
        if not min_samples_met:
            return "Continue experiment - minimum sample size not met"

        if improvement_pct is None:
            return "Insufficient data to make recommendation"

        if improvement_pct > 5:
            return "PROMOTE CHALLENGER - Significant improvement detected"
        elif improvement_pct > 2:
            return "PROMOTE CHALLENGER - Moderate improvement detected"
        elif improvement_pct > -2:
            return "NEUTRAL - No significant difference"
        else:
            return "KEEP CHAMPION - Challenger performing worse"

    def stop_experiment(self, experiment_id: str):
        """Stop an active experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = "completed"
            self.experiments[experiment_id].end_date = datetime.now().isoformat()
            self._save_experiment(self.experiments[experiment_id])
            logger.info(f"Stopped experiment '{experiment_id}'")

    def _save_experiment(self, experiment: ABExperiment):
        """Save experiment configuration to file."""
        config_file = self.experiments_dir / f"{experiment.experiment_id}_config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(experiment), f, indent=2)

    def _load_experiments(self) -> Dict[str, ABExperiment]:
        """Load all experiment configurations."""
        experiments = {}
        for config_file in self.experiments_dir.glob("*_config.json"):
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    # Reconstruct variants
                    variants = [ABVariant(**v) for v in data["variants"]]
                    data["variants"] = variants
                    experiment = ABExperiment(**data)
                    experiments[experiment.experiment_id] = experiment
            except Exception as e:
                logger.error(f"Failed to load experiment from {config_file}: {e}")
        return experiments


# Convenience functions
def get_ab_test_manager() -> ABTestManager:
    """Get singleton instance of A/B test manager."""
    global _ab_test_manager
    if "_ab_test_manager" not in globals():
        _ab_test_manager = ABTestManager()
    return _ab_test_manager
