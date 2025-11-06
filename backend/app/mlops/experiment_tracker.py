"""
Experiment Tracker for ML Model Experimentation

Tracks model training experiments, hyperparameters, and results.
"""
import mlflow
from typing import Dict, Any, Optional, Callable
from loguru import logger
import time
from contextlib import contextmanager


class ExperimentTracker:
    """
    Tracks ML experiments with MLflow.

    Features:
    - Automatic metric logging
    - Parameter tracking
    - Artifact management
    - Experiment comparison
    """

    def __init__(self, experiment_name: str = "clarify_products_ml"):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment tracker initialized: {experiment_name}")

    @contextmanager
    def track_experiment(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for tracking an experiment run.

        Args:
            run_name: Name of this experimental run
            params: Model/experiment parameters
            tags: Metadata tags

        Example:
            >>> tracker = ExperimentTracker()
            >>> with tracker.track_experiment("test_model", params={"lr": 0.001}):
            ...     # Train model
            ...     tracker.log_metric("accuracy", 0.95)
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Track start time
            start_time = time.time()
            mlflow.log_param("start_time", start_time)

            logger.info(f"Started experiment run: {run_name}")

            try:
                yield run
            finally:
                # Log duration
                duration = time.time() - start_time
                mlflow.log_metric("duration_seconds", duration)
                logger.info(
                    f"Completed experiment run: {run_name} "
                    f"(duration: {duration:.2f}s)"
                )

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (for tracking over time)
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to local file/directory
            artifact_path: Optional subdirectory in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        test_data_size: int,
        inference_time_ms: float
    ):
        """
        Log comprehensive model performance metrics.

        Args:
            model_name: Name of the model
            metrics: Performance metrics (accuracy, f1, etc.)
            test_data_size: Number of test samples
            inference_time_ms: Average inference time in milliseconds
        """
        # Log all metrics
        self.log_metrics(metrics)

        # Log additional context
        self.log_param("model_name", model_name)
        self.log_param("test_data_size", test_data_size)
        self.log_metric("inference_time_ms", inference_time_ms)

        logger.info(
            f"Logged performance for {model_name}: "
            f"{metrics} on {test_data_size} samples"
        )

    def compare_runs(
        self,
        experiment_name: Optional[str] = None,
        metric: str = "accuracy",
        limit: int = 10
    ) -> list:
        """
        Compare experiment runs by a specific metric.

        Args:
            experiment_name: Name of experiment (uses current if None)
            metric: Metric to compare
            limit: Number of runs to return

        Returns:
            List of runs sorted by metric
        """
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)

        if not experiment:
            logger.warning(f"Experiment {exp_name} not found")
            return []

        # Search runs in this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=limit
        )

        logger.info(f"Retrieved {len(runs)} runs from {exp_name}")
        return runs.to_dict('records') if not runs.empty else []


# Singleton instance
_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker(experiment_name: str = "clarify_products_ml") -> ExperimentTracker:
    """Get the singleton experiment tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(experiment_name)
    return _tracker
