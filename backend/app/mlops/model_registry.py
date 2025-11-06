"""
Model Registry for Managing ML Models

Handles model versioning, storage, and retrieval using MLflow.
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
import os


class ModelRegistry:
    """
    Centralized model registry for managing ML models.

    Features:
    - Model versioning
    - Experiment tracking
    - Model comparison
    - Production deployment tracking
    """

    def __init__(self, tracking_uri: str = "file:///./mlruns"):
        """
        Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI (default: local file storage)
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        logger.info(f"Model registry initialized with tracking URI: {tracking_uri}")

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model (e.g., "sentiment_analyzer")
            model_path: Path to the model artifacts
            metrics: Performance metrics (e.g., {"accuracy": 0.91})
            params: Model parameters (e.g., {"max_length": 512})
            tags: Additional metadata tags
            artifacts: Additional artifacts to log (e.g., {"vocab": "vocab.json"})

        Returns:
            Model version string
        """
        with mlflow.start_run(run_name=f"{model_name}_registration") as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log model artifacts
            mlflow.log_artifacts(model_path, artifact_path="model")

            # Log additional artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_path=artifact_name)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            logger.info(
                f"Registered {model_name} version {mv.version} "
                f"with metrics: {metrics}"
            )

            return mv.version

    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = "Production"
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model
            version: Specific version (if None, uses stage)
            stage: Model stage ("Production", "Staging", "Archived")

        Returns:
            Loaded model object
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded {model_name} from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def compare_models(
        self,
        model_name: str,
        metric: str = "accuracy",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Compare model versions by a specific metric.

        Args:
            model_name: Name of the model
            metric: Metric to compare (e.g., "accuracy", "f1_score")
            limit: Number of versions to return

        Returns:
            List of model versions with metrics, sorted by the metric
        """
        try:
            # Get all versions of the model
            versions = self.client.search_model_versions(f"name='{model_name}'")

            results = []
            for v in versions[:limit]:
                run_id = v.run_id
                run = self.client.get_run(run_id)

                results.append({
                    "version": v.version,
                    "run_id": run_id,
                    "stage": v.current_stage,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "created_at": v.creation_timestamp
                })

            # Sort by metric (descending)
            results.sort(
                key=lambda x: x["metrics"].get(metric, 0),
                reverse=True
            )

            logger.info(f"Compared {len(results)} versions of {model_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return []

    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ):
        """
        Promote a model version to a specific stage.

        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage ("Staging", "Production", "Archived")
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def get_production_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about the current production model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model info
        """
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )

            if not versions:
                return {"error": "No production model found"}

            v = versions[0]
            run = self.client.get_run(v.run_id)

            return {
                "version": v.version,
                "run_id": v.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "created_at": v.creation_timestamp,
                "description": v.description or "No description"
            }

        except Exception as e:
            logger.error(f"Failed to get production model info: {e}")
            return {"error": str(e)}


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the singleton model registry instance."""
    global _registry
    if _registry is None:
        # Use environment variable or default to local storage
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///./mlruns")
        _registry = ModelRegistry(tracking_uri)
    return _registry
