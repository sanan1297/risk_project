import os
import logging
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "risk-predictor")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "risk-predictor-svr")
FALLBACK_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_MODEL_REGISTRY: dict = {
    "model_version": None,
    "run_id": None,
    "experiment_id": None,
    "mlflow_available": False,
}


def get_model_registry():
    return dict(_MODEL_REGISTRY)


@lru_cache(maxsize=1)
def load_artifacts():
    if _MODEL_REGISTRY["mlflow_available"]:
        try:
            return _load_from_mlflow()
        except Exception as e:
            logger.warning("Error loading from MLflow, falling back to local: %s", e)
    return _load_from_local()


def _load_from_local():
    logger.info("Loading models from local files (%s)", FALLBACK_MODELS_DIR)
    return (
        joblib.load(FALLBACK_MODELS_DIR / "svr_regressor.pkl"),
        joblib.load(FALLBACK_MODELS_DIR / "classifier.pkl"),
        joblib.load(FALLBACK_MODELS_DIR / "scaler.pkl"),
        joblib.load(FALLBACK_MODELS_DIR / "feature_names.pkl"),
    )


def _load_from_mlflow():
    client = MlflowClient()
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
    if not versions:
        logger.info("No registered model found in MLflow, falling back to local")
        _MODEL_REGISTRY["mlflow_available"] = False
        return _load_from_local()

    mv = versions[0]
    run_id = mv.run_id
    _MODEL_REGISTRY["model_version"] = mv.version
    _MODEL_REGISTRY["run_id"] = run_id
    _MODEL_REGISTRY["experiment_id"] = client.get_run(run_id).info.experiment_id

    artifact_uri = client.get_run(run_id).info.artifact_uri
    parsed = urlparse(artifact_uri)
    if parsed.scheme in ("file", ""):
        artifact_path = Path(parsed.path) if parsed.path else Path(artifact_uri)
    else:
        artifact_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id))

    logger.info("Loading model %s v%s from MLflow (run_id=%s)", MLFLOW_MODEL_NAME, mv.version, run_id)
    return (
        joblib.load(artifact_path / "svr_regressor.pkl"),
        joblib.load(artifact_path / "classifier.pkl"),
        joblib.load(artifact_path / "scaler.pkl"),
        joblib.load(artifact_path / "feature_names.pkl"),
    )


@lru_cache(maxsize=1)
def load_permutation_importance():
    try:
        return _load_permutation_from_mlflow()
    except Exception:
        logger.debug("Permutation importance from MLflow unavailable, using local")
    df = pd.read_csv(FALLBACK_MODELS_DIR / "permutation_importance.csv")
    return df.sort_values("importance", ascending=False)


def _load_permutation_from_mlflow():
    client = MlflowClient()
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
    if not versions:
        raise FileNotFoundError("No model registered")

    mv = versions[0]
    artifact_uri = client.get_run(mv.run_id).info.artifact_uri
    parsed = urlparse(artifact_uri)
    if parsed.scheme in ("file", ""):
        artifact_path = Path(parsed.path) if parsed.path else Path(artifact_uri)
    else:
        artifact_path = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id))

    df = pd.read_csv(artifact_path / "permutation_importance.csv")
    return df.sort_values("importance", ascending=False)


def init_from_mlflow():
    if not MLFLOW_TRACKING_URI:
        _MODEL_REGISTRY["mlflow_available"] = False
        logger.info("MLFLOW_TRACKING_URI not set, using local models")
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        _MODEL_REGISTRY["mlflow_available"] = True
        logger.info("Connected to MLflow at %s", MLFLOW_TRACKING_URI)

        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if versions:
            mv = versions[0]
            _MODEL_REGISTRY["model_version"] = mv.version
            _MODEL_REGISTRY["run_id"] = mv.run_id
            _MODEL_REGISTRY["experiment_id"] = client.get_run(mv.run_id).info.experiment_id
            logger.info("Production model: %s v%s", MLFLOW_MODEL_NAME, mv.version)
        else:
            logger.info("No production model registered yet")
    except Exception as e:
        _MODEL_REGISTRY["mlflow_available"] = False
        logger.warning("MLflow unavailable (%s), using local models", e)
