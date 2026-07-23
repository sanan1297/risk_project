from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from . import mlflow_tracker

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_META = {
    "modelo": "RandomForest (nested CV campeon)",
    "r2_cv": 0.235,
    "auc_cv": 0.591,
    "rmse": 11.4,
    "features": 38,
    "tipo_control": "rango_fechas",
}


def _load_artifacts():
    return mlflow_tracker.load_artifacts()


def _load_permutation_importance():
    return mlflow_tracker.load_permutation_importance()


def load():
    return _load_artifacts()


def predict(df_features: pd.DataFrame) -> dict:
    regressor, classifier, scaler, feature_names = _load_artifacts()

    X = df_features[feature_names].values
    X_scaled = scaler.transform(X)

    preds = np.round(regressor.predict(X_scaled), 2)
    probas = np.round(classifier.predict_proba(X_scaled)[:, 1], 4)
    alerts = ["ALTO RIESGO" if p > 0.5 else "RIESGO MODERADO" for p in probas]

    imp_df = _load_permutation_importance()
    top_pos = imp_df.head(5)
    top_neg = imp_df.tail(5).sort_values("importance", ascending=True)

    registry = mlflow_tracker.get_model_registry()

    return {
        "predicciones": preds.tolist(),
        "probabilidades": probas.tolist(),
        "alertas": alerts,
        "contratos": df_features["id_contrato"].tolist(),
        "n_riesgos": df_features["n_riesgos"].tolist(),
        "explicacion": {
            "aumentan": [{"feature": r["feature"], "coef": round(r["importance"], 4)} for _, r in top_pos.iterrows()],
            "disminuyen": [{"feature": r["feature"], "coef": round(-r["importance"], 4)} for _, r in top_neg.iterrows()],
        },
        "model_version": registry.get("model_version"),
        "model_run_id": registry.get("run_id"),
        **MODEL_META,
    }
