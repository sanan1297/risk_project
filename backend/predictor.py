from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_META = {
    "modelo": "Ridge + LogisticRegression",
    "r2_cv": 0.103,
    "auc_cv": 0.639,
    "accuracy": 0.706,
}


@lru_cache(maxsize=1)
def _load_artifacts():
    return (
        joblib.load(MODELS_DIR / "ridge_regressor.pkl"),
        joblib.load(MODELS_DIR / "ridge_classifier.pkl"),
        joblib.load(MODELS_DIR / "scaler.pkl"),
        joblib.load(MODELS_DIR / "feature_names.pkl"),
    )


def load():
    return _load_artifacts()


def predict(df_features: pd.DataFrame) -> dict:
    regressor, classifier, scaler, feature_names = load()

    X = df_features[feature_names].values
    X_scaled = scaler.transform(X)

    preds = np.round(regressor.predict(X_scaled), 2)
    probas = np.round(classifier.predict_proba(X_scaled)[:, 1], 4)
    alerts = ["ALTO RIESGO" if p > 0.5 else "RIESGO MODERADO" for p in probas]

    coef_df = pd.DataFrame({"feature": feature_names, "coef": regressor.coef_})
    top_pos = coef_df.sort_values("coef", ascending=False).head(5)
    top_neg = coef_df.sort_values("coef", ascending=True).head(5)

    return {
        "predicciones": preds.tolist(),
        "probabilidades": probas.tolist(),
        "alertas": alerts,
        "contratos": df_features["id_contrato"].tolist(),
        "n_riesgos": df_features["n_riesgos"].tolist(),
        "explicacion": {
            "aumentan": [{"feature": r["feature"], "coef": round(r["coef"], 4)} for _, r in top_pos.iterrows()],
            "disminuyen": [{"feature": r["feature"], "coef": round(r["coef"], 4)} for _, r in top_neg.iterrows()],
        },
        **MODEL_META,
    }
