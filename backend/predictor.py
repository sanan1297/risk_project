from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_META = {
    "modelo": "SVR (kernel RBF)",
    "r2_cv": 0.068,
    "auc_cv": 0.673,
    "rmse": 17.1,
    "features": 35,
    "tipo_control": "rango_fechas",
}


@lru_cache(maxsize=1)
def _load_artifacts():
    return (
        joblib.load(MODELS_DIR / "svr_regressor.pkl"),
        joblib.load(MODELS_DIR / "classifier.pkl"),
        joblib.load(MODELS_DIR / "scaler.pkl"),
        joblib.load(MODELS_DIR / "feature_names.pkl"),
    )


@lru_cache(maxsize=1)
def _load_permutation_importance():
    df = pd.read_csv(MODELS_DIR / "permutation_importance.csv")
    return df.sort_values("importance", ascending=False)


def load():
    return _load_artifacts()


def predict(df_features: pd.DataFrame) -> dict:
    regressor, classifier, scaler, feature_names = load()

    X = df_features[feature_names].values
    X_scaled = scaler.transform(X)

    preds = np.round(regressor.predict(X_scaled), 2)
    probas = np.round(classifier.predict_proba(X_scaled)[:, 1], 4)
    alerts = ["ALTO RIESGO" if p > 0.5 else "RIESGO MODERADO" for p in probas]

    imp_df = _load_permutation_importance()
    top_pos = imp_df.head(5)
    top_neg = imp_df.tail(5).sort_values("importance", ascending=True)

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
        **MODEL_META,
    }
