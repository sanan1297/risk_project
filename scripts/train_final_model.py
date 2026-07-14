import os
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, str(Path.cwd()))
from estudio_data.features import engineer_features, STOP_WORDS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "docs"

MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "risk-predictor")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "risk-predictor-svr")

IPC_TRM = {
    2000: {"ipc": 8.75, "trm": 2052}, 2001: {"ipc": 7.65, "trm": 2200},
    2002: {"ipc": 6.99, "trm": 2504}, 2003: {"ipc": 6.49, "trm": 2878},
    2004: {"ipc": 5.50, "trm": 2628}, 2005: {"ipc": 4.85, "trm": 2322},
    2006: {"ipc": 4.48, "trm": 2358}, 2007: {"ipc": 5.69, "trm": 2014},
    2008: {"ipc": 7.67, "trm": 1973}, 2009: {"ipc": 2.00, "trm": 2047},
    2010: {"ipc": 3.17, "trm": 1898}, 2011: {"ipc": 3.73, "trm": 1848},
    2012: {"ipc": 2.44, "trm": 1798}, 2013: {"ipc": 1.94, "trm": 1887},
    2014: {"ipc": 3.66, "trm": 2020}, 2015: {"ipc": 6.77, "trm": 2742},
    2016: {"ipc": 5.75, "trm": 3055}, 2017: {"ipc": 4.09, "trm": 2951.32},
    2018: {"ipc": 3.18, "trm": 2956.55}, 2019: {"ipc": 3.80, "trm": 3281.09},
    2020: {"ipc": 1.61, "trm": 3693.36}, 2021: {"ipc": 5.62, "trm": 3743.09},
    2022: {"ipc": 13.12, "trm": 4255.44}, 2023: {"ipc": 9.28, "trm": 4325.05},
    2024: {"ipc": 5.20, "trm": 4071.28}, 2025: {"ipc": 5.10, "trm": 4052.86},
    2026: {"ipc": 6.40, "trm": 4200}, 2027: {"ipc": 4.80, "trm": 4100},
}

TOP_30_FEATURES = [
    "tfidf_desarrollo", "interaccion_prob_x_impacto", "tfidf_insumos",
    "prob_std", "tfidf_expedicion", "tfidf_materiales", "imp_promedio",
    "tfidf_obra", "tfidf_ejecucion", "tfidf_contrato",
    "prop_tipo_operacional", "prob_promedio", "prop_cate_bajo",
    "valor_inicial", "tfidf_riesgo", "tfidf_tecnicas", "tfidf_municipio",
    "tfidf_obras", "tfidf_informacion", "prop_fuen_externo", "tfidf_cuando",
    "tfidf_disenos", "tfidf_ejecucion contrato", "tfidf_calidad",
    "tfidf_manejo", "prop_cate_alto", "tfidf_pago", "prop_tipo_economico",
    "prop_asig_entidad", "tfidf_falta",
]

CONTROL_VARS_RANGO = ["anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]
FEATURES_35 = TOP_30_FEATURES + CONTROL_VARS_RANGO


def compute_range_features(anio_inicio: int, anio_fin: int, max_duracion: int = 5) -> dict:
    if anio_fin < anio_inicio:
        anio_fin = anio_inicio
    duracion = anio_fin - anio_inicio
    if duracion > max_duracion:
        anio_fin = anio_inicio + max_duracion
        duracion = max_duracion
    ipc_acum = 1.0
    trm_vals = []
    for y in range(anio_inicio, anio_fin + 1):
        d = IPC_TRM.get(y, {"ipc": 3.0, "trm": 4000})
        ipc_acum *= (1 + d["ipc"] / 100)
        trm_vals.append(d["trm"])
    ipc_acum = (ipc_acum - 1) * 100
    trm_prom = float(np.mean(trm_vals))
    return {
        "anio_inicio": anio_inicio,
        "anio_fin": anio_fin,
        "duracion": duracion,
        "ipc_acumulado": round(ipc_acum, 2),
        "trm_promedio": round(trm_prom, 2),
    }


def add_macro_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(macro_df, on="id_contrato", how="left")
    for c in CONTROL_VARS_RANGO:
        if c not in df.columns:
            fallback = compute_range_features(2022, 2022)
            df[c] = fallback[c]
    return df


def main() -> None:
    logger.info("=" * 55)
    logger.info("  ENTRENAMIENTO FINAL — SVR (kernel RBF)")
    logger.info("  Features: 35 (30 TF-IDF + 5 rango de fechas)")
    logger.info("=" * 55)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        mlflow.log_params({
            "model_type": "SVR",
            "kernel": "rbf",
            "C": 10,
            "gamma": "scale",
            "feature_count": len(FEATURES_35),
            "feature_set": "30_tfidf_5_rango",
            "top_features": len(TOP_30_FEATURES),
            "control_vars": len(CONTROL_VARS_RANGO),
            "max_features_tfidf": 100,
            "ngram_range": "(1,2)",
            "ridge_alpha": 244.0,
            "classifier": "LogisticRegression",
            "classifier_C": 1.0,
            "classifier_max_iter": 1000,
            "classifier_class_weight": "balanced",
            "random_state": 42,
            "outlier_threshold": 200,
            "n_repeats_permutation": 10,
        })

        logger.info("\n1. Cargando matriz_clean.csv...")
        matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
        n_riesgos_total = len(matriz)
        n_contratos_total = int(matriz['id_contrato'].nunique())
        mlflow.log_metrics({"n_riesgos_total": n_riesgos_total, "n_contratos_total": n_contratos_total})
        logger.info("   Riesgos: %s | Contratos: %s", f"{n_riesgos_total:,}", n_contratos_total)

        logger.info("\n2. Cargando features de rango macro...")
        macro = pd.read_csv(DATA_DIR / "contratos_macro.csv")
        logger.info("   %s contratos con ipc_acumulado + trm_promedio", len(macro))

        logger.info("\n3. Feature engineering...")
        df_feat = engineer_features(matriz)
        df_feat = add_macro_features(df_feat, macro)
        n_antes = len(df_feat)
        df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
        n_despues = len(df_feat)
        mlflow.log_metrics({"contratos_antes_outlier": n_antes, "contratos_despues_outlier": n_despues})
        logger.info("   Features: %s | Contratos: %s (outliers: %s)", df_feat.shape[1], n_despues, n_antes - n_despues)

        missing = [c for c in FEATURES_35 if c not in df_feat.columns]
        if missing:
            logger.info("   ERROR: Faltan columnas: %s", missing)
            return

        X = df_feat[FEATURES_35].values
        y = df_feat["sobrecosto"].values
        y_bin = (y > 25).astype(int)

        mlflow.log_metrics({
            "target_mean": float(np.mean(y)),
            "target_median": float(np.median(y)),
            "target_std": float(np.std(y)),
            "alto_riesgo_ratio": float(np.mean(y_bin)),
        })

        logger.info("\n4. Guardando vectorizador TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=100, stop_words=STOP_WORDS, ngram_range=(1, 2))
        textos = matriz.groupby("id_contrato")["descripcion_riesgo"].apply(
            lambda g: " ".join(g.dropna().astype(str))
        )
        vectorizer.fit(textos)
        n_tokens = len(vectorizer.get_feature_names_out())
        mlflow.log_metric("tfidf_tokens", n_tokens)
        logger.info("   %s tokens", n_tokens)

        logger.info("\n5. Escalando features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info("\n6. Entrenando modelo campeon: SVR (kernel RBF)...")
        svr = SVR(kernel="rbf", C=10, gamma="scale")
        svr.fit(X_scaled, y)
        y_pred = svr.predict(X_scaled)
        r2_full = r2_score(y, y_pred)
        cv_r2 = cross_val_score(svr, X_scaled, y, cv=5, scoring="r2")
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        mlflow.log_metrics({
            "r2_full": r2_full,
            "r2_cv_mean": float(cv_r2.mean()),
            "r2_cv_std": float(cv_r2.std()),
            "rmse": rmse,
        })
        logger.info("   R² full: %.4f | R² CV: %.4f ± %.4f | RMSE: %.2f", r2_full, cv_r2.mean(), cv_r2.std(), rmse)

        logger.info("\n7. Entrenando clasificador (LogisticRegression)...")
        classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        classifier.fit(X_scaled, y_bin)
        cv_auc = cross_val_score(classifier, X_scaled, y_bin, cv=5, scoring="roc_auc")
        y_prob = classifier.predict_proba(X_scaled)[:, 1]
        auc_full = roc_auc_score(y_bin, y_prob)

        mlflow.log_metrics({
            "auc_cv_mean": float(cv_auc.mean()),
            "auc_cv_std": float(cv_auc.std()),
            "auc_full": float(auc_full),
        })
        logger.info("   AUC CV: %.4f ± %.4f | AUC full: %.4f", cv_auc.mean(), cv_auc.std(), auc_full)

        logger.info("\n8. Permutation importance global (10 repeticiones)...")
        imp = permutation_importance(svr, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            "feature": FEATURES_35,
            "importance": imp.importances_mean,
            "std": imp.importances_std,
        }).sort_values("importance", ascending=False)

        perm_path = MODELS_DIR / "permutation_importance.csv"
        imp_df.to_csv(perm_path, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(str(perm_path))

        logger.info("\n9. Ridge de referencia (para coefs interpretables)...")
        ridge = Ridge(alpha=244.0, random_state=42)
        ridge.fit(X_scaled, y)
        ridge_coefs = pd.DataFrame({"feature": FEATURES_35, "coef": ridge.coef_})
        ridge_path = MODELS_DIR / "coeficientes_ridge.csv"
        ridge_coefs.to_csv(ridge_path, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(str(ridge_path))

        logger.info("\n10. Guardando artefactos locales...")
        for name, obj in [
            ("svr_regressor.pkl", svr),
            ("classifier.pkl", classifier),
            ("scaler.pkl", scaler),
            ("feature_names.pkl", FEATURES_35),
            ("tfidf_vectorizer.pkl", vectorizer),
            ("ipc_trm.pkl", IPC_TRM),
            ("ridge_reference.pkl", ridge),
        ]:
            path = MODELS_DIR / name
            joblib.dump(obj, path)
            mlflow.log_artifact(str(path))

        for f in sorted(MODELS_DIR.iterdir()):
            logger.info("   %s", f.name)

        logger.info("\n11. Registrando modelo en MLflow Model Registry...")
        mlflow.sklearn.log_model(svr, "svr_regressor")
        mlflow.sklearn.log_model(classifier, "classifier")

        try:
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/svr_regressor",
                name=MLFLOW_MODEL_NAME,
            )
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=result.version,
                stage="Production",
            )
            logger.info("   Modelo registrado: %s v%s", MLFLOW_MODEL_NAME, result.version)
        except Exception as e:
            logger.warning("   No se pudo registrar el modelo en MLflow: %s", e)

        logger.info("\n12. Top features por permutation importance:")
        for _, r in imp_df.head(10).iterrows():
            logger.info("   + %-35s %.4f ± %.4f", r['feature'], r['importance'], r['std'])
        for _, r in imp_df.tail(5).iterrows():
            logger.info("   - %-35s %.4f ± %.4f", r['feature'], r['importance'], r['std'])

        logger.info("\n13. Resumen:")
        logger.info("   Regresion: SVR R² CV=%.4f | AUC clasif CV=%.4f", cv_r2.mean(), cv_auc.mean())
        logger.info("   Modelo: SVR (kernel RBF, C=10, gamma=scale)")
        logger.info("   MLflow run: %s", run_id)
        logger.info("   MLflow UI: %s/#/experiments/%s/runs/%s", MLFLOW_TRACKING_URI, exp.experiment_id, run_id)

    logger.info("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
