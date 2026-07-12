from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, str(Path.cwd()))
from estudio_data.features import engineer_features, STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "docs"

MODELS_DIR.mkdir(exist_ok=True)

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
    print("=" * 55)
    print("  ENTRENAMIENTO FINAL — SVR (kernel RBF)")
    print("  Features: 35 (30 TF-IDF + 5 rango de fechas)")
    print("=" * 55)

    print("\n1. Cargando matriz_clean.csv...")
    matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
    print(f"   Riesgos: {len(matriz):,} | Contratos: {matriz['id_contrato'].nunique()}")

    print("\n2. Cargando features de rango macro...")
    macro = pd.read_csv(DATA_DIR / "contratos_macro.csv")
    print(f"   {len(macro)} contratos con ipc_acumulado + trm_promedio")

    print("\n3. Feature engineering...")
    df_feat = engineer_features(matriz)
    df_feat = add_macro_features(df_feat, macro)
    n_antes = len(df_feat)
    df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
    print(f"   Features: {df_feat.shape[1]} | Contratos: {len(df_feat)} (outliers: {n_antes - len(df_feat)})")

    missing = [c for c in FEATURES_35 if c not in df_feat.columns]
    if missing:
        print(f"   ERROR: Faltan columnas: {missing}")
        return

    X = df_feat[FEATURES_35].values
    y = df_feat["sobrecosto"].values
    y_bin = (y > 25).astype(int)

    print("\n4. Guardando vectorizador TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words=STOP_WORDS, ngram_range=(1, 2))
    textos = matriz.groupby("id_contrato")["descripcion_riesgo"].apply(
        lambda g: " ".join(g.dropna().astype(str))
    )
    vectorizer.fit(textos)
    print(f"   {len(vectorizer.get_feature_names_out())} tokens")

    print("\n5. Escalando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n6. Entrenando modelo campeon: SVR (kernel RBF)...")
    svr = SVR(kernel="rbf", C=10, gamma="scale")
    svr.fit(X_scaled, y)
    y_pred = svr.predict(X_scaled)
    r2_full = r2_score(y, y_pred)
    cv_r2 = cross_val_score(svr, X_scaled, y, cv=5, scoring="r2")
    print(f"   R² full: {r2_full:.4f} | R² CV: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    print("\n7. Entrenando clasificador (LogisticRegression)...")
    classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    classifier.fit(X_scaled, y_bin)
    cv_auc = cross_val_score(classifier, X_scaled, y_bin, cv=5, scoring="roc_auc")
    print(f"   AUC CV: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    print("\n8. Permutation importance global (10 repeticiones)...")
    imp = permutation_importance(svr, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": FEATURES_35,
        "importance": imp.importances_mean,
        "std": imp.importances_std,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(MODELS_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    print("\n9. Ridge de referencia (para coefs interpretables)...")
    ridge = Ridge(alpha=244.0, random_state=42)
    ridge.fit(X_scaled, y)
    ridge_coefs = pd.DataFrame({"feature": FEATURES_35, "coef": ridge.coef_})
    ridge_coefs.to_csv(MODELS_DIR / "coeficientes_ridge.csv", index=False, encoding="utf-8-sig")

    print("\n10. Guardando artefactos...")
    joblib.dump(svr, MODELS_DIR / "svr_regressor.pkl")
    joblib.dump(classifier, MODELS_DIR / "classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURES_35, MODELS_DIR / "feature_names.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(IPC_TRM, MODELS_DIR / "ipc_trm.pkl")
    joblib.dump(ridge, MODELS_DIR / "ridge_reference.pkl")

    for f in sorted(MODELS_DIR.iterdir()):
        print(f"   {f.name}")

    print("\n11. Top features por permutation importance:")
    for _, r in imp_df.head(10).iterrows():
        print(f"   + {r['feature']:35s} {r['importance']:.4f} ± {r['std']:.4f}")
    for _, r in imp_df.tail(5).iterrows():
        print(f"   - {r['feature']:35s} {r['importance']:.4f} ± {r['std']:.4f}")

    print(f"\n12. Resumen:")
    print(f"   Regresion: SVR R² CV={cv_r2.mean():.4f} | AUC clasif CV={cv_auc.mean():.4f}")
    print(f"   Modelo campeon: SVR (kernel RBF, C=10, gamma=scale)")
    print(f"   Interpretabilidad: permutation importance (10 reps)")
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
