from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

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

CONTROL_VARS = ["anio", "ipc", "trm"]
FEATURES_33 = TOP_30_FEATURES + CONTROL_VARS


def add_ipc_trm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anio"] = df.get("anio", 2022)
    df["ipc"] = df["anio"].map(lambda y: IPC_TRM.get(int(y), {"ipc": 0})["ipc"])
    df["trm"] = df["anio"].map(lambda y: IPC_TRM.get(int(y), {"trm": 0})["trm"])
    return df


def main() -> None:
    print("=" * 55)
    print("  ENTRENAMIENTO FINAL — Ridge + LogisticRegression")
    print("=" * 55)

    print("\n1. Cargando matriz_clean.csv...")
    matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
    print(f"   Riesgos: {len(matriz):,} | Contratos: {matriz['id_contrato'].nunique()}")

    print("\n2. Feature engineering...")
    df_feat = engineer_features(matriz)
    df_feat = add_ipc_trm(df_feat)
    n_antes = len(df_feat)
    df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
    print(f"   Features: {df_feat.shape[1]} | Contratos: {len(df_feat)} (outliers: {n_antes - len(df_feat)})")

    missing = [c for c in FEATURES_33 if c not in df_feat.columns]
    if missing:
        print(f"   ERROR: Faltan columnas: {missing}")
        return

    X = df_feat[FEATURES_33].values
    y = df_feat["sobrecosto"].values
    y_bin = (y > 25).astype(int)

    print("\n3. Guardando vectorizador TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words=STOP_WORDS, ngram_range=(1, 2))
    textos = matriz.groupby("id_contrato")["descripcion_riesgo"].apply(
        lambda g: " ".join(g.dropna().astype(str))
    )
    vectorizer.fit(textos)
    print(f"   {len(vectorizer.get_feature_names_out())} tokens")

    print("\n4. Escalando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n5. Entrenando Ridge (regresion)...")
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="neg_root_mean_squared_error")
    ridge.fit(X_scaled, y)
    r2 = ridge.score(X_scaled, y)
    cv_r2 = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")
    print(f"   Alpha: {ridge.alpha_:.4f} | R² full: {r2:.4f} | R² CV: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    print("\n6. Entrenando LogisticRegression (clasificacion)...")
    classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    classifier.fit(X_scaled, y_bin)
    cv_auc = cross_val_score(classifier, X_scaled, y_bin, cv=5, scoring="roc_auc")
    print(f"   AUC CV: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    print("\n7. Guardando artefactos...")
    joblib.dump(ridge, MODELS_DIR / "ridge_regressor.pkl")
    joblib.dump(classifier, MODELS_DIR / "ridge_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(FEATURES_33, MODELS_DIR / "feature_names.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(IPC_TRM, MODELS_DIR / "ipc_trm.pkl")

    for f in sorted(MODELS_DIR.iterdir()):
        print(f"   {f.name}")

    coefs = pd.DataFrame({"feature": FEATURES_33, "coef": ridge.coef_})
    coefs.to_csv(MODELS_DIR / "coeficientes_ridge.csv", index=False, encoding="utf-8-sig")

    print("\n8. Top coeficientes:")
    top = coefs.sort_values("coef", ascending=False)
    for _, r in top.head(10).iterrows():
        print(f"   + {r['feature']:30s} {r['coef']:.4f}")
    for _, r in top.tail(10).iterrows():
        print(f"   - {r['feature']:30s} {r['coef']:.4f}")

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
