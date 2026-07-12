# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Evaluación de Modelos — Risk Predictor
# ## Comparación Ridge vs Random Forest vs Gradient Boosting vs ElasticNet
# Misma data, mismas 35 features, misma metodología de validación cruzada.

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path.cwd().resolve()))
from estudio_data.features import engineer_features, STOP_WORDS

ROOT = Path.cwd().resolve()
DATA_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"

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
MAX_DURACION = 5

# %% [markdown]
# ## 1. Carga de datos

# %%
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)

matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
print(f"Matriz: {len(matriz):,} riesgos | {matriz['id_contrato'].nunique()} contratos")

macro = pd.read_csv(DATA_DIR / "contratos_macro.csv")
print(f"Macro: {len(macro)} contratos con ipc_acumulado + trm_promedio")

# %% [markdown]
# ## 2. Feature engineering

# %%
def compute_range_features(anio_inicio, anio_fin):
    if anio_fin < anio_inicio:
        anio_fin = anio_inicio
    duracion = anio_fin - anio_inicio
    if duracion > MAX_DURACION:
        anio_fin = anio_inicio + MAX_DURACION
        duracion = MAX_DURACION
    ipc_acum = 1.0
    trm_vals = []
    for y in range(anio_inicio, anio_fin + 1):
        d = IPC_TRM.get(y, {"ipc": 3.0, "trm": 4000})
        ipc_acum *= (1 + d["ipc"] / 100)
        trm_vals.append(d["trm"])
    ipc_acum = (ipc_acum - 1) * 100
    trm_prom = float(np.mean(trm_vals))
    return {"anio_inicio": anio_inicio, "anio_fin": anio_fin,
            "duracion": duracion, "ipc_acumulado": round(ipc_acum, 2),
            "trm_promedio": round(trm_prom, 2)}

print("=" * 60)
print("  FEATURE ENGINEERING")
print("=" * 60)

df_feat = engineer_features(matriz)
print(f"Features base: {df_feat.shape}")

df_feat = df_feat.merge(macro, on="id_contrato", how="left")
for c in CONTROL_VARS_RANGO:
    if c not in df_feat.columns:
        fb = compute_range_features(2022, 2022)
        df_feat[c] = fb[c]

n_antes = len(df_feat)
df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
print(f"Outliers removidos: {n_antes - len(df_feat)}")
print(f"Dataset final: {df_feat.shape}")

missing = [c for c in FEATURES_35 if c not in df_feat.columns]
if missing:
    raise ValueError(f"Faltan: {missing}")

X = df_feat[FEATURES_35].values
y = df_feat["sobrecosto"].values
y_bin = (y > 25).astype(int)

print(f"\nX shape: {X.shape}")
print(f"y range: {y.min():.1f}% – {y.max():.1f}%")
print(f"Alto riesgo (>25%): {y_bin.mean():.1%}")

# %% [markdown]
# ## 3. Escalado y modelos

# %%
print("=" * 60)
print("  ENTRENAMIENTO Y CV")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Ridge (actual)": RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="neg_root_mean_squared_error"),
    "Ridge alpha=1.0": Ridge(alpha=1.0, random_state=42),
    "Ridge alpha=10": Ridge(alpha=10, random_state=42),
    "ElasticNet": ElasticNetCV(alphas=np.logspace(-3, 2, 30), l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], random_state=42, max_iter=10000),
    "RandomForest 100": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "RandomForest 300": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    "GradientBoosting shallow": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
}

results = []
all_preds = {}

for name, model in models.items():
    # Cross-validation R²
    cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    
    # Cross-validation RMSE
    cv_rmse = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error")
    
    # Full fit + prediction for AUC (need predict for classification)
    model.fit(X_scaled, y)
    y_pred_full = model.predict(X_scaled)
    r2_full = r2_score(y, y_pred_full)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    mae_full = mean_absolute_error(y, y_pred_full)
    
    # AUC: usamos la predicción de regresión como score para ROC
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=5)
    auc = roc_auc_score(y_bin, y_pred_cv)
    
    results.append({
        "modelo": name,
        "R² full": round(r2_full, 4),
        "R² CV": f"{cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
        "R² CV medio": round(cv_r2.mean(), 4),
        "RMSE CV": f"{-cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}",
        "RMSE full": round(rmse_full, 2),
        "MAE full": round(mae_full, 2),
        "AUC CV": round(auc, 4),
    })
    all_preds[name] = y_pred_cv
    
    print(f"\n{name}:")
    print(f"  R² full: {r2_full:.4f} | R² CV: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  RMSE CV: {-cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
    print(f"  AUC CV:  {auc:.4f}")

df_results = pd.DataFrame(results).sort_values("R² CV medio", ascending=False)
print("\n" + "=" * 60)
print("  TABLA COMPARATIVA")
print("=" * 60)
print(df_results[["modelo", "R² CV medio", "R² full", "RMSE full", "MAE full", "AUC CV"]].to_string(index=False))

# %% [markdown]
# ## 4. Feature Importance (Random Forest)

# %%
print("\n" + "=" * 60)
print("  FEATURE IMPORTANCE — Random Forest 300")
print("=" * 60)

rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

fi = pd.DataFrame({"feature": FEATURES_35, "importance": rf.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
for _, r in fi.head(15).iterrows():
    print(f"  + {r['feature']:35s} {r['importance']:.4f}")
print("  " + "-" * 45)
for _, r in fi.tail(5).iterrows():
    print(f"  - {r['feature']:35s} {r['importance']:.4f}")

# %% [markdown]
# ## 5. Comparación Ridge vs RandomForest — Coeficientes

# %%
print("\n" + "=" * 60)
print("  TOP COEFICIENTES — Ridge actual")
print("=" * 60)

ridge_actual = RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="neg_root_mean_squared_error")
ridge_actual.fit(X_scaled, y)
coefs = pd.DataFrame({"feature": FEATURES_35, "coef": ridge_actual.coef_})
coefs = coefs.sort_values("coef", ascending=False)
print(f"Alpha seleccionado: {ridge_actual.alpha_:.4f}")
for _, r in coefs.head(10).iterrows():
    print(f"  + {r['feature']:35s} {r['coef']:.4f}")
for _, r in coefs.tail(10).iterrows():
    print(f"  - {r['feature']:35s} {r['coef']:.4f}")

# %% [markdown]
# ## 6. ¿Modelo final recomendado?

# %%
print("\n" + "=" * 60)
print("  RESUMEN Y RECOMENDACIÓN")
print("=" * 60)

best_cv = df_results.iloc[0]
best_r2_name = best_cv["modelo"]
best_r2_val = best_cv["R² CV medio"]
best_auc = df_results.loc[df_results["AUC CV"].idxmax()]

print(f"Mejor R² CV: {best_r2_name} → {best_r2_val:.4f}")
print(f"Mejor AUC CV: {best_auc['modelo']} → {best_auc['AUC CV']:.4f}")

ridge_row = df_results[df_results["modelo"] == "Ridge (actual)"].iloc[0]
print(f"\nRidge actual: R² CV={ridge_row['R² CV medio']:.4f}, AUC={ridge_row['AUC CV']:.4f}")

print("\n")
print("El mejor modelo en R² CV gana si priorizamos precisión numérica.")
print("El mejor modelo en AUC CV gana si priorizamos ranking de riesgo.")
print("Podemos usar el que gane en ambas o un ensemble.")
print("La decisión final es tuya — ajustamos train_final_model.py y backend.")
