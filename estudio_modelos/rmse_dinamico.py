# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # RMSE Dinámico — Predicción del Error Esperado por Contrato
#
# **Problema:** El RMSE actual es un valor fijo por bucket de `n_riesgos`:
# - 1–10 → 12 pp, 11–20 → 16 pp, 21–30 → 20 pp, >30 → 24 pp
#
# **Propuesta:** Entrenar un modelo ML que prediga el error esperado (`|real - svr|`)
# para cada contrato a partir de sus 35 features, reemplazando la heurística estática.
#
# Hipótesis: el error del SVR no depende solo de `n_riesgos`, sino también del
# tipo de riesgos, las categorías, el texto de las descripciones y el contexto macro.

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, KFold, train_test_split
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()

# %% [markdown]
# ## 1. Cargar datos y generar predicciones SVR

# %%
matriz = pd.read_csv(ROOT / "docs" / "matriz_clean.csv")
print(f"Matriz riesgos: {len(matriz)} filas, {matriz['id_contrato'].nunique()} contratos")

# Cargar artefactos del modelo
svr = joblib.load(ROOT / "models" / "svr_regressor.pkl")
scaler = joblib.load(ROOT / "models" / "scaler.pkl")
feature_names = joblib.load(ROOT / "models" / "feature_names.pkl")
tfidf_vec = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")
ipc_trm = joblib.load(ROOT / "models" / "ipc_trm.pkl")

print(f"Features del modelo: {len(feature_names)}")
print(feature_names)

# %% [markdown]
# ## 2. Replicar feature engineering para los 351 contratos
#
# Usamos `contratos_macro.csv` que ya tiene `anio_inicio`, `anio_fin`,
# `ipc_acumulado`, `trm_promedio` por contrato. Re-agregamos básicos y TF-IDF.

# %%
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
print(f"Macro: {len(macro)} contratos")
print(f"Columnas macro: {list(macro.columns)}")

# %%
def build_features(matriz, macro, tfidf_vec, feature_names):
    """Replica aggregate_risks() para todos los contratos."""

    # Coerce numeric columns
    for c in ["valor_inicial", "probabilidad", "impacto", "valoracion"]:
        matriz[c] = pd.to_numeric(matriz[c], errors="coerce").fillna(0)

    contratos = matriz.groupby("id_contrato", sort=False)

    # --- Basic stats ---
    basic = contratos.agg(
        valor_inicial=("valor_inicial", "first"),
        n_riesgos=("id_riesgo", "nunique"),
        prob_promedio=("probabilidad", "mean"),
        prob_std=("probabilidad", "std"),
        imp_promedio=("impacto", "mean"),
        imp_std=("impacto", "std"),
        val_promedio=("valoracion", "mean"),
        val_std=("valoracion", "std"),
    ).reset_index()
    basic["interaccion_prob_x_impacto"] = basic["prob_promedio"] * basic["imp_promedio"]
    basic["prob_std"] = basic["prob_std"].fillna(0)
    basic["imp_std"] = basic["imp_std"].fillna(0)
    basic["val_std"] = basic["val_std"].fillna(0)

    # Category counts
    for c in ["bajo", "medio", "alto", "extremo", "no especificado"]:
        cnts = matriz[matriz["categoria"] == c].groupby("id_contrato").size()
        basic[f"n_categoria_{c}"] = basic["id_contrato"].map(cnts).fillna(0).astype(int)

    # --- Categorical proportions ---
    cat_cols = ["tipo", "clase", "asignacion", "fuente_riesgo", "etapa", "categoria"]
    prefix_map = {"tipo": "prop_tipo", "clase": "prop_clas", "asignacion": "prop_asig",
                  "fuente_riesgo": "prop_fuen", "etapa": "prop_etap", "categoria": "prop_cate"}
    all_props = []
    for col in cat_cols:
        dummies = pd.get_dummies(matriz[col], prefix=prefix_map[col])
        dummies["id_contrato"] = matriz["id_contrato"]
        props = dummies.groupby("id_contrato").mean().reset_index()
        all_props.append(props)

    # Merge props
    from functools import reduce
    props_merged = reduce(lambda left, right: pd.merge(left, right, on="id_contrato", how="outer"), all_props)
    props_merged = props_merged.fillna(0)

    # --- TF-IDF ---
    text_series = matriz["descripcion_riesgo"].fillna("").astype(str)
    texts = text_series.groupby(matriz["id_contrato"]).apply(lambda x: " ".join(x)).reset_index()
    texts.columns = ["id_contrato", "descripcion_riesgo"]
    tfidf_matrix = tfidf_vec.transform(texts["descripcion_riesgo"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf_vec.get_feature_names_out()]
    )
    tfidf_df["id_contrato"] = texts["id_contrato"]

    # --- Merge all ---
    merged = basic.merge(props_merged, on="id_contrato", how="left")
    merged = merged.merge(tfidf_df, on="id_contrato", how="left")
    merged = merged.merge(macro, on="id_contrato", how="left")
    merged = merged.fillna(0)

    # --- Align to exact 35 feature names ---
    for f in feature_names:
        if f not in merged.columns:
            merged[f] = 0.0

    # Preserve n_riesgos for heuristic comparison (not a model feature)
    merged["n_riesgos"] = basic["n_riesgos"].values
    return merged[feature_names + ["id_contrato", "n_riesgos"]]

# %%
print("Construyendo features para todos los contratos...")
features_df = build_features(matriz, macro, tfidf_vec, feature_names)
print(f"Features: {features_df.shape}")
print(f"Contratos: {features_df['id_contrato'].nunique()}")

# %%
# Obtener sobrecosto real por contrato
real_df = matriz.groupby("id_contrato", sort=False).agg(
    sobrecosto=("sobrecosto", "first")
).reset_index()

# Merge features + real
data = features_df.merge(real_df, on="id_contrato", how="inner")
print(f"Dataset completo: {data.shape}")

# %%
# Predecir con SVR
X = data[feature_names].values
X_scaled = scaler.transform(X)
preds = svr.predict(X_scaled)

data["svr_pred"] = np.round(preds, 2)
data["abs_error"] = np.abs(data["sobrecosto"] - data["svr_pred"])

# %%
# Estadísticas del error actual
print("Estadísticas del error absoluto (|real - svr|):")
print(data["abs_error"].describe())
print(f"\nRMSE global: {np.sqrt(mean_squared_error(data['sobrecosto'], data['svr_pred'])):.2f} pp")
print(f"MAE global: {mean_absolute_error(data['sobrecosto'], data['svr_pred']):.2f} pp")

# %% [markdown]
# ## 3. Comparar heurística actual vs error real

# %%
def heuristic_rmse(n):
    if n <= 10: return 12.0
    elif n <= 20: return 16.0
    elif n <= 30: return 20.0
    else: return 24.0

data["rmse_heuristic"] = data["n_riesgos"].apply(heuristic_rmse)

# Agrupar por bucket heurístico y ver error real promedio
data["bucket"] = pd.cut(data["n_riesgos"], bins=[0, 10, 20, 30, 200],
                        labels=["1-10", "11-20", "21-30", ">30"])
print("Error real por bucket vs RMSE heurístico:")
print(data.groupby("bucket", observed=True).agg(
    n=("abs_error", "count"),
    error_prom=("abs_error", "mean"),
    error_std=("abs_error", "std"),
    error_p90=("abs_error", lambda x: np.percentile(x, 90)),
    rmse_asignado=("rmse_heuristic", "first"),
))

# %% [markdown]
# ## 4. Entrenar modelos para predecir el error
#
# Target: `abs_error` (error absoluto de la predicción SVR)
# Features: las 35 features del modelo (mismas que usa el SVR)

# %%
y = data["abs_error"].values
X_model = X_scaled  # ya escaladas

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# %%
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    "SVR RBF": SVR(kernel="rbf", C=10, gamma="scale"),
    "SVR Linear": SVR(kernel="linear", C=10),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="neg_mean_absolute_error")
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="r2")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    results.append({
        "modelo": name,
        "MAE CV": f"{-scores.mean():.2f} ± {scores.std():.2f}",
        "R² CV": f"{r2_scores.mean():.3f} ± {r2_scores.std():.3f}",
        "MAE test": f"{test_mae:.2f}",
        "R² test": f"{test_r2:.3f}",
    })
    print(f"{name:20s} | MAE CV={-scores.mean():.2f}±{scores.std():.2f} | R² CV={r2_scores.mean():.3f}±{r2_scores.std():.3f} | Test MAE={test_mae:.2f} R²={test_r2:.3f}")

# %%
results_df = pd.DataFrame(results)
print("\n=== COMPARATIVA DE MODELOS ===")
print(results_df.to_string(index=False))

# %% [markdown]
# ## 5. Evaluar contra la heurística actual

# %%
# Comparar en todo el dataset
heuristic_vals = np.array([heuristic_rmse(n) for n in data["n_riesgos"]])
heuristic_mae = mean_absolute_error(y, heuristic_vals)
heuristic_corr = pearsonr(heuristic_vals, y)[0]

# Best model on full dataset via CV
best_idx = np.argmin([float(r["MAE test"]) for r in results])
best_model_name = results[best_idx]["modelo"]
best_model = list(models.values())[best_idx]
print(f"\nMejor modelo: {best_model_name}")
best_model.fit(X_train, y_train)

print(f"\n{'Métrica':<30} {'Heurística':<15} {best_model_name:<15}")
print("-" * 60)
print(f"{'MAE (pp)':<30} {heuristic_mae:<15.2f} {results[best_idx]['MAE test']}")
print(f"{'Correlación con error real':<30} {heuristic_corr:<15.3f} {pearsonr(best_model.predict(X_test), y_test)[0]:<15.3f}")

# %% [markdown]
# ## 6. Visualizar resultados

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribución del error real
ax = axes[0, 0]
ax.hist(data["abs_error"], bins=30, edgecolor="white", color="steelblue")
ax.axvline(data["abs_error"].mean(), color="red", ls="--", label=f"Media={data['abs_error'].mean():.1f}")
ax.set_xlabel("Error absoluto (pp)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del error real |real - SVR|")
ax.legend()

# 2. Error real vs n_riesgos (con heurística superpuesta)
ax = axes[0, 1]
ax.scatter(data["n_riesgos"], data["abs_error"], alpha=0.5, s=20)
# Heuristic line
for n in range(1, 80):
    h = heuristic_rmse(n)
    ax.scatter(n, h, color="red", s=10, alpha=0.3)
ax.set_xlabel("n_riesgos")
ax.set_ylabel("Error absoluto (pp)")
ax.set_title("Error real vs n_riesgos (rojo=heurística)")

# 3. Error real vs predicción SVR
ax = axes[1, 0]
ax.scatter(data["svr_pred"], data["abs_error"], alpha=0.5, s=20)
ax.set_xlabel("Predicción SVR (%)")
ax.set_ylabel("Error absoluto (pp)")
ax.set_title("¿El error depende del nivel de sobrecosto?")

# 4. Predicción vs Real del mejor modelo
ax = axes[1, 1]
best_cls = list(models.values())[best_idx]
y_full_pred = cross_val_predict(best_cls.__class__(**best_cls.get_params()),
                                 X_model, y, cv=5)
ax.scatter(y, y_full_pred, alpha=0.5, s=20)
ax.plot([0, 60], [0, 60], "r--", alpha=0.5)
ax.set_xlabel("Error real (pp)")
ax.set_ylabel("Error predicho (pp)")
ax.set_title(f"Mejor modelo: {best_model_name}")

plt.tight_layout()
plt.savefig(ROOT / "estudio_modelos" / "resultados_v2" / "rmse_dinamico.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Conclusión y viabilidad
#
# ### ¿Qué tan factible es?
#
# - **Datos disponibles**: tenemos 351 contratos con error conocido → dataset pequeño pero
#   suficiente para un PoC
# - **Features**: las mismas 35 que ya usa el SVR, sin necesidad de ingeniería adicional
# - **Target claro**: `abs_error = |real - svr|`
#
# ### Riesgos
#
# - **Sobreajuste**: el error del SVR puede tener mucho ruido. Si R² < 0, el modelo
#   predictivo es peor que usar la media (o la heurística actual)
# - **Generalización**: el modelo de error se entrenaría con los mismos contratos que el SVR.
#   Si el SVR falla de forma distinta en datos no vistos, el predictor de error también fallará.
# - **Complejidad añadida**: reemplazar 4 líneas de heurística por un modelo serializado
#   con su propio scaler y feature engineering

print("\n=== CONCLUSIÓN ===")
best_test_mae = float(results[best_idx]["MAE test"])
print(f"MAE de la heurística actual: {heuristic_mae:.2f} pp")
print(f"MAE del mejor modelo ({best_model_name}) en test: {best_test_mae:.2f} pp")
mejora_pct = (heuristic_mae - best_test_mae) / heuristic_mae * 100
if mejora_pct > 0:
    print(f"Mejora: {mejora_pct:.1f}%")
else:
    print(f"Sin mejora significativa (el modelo no supera la heurística)")
print(f"\nImplementación: cambiar backend/quantitative_analysis.py:_rmse_por_contrato()")
print(f"por un modelo .pkl cargado desde models/rmse_predictor.pkl")
