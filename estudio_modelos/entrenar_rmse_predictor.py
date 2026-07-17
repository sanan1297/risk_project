"""
PoC: Entrenar modelo de error (RMSE dinámico) usando residuales
del SVR de sobrecosto sobre datos históricos.

Filosofía:
  - El SVR de sobrecosto se entrenó con datos históricos (sobrecosto_real conocido).
  - Para esos mismos contratos, calculamos |sobrecosto_real - svr_predicción|.
  - Entrenamos un nuevo modelo que aprende: "dado este perfil de contrato,
    el SVR suele tener este error".
  - En inferencia (contrato NUEVO), usamos solo sus features → el modelo
    de error predice cuánto esperamos que se equivoque el SVR.

Resultado: models/rmse_predictor.pkl + comparación con heurística.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from backend.feature_engineering import aggregate_risks
from backend.predictor import load as load_model

# ── 1. Cargar modelos existentes ──────────────────────────────────
print("=" * 60)
print("ENTRENAMIENTO DE RMSE PREDICTOR (residuales)")
print("=" * 60)

regressor, classifier, scaler, feature_names = load_model()
print(f"Features del SVR: {len(feature_names)}")

# ── 2. Cargar datos históricos ────────────────────────────────────
matriz = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
macro_map = macro.set_index("id_contrato").to_dict("index")

contratos = matriz["id_contrato"].unique()
print(f"Contratos históricos: {len(contratos)}")

# ── 3. Generar residuales ─────────────────────────────────────────
def heuristic_rmse(n_riesgos):
    if n_riesgos <= 10: return 12.0
    elif n_riesgos <= 20: return 16.0
    elif n_riesgos <= 30: return 20.0
    else: return 24.0

rows = []
for i, cid in enumerate(contratos):
    sub = matriz[matriz["id_contrato"] == cid].copy()
    sobrecosto_real = float(sub["sobrecosto"].iloc[0])

    mc = macro_map.get(cid, {"anio_inicio": 2022, "anio_fin": 2022})
    feats = aggregate_risks(sub, anio_inicio=mc["anio_inicio"], anio_fin=mc["anio_fin"])

    n_riesgos = len(sub)

    X = feats[feature_names].values.astype(np.float64).reshape(1, -1)
    svr_pred = float(regressor.predict(scaler.transform(X))[0])

    abs_error = abs(sobrecosto_real - svr_pred)

    rows.append({
        "id_contrato": cid,
        "sobrecosto_real": sobrecosto_real,
        "svr_pred": svr_pred,
        "abs_error": abs_error,
        "n_riesgos": n_riesgos,
        "rmse_heuristic": heuristic_rmse(n_riesgos),
        **{f: feats[f].iloc[0] for f in feature_names},
    })

    if (i + 1) % 50 == 0:
        print(f"  Procesados {i + 1}/{len(contratos)} contratos...")

data = pd.DataFrame(rows)
print(f"\nDataset: {len(data)} contratos")

# ── 4. Estadísticas del error ─────────────────────────────────────
print(f"\nError absoluto del SVR en entrenamiento:")
print(f"  Media: {data['abs_error'].mean():.2f} pp")
print(f"  Mediana: {data['abs_error'].median():.2f} pp")
print(f"  Std: {data['abs_error'].std():.2f} pp")
print(f"  Min: {data['abs_error'].min():.2f} pp")
print(f"  Max: {data['abs_error'].max():.2f} pp")
print(f"\nRMSE del SVR en entrenamiento: {np.sqrt(mean_squared_error(data['sobrecosto_real'], data['svr_pred'])):.2f} pp")

# ── 5. Heurística vs error real por bucket ────────────────────────
print(f"\nError real vs RMSE heurístico asignado:")
print(f"{'Bucket':<10} {'n':<5} {'error_prom':<12} {'error_p90':<12} {'heur':<8} {'MAE_heur':<10}")
print("-" * 57)
for b, grp in data.groupby(pd.cut(data["n_riesgos"], bins=[0, 10, 20, 30, 999], labels=["1-10","11-20","21-30",">30"]), observed=True):
    mae_h = mean_absolute_error(grp["abs_error"], grp["rmse_heuristic"])
    print(f"{str(b):<10} {len(grp):<5} {grp['abs_error'].mean():<12.2f} {grp['abs_error'].quantile(0.9):<12.2f} {grp['rmse_heuristic'].iloc[0]:<8.0f} {mae_h:<10.2f}")

# ── 6. Entrenar modelo de error ───────────────────────────────────
X = data[feature_names].values.astype(np.float64)
y = data["abs_error"].values

# Escalar features para el SVR de error
scaler_error = joblib.load(ROOT / "models" / "scaler.pkl")
# Reuse same scaler from SVR (fitted on same features)

# Train/validation split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVR Linear": SVR(kernel="linear", C=1.0),
    "SVR RBF": SVR(kernel="rbf", C=1.0, gamma="scale"),
}

results = []
for name, model in models.items():
    model.fit(scaler_error.transform(X_train), y_train)
    y_pred = model.predict(scaler_error.transform(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mae, r2))
    print(f"{name:<15} | Test MAE={mae:.2f} | Test R2={r2:.4f}")

# CV on full dataset
print(f"\nCV (5-fold) en dataset completo:")
for name, model_cls in models.items():
    cv_scores = cross_val_score(model_cls, scaler_error.transform(X), y, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    print(f"  {name:<15} | MAE CV = {-cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

# ── 7. Elegir y guardar mejor modelo ─────────────────────────────
best_model = SVR(kernel="linear", C=1.0)
best_model.fit(scaler_error.transform(X), y)

joblib.dump(best_model, ROOT / "models" / "rmse_predictor.pkl")
print(f"\nModelo guardado en: models/rmse_predictor.pkl")

# ── 8. Comparación final ─────────────────────────────────────────
y_pred_full = best_model.predict(scaler_error.transform(X))
heuristic_vals = data["rmse_heuristic"].values

print(f"\n{'=' * 60}")
print(f"COMPARACIÓN FINAL (351 contratos de entrenamiento)")
print(f"{'=' * 60}")
print(f"{'Métrica':<30} {'Heurística':<15} {'RMSE Predictor':<15}")
print("-" * 60)
print(f"{'MAE (pp)':<30} {mean_absolute_error(y, heuristic_vals):<15.2f} {mean_absolute_error(y, y_pred_full):<15.2f}")
print(f"{'RMSE (pp)':<30} {np.sqrt(mean_squared_error(y, heuristic_vals)):<15.2f} {np.sqrt(mean_squared_error(y, y_pred_full)):<15.2f}")
print(f"{'Correlación con error real':<30} {pearsonr(heuristic_vals, y)[0]:<15.4f} {pearsonr(y_pred_full, y)[0]:<15.4f}")

# Per bucket
print(f"\nDesglose por bucket:")
print(f"{'Bucket':<10} {'n':<5} {'MAE heur':<10} {'MAE pred':<10} {'mejora':<8}")
print("-" * 43)
for b, grp in data.groupby(pd.cut(data["n_riesgos"], bins=[0, 10, 20, 30, 999], labels=["1-10","11-20","21-30",">30"]), observed=True):
    idx = grp.index
    mae_h = mean_absolute_error(y[idx], heuristic_vals[idx])
    mae_p = mean_absolute_error(y[idx], y_pred_full[idx])
    mejora = (mae_h - mae_p) / mae_h * 100
    print(f"{str(b):<10} {len(grp):<5} {mae_h:<10.2f} {mae_p:<10.2f} {mejora:<+7.1f}%")

print(f"\n{'=' * 60}")
mejora_global = (mean_absolute_error(y, heuristic_vals) - mean_absolute_error(y, y_pred_full)) / mean_absolute_error(y, heuristic_vals) * 100
print(f"MEJORA GLOBAL EN MAE: {mejora_global:.1f}%")
print(f"{'=' * 60}")
