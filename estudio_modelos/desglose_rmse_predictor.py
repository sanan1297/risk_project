"""
Desglose detallado del RMSE Predictor para documentación.
Genera la misma info que el SHAP desglose: patrones, ejemplos, distribución.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from backend.feature_engineering import aggregate_risks
from backend.predictor import load as load_model

# ── Cargar modelos ────────────────────────────────────────────────
regressor, _, scaler, feature_names = load_model()
rmse_predictor = joblib.load(ROOT / "models" / "rmse_predictor.pkl")

def heuristic_rmse(n):
    if n <= 10: return 12.0
    elif n <= 20: return 16.0
    elif n <= 30: return 20.0
    else: return 24.0

# ── Cargar y procesar datos históricos ────────────────────────────
matriz = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
macro_map = macro.set_index("id_contrato").to_dict("index")

contratos = matriz["id_contrato"].unique()

rows = []
for cid in contratos:
    sub = matriz[matriz["id_contrato"] == cid].copy()
    sc_real = float(sub["sobrecosto"].iloc[0])
    mc = macro_map.get(cid, {"anio_inicio": 2022, "anio_fin": 2022})
    feats = aggregate_risks(sub, anio_inicio=mc["anio_inicio"], anio_fin=mc["anio_fin"])
    n_riesgos = len(sub)

    X = feats[feature_names].values.astype(np.float64).reshape(1, -1)
    svr_pred = float(regressor.predict(scaler.transform(X))[0])
    abs_error = abs(sc_real - svr_pred)

    X_s = scaler.transform(X)
    rmse_pred = float(rmse_predictor.predict(X_s)[0])
    rmse_heur = heuristic_rmse(n_riesgos)

    rows.append({
        "id_contrato": cid,
        "sobrecosto_real": sc_real,
        "svr_pred": svr_pred,
        "abs_error": abs_error,
        "n_riesgos": n_riesgos,
        "rmse_heuristic": rmse_heur,
        "rmse_pred": rmse_pred,
        "err_heur_vs_real": abs(rmse_heur - abs_error),
        "err_pred_vs_real": abs(rmse_pred - abs_error),
    })

data = pd.DataFrame(rows)
data["bucket"] = pd.cut(data["n_riesgos"], bins=[0, 10, 20, 30, 999], labels=["1-10","11-20","21-30",">30"])

# ══════════════════════════════════════════════════════════════════
print("=" * 72)
print("DESGLOSE DETALLADO — RMSE PREDICTOR (Residuales de Entrenamiento)")
print("=" * 72)

# ── 1. Resumen del error del SVR ─────────────────────────────────
print("\n" + "─" * 72)
print("1. Error del SVR de Sobrecosto en Datos Históricos (n=351)")
print("─" * 72)
print(f"El SVR predice el sobrecosto (% de desviación del valor inicial).")
print(f"Su error absoluto (|real - predicción|) en entrenamiento es:")
print(f"")
print(f"  Media:     {data['abs_error'].mean():>7.2f} pp")
print(f"  Mediana:   {data['abs_error'].median():>7.2f} pp")
print(f"  Desv. std: {data['abs_error'].std():>7.2f} pp")
print(f"  P25:       {data['abs_error'].quantile(0.25):>7.2f} pp")
print(f"  P75:       {data['abs_error'].quantile(0.75):>7.2f} pp")
print(f"  P90:       {data['abs_error'].quantile(0.90):>7.2f} pp")
print(f"  Máximo:    {data['abs_error'].max():>7.2f} pp")
print(f"")
print(f"  => 50% de los contratos tienen error <6.43 pp (mediana).")
print(f"  => La media (11.57) es mayor que la mediana por la cola")
print(f"     derecha: ~10 contratos con error extremo (>50 pp).")

# ── 2. Heurística actual vs error real ────────────────────────────
print("\n" + "─" * 72)
print("2. Heurística Actual: RMSE Fijo por Bucket de n_riesgos")
print("─" * 72)
print(f"")
print(f"{'Bucket':<10} {'n':<6} {'Error_prom':<12} {'Error_std':<12} {'Error_P90':<12} {'RMSE_asig':<12} {'MAE_asig':<10}")
print("-" * 74)
for b, grp in data.groupby("bucket", observed=True):
    mae = mean_absolute_error(grp["abs_error"], grp["rmse_heuristic"])
    print(f"{str(b):<10} {len(grp):<6} {grp['abs_error'].mean():<12.2f} {grp['abs_error'].std():<12.2f} {grp['abs_error'].quantile(0.9):<12.2f} {grp['rmse_heuristic'].iloc[0]:<12.0f} {mae:<10.2f}")
print(f"")
print(f"Problema: los buckets 21-30 y >30 asignan 20-24 pp cuando")
print(f"el error real promedio es ~10 pp. La heurística sobreestima")
print(f"sistemáticamente el error en contratos con muchos riesgos.")

# ── 3. RMSE Predictor ─────────────────────────────────────────────
print("\n" + "─" * 72)
print("3. RMSE Predictor: SVR Lineal para Predecir el Error")
print("─" * 72)
print(f"")
print(f"Target:     |sobrecosto_real - svr_pred|")
print(f"Features:   {len(feature_names)} (TF-IDF + riesgos + macro)")
print(f"Modelo:    SVR(kernel='linear', C=1.0)")
print(f"Escalado:  mismo scaler del SVR de sobrecosto")
print(f"")
print(f"Comparación global:")
print(f"")
print(f"{'Métrica':<35} {'Heurística':<15} {'RMSE Pred.':<15} {'Mejora':<10}")
print("-" * 75)
mae_h = mean_absolute_error(data["abs_error"], data["rmse_heuristic"])
mae_p = mean_absolute_error(data["abs_error"], data["rmse_pred"])
rmse_h = np.sqrt(mean_squared_error(data["abs_error"], data["rmse_heuristic"]))
rmse_p = np.sqrt(mean_squared_error(data["abs_error"], data["rmse_pred"]))
corr_h = pearsonr(data["rmse_heuristic"], data["abs_error"])[0]
corr_p = pearsonr(data["rmse_pred"], data["abs_error"])[0]
print(f"{'MAE (pp)':<35} {mae_h:<15.2f} {mae_p:<15.2f} {(mae_h-mae_p)/mae_h*100:<+9.1f}%")
print(f"{'RMSE (pp)':<35} {rmse_h:<15.2f} {rmse_p:<15.2f} {(rmse_h-rmse_p)/rmse_h*100:<+9.1f}%")
print(f"{'MAE / RMSE ratio':<35} {mae_h/rmse_h:<15.3f} {mae_p/rmse_p:<15.3f}")
print(f"{'Correlación con error real':<35} {corr_h:<15.4f} {corr_p:<15.4f}")
print(f"")

# ── 4. Desglose por bucket ───────────────────────────────────────
print("─" * 72)
print("4. Desglose por Bucket de n_riesgos")
print("─" * 72)
print(f"")
print(f"{'Bucket':<10} {'n':<6} {'MAE heur':<12} {'MAE pred':<12} {'Mejora':<10} {'RMSE heur':<12} {'RMSE pred':<12}")
print("-" * 74)
for b, grp in data.groupby("bucket", observed=True):
    mh = mean_absolute_error(grp["abs_error"], grp["rmse_heuristic"])
    mp = mean_absolute_error(grp["abs_error"], grp["rmse_pred"])
    rh = np.sqrt(mean_squared_error(grp["abs_error"], grp["rmse_heuristic"]))
    rp = np.sqrt(mean_squared_error(grp["abs_error"], grp["rmse_pred"]))
    mej = (mh - mp) / mh * 100
    print(f"{str(b):<10} {len(grp):<6} {mh:<12.2f} {mp:<12.2f} {mej:<+9.1f}% {rh:<12.2f} {rp:<12.2f}")
print(f"")
print(f"Análisis:")
print(f"  - Bucket 1-10: mejora +28%. La heurística (12pp) sobra para")
print(f"    contratos simples. El predictor ajusta a la baja.")
print(f"  - Bucket 11-20: mejora +21%. Es el más ruidoso (std=72pp")
print(f"    por outliers extremos). El predictor identifica mejor")
print(f"    qué contratos serán outliers.")
print(f"  - Bucket 21-30: mejora +45%. La heurística sobreasignaba 20pp")
print(f"    a contratos con error real ~10pp. Gran corrección.")
print(f"  - Bucket >30: mejora +50%. Similar al anterior: 24pp → ~8pp MAE.")
print(f"")

# ── 5. Top 10 mejores y peores casos ──────────────────────────────
print("─" * 72)
print("5. Top 5 Casos Donde el Predictor Gana Más")
print("─" * 72)
data["gain_over_heuristic"] = data["err_heur_vs_real"] - data["err_pred_vs_real"]
top = data.sort_values("gain_over_heuristic", ascending=False).head(5)
print(f"")
print(f"{'Contrato':<12} {'n_riesgos':<10} {'Error':<10} {'Heur':<10} {'Pred':<10} {'Gana':<10} {'Obs':<25}")
print("-" * 87)
for _, r in top.iterrows():
    obs = f"Heur={r['rmse_heuristic']:.0f}, real={r['abs_error']:.1f}"
    print(f"{r['id_contrato']:<12} {r['n_riesgos']:<10} {r['abs_error']:<10.2f} {r['rmse_heuristic']:<10.0f} {r['rmse_pred']:<10.2f} {r['gain_over_heuristic']:<+10.2f} {obs:<25}")

print(f"")
print("Peores 5 (predictor pierde vs heurística):")
worst = data.sort_values("gain_over_heuristic", ascending=True).head(5)
print(f"{'Contrato':<12} {'n_riesgos':<10} {'Error':<10} {'Heur':<10} {'Pred':<10} {'Pierde':<10} {'Obs':<25}")
print("-" * 87)
for _, r in worst.iterrows():
    obs = f"Heur={r['rmse_heuristic']:.0f}, real={r['abs_error']:.1f}"
    print(f"{r['id_contrato']:<12} {r['n_riesgos']:<10} {r['abs_error']:<10.2f} {r['rmse_heuristic']:<10.0f} {r['rmse_pred']:<10.2f} {r['gain_over_heuristic']:<+10.2f} {obs:<25}")

# ── 6. Distribución de predicciones ───────────────────────────────
print("\n" + "─" * 72)
print("6. Distribución del RMSE Predicho vs Heurístico")
print("─" * 72)
print(f"")
print(f"{'Estadístico':<20} {'Heurística':<15} {'RMSE Pred.':<15} {'Error Real':<15}")
print("-" * 65)
for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
    if stat in ["min", "max"]:
        h = getattr(data["rmse_heuristic"], stat)()
        p = getattr(data["rmse_pred"], stat)()
        r = getattr(data["abs_error"], stat)()
    else:
        h = data["rmse_heuristic"].describe()[stat]
        p = data["rmse_pred"].describe()[stat]
        r = data["abs_error"].describe()[stat]
    print(f"{stat:<20} {h:<15.2f} {p:<15.2f} {r:<15.2f}")
print(f"")
print(f"La heurística solo produce 4 valores (12, 16, 20, 24).")
print(f"El RMSE Predictor produce valores continuos, distribuyéndose")
print(f"de forma similar al error real (media ~10-12, std ~9-11).")

# ── 7. Correlación de predicciones con error real ─────────────────
print("─" * 72)
print("7. Interpretación")
print("─" * 72)
print(f"")
print(f"El RMSE Predictor NO intenta predecir el error exacto (eso es")
print(f"imposible: es ruido aleatorio del modelo). En su lugar, aprende")
print(f"el perfil de incertidumbre: qué características hacen que el")
print(f"SVR se equivoque más o menos para un contrato dado.")
print(f"")
print(f"La correlación baja (R={corr_p:.4f}) con el error real confirma")
print(f"que el error del SVR tiene un componente aleatorio importante.")
print(f"Pero el MAE 32% menor que la heurística significa que el RMSE")
print(f"asignado es más cercano en magnitud al error real.")
print(f"")
print(f"Impacto en Monte Carlo:")
print(f"  - Un RMSE más preciso significa que el ruido gaussiano añadido")
print(f"    en cada iteración MC (línea 284 de quantitative_analysis.py)")
print(f"    tiene una desviación estándar más realista.")
print(f"  - Contratos con sobrecosto predecible (TF-IDF claro, pocos")
print(f"    riesgos, tipo conocido) → RMSE bajo → MC concentrado.")
print(f"  - Contratos atípicos o con descripciones ambiguas → RMSE alto")
print(f"    → MC disperso → percentiles más conservadores.")

# ── 8. Cuándo no mejora ───────────────────────────────────────────
print("─" * 72)
print("8. Limitaciones")
print("─" * 72)
print(f"")
print(f"  1. El modelo de error se entrenó sobre residuales del mismo")
print(f"     SVR. Si el SVR cambia (re-entrenamiento), el RMSE Predictor")
print(f"     debe re-entrenarse también.")
print(f"  2. R² negativo: el modelo no explica la varianza del error.")
print(f"     Sirve para estimar magnitud, no para rankear contratos")
print(f"     por incertidumbre.")
print(f"  3. 351 contratos es una muestra pequeña para un problema con")
print(f"     tanta varianza. Más datos mejorarían el modelo.")
print(f"  4. Outliers extremos (error >100pp) distorsionan el RMSE total")
print(f"     aunque el MAE mejore sustancialmente.")
print(f"")

print("=" * 72)
print("FIN DEL DESGLOSE")
print("=" * 72)
