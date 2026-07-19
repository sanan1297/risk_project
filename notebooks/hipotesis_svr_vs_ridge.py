# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Pruebas de Hipotesis: SVR vs Ridge
# ## Comparacion estadistica formal con 35 features (rango de fechas)
#
# **Pregunta:** ?SVR (kernel RBF) supera **significativamente** a Ridge (L2) en la prediccion de sobrecostos?
#
# **Tests incluidos:**
# 1. **Paired t-test** -- diferencia de medias en folds de CV
# 2. **Wilcoxon signed-rank** -- alternativa no parametrica
# 3. **Diebold-Mariano** -- comparacion de precision predictiva
# 4. **McNemar** -- concordancia en clasificacion binaria (>25%)
# 5. **Bootstrap** -- distribucion bootstrap de la diferencia
# 6. **Cohen's d** -- tamano del efecto

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, norm

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, RepeatedKFold,
    KFold
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)

import sys
sys.path.insert(0, str(Path.cwd().resolve()))
from estudio_data.features import engineer_features, STOP_WORDS

import matplotlib.pyplot as plt
import seaborn as sns

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
# ## 1. Carga de datos y feature engineering

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
print("  CARGA DE DATOS")
print("=" * 60)

matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
print(f"Matriz: {len(matriz):,} riesgos | {matriz['id_contrato'].nunique()} contratos")

macro = pd.read_csv(DATA_DIR / "contratos_macro.csv")
print(f"Macro: {len(macro)} contratos con ipc_acumulado + trm_promedio")

print("\nFeature engineering...")
df_feat = engineer_features(matriz)
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

print(f"\nX: {X.shape}  |  y: {y.min():.1f}% - {y.max():.1f}%")
print(f"Alto riesgo (>25%): {y_bin.mean():.1%}")

# %% [markdown]
# ## 2. Configuracion experimental
#
# Usamos **RepeatedKFold (10 rep x 5 folds = 50 pares)** para obtener
# distribuciones robustas de metricas y poder hacer tests estadisticos.

# %%
N_REPEATS = 5
N_SPLITS = 5
RANDOM_STATE = 42

rkf = RepeatedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

print("=" * 60)
print("  CONFIGURACIÓN EXPERIMENTAL")
print("=" * 60)
print(f"  Repeated K-Fold: {N_SPLITS} splits x {N_REPEATS} reps = {N_SPLITS * N_REPEATS} folds")
print(f"  Modelos: SVR (kernel RBF, C=10, gamma=scale) vs Ridge (alpha=244.0)")

# %% [markdown]
# ## 3. Entrenamiento y evaluacion con CV repetida

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svr = SVR(kernel="rbf", C=10, gamma="scale")
ridge = Ridge(alpha=244.0, random_state=RANDOM_STATE)

models = {
    "SVR (RBF)": svr,
    "Ridge (L2)": ridge,
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")

    r2_scores = []
    rmse_scores = []
    mae_scores = []
    auc_scores = []
    y_pred_all = np.zeros_like(y)
    y_true_all = y.copy()

    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_bin_test = y_bin[test_idx]

        if name == "SVR (RBF)":
            m = SVR(kernel="rbf", C=10, gamma="scale")
        else:
            m = Ridge(alpha=244.0, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

        try:
            auc_scores.append(roc_auc_score(y_bin_test, y_pred))
        except ValueError:
            auc_scores.append(np.nan)

    results[name] = {
        "r2": np.array(r2_scores),
        "rmse": np.array(rmse_scores),
        "mae": np.array(mae_scores),
        "auc": np.array(auc_scores),
    }

    print(f"  R2:  {r2_scores[-N_SPLITS:]} ...")
    print(f"  R2:  {np.mean(r2_scores):.4f} +- {np.std(r2_scores):.4f}")
    print(f"  RMSE: {np.mean(rmse_scores):.2f} +- {np.std(rmse_scores):.2f}")
    print(f"  MAE:  {np.mean(mae_scores):.2f} +- {np.std(mae_scores):.2f}")
    print(f"  AUC:  {np.nanmean(auc_scores):.4f} +- {np.nanstd(auc_scores):.4f}")

# %% [markdown]
# ## 4. Tabla resumen

# %%
summary = []
for name in models:
    r = results[name]
    summary.append({
        "Modelo": name,
        "R2": f"{r['r2'].mean():.4f} +- {r['r2'].std():.4f}",
        "RMSE": f"{r['rmse'].mean():.2f} +- {r['rmse'].std():.2f}",
        "MAE": f"{r['mae'].mean():.2f} +- {r['mae'].std():.2f}",
        "AUC": f"{np.nanmean(r['auc']):.4f} +- {np.nanstd(r['auc']):.4f}",
    })

print("=" * 70)
print("  RESUMEN -- CV Repetida (10x5 = 50 folds)")
print("=" * 70)
print(f"{'Modelo':<20} {'R2':<18} {'RMSE':<18} {'MAE':<18} {'AUC':<18}")
print("-" * 70)
for s in summary:
    print(f"{s['Modelo']:<20} {s['R2']:<18} {s['RMSE']:<18} {s['MAE']:<18} {s['AUC']:<18}")

# %% [markdown]
# ## 5. Tests de Hipotesis
#
# Para cada metrica, probamos:
# - **H₀:** SVR no supera a Ridge (diferencia ≤ 0)
# - **H1:** SVR supera a Ridge (diferencia > 0)
#
# Es decir, test **unilateral** (one-tailed).

# %%
def interpret_p(p):
    if p < 0.001:
        return "*** p < 0.001"
    if p < 0.01:
        return "** p < 0.01"
    if p < 0.05:
        return "* p < 0.05"
    if p < 0.10:
        return "† p < 0.10"
    return f"n.s. (p = {p:.4f})"

def cohens_d(x, y):
    n = len(x)
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

print("=" * 70)
print("  PRUEBAS DE HIPÓTESIS ESTADÍSTICAS")
print("  H_1: SVR supera a Ridge en la metrica indicada (one-tailed)")
print("=" * 70)

metrics_info = [
    ("R2", results["SVR (RBF)"]["r2"], results["Ridge (L2)"]["r2"],
     lambda svr_m, ridge_m: np.mean(svr_m) - np.mean(ridge_m),
     "mayor"),
    ("RMSE", results["SVR (RBF)"]["rmse"], results["Ridge (L2)"]["rmse"],
     lambda svr_m, ridge_m: float(np.mean(ridge_m - svr_m)),
     "menor"),
    ("MAE", results["SVR (RBF)"]["mae"], results["Ridge (L2)"]["mae"],
     lambda svr_m, ridge_m: float(np.mean(ridge_m - svr_m)),
     "menor"),
    ("AUC", results["SVR (RBF)"]["auc"], results["Ridge (L2)"]["auc"],
     lambda svr_m, ridge_m: np.mean(svr_m) - np.mean(ridge_m),
     "mayor"),
]

for metric_name, svr_vals, ridge_vals, diff_func, better_dir in metrics_info:
    svr_clean = svr_vals[~np.isnan(svr_vals)]
    ridge_clean = ridge_vals[~np.isnan(ridge_vals)]
    min_len = min(len(svr_clean), len(ridge_clean))
    svr_clean = svr_clean[:min_len]
    ridge_clean = ridge_clean[:min_len]

    diff = svr_clean - ridge_clean if better_dir == "mayor" else ridge_clean - svr_clean
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)

    t_stat, p_paired = ttest_rel(svr_clean, ridge_clean, alternative="greater" if better_dir == "mayor" else "less")
    if metric_name == "R2":
        _, p_wilcox = wilcoxon(svr_clean, ridge_clean, alternative="greater")
    elif metric_name == "RMSE":
        _, p_wilcox = wilcoxon(ridge_clean, svr_clean, alternative="greater")
    elif metric_name == "MAE":
        _, p_wilcox = wilcoxon(ridge_clean, svr_clean, alternative="greater")
    else:
        _, p_wilcox = wilcoxon(svr_clean, ridge_clean, alternative="greater")

    d = cohens_d(svr_clean, ridge_clean)
    if better_dir == "menor":
        d = cohens_d(ridge_clean, svr_clean)
    if d < 0:
        eff_size = f"negativo (favorece a Ridge, d={d:.3f})"
    elif d < 0.2:
        eff_size = f"despreciable (d={d:.3f})" if d >= 0 else f"negativo (d={d:.3f})"
    elif d < 0.5:
        eff_size = f"pequeno (d={d:.3f})"
    elif d < 0.8:
        eff_size = f"mediano (d={d:.3f})"
    else:
        eff_size = f"grande (d={d:.3f})"

    actual_diff = diff_func(svr_clean, ridge_clean)
    pct_better = (diff > 0).mean()

    print(f"\n{metric_name}:")
    print(f"  Diferencia media: {actual_diff:.4f} ({pct_better:.0%} de folds favorecen)")
    print(f"  Paired t-test:    t = {t_stat:.3f}, {interpret_p(p_paired)}")
    print(f"  Wilcoxon:         {interpret_p(p_wilcox)}")
    print(f"  Cohen's d:        {eff_size}")

# %% [markdown]
# ## 6. Test de Diebold-Mariano
#
# Compara la precision predictiva de dos modelos usando la funcion de perdida
# cuadratica. Evalua si la diferencia en MSE es estadisticamente significativa.

# %%
def diebold_mariano(e1, e2, h=1, alternative="two-sided"):
    n = len(e1)
    d = e1**2 - e2**2
    d_mean = d.mean()
    if n > 1:
        acf = np.correlate(d - d.mean(), d - d.mean(), mode="full")
        acf = acf[acf.size // 2:]
        var_d = acf[0] + 2 * sum(acf[1:h + 1])
        var_d /= n
    else:
        var_d = 0
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(var_d)
    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(dm_stat)))
    elif alternative == "greater":
        p = 1 - norm.cdf(dm_stat)
    else:
        p = norm.cdf(dm_stat)
    return dm_stat, p

print("=" * 70)
print("  TEST DE DIEBOLD-MARIANO")
print("  H1: SVR tiene menor MSE que Ridge (one-tailed)")
print("  DM < 0 -> SVR es mas preciso")
print("=" * 70)

mse_svr = results["SVR (RBF)"]["rmse"] ** 2
mse_ridge = results["Ridge (L2)"]["rmse"] ** 2

for metric_name, e_svr, e_ridge in [
    ("RMSE (cuadratico)", results["SVR (RBF)"]["rmse"], results["Ridge (L2)"]["rmse"]),
    ("MAE (absoluto)", results["SVR (RBF)"]["mae"], results["Ridge (L2)"]["mae"]),
]:
    dm_stat, p_dm = diebold_mariano(e_svr, e_ridge, alternative="less")
    print(f"\n{metric_name}:")
    print(f"  DM statistic: {dm_stat:.3f}")
    print(f"  p-value:      {interpret_p(p_dm)}")
    if dm_stat < 0:
        print(f"  -> SVR tiene menor error (favorece a SVR)")
    else:
        print(f"  -> Ridge tiene menor error (favorece a Ridge)")

# %% [markdown]
# ## 7. Test de McNemar -- Clasificacion binaria (>25% sobrecosto)
#
# Evalua si la diferencia en tasa de aciertos (alertas correctas) entre
# SVR y Ridge es estadisticamente significativa.

# %%
print("=" * 70)
print("  TEST DE MCNEMAR -- Clasificacion Binaria")
print("  Umbral: sobrecosto > 25% = ALTO RIESGO")
print("=" * 70)

rkf_mc = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
y_pred_svr_bin = np.zeros_like(y_bin, dtype=float)
y_pred_ridge_bin = np.zeros_like(y_bin, dtype=float)

for train_idx, test_idx in rkf_mc.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = y[train_idx]

    svr_cv = SVR(kernel="rbf", C=10, gamma="scale")
    svr_cv.fit(X_train, y_train)
    y_pred_svr_bin[test_idx] = svr_cv.predict(X_test)

    ridge_cv = Ridge(alpha=244.0, random_state=RANDOM_STATE)
    ridge_cv.fit(X_train, y_train)
    y_pred_ridge_bin[test_idx] = ridge_cv.predict(X_test)

y_pred_svr_bin_class = (y_pred_svr_bin > 25).astype(int)
y_pred_ridge_bin_class = (y_pred_ridge_bin > 25).astype(int)

tn, fp, fn, tp = confusion_matrix(y_bin, y_pred_svr_bin_class).ravel()
tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_bin, y_pred_ridge_bin_class).ravel()

print(f"\nSVR -- Accuracy: {(tp + tn) / len(y_bin):.1%}")
print(f"       Precision: {tp / (tp + fp):.1%}  Recall: {tp / (tp + fn):.1%}")
print(f"       F1-score: {2 * tp / (2 * tp + fp + fn):.3f}")
print(f"\nRidge -- Accuracy: {(tp_r + tn_r) / len(y_bin):.1%}")
print(f"       Precision: {tp_r / (tp_r + fp_r):.1%}  Recall: {tp_r / (tp_r + fn_r):.1%}")
print(f"       F1-score: {2 * tp_r / (2 * tp_r + fp_r + fn_r):.3f}")

a = (y_pred_svr_bin_class == 1) & (y_pred_ridge_bin_class == 0)
b = (y_pred_svr_bin_class == 0) & (y_pred_ridge_bin_class == 1)

b_obs = a.sum()
c_obs = b.sum()

if b_obs + c_obs > 0:
    mcnemar_stat = ((b_obs - c_obs) ** 2) / (b_obs + c_obs)
    p_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    kappa = cohen_kappa_score(y_pred_svr_bin_class, y_pred_ridge_bin_class)
else:
    mcnemar_stat = 0.0
    p_mcnemar = 1.0
    kappa = 1.0

print(f"\nDiscordancia SVR-1 & Ridge-0: {b_obs}")
print(f"Discordancia SVR-0 & Ridge-1: {c_obs}")
print(f"McNemar chi2 = {mcnemar_stat:.3f}  p = {p_mcnemar:.4f}  {interpret_p(p_mcnemar)}")
print(f"Cohen's kappa = {kappa:.4f}")

# %% [markdown]
# ## 8. Bootstrap -- distribucion de la diferencia
#
# Generamos la distribucion bootstrap de `diferencia = R2_SVR - R2_Ridge`
# para estimar intervalos de confianza sin supuestos parametricos.

# %%
N_BOOTSTRAP = 2000
rng = np.random.default_rng(RANDOM_STATE)

svr_r2_full = results["SVR (RBF)"]["r2"]
ridge_r2_full = results["Ridge (L2)"]["r2"]

boot_diffs = np.zeros(N_BOOTSTRAP)
boot_diffs_rmse = np.zeros(N_BOOTSTRAP)
boot_diffs_auc = np.zeros(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(svr_r2_full), size=len(svr_r2_full))
    svr_boot = svr_r2_full[idx]
    ridge_boot = ridge_r2_full[idx]
    boot_diffs[i] = svr_boot.mean() - ridge_boot.mean()

    rmse_svr_boot = results["SVR (RBF)"]["rmse"][idx]
    rmse_ridge_boot = results["Ridge (L2)"]["rmse"][idx]
    boot_diffs_rmse[i] = rmse_ridge_boot.mean() - rmse_svr_boot.mean()

    auc_svr_boot = results["SVR (RBF)"]["auc"][idx]
    auc_ridge_boot = results["Ridge (L2)"]["auc"][idx]
    boot_diffs_auc[i] = auc_svr_boot.mean() - auc_ridge_boot.mean()

def ci_percentile(samples, alpha=0.05):
    lo = np.percentile(samples, 100 * alpha / 2)
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return lo, hi

print("=" * 70)
print("  BOOTSTRAP -- Intervalos de Confianza (95%)")
print(f"  {N_BOOTSTRAP:,} iteraciones")
print("=" * 70)

for name, diffs, label_winner in [
    ("Delta R2 (SVR - Ridge)", boot_diffs, "mayor"),
    ("Delta RMSE (Ridge - SVR)", boot_diffs_rmse, "menor"),
    ("Delta AUC (SVR - Ridge)", boot_diffs_auc, "mayor"),
]:
    lo, hi = ci_percentile(diffs)
    mean_diff = diffs.mean()
    pct_pos = (diffs > 0).mean()
    print(f"\n{name}:")
    print(f"  Media: {mean_diff:.4f}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  % positivo: {pct_pos:.1%}")
    if lo > 0 and hi > 0:
        print(f"  -> Diferencia SIGNIFICATIVA (IC no cruza cero)")
    elif lo < 0 and hi < 0:
        print(f"  -> Ridge supera a SVR significativamente")
    else:
        print(f"  -> NO significativa (IC cruza cero)")

# %% [markdown]
# ## 9. Visualizaciones

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_plot = [
    ("R2", "R2 (mayor es mejor)", "#2ecc71"),
    ("RMSE", "RMSE (menor es mejor)", "#e74c3c"),
    ("MAE", "MAE (menor es mejor)", "#f39c12"),
    ("AUC", "AUC (mayor es mejor)", "#9b59b6"),
]

for idx, (metric, title, color) in enumerate(metrics_plot):
    ax = axes[idx // 2, idx % 2]
    ax.violinplot(
        [results["SVR (RBF)"][metric.lower()],
         results["Ridge (L2)"][metric.lower()]],
        positions=[1, 2], showmeans=True, showmedians=True
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["SVR (RBF)", "Ridge (L2)"], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

ax = axes[1, 2]
ax.hist(boot_diffs, bins=50, alpha=0.7, color="#3498db", edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="cero")
lo, hi = ci_percentile(boot_diffs)
ax.axvline(lo, color="orange", linestyle=":", linewidth=1.5, label=f"CI 95%")
ax.axvline(hi, color="orange", linestyle=":", linewidth=1.5)
ax.set_title("Bootstrap: Delta R2 (SVR - Ridge)", fontsize=13, fontweight="bold")
ax.set_xlabel("Delta R2", fontsize=12)
ax.set_ylabel("Frecuencia", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(DATA_DIR / "hipotesis_svr_vs_ridge.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Grafico guardado: {DATA_DIR / 'hipotesis_svr_vs_ridge.png'}")

# %% [markdown]
# ## 10. Conclusion

# %%
print("=" * 70)
print("  CONCLUSIÓN -- SVR vs Ridge (35 features, rango de fechas)")
print("=" * 70)

better_r2 = results["SVR (RBF)"]["r2"].mean() > results["Ridge (L2)"]["r2"].mean()
better_rmse = results["SVR (RBF)"]["rmse"].mean() < results["Ridge (L2)"]["rmse"].mean()
better_mae = results["SVR (RBF)"]["mae"].mean() < results["Ridge (L2)"]["mae"].mean()
better_auc = np.nanmean(results["SVR (RBF)"]["auc"]) > np.nanmean(results["Ridge (L2)"]["auc"])

n_wins = sum([better_r2, better_rmse, better_mae, better_auc])

print(f"\nSVR gana en {n_wins}/4 metricas:")
print(f"  R2:  {'SVR V' if better_r2 else 'Ridge'}")
print(f"  RMSE: {'SVR V' if better_rmse else 'Ridge'}")
print(f"  MAE:  {'SVR V' if better_mae else 'Ridge'}")
print(f"  AUC:  {'SVR V' if better_auc else 'Ridge'}")

_, p_r2 = ttest_rel(
    results["SVR (RBF)"]["r2"],
    results["Ridge (L2)"]["r2"],
    alternative="greater"
)

print(f"\n{'-> SVR supera significativamente a Ridge' if p_r2 < 0.05 else '-> NO hay evidencia suficiente para concluir que SVR supera a Ridge'}")
print(f"  Paired t-test R2 (one-tailed): p = {p_r2:.4f}")
print(f"\nNota: La decision de usar SVR como modelo campeon se basa en")
print(f"su capacidad para capturar relaciones no lineales (rango de fechas),")
print(f"no necesariamente en superioridad estadistica sobre Ridge.")

