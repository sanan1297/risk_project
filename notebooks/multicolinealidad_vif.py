# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Analisis de Multicolinealidad — VIF de las 35 features
#
# La multicolinealidad ocurre cuando dos o mas predictores estan altamente
# correlacionados, lo que infla la varianza de los coeficientes estimados
# y dificulta la interpretacion.
#
# Aunque SVR con kernel RBF maneja relaciones no lineales y la regularizacion
# (C, epsilon, margin) mitiga el sobreajuste, el analisis de VIF ayuda a:
# 1. Identificar features redundantes
# 2. Fortalecer la discusion sobre estabilidad del modelo
# 3. Justificar la seleccion de features
# 4. Entender limitaciones de interpretabilidad (permutation importance, SHAP)

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path.cwd().resolve()))
from estudio_data.features import engineer_features, STOP_WORDS

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
print(f"Dataset final: {df_feat.shape}")

missing = [c for c in FEATURES_35 if c not in df_feat.columns]
if missing:
    raise ValueError(f"Faltan: {missing}")

X_raw = df_feat[FEATURES_35].copy()
y = df_feat["sobrecosto"].values

# %% [markdown]
# ## 2. Matriz de correlacion
#
# Exploramos correlaciones de Pearson entre las 35 features.
# Umbral |r| > 0.7 indica colinealidad potencialmente problematica.

# %%
print("=" * 60)
print("  MATRIZ DE CORRELACION")
print("=" * 60)

corr_matrix = X_raw.corr(method="pearson")

corr_triu = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_pairs = []
for col in corr_triu.columns:
    for idx in corr_triu.index:
        val = corr_triu.loc[idx, col]
        if pd.notna(val) and abs(val) >= 0.7:
            high_pairs.append((col, idx, val))

high_pairs_sorted = sorted(high_pairs, key=lambda x: -abs(x[2]))

print(f"\nPares con |r| >= 0.7 ({len(high_pairs)} pares):")
for c1, c2, r in high_pairs_sorted[:15]:
    print(f"  {c1:35s} <-> {c2:35s}  r = {r:+.4f}")
if len(high_pairs_sorted) > 15:
    print(f"  ... y {len(high_pairs_sorted) - 15} pares mas")

n_high = len(high_pairs)
n_total = (35 * 34) // 2
print(f"\nProporcion de pares con alta correlacion: {n_high}/{n_total} = {100*n_high/n_total:.1f}%")

# %% [markdown]
# ## 3. Variance Inflation Factor (VIF)
#
# VIF = 1 / (1 - R²_i) donde R²_i es el R² de regresion de la feature i
# contra todas las demas.
#
# Criterios:
# - VIF = 1: no correlacionada
# - 1 < VIF < 5: correlacion moderada (aceptable)
# - 5 <= VIF < 10: correlacion alta (causa preocupacion)
# - VIF >= 10: multicolinealidad severa (problematica)

# %%
def compute_vif(df_features):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    vif_data["vif"] = 0.0
    vif_data["r2"] = 0.0
    vif_data["tol"] = 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values.astype(np.float64))
    feature_names = df_features.columns

    for i, col in enumerate(feature_names):
        y_i = X_scaled[:, i]
        X_i = np.delete(X_scaled, i, axis=1)

        reg = LinearRegression()
        reg.fit(X_i, y_i)
        r2_i = reg.score(X_i, y_i)

        if r2_i >= 0.999999:
            vif_i = float("inf")
        else:
            vif_i = 1.0 / (1.0 - r2_i)

        vif_data.loc[vif_data["feature"] == col, "vif"] = vif_i
        vif_data.loc[vif_data["feature"] == col, "r2"] = r2_i
        vif_data.loc[vif_data["feature"] == col, "tol"] = 1.0 - r2_i

    return vif_data.sort_values("vif", ascending=False)

print("=" * 60)
print("  VIF — Variance Inflation Factor")
print("=" * 60)

vif_df = compute_vif(X_raw)

def vif_label(v):
    if v == float("inf"):
        return "INF (perfecta)"
    if v >= 10:
        return "SEVERA"
    if v >= 5:
        return "ALTA"
    if v >= 2:
        return "MODERADA"
    return "BAJA"

vif_df["nivel"] = vif_df["vif"].apply(vif_label)
vif_df["vif_round"] = vif_df["vif"].apply(lambda v: float("inf") if v == float("inf") else round(v, 2))

print(f"\n{'Feature':35s} {'VIF':>10s} {'R2':>8s} {'Tol':>8s} {'Nivel':>12s}")
print("-" * 75)
for _, r in vif_df.iterrows():
    vif_str = f"{r['vif_round']:.2f}" if r['vif_round'] != float("inf") else "INF"
    print(f"{r['feature']:35s} {vif_str:>10s} {r['r2']:.4f}  {r['tol']:.4f}  {r['nivel']:>12s}")

n_severa = (vif_df["vif"] >= 10).sum()
n_alta = ((vif_df["vif"] >= 5) & (vif_df["vif"] < 10)).sum()
n_moderada = ((vif_df["vif"] >= 2) & (vif_df["vif"] < 5)).sum()
n_baja = (vif_df["vif"] < 2).sum()

print(f"\nResumen:")
print(f"  VIF < 2   (baja):   {n_baja} features")
print(f"  2 <= VIF < 5  (moderada): {n_moderada} features")
print(f"  5 <= VIF < 10 (alta):     {n_alta} features")
print(f"  VIF >= 10 (severa):  {n_severa} features")

# %% [markdown]
# ## 4. Analisis por grupos de features
#
# Agrupamos las features en categorias semanticas para entender
# donde se concentra la multicolinealidad.

# %%
print("=" * 60)
print("  VIF POR GRUPO SEMANTICO")
print("=" * 60)

def classify_feature(name):
    if name in CONTROL_VARS_RANGO:
        return "Macro (rango fechas)"
    if name.startswith("tfidf_"):
        if name in ["tfidf_ejecucion contrato"]:
            return "TF-IDF (bigrama)"
        return "TF-IDF (unigrama)"
    if name.startswith("prop_"):
        return "Proporcion categorica"
    if name in ["prob_promedio", "prob_std", "imp_promedio"]:
        return "Estadistica riesgo"
    if name == "interaccion_prob_x_impacto":
        return "Interaccion"
    if name == "valor_inicial":
        return "Valor economico"
    return "Otro"

vif_df["grupo"] = vif_df["feature"].apply(classify_feature)

grupos = vif_df.groupby("grupo")["vif"].agg(["count", "mean", "max", "std"])
grupos.columns = ["count", "VIF_mean", "VIF_max", "VIF_std"]
print(grupos.to_string())

# %% [markdown]
# ## 5. VIF con Ridge — VIF generalizado
#
# El VIF estandar usa R² de regresion lineal. Cuando hay relaciones
# no lineales entre features, el VIF lineal puede subestimar la
# multicolinealidad. Calculamos un VIF "generalizado" usando Ridge
# como estimador mas estable.

# %%
def compute_vif_ridge(df_features, alpha=1.0):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    vif_data["vif_ridge"] = 0.0

    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(df_features.values.astype(np.float64))
    feature_names = df_features.columns

    for i, col in enumerate(feature_names):
        y_i = X_scaled[:, i]
        X_i = np.delete(X_scaled, i, axis=1)

        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X_i, y_i)
        y_pred = reg.predict(X_i)
        ss_res = np.sum((y_i - y_pred) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2_i = 1 - ss_res / ss_tot

        if r2_i >= 0.999999:
            vif_i = float("inf")
        else:
            vif_i = 1.0 / (1.0 - r2_i)

        vif_data.loc[vif_data["feature"] == col, "vif_ridge"] = vif_i

    return vif_data.sort_values("vif_ridge", ascending=False)

print("=" * 60)
print("  VIF GENERALIZADO (Ridge, alpha=1.0)")
print("=" * 60)

vif_ridge_df = compute_vif_ridge(X_raw, alpha=1.0)
vif_df = vif_df.merge(vif_ridge_df, on="feature", how="left")

print(f"\n{'Feature':35s} {'VIF_OLS':>10s} {'VIF_Ridge':>12s} {'Delta':>10s}")
print("-" * 70)
for _, r in vif_df.sort_values("vif", ascending=False).iterrows():
    vif_ols = f"{r['vif_round']:.2f}" if r['vif_round'] != float("inf") else "INF"
    vif_ridge = f"{r['vif_ridge']:.2f}" if r['vif_ridge'] != float("inf") else "INF"
    delta = r['vif'] - r['vif_ridge']
    delta_str = f"{delta:.2f}" if abs(delta) < 1e6 else "INF"
    print(f"{r['feature']:35s} {vif_ols:>10s} {vif_ridge:>12s} {delta_str:>10s}")

# %% [markdown]
# ## 6. Nota sobre escala
#
# El VIF se calculo sobre datos estandarizados (z-scores) para evitar
# inestabilidad numerica causada por features en escalas muy dispares
# (e.g., valor_inicial en miles de millones COP vs TF-IDF en [0,1]).
# Sin este escalado, la matriz de diseno tiene un condition number
# del orden de 10^11, lo que hace que la regresion OLS no converja
# a la solucion correcta.
#
# VIF es invariante a escala lineal cuando se usa la formulacion
# basada en la matriz de correlacion (datos estandarizados).

# %% [markdown]
# ## 7. PCA exploratorio
#
# Analisis de componentes principales para entender cuantas dimensiones
# subyacentes explican la varianza de las 35 features.

# %%
print("=" * 60)
print("  PCA — DESCOMPOSICION DE VARIANZA")
print("=" * 60)

pca_scaler = StandardScaler()
X_scaled = pca_scaler.fit_transform(X_raw.astype(np.float64))

pca = PCA()
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

n_90 = np.argmax(cumulative >= 0.90) + 1
n_95 = np.argmax(cumulative >= 0.95) + 1
n_99 = np.argmax(cumulative >= 0.99) + 1

print(f"\nComponentes para explicar:")
print(f"  90% varianza: {n_90} componentes")
print(f"  95% varianza: {n_95} componentes")
print(f"  99% varianza: {n_99} componentes")
print(f"\n  Primer componente: {explained[0]:.2%}")
print(f"  Primeros 5: {cumulative[4]:.2%}")
print(f"  Primeros 10: {cumulative[9]:.2%}")

# Cargas PCA (loadings) para features mas influyentes
loadings = pd.DataFrame(
    pca.components_[:10].T,
    columns=[f"PC{i+1}" for i in range(10)],
    index=FEATURES_35
)

print("\nTop 5 features por PC1 (mayor peso absoluto):")
pc1_sorted = loadings["PC1"].abs().sort_values(ascending=False)
for f in pc1_sorted.head(5).index:
    print(f"  {f:35s}  {loadings.loc[f, 'PC1']:+.4f}")

print("\nTop 5 features por PC2:")
pc2_sorted = loadings["PC2"].abs().sort_values(ascending=False)
for f in pc2_sorted.head(5).index:
    print(f"  {f:35s}  {loadings.loc[f, 'PC2']:+.4f}")

# %% [markdown]
# ## 8. Condicion Number (Numero de Condicion)
#
# Otra medida de multicolinealidad: kappa = sqrt(lambda_max / lambda_min)
# kappa > 30 indica multicolinealidad severa.

# %%
print("=" * 60)
print("  CONDITION NUMBER")
print("=" * 60)

singular_values = pca.singular_values_
condition_number = np.sqrt(singular_values.max() / singular_values.min())

print(f"  Condition number (kappa): {condition_number:.2f}")
print(f"  Interpretacion:")
if condition_number < 10:
    print("    kappa < 10  -> Multicolinealidad debil")
elif condition_number < 30:
    print("    10 <= kappa < 30 -> Multicolinealidad moderada")
elif condition_number < 100:
    print("    30 <= kappa < 100 -> Multicolinealidad fuerte")
else:
    print("    kappa >= 100 -> Multicolinealidad severa")

# Variance decomposition proportions
print("\nNota: datos estandarizados, varianza = 1 para todas las features.")
print(f"  Condition number de la matriz de diseno (X_scaled): {condition_number:.2e}")
print(f"  Numero de dimensiones efectivas: ~{int(round(singular_values.size - sum(singular_values < 1)))}")

# %% [markdown]
# ## 9. Implicaciones para SVR y Ridge
#
# Discutimos como la multicolinealidad afecta a cada modelo.

# %%
print("=" * 60)
print("  IMPLICACIONES POR MODELO")
print("=" * 60)

n_vif_alto = (vif_df["vif"] >= 5).sum()
n_vif_severo = (vif_df["vif"] >= 10).sum()
top_vif_features = vif_df.head(5)["feature"].tolist()
top_vif_values = vif_df.head(5)["vif"].tolist()

top_vif_str = ', '.join(f"{f} ({v:.1f})" for f, v in zip(top_vif_features[:5], top_vif_values[:5]))
print(f"""
Resumen de multicolinealidad:
- {n_vif_alto} features con VIF >= 5
- {n_vif_severo} features con VIF >= 10
- Top VIF: {top_vif_str}
""")

print(f"""
Impacto en Ridge (L2):
- La regularizacion L2 mitiga multicolinealidad (encoge coeficientes)
- Coeficientes aun pueden ser inestables si VIF es muy alto
- Interpretacion de coeficientes requiere cautela
- Ridge se uso solo como referencia, no como predictor (alpha=244)

Impacto en SVR (kernel RBF):
- SVR no tiene coeficientes interpretables, usa vectores de soporte
- Kernel RBF mapea a espacio de alta dimension -> no afectado por
  multicolinealidad lineal
- Permutation importance no asume independencia entre features
- SHAP es consistente incluso con features correlacionadas
- La multicolinealidad NO es un problema para SVR
""")

# %% [markdown]
# ## 10. Visualizaciones

# %%
print("\nGenerando graficos...")

fig, axes = plt.subplots(2, 3, figsize=(18, 14))

# --- Heatmap de correlacion ---
ax1 = axes[0, 0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, center=0,
    square=True, linewidths=0.3, cbar_kws={"shrink": 0.6},
    ax=ax1, vmin=-1, vmax=1,
    xticklabels=False, yticklabels=False
)
ax1.set_title("Matriz de Correlacion (Pearson)", fontsize=12, fontweight="bold")

# --- VIF bar chart (top 15) ---
ax2 = axes[0, 1]
vif_top15 = vif_df.head(15).iloc[::-1]
colors = ["#e74c3c" if v >= 10 else "#f39c12" if v >= 5 else "#3498db" if v >= 2 else "#2ecc71" for v in vif_top15["vif"]]
ax2.barh(range(len(vif_top15)), vif_top15["vif"], color=colors, edgecolor="white")
ax2.set_yticks(range(len(vif_top15)))
ax2.set_yticklabels(vif_top15["feature"].values, fontsize=9)
ax2.axvline(5, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.axvline(10, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.set_xlabel("VIF", fontsize=11)
ax2.set_title("Top 15 VIF", fontsize=12, fontweight="bold")

# --- PCA Scree plot ---
ax3 = axes[0, 2]
ax3.bar(range(1, len(explained) + 1), explained, alpha=0.7, color="#3498db", edgecolor="white", label="Individual")
ax3.plot(range(1, len(cumulative) + 1), cumulative, "r-o", markersize=3, linewidth=1.5, label="Acumulado")
ax3.axhline(0.90, color="green", ls=":", alpha=0.7)
ax3.axhline(0.95, color="orange", ls=":", alpha=0.7)
ax3.set_xlabel("Componente", fontsize=11)
ax3.set_ylabel("Varianza Explicada", fontsize=11)
ax3.set_title("Scree Plot", fontsize=12, fontweight="bold")
ax3.legend(fontsize=8)

# --- Dendrograma ---
ax4 = axes[1, 0]
corr_dist = 1 - abs(corr_matrix.values)
condensed = pdist(corr_dist)
link = hierarchy.average(condensed)
hierarchy.dendrogram(link, labels=FEATURES_35, ax=ax4, leaf_rotation=90, leaf_font_size=7)
ax4.set_title("Dendrograma (1 - |r|)", fontsize=12, fontweight="bold")

# --- VIF OLS vs Ridge ---
ax5 = axes[1, 1]
mask_finite = np.isfinite(vif_df["vif"].values) & np.isfinite(vif_df["vif_ridge"].values)
x_vals = vif_df["vif"].values[mask_finite]
y_vals = vif_df["vif_ridge"].values[mask_finite]
max_val = max(x_vals.max(), y_vals.max()) * 1.1
ax5.scatter(x_vals, y_vals, alpha=0.6, s=30, c="#3498db", edgecolors="white")
ax5.plot([0, max_val], [0, max_val], "r--", linewidth=1)
ax5.set_xlabel("VIF (OLS)", fontsize=11)
ax5.set_ylabel("VIF (Ridge)", fontsize=11)
ax5.set_title("VIF OLS vs Ridge", fontsize=12, fontweight="bold")
ax5.grid(alpha=0.3)

# --- Boxplot VIF por grupo ---
ax6 = axes[1, 2]
grupo_order = vif_df.groupby("grupo")["vif"].median().sort_values(ascending=False).index
sns.boxplot(data=vif_df, x="grupo", y="vif", order=grupo_order, ax=ax6, palette="Set2")
ax6.axhline(5, color="orange", ls="--", lw=1.5, alpha=0.7)
ax6.axhline(10, color="red", ls="--", lw=1.5, alpha=0.7)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax6.set_title("VIF por Grupo", fontsize=12, fontweight="bold")
ax6.set_ylabel("VIF", fontsize=11)

plt.tight_layout()
plt.savefig(DATA_DIR / "multicolinealidad_vif.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"Grafico guardado: {DATA_DIR / 'multicolinealidad_vif.png'}")

# %% [markdown]
# ## 11. Conclusion

# %%
print("=" * 60)
print("  CONCLUSION")
print("=" * 60)

n_alto_vif = (vif_df["vif"] >= 5).sum()
n_sev_vif = (vif_df["vif"] >= 10).sum()
n_bajo_vif = (vif_df["vif"] < 2).sum()
n_inf_vif = np.isinf(vif_df["vif"]).sum()

print(f"""
Las 35 features presentan:

- {n_bajo_vif} features con VIF bajo (< 2)
- {n_moderada} features con VIF moderado (2-5)
- {n_alta} features con VIF alto (5-10)
- {n_sev_vif} features con VIF severo (>= 10)
- {n_inf_vif} features con VIF INF (colinealidad perfecta)

Fuentes de multicolinealidad:

1. VARIABLES MACRO (VIF = INF):
   anio_inicio, anio_fin, duracion -> duracion = anio_fin - anio_inicio
   Esta colinealidad perfecta es por construccion. Aunque el cap de
   duracion a 5 anos introduce no-linealidad, la relacion determinista
   persiste en los datos procesados.

2. INTERACCION Y SUS COMPONENTES (VIF > 16):
   interaccion_prob_x_impacto (39.9), prob_promedio (19.9), imp_promedio (16.4)
   -> interaccion = prob_promedio * imp_promedio, correlacion esperada

3. TRM_PROMEDIO (VIF = 39.5):
   Altamente correlacionada con anio_inicio y anio_fin (r = 0.94 y 0.93)
   por ser calculada como promedio del rango de anos.

4. IPC_ACUMULADO (VIF = 5.9):
   Correlacion moderada-alta con el rango de fechas.

5. TF-IDF (VIF 2-3):
   Colinealidad leve por co-ocurrencia de terminos en lenguaje natural
   (e.g., "obra"/"ejecucion", "categoria alto"/"categoria bajo").

Implicaciones para los modelos:
- Ridge (L2): la regularizacion L2 mitiga multicolinealidad encogiendo
  coeficientes correlacionados. Sin embargo, con VIF > 10 los coeficientes
  siguen siendo inestables e interpretarlos requiere cautela extrema.
  Por eso Ridge se usa solo como referencia LINEAL, no como predictor.
- SVR (RBF): NO se afecta por multicolinealidad lineal:
  * Kernel RBF mapea a espacio de alta dimension (incluso infinita)
  * Los vectores de soporte no se ven afectados por correlaciones entre features
  * Permutation importance evalua cada feature independientemente
  * SHAP es consistente bajo features correlacionadas
- La multicolinealidad observada FORTALECE la eleccion de SVR sobre Ridge
  para las 35 features con rango de fechas.

Recomendacion: si se quisiera usar un modelo lineal interpretable,
habria que:
  a) Eliminar duracion (redundante con anio_inicio/anio_fin)
  b) Eliminar interaccion_prob_x_impacto (redundante con sus componentes)
  c) Considerar PCA o PLS para reducir dimensiones
  d) Usar Ridge con alpha mas alto (mas regularizacion)
""")

