# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Exposicion Financiera Detectada — Value of Information (VoI)
#
# **Pregunta:** ¿Cual es el valor economico de la informacion que genera
# el modelo SVR? Es decir, ¿cuanto sobrecosto NO detectado por la matriz
# de riesgo tradicional (categoria) es identificado por el modelo?
#
# ## Metodologia
#
# 1. Clasificamos cada contrato segun su categoria de riesgo (BAJO/MEDIO = seguro,
#    ALTO/EXTREMO = riesgo) — esta es la "alerta tradicional"
# 2. Clasificamos cada contrato segun la prediccion SVR (>25% = alerta)
# 3. Comparamos contra la realidad (sobrecosto real > 25%)
# 4. Calculamos el valor total y sobrecosto de los contratos que la matriz
#    tradicional NO detecto, y cuantos de ellos SVR si detecta
#
# Esto cuantifica el **Valor de la Informacion (VoI)** que aporta el modelo
# sobre la evaluacion tradicional de riesgos.

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path.cwd().resolve()))
from estudio_data.features import engineer_features, STOP_WORDS

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path.cwd().resolve()
DATA_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"

UMBRAL_ALERTA = 25.0

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

# %% [markdown]
# ## 1. Carga y exploracion de datos

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
print(f"Matriz: {len(matriz):,} riesgos, {matriz['id_contrato'].nunique()} contratos")

macro = pd.read_csv(DATA_DIR / "contratos_macro.csv")
print(f"Macro: {len(macro)} contratos")

df_info = matriz.groupby("id_contrato").agg({
    "valor_inicial": "first",
    "sobrecosto": "first",
    "categoria": lambda x: x.mode().iloc[0] if not x.mode().empty else "no especificado",
    "valoracion": "mean",
}).reset_index()

df_info["categoria_riesgo"] = df_info["categoria"].str.lower().str.strip()
df_info["tradicional_alerta"] = df_info["categoria_riesgo"].isin(["alto", "extremo"])
df_info["real_sobrecosto_alto"] = df_info["sobrecosto"] > UMBRAL_ALERTA

print(f"\nDistribucion de categoria dominante:")
print(df_info["categoria_riesgo"].value_counts().to_string())

print(f"\nSobrecosto real > {UMBRAL_ALERTA}%:  {df_info['real_sobrecosto_alto'].sum()}/{len(df_info)} contratos")
print(f"Alerta tradicional (Alto/Extremo): {df_info['tradicional_alerta'].sum()}/{len(df_info)} contratos")

# Tabla de contingencia tradicional vs realidad
print("\nMatriz de confusion: Alerta Tradicional vs Realidad")
ct_trad = pd.crosstab(
    df_info["tradicional_alerta"].map({True: "ALERTA", False: "SEGURO"}),
    df_info["real_sobrecosto_alto"].map({True: ">25%", False: "<=25%"}),
    margins=True
)
print(ct_trad.to_string())

# %% [markdown]
# ## 2. Predicciones con SVR (modelo campeon)
#
# Cargamos el modelo entrenado y generamos predicciones para todos
# los contratos del dataset.

# %%
print("=" * 60)
print("  PREDICCIONES CON SVR")
print("=" * 60)

print("\nFeature engineering...")
df_feat = engineer_features(matriz)
df_feat = df_feat.merge(macro, on="id_contrato", how="left")
for c in CONTROL_VARS_RANGO:
    if c not in df_feat.columns:
        fb = compute_range_features(2022, 2022)
        df_feat[c] = fb[c]

df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()

missing = [c for c in FEATURES_35 if c not in df_feat.columns]
if missing:
    raise ValueError(f"Faltan columns: {missing}")

print(f"Dataset final: {df_feat.shape}")

X = df_feat[FEATURES_35].values
y_true = df_feat["sobrecosto"].values
contrato_ids = df_feat["id_contrato"].values

svr = joblib.load(MODELS_DIR / "svr_regressor.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

X_scaled = scaler.transform(X)
y_pred = svr.predict(X_scaled)

pred_df = pd.DataFrame({
    "id_contrato": contrato_ids,
    "sobrecosto_real": y_true,
    "prediccion_svr": y_pred,
})

pred_df["svr_alerta"] = pred_df["prediccion_svr"] > UMBRAL_ALERTA
pred_df["real_alerta"] = pred_df["sobrecosto_real"] > UMBRAL_ALERTA

print(f"\nPredicciones generadas para {len(pred_df)} contratos")
print(f"SVR alerta (> {UMBRAL_ALERTA}%): {pred_df['svr_alerta'].sum()}/{len(pred_df)}")
print(f"Real alerta (> {UMBRAL_ALERTA}%): {pred_df['real_alerta'].sum()}/{len(pred_df)}")

ct_svr = pd.crosstab(
    pred_df["svr_alerta"].map({True: "ALERTA", False: "SEGURO"}),
    pred_df["real_alerta"].map({True: ">25%", False: "<=25%"}),
    margins=True
)
print("\nMatriz de confusion: SVR vs Realidad")
print(ct_svr.to_string())

# %% [markdown]
# ## 3. Merge con datos financieros e informacion de categoria

# %%
df_full = pred_df.merge(
    df_info[["id_contrato", "valor_inicial", "categoria_riesgo", "tradicional_alerta"]],
    on="id_contrato", how="left"
)

print("=" * 60)
print("  DATOS COMBINADOS")
print("=" * 60)
print(f"Contratos: {len(df_full)}")
print(f"Columnas: {list(df_full.columns)}")
print(f"\nPrimeros 5:")
print(df_full.head().to_string())

# %% [markdown]
# ## 4. Analisis de Exposicion Financiera
#
# Identificamos contratos donde:
# - **Grupo A**: Categoria BAJO/MEDIO pero sobrecosto real > 25% (falsos
#   negativos de la matriz tradicional)
# - **Grupo B**: Categoria BAJO/MEDIO pero SVR predice > 25% (SVR los
#   detecta aunque la matriz no)

# %%
# Grupo A: No detectados por matriz tradicional
grupo_a = df_full[
    (~df_full["tradicional_alerta"]) &
    (df_full["real_alerta"])
].copy()

# Grupo B: Detectados por SVR que la matriz tradicional no vio
grupo_b = df_full[
    (~df_full["tradicional_alerta"]) &
    (df_full["svr_alerta"]) &
    (df_full["real_alerta"])
].copy()

# Grupo C: SVR alerta donde tradicional dijo seguro (todos, incluyendo falsos positivos)
grupo_c = df_full[
    (~df_full["tradicional_alerta"]) &
    (df_full["svr_alerta"])
].copy()

print("=" * 70)
print("  EXPOSICION FINANCIERA — RESULTADOS")
print("=" * 70)

def fmt_cop(val):
    if abs(val) >= 1e12:
        return f"{val/1e12:.2f} billones COP"
    elif abs(val) >= 1e9:
        return f"{val/1e9:.2f} mil millones COP"
    elif abs(val) >= 1e6:
        return f"{val/1e6:.0f} millones COP"
    else:
        return f"{val:,.0f} COP"

def analizar_grupo(df_grupo, nombre, descripcion):
    n = len(df_grupo)
    if n == 0:
        print(f"\n{nombre}: 0 contratos")
        return {
            "nombre": nombre, "n": 0,
            "valor_total": 0, "sobrecosto_total_cop": 0,
            "sobrecosto_promedio": 0, "exposicion_pct": 0,
            "valor_promedio": 0,
        }

    valor_total = df_grupo["valor_inicial"].sum()
    sobrecosto_total_cop = (df_grupo["valor_inicial"] * df_grupo["sobrecosto_real"] / 100).sum()
    sobrecosto_promedio = df_grupo["sobrecosto_real"].mean()
    valor_promedio = df_grupo["valor_inicial"].mean()
    exposicion_pct = (sobrecosto_total_cop / valor_total) * 100 if valor_total > 0 else 0

    print(f"\n{nombre}: {n} contratos")
    print(f"  {descripcion}")
    print(f"  Valor total:        {fmt_cop(valor_total)}")
    print(f"  Valor promedio:     {fmt_cop(valor_promedio)}")
    print(f"  Sobrecosto real total: {fmt_cop(sobrecosto_total_cop)}")
    print(f"  Sobrecosto promedio:   {sobrecosto_promedio:.2f}%")
    print(f"  Exposicion (% del valor): {exposicion_pct:.1f}%")

    return {
        "nombre": nombre, "n": n,
        "valor_total": valor_total,
        "sobrecosto_total_cop": sobrecosto_total_cop,
        "sobrecosto_promedio": sobrecosto_promedio,
        "exposicion_pct": exposicion_pct,
        "valor_promedio": valor_promedio,
    }

res_a = analizar_grupo(grupo_a,
    "Grupo A - NO detectados por matriz tradicional",
    "Categoria Bajo/Medio pero sobrecosto real > 25%")

res_b = analizar_grupo(grupo_b,
    "Grupo B - SVR los detecta (VoI)",
    "Categoria Bajo/Medio, sobrecosto real > 25%, SVR los alerta")

res_c = analizar_grupo(grupo_c,
    "Grupo C - Todos los que SVR alerta (incluyendo FP)",
    "Categoria Bajo/Medio pero SVR predice > 25%")

# Valor total del portafolio
valor_portafolio = df_full["valor_inicial"].sum()
print(f"\n---")
print(f"Valor total del portafolio ({len(df_full)} contratos): {fmt_cop(valor_portafolio)}")

# %% [markdown]
# ## 5. Value of Information (VoI) — Metricas clave
#
# Calculamos el VoI como la proporcion del sobrecosto NO detectado por
# la matriz tradicional que SVR SI logra identificar.

# %%
print("=" * 70)
print("  VALUE OF INFORMATION (VoI) — SVR vs Matriz Tradicional")
print("=" * 70)

if res_a["n"] > 0:
    # Cobertura de SVR sobre el grupo no detectado
    cobertura_svr = res_b["n"] / res_a["n"] * 100

    # Sobrecosto recuperado por SVR (evitable con alerta temprana)
    sobrecosto_recuperado = res_b["sobrecosto_total_cop"]
    sobrecosto_no_detectado = res_a["sobrecosto_total_cop"]
    pct_recuperado = (sobrecosto_recuperado / sobrecosto_no_detectado * 100) if sobrecosto_no_detectado > 0 else 0

    print(f"\nContratos con sobrecosto real > {UMBRAL_ALERTA}% no alertados por matriz tradicional:")
    print(f"  Total: {res_a['n']} contratos")
    print(f"  Valor total: {fmt_cop(res_a['valor_total'])}")
    print(f"  Sobrecosto total no detectado: {fmt_cop(sobrecosto_no_detectado)}")

    print(f"\nDe ellos, SVR alerta correctamente:")
    print(f"  {res_b['n']}/{res_a['n']} contratos ({cobertura_svr:.0f}%)")
    print(f"  Sobrecosto asociado: {fmt_cop(sobrecosto_recuperado)}")
    print(f"  Porcentaje del sobrecosto no detectado que SVR recupera: {pct_recuperado:.0f}%")

    # Eficiencia de SVR (precision en este subgrupo)
    svr_aciertos = len(grupo_b)
    svr_total_alertas = len(grupo_c)
    precision_svr = (svr_aciertos / svr_total_alertas * 100) if svr_total_alertas > 0 else 0
    print(f"\nEficiencia de SVR en contratos de bajo riesgo:")
    print(f"  Alertas SVR en categoria Bajo/Medio: {svr_total_alertas}")
    print(f"  Correctas (TP): {svr_aciertos}")
    print(f"  Falsas alarmas (FP): {svr_total_alertas - svr_aciertos}")
    print(f"  Precision: {precision_svr:.0f}%")

print(f"\n{'='*70}")
print(f"  VoI: SVR detecta {cobertura_svr:.0f}% de los contratos con sobrecosto")
print(f"       que la matriz tradicional clasifica como BAJO/MEDIO.")
print(f"       Esto representa {fmt_cop(sobrecosto_recuperado)} en sobrecosto real")
print(f"       sobre {fmt_cop(res_a['valor_total'])} en valor de contratos.")
print(f"{'='*70}")

# %% [markdown]
# ## 6. Desglose detallado de contratos inconsistentes
#
# Mostramos los contratos donde la categoria tradicional y la realidad
# NO coinciden. Estos son los casos donde el modelo agrega valor.

# %%
print("=" * 70)
print("  DESGLOSE DE CONTRATOS INCONSISTENTES")
print("=" * 70)

inconsistentes = df_full[
    (~df_full["tradicional_alerta"]) &
    (df_full["real_alerta"])
].sort_values("sobrecosto_real", ascending=False)

print(f"\nContratos con categoria BAJO/MEDIO pero sobrecosto real > {UMBRAL_ALERTA}%:")
print(f"  Total: {len(inconsistentes)} contratos")
print(f"\n{'ID':<12} {'Categoria':<12} {'Valor Inicial':<20} {'Sobrecosto':<12} {'Pred SVR':<12} {'Alerta':<8}")
print("-" * 80)
for _, r in inconsistentes.head(20).iterrows():
    alerta_svr = "***" if r["svr_alerta"] else ""
    print(f"{r['id_contrato']:<12} {r['categoria_riesgo']:<12} {fmt_cop(r['valor_inicial']):<20} {r['sobrecosto_real']:>6.2f}%  {r['prediccion_svr']:>6.2f}%  {alerta_svr:<8}")

if len(inconsistentes) > 20:
    print(f"... y {len(inconsistentes) - 20} mas")

# %% [markdown]
# ## 7. Comparacion completa de sistemas de alerta
#
# Comparamos tres sistemas de alerta lado a lado:
# 1) Matriz tradicional (categoria)
# 2) SVR
# 3) Combinado (categoria OR SVR)

# %%
print("=" * 70)
print("  COMPARACION DE SISTEMAS DE ALERTA")
print("=" * 70)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

y_true_bin = df_full["real_alerta"].astype(int)
y_trad = df_full["tradicional_alerta"].astype(int)
y_svr = df_full["svr_alerta"].astype(int)
y_combined = (y_trad | y_svr).astype(int)

sistemas = {
    "Matriz Tradicional": y_trad,
    "SVR": y_svr,
    "Combinado (Trad OR SVR)": y_combined,
}

resultados_sistemas = []
for nombre, y_pred in sistemas.items():
    resultados_sistemas.append({
        "Sistema": nombre,
        "Accuracy": accuracy_score(y_true_bin, y_pred),
        "Precision": precision_score(y_true_bin, y_pred, zero_division=0),
        "Recall": recall_score(y_true_bin, y_pred, zero_division=0),
        "F1-score": f1_score(y_true_bin, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true_bin, y_pred),
    })

df_sistemas = pd.DataFrame(resultados_sistemas)
print(f"\n{df_sistemas.to_string(index=False)}")

# %% [markdown]
# ## 8. Analisis por umbrales de alerta
#
# Evaluamos como cambia el VoI si ajustamos el umbral de alerta del SVR.

# %%
print("=" * 70)
print("  ANALISIS DE SENSIBILIDAD — UMBRAL DE ALERTA SVR")
print("=" * 70)

umbrales = np.arange(10, 51, 5)
sensibilidad = []

for umbral in umbrales:
    alerta = df_full["prediccion_svr"] > umbral
    n_alertas = alerta.sum()
    tp = (alerta & df_full["real_alerta"]).sum()
    fp = (alerta & ~df_full["real_alerta"]).sum()
    fn = (~alerta & df_full["real_alerta"]).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    valor_alerta = df_full.loc[alerta, "valor_inicial"].sum()
    sobrecosto_detectado = (df_full.loc[alerta, "valor_inicial"] * df_full.loc[alerta, "sobrecosto_real"] / 100).sum()

    sensibilidad.append({
        "Umbral (%)": umbral,
        "Alertas": n_alertas,
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        "Valor Alertado": valor_alerta,
        "Sobrecosto Detectado": sobrecosto_detectado,
    })

df_sens = pd.DataFrame(sensibilidad)
print(f"\n{df_sens[['Umbral (%)', 'Alertas', 'TP', 'FP', 'Precision', 'Recall', 'F1']].to_string(index=False)}")

# %% [markdown]
# ## 9. Visualizaciones

# %%
print("Generando visualizaciones...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- 1. Confusion matrix: Tradicional vs Realidad ---
ax1 = axes[0, 0]
ct_trad = pd.crosstab(
    df_full["tradicional_alerta"].map({True: "Alerta", False: "Seguro"}),
    df_full["real_alerta"].map({True: ">25%", False: "<=25%"}),
)
sns.heatmap(ct_trad, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=False)
ax1.set_title("Matriz Tradicional (categoria)\nvs Realidad", fontsize=12, fontweight="bold")
ax1.set_ylabel("Alerta Tradicional")
ax1.set_xlabel("Sobrecosto Real")

# --- 2. Confusion matrix: SVR vs Realidad ---
ax2 = axes[0, 1]
ct_svr = pd.crosstab(
    df_full["svr_alerta"].map({True: "Alerta", False: "Seguro"}),
    df_full["real_alerta"].map({True: ">25%", False: "<=25%"}),
)
sns.heatmap(ct_svr, annot=True, fmt="d", cmap="Greens", ax=ax2, cbar=False)
ax2.set_title("SVR\nvs Realidad", fontsize=12, fontweight="bold")
ax2.set_ylabel("Alerta SVR")
ax2.set_xlabel("Sobrecosto Real")

# --- 3. Comparacion de sistemas ---
ax3 = axes[0, 2]
df_plot = df_sistemas.melt(id_vars=["Sistema"], var_name="Metrica", value_name="Valor")
sns.barplot(data=df_plot, x="Metrica", y="Valor", hue="Sistema", ax=ax3)
ax3.set_title("Comparacion de Sistemas de Alerta", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 1)
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)

# --- 4. Sobrecosto real vs predicho ---
ax4 = axes[1, 0]
colors = ["#e74c3c" if r["real_alerta"] else "#2ecc71" for _, r in df_full.iterrows()]
ax4.scatter(df_full["prediccion_svr"], df_full["sobrecosto_real"],
            c=colors, alpha=0.6, s=20, edgecolors="white")
ax4.axhline(UMBRAL_ALERTA, color="red", linestyle="--", linewidth=1, alpha=0.5, label=f"Umbral {UMBRAL_ALERTA}%")
ax4.axvline(UMBRAL_ALERTA, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax4.plot([0, 120], [0, 120], "gray", linestyle=":", linewidth=1, alpha=0.5, label="Ideal")
ax4.set_xlabel("Prediccion SVR (%)", fontsize=11)
ax4.set_ylabel("Sobrecosto Real (%)", fontsize=11)
ax4.set_title("Sobrecosto Real vs Predicho", fontsize=12, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# --- 5. Sensibilidad de umbral ---
ax5 = axes[1, 1]
ax5.plot(df_sens["Umbral (%)"], df_sens["Precision"], "b-o", label="Precision", markersize=4)
ax5.plot(df_sens["Umbral (%)"], df_sens["Recall"], "r-o", label="Recall", markersize=4)
ax5.plot(df_sens["Umbral (%)"], df_sens["F1"], "g-o", label="F1-score", markersize=4)
ax5.axvline(UMBRAL_ALERTA, color="gray", linestyle=":", alpha=0.7, label=f"Umbral actual ({UMBRAL_ALERTA}%)")
ax5.set_xlabel("Umbral de alerta SVR (%)", fontsize=11)
ax5.set_ylabel("Metrica", fontsize=11)
ax5.set_title("Analisis de Sensibilidad", fontsize=12, fontweight="bold")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# --- 6. Valor financiero por grupo ---
ax6 = axes[1, 2]
grupos_barras = ["Portafolio\nTotal", "No detectados\npor Tradicional", "SVR los\ndetecta (VoI)", "SVR alerta\n(Todos)"]
valores = [
    valor_portafolio,
    res_a["valor_total"],
    res_b["valor_total"],
    res_c["valor_total"],
]
colores = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
bars = ax6.bar(range(len(grupos_barras)), valores, color=colores, edgecolor="white")
ax6.set_xticks(range(len(grupos_barras)))
ax6.set_xticklabels(grupos_barras, fontsize=8)
ax6.set_title("Valor de Contratos por Grupo", fontsize=12, fontweight="bold")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e9:.0f}B"))
ax6.grid(axis="y", alpha=0.3)

# Anotar valores en las barras
for bar, val in zip(bars, valores):
    if val > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 fmt_cop(val), ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(DATA_DIR / "exposicion_financiera_voi.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafico guardado: {DATA_DIR / 'exposicion_financiera_voi.png'}")

# %% [markdown]
# ## 10. Conclusion

# %%
print("=" * 70)
print("  CONCLUSION — Value of Information (VoI)")
print("=" * 70)

n_falsos_negativos_trad = len(inconsistentes)
n_detectados_por_svr = len(grupo_b)
pct_recuperado_final = (n_detectados_por_svr / n_falsos_negativos_trad * 100) if n_falsos_negativos_trad > 0 else 0

print(f"""
SINTESIS:

La matriz de riesgo tradicional (categoria BAJO/MEDIO = seguro,
ALTO/EXTREMO = riesgo) NO logra detectar {n_falsos_negativos_trad} contratos
que terminaron con sobrecosto real superior a {UMBRAL_ALERTA}%.

El modelo SVR, aplicado sobre las mismas matrices de riesgo,
logra identificar {n_detectados_por_svr} de esos {n_falsos_negativos_trad} contratos
({pct_recuperado_final:.0f}% de cobertura).

Esto representa un Valor de la Informacion (VoI) medible:
- Contratos no detectados por metodo tradicional:
  * Cantidad: {res_a['n']}
  * Valor total: {fmt_cop(res_a['valor_total'])}
  * Sobrecosto real acumulado: {fmt_cop(res_a['sobrecosto_total_cop'])}

- De ellos, detectados por SVR:
  * Cantidad: {res_b['n']}
  * Valor total: {fmt_cop(res_b['valor_total'])}
  * Sobrecosto asociado: {fmt_cop(res_b['sobrecosto_total_cop'])}

- Eficiencia del SVR en categoria Bajo/Medio:
  * Precision: {precision_svr:.0f}%
  * Recall: {res_b['n']}/{res_a['n']} ({pct_recuperado_final:.0f}%)

NOTA: Estas cifras representan la exposicion financiera detectada,
no el ahorro real (que requeriria datos de mitigacion). Sin embargo,
demuestran que el modelo SVR captura informacion NO contenida en la
categoria de riesgo tradicional, agregando valor diagnostico.
""")

