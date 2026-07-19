# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Análisis de Representatividad de la Muestra (n=351)
# ## Comparación sistemática contra la población de contratos SECOP I
#
# **Objetivo**: Demostrar que la muestra final de 351 contratos con matrices de riesgo
# no está sesgada sistemáticamente respecto a la población de contratos de obra pública
# en Colombia, o reconocer explícitamente las limitaciones.
#
# Se comparan 4 dimensiones:
# 1. **Distribución geográfica** (departamento)
# 2. **Cuantía** (valor inicial del contrato en rangos)
# 3. **Distribución temporal** (año de inicio)
# 4. **Sobrecosto** (variable target)

# %%
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5), "font.size": 11})

ROOT = Path.cwd().resolve()

# %% [markdown]
# ## 1. Carga de datos

# %%
# --- POBLACIÓN: proyectos depurados de SECOP I (pool completo con sobrecosto) ---
dep = pd.read_csv(ROOT / "contratos" / "proyectos_depurados.csv", encoding="utf-8-sig")
print(f"Población SECOP I (depurados): {len(dep):,} registros")

# Para la población, usamos cada registro como un contrato único (sin duplicados por constancia)
dep_unique = dep.drop_duplicates(subset=["url"]).copy()
print(f"  Contratos únicos: {len(dep_unique):,}")

# --- MUESTRA: 351 contratos con matrices de riesgo ---
mc = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
print(f"Muestra (matriz_clean): {mc['id_contrato'].nunique()} contratos, {len(mc):,} riesgos")

# Extraer constancia de la URL para hacer match
def extract_constancia(url):
    if pd.isna(url):
        return None
    s = str(url)
    if "numConstancia=" in s:
        return s.split("numConstancia=")[1].split("&")[0].strip()
    return s.strip()

mc["constancia"] = mc["url"].apply(extract_constancia)
dep_unique["constancia"] = dep_unique["url"].apply(extract_constancia)

# Obtener sobrecosto por contrato
sc_map = mc.groupby("id_contrato")["sobrecosto"].first().to_dict()

# Merge muestra con datos de poblacion (un row por contrato)
muestra = mc[["constancia", "id_contrato"]].drop_duplicates(subset="id_contrato").copy()
muestra["sobrecosto_pct"] = muestra["id_contrato"].map(sc_map)

# Renombrar para evitar conflicto con columna de dep_unique
dep_unique_join = dep_unique.drop(columns=["sobrecosto_pct"], errors="ignore")

matched = muestra.merge(dep_unique_join, on="constancia", how="left")
match_count = matched["entidad"].notna().sum()
print(f"  Contratos de muestra con match en poblacion: {match_count} / {len(muestra)}")
print(f"  Sin match: {len(muestra) - match_count}")

# Dataset de muestra enriquecido (un row por contrato)
muestra = matched.copy()

print(f"\nMuestra final para análisis: {len(muestra)} contratos")

# %% [markdown]
# ## 2. Distribución Geográfica (Departamento)

# %%
print("=" * 65)
print("  DISTRIBUCIÓN POR DEPARTAMENTO")
print("=" * 65)

# Agrupar departamentos pequeños en "Otros" para visualización
TOP_N = 12

pob_dept = dep_unique["departamento"].value_counts()
pob_top = pob_dept.head(TOP_N)
pob_otros = pob_dept.iloc[TOP_N:].sum()

mue_dept = muestra["departamento"].value_counts()
mue_top = mue_dept.head(TOP_N)
mue_otros = mue_dept.iloc[TOP_N:].sum()

# Tabla comparativa
pob_n_list = pob_top.tolist() + [pob_otros]
pob_n_arr = np.array(pob_n_list, dtype=float)
dept_comparison = pd.DataFrame({
    "Departamento": pob_top.index.tolist() + ["Otros"],
    "Poblacion_n": pob_n_list,
    "Poblacion_%": pob_n_arr / len(dep_unique) * 100,
})
dept_comparison["Muestra_n"] = dept_comparison["Departamento"].map(
    lambda d: mue_top.get(d) if d != "Otros" else mue_otros
).fillna(0).astype(int)
dept_comparison["Muestra_%"] = dept_comparison["Muestra_n"] / len(muestra) * 100
dept_comparison["Diferencia_pp"] = (dept_comparison["Muestra_%"] - dept_comparison["Poblacion_%"]).round(1)

print(dept_comparison.to_string(index=False))
print(f"\nCorrelación entre % población y % muestra: {dept_comparison['Poblacion_%'].corr(dept_comparison['Muestra_%']):.3f}")

# --- Prueba Chi-cuadrado ---
# Normalizar departamentos a mismos grupos
dep_unique["departamento"] = dep_unique["departamento"].astype(str).fillna("No especificado")
muestra["departamento"] = muestra["departamento"].astype(str).fillna("No especificado")
all_depts = sorted(set(dep_unique["departamento"].unique()) | set(muestra["departamento"].unique()))
obs_pob = np.array([(dep_unique["departamento"] == d).sum() for d in all_depts])
obs_mue = np.array([(muestra["departamento"] == d).sum() for d in all_depts])
# Para chi-cuadrado, ajustamos frecuencias esperadas de la muestra basadas en población
expected = obs_pob / obs_pob.sum() * obs_mue.sum()
mask = expected > 5  # Solo categorías con frecuencia esperada > 5
chi2, p_val, _, _ = chi2_contingency(pd.crosstab(
    index=["Poblacion"] * len(dep_unique) + ["Muestra"] * len(muestra),
    columns=pd.concat([
        dep_unique["departamento"],
        muestra["departamento"]
    ]),
))
# Mejor: agrupar en top-N + Otros para chi-cuadrado
def group_dept(d, top_set):
    return d if d in top_set else "Otros"

top10 = set(pob_dept.head(10).index)
pob_grouped = dep_unique["departamento"].apply(lambda d: group_dept(d, top10))
mue_grouped = muestra["departamento"].apply(lambda d: group_dept(d, top10))

obs = pd.crosstab(
    index=["Poblacion"] * len(pob_grouped) + ["Muestra"] * len(mue_grouped),
    columns=pd.concat([pob_grouped, mue_grouped])
)
chi2, p_val, dof, expected = chi2_contingency(obs)
print(f"\nChi-cuadrado (departamentos agrupados, top-10 + Otros):")
print(f"  Chi2 = {chi2:.2f}, p-valor = {p_val:.4f}, gl = {dof}")
if p_val >= 0.05:
    print("  [OK] No hay evidencia de diferencia significativa (p>=0.05)")
else:
    print("  [ATENCION] Diferencia significativa (p<0.05)")

# --- Visualización ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
top12_depts = pob_dept.head(12).index.tolist()
pob_vals = pob_dept.head(12).values
mue_vals = [mue_dept.get(d, 0) for d in top12_depts]
mue_vals_rest = len(muestra) - sum(mue_vals)

x = np.arange(len(top12_depts))
width = 0.35
bars1 = ax.bar(x - width/2, pob_vals / len(dep_unique) * 100, width, label="Población", color="#4facfe", alpha=0.85)
bars2 = ax.bar(x + width/2, mue_vals + [mue_vals_rest] if len(mue_vals) < len(top12_depts) else mue_vals, width, label="Muestra (n=351)", color="#7B5CE4", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(top12_depts, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("% del total")
ax.set_title("Distribución por Departamento", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
comparison = dept_comparison.sort_values("Diferencia_pp", ascending=False)
colors = ["#EF4444" if v > 0 else "#1ABC9C" for v in comparison["Diferencia_pp"]]
ax.barh(comparison["Departamento"], comparison["Diferencia_pp"], color=colors, alpha=0.8)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Diferencia (pp): Muestra - Población")
ax.set_title("Sesgo por Departamento", fontweight="bold")

plt.tight_layout()
plt.savefig(ROOT / "docs" / "representatividad_departamento.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGrafico guardado: docs/representatividad_departamento.png")

# %% [markdown]
# ## 3. Distribución por Cuantía (Valor Inicial)

# %%
print("=" * 65)
print("  DISTRIBUCIÓN POR CUANTÍA (VALOR INICIAL)")
print("=" * 65)

# Crear rangos de cuantía en COP (escala logarítmica natural)
billon = 1_000_000_000
ranges = [0, 500e6, 1e9, 2e9, 5e9, 10e9, 20e9, 50e9, 100e9, 1e12]
labels_r = ["<500M", "500M–1B", "1B–2B", "2B–5B", "5B–10B", "10B–20B", "20B–50B", "50B–100B", ">100B"]

pob_val = dep_unique["valor_inicial"].dropna()
mue_val = muestra["valor_inicial"].dropna()

pob_bins = pd.cut(pob_val, bins=ranges, labels=labels_r)
mue_bins = pd.cut(mue_val, bins=ranges, labels=labels_r)

pob_dist = pob_bins.value_counts().sort_index()
mue_dist = mue_bins.value_counts().sort_index()

cuantia_df = pd.DataFrame({
    "Rango": labels_r,
    "Poblacion_n": [pob_dist.get(l, 0) for l in labels_r],
    "Poblacion_%": [pob_dist.get(l, 0) / len(pob_val) * 100 for l in labels_r],
    "Muestra_n": [mue_dist.get(l, 0) for l in labels_r],
    "Muestra_%": [mue_dist.get(l, 0) / len(mue_val) * 100 for l in labels_r],
})
cuantia_df["Diferencia_pp"] = (cuantia_df["Muestra_%"] - cuantia_df["Poblacion_%"]).round(1)
print(cuantia_df.to_string(index=False))

# Estadísticas descriptivas
print(f"\nPoblación:  media=${pob_val.mean()/1e9:.1f}B, mediana=${pob_val.median()/1e9:.1f}B")
print(f"Muestra:    media=${mue_val.mean()/1e9:.1f}B, mediana=${mue_val.median()/1e9:.1f}B")

# Kolmogorov-Smirnov test
ks_stat, ks_p = ks_2samp(pob_val / 1e9, mue_val / 1e9)
print(f"\nKolmogorov-Smirnov test (valor inicial en miles de millones):")
print(f"  KS = {ks_stat:.4f}, p-valor = {ks_p:.4f}")
if ks_p >= 0.05:
    print("  [OK] No hay evidencia de diferencia significativa (p>=0.05)")
else:
    print("  [ATENCION] Diferencia significativa (p<0.05)")

# --- Visualización ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
x = np.arange(len(labels_r))
width = 0.35
ax.bar(x - width/2, cuantia_df["Poblacion_%"], width, label="Población", color="#4facfe", alpha=0.85)
ax.bar(x + width/2, cuantia_df["Muestra_%"], width, label="Muestra (n=351)", color="#7B5CE4", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_r, rotation=45, ha="right")
ax.set_ylabel("% del total")
ax.set_title("Distribución por Rango de Cuantía", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(pob_val / 1e9, bins=50, alpha=0.5, label="Población", color="#4facfe", density=True)
ax.hist(mue_val / 1e9, bins=20, alpha=0.6, label="Muestra", color="#7B5CE4", density=True)
ax.set_xlabel("Valor inicial (miles de millones COP)")
ax.set_ylabel("Densidad")
ax.set_title("Distribución de Valor Inicial", fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(ROOT / "docs" / "representatividad_cuantia.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGráfico guardado: docs/representatividad_cuantia.png")

# %% [markdown]
# ## 4. Distribución Temporal (Año de Inicio)

# %%
print("=" * 65)
print("  DISTRIBUCIÓN TEMPORAL")
print("=" * 65)

# Año de inicio para población
pob_fechas = pd.to_datetime(dep_unique["fecha_inicio"], errors="coerce")
pob_anio = pob_fechas.dt.year.dropna().astype(int)

# Año de inicio para muestra (del macro)
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
mue_anio = macro["anio_inicio"]

pob_anio_dist = pob_anio.value_counts().sort_index()
mue_anio_dist = mue_anio.value_counts().sort_index()

all_years = sorted(set(pob_anio_dist.index) | set(mue_anio_dist.index))
anio_df = pd.DataFrame({
    "Año": all_years,
    "Poblacion_n": [pob_anio_dist.get(y, 0) for y in all_years],
    "Poblacion_%": [pob_anio_dist.get(y, 0) / len(pob_anio) * 100 for y in all_years],
    "Muestra_n": [mue_anio_dist.get(y, 0) for y in all_years],
    "Muestra_%": [mue_anio_dist.get(y, 0) / len(mue_anio) * 100 for y in all_years],
})
anio_df["Diferencia_pp"] = (anio_df["Muestra_%"] - anio_df["Poblacion_%"]).round(1)
print(anio_df.to_string(index=False))

# KS test
ks_stat_a, ks_p_a = ks_2samp(pob_anio.values, mue_anio.values)
print(f"\nKolmogorov-Smirnov test (año):")
print(f"  KS = {ks_stat_a:.4f}, p-valor = {ks_p_a:.4f}")
print(f"  {'[OK]  No hay evidencia de diferencia significativa (p≥0.05)' if ks_p_a >= 0.05 else '[!]  Diferencia significativa (p<0.05)'}")

# Visualización
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(all_years))
width = 0.35
ax.bar(x - width/2, anio_df["Poblacion_%"], width, label="Población", color="#4facfe", alpha=0.85)
ax.bar(x + width/2, anio_df["Muestra_%"], width, label="Muestra (n=351)", color="#7B5CE4", alpha=0.85)
for i, row in anio_df.iterrows():
    if row["Muestra_%"] > 0:
        ax.annotate(f'{row["Muestra_n"]}', (i + width/2, row["Muestra_%"] + 0.3),
                   ha="center", fontsize=8, color="#7B5CE4", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(all_years)
ax.set_ylabel("% del total")
ax.set_title("Distribución por Año de Inicio", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(ROOT / "docs" / "representatividad_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGráfico guardado: docs/representatividad_temporal.png")

# %% [markdown]
# ## 5. Distribución del Sobrecosto

# %%
print("=" * 65)
print("  DISTRIBUCIÓN DEL SOBRECOSTO")
print("=" * 65)

pob_sc = dep_unique["sobrecosto_pct"].dropna()
mue_sc = muestra["sobrecosto_pct"].dropna()  # ya en %

print(f"Población: mean={pob_sc.mean():.1f}%, median={pob_sc.median():.1f}%, std={pob_sc.std():.1f}%")
print(f"Muestra:   mean={mue_sc.mean():.1f}%, median={mue_sc.median():.1f}%, std={mue_sc.std():.1f}%")
print(f"(Nota: la muestra sobreselecciona contratos con sobrecosto >0 por diseño experimental)")

# Proporción de alto riesgo (>25%) en cada grupo
pob_alto = (pob_sc > 25).mean() * 100
mue_alto = (mue_sc > 25).mean() * 100
print(f"\nProporción alto riesgo (>25%):")
print(f"  Población: {pob_alto:.1f}%")
print(f"  Muestra:   {mue_alto:.1f}%")

# KS test
ks_stat_sc, ks_p_sc = ks_2samp(pob_sc, mue_sc)
print(f"\nKolmogorov-Smirnov (sobrecosto): KS={ks_stat_sc:.4f}, p={ks_p_sc:.4f}")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(pob_sc.clip(0, 200), bins=40, alpha=0.5, label="Población", color="#4facfe", density=True)
ax.hist(mue_sc.clip(0, 200), bins=20, alpha=0.6, label="Muestra", color="#7B5CE4", density=True)
ax.set_xlabel("Sobrecosto (%)")
ax.set_ylabel("Densidad")
ax.set_title("Distribución del Sobrecosto", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
# Proporción de contratos con sobrecosto > umbral
thresholds = range(0, 101, 10)
pob_props = [(pob_sc > t).mean() * 100 for t in thresholds]
mue_props = [(mue_sc > t).mean() * 100 for t in thresholds]
ax.plot(thresholds, pob_props, "o-", label="Población", color="#4facfe", linewidth=2)
ax.plot(thresholds, mue_props, "s-", label="Muestra", color="#7B5CE4", linewidth=2)
ax.set_xlabel("Umbral de sobrecosto (%)")
ax.set_ylabel("% de contratos que exceden el umbral")
ax.set_title("Curva de Excedencia", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / "docs" / "representatividad_sobrecosto.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGráfico guardado: docs/representatividad_sobrecosto.png")

# %% [markdown]
# ## 6. Distribución por Entidad (Tipo de Organismo)

# %%
print("=" * 65)
print("  DISTRIBUCIÓN POR ENTIDAD CONTRATANTE")
print("=" * 65)

pob_ent = dep_unique["entidad"].value_counts()
mue_ent = muestra["entidad"].value_counts()

print(f"Entidades únicas en población: {len(pob_ent)}")
print(f"Entidades únicas en muestra: {len(mue_ent)}")

# Top 10 entidades en población y su presencia en muestra
top10_ent = pob_ent.head(10).index
print(f"\nTop 10 entidades (población) vs muestra:")
for e in top10_ent:
    pn = pob_ent.get(e, 0)
    mn = mue_ent.get(e, 0)
    print(f"  {e[:55]:55s}  pob:{pn:4d}  muestra:{mn:3d}  ({mn/max(pn,1)*100:.0f}%)")

# %% [markdown]
# ## 7. Resumen y Conclusiones

# %%
print("=" * 65)
print("  RESUMEN DE PRUEBAS ESTADÍSTICAS")
print("=" * 65)

resultados = []

# Departamento (Chi-cuadrado)
chi2, p_chi2, _, _ = chi2_contingency(pd.crosstab(
    index=["Pob"] * len(pob_grouped) + ["Mue"] * len(mue_grouped),
    columns=pd.concat([pob_grouped, mue_grouped])
))
resultados.append(("Departamento (Chi2)", f"Chi2={chi2:.2f}, p={p_chi2:.4f}",
                   "No sig." if p_chi2 >= 0.05 else "Significativo"))

# Cuantía (KS)
ks_c, p_c = ks_2samp(pob_val / 1e9, mue_val / 1e9)
resultados.append(("Valor inicial (KS)", f"D={ks_c:.4f}, p={p_c:.4f}",
                   "No significativo" if p_c >= 0.05 else "Significativo"))

# Año (KS)
ks_a, p_a = ks_2samp(pob_anio.values.astype(float), mue_anio.values.astype(float))
resultados.append(("Año inicio (KS)", f"D={ks_a:.4f}, p={p_a:.4f}",
                   "No significativo" if p_a >= 0.05 else "Significativo"))

# Sobrecosto (KS) - se espera sesgo porque la muestra sobremuestrea sobrecosto > 0
# pero eso es por diseño
ks_sc, p_sc = ks_2samp(pob_sc, mue_sc)
resultados.append(("Sobrecosto (KS)", f"D={ks_sc:.4f}, p={p_sc:.4f}",
                   "[WARN]  Diferencia esperada (sesgo de selección: solo contratos con matriz de riesgo disponible)"))

res_df = pd.DataFrame(resultados, columns=["Dimensión", "Estadístico", "Interpretación"])
print(res_df.to_string(index=False))

print("\n" + "=" * 65)
print("  CONCLUSIONES")
print("=" * 65)
print("""
1. Distribucion geografica: La muestra cubre menos departamentos que la
   poblacion y esta concentrada en Antioquia (27% vs 11%). Bogota esta
   subrepresentada (0% vs 21% sumando Bogota D.C. y Distrito Capital).
   Esto se debe a que la disponibilidad de matrices de riesgo en PDF
   depende de la entidad contratante y el departamento.

2. Cuantia: La muestra esta sesgada hacia contratos de mayor valor.
   Mediana pob: $1.1B vs muestra: $3.8B. Esto es esperado porque los
   contratos mas grandes tienden a tener matrices de riesgo mas formales
   y documentadas. El KS test confirma diferencia significativa (p<0.001).

3. Temporal: La muestra se concentra en 2018-2020 (64% del total), mientras
   que la poblacion tiene distribucion mas uniforme entre 2018-2023.
   Esto refleja que las matrices historicas son mas accesibles que las
   de anos recientes (2023+).

4. Sobrecosto: La muestra tiene sobrecosto promedio mas alto que la
   poblacion (media 28% vs 12%). Esto es por diseno: el estudio
   selecciono contratos con sobrecosto > 0 y con matrices de riesgo
   disponibles (contratos mas estructurados tienden a ser mas grandes
   y con mayor sobrecosto absoluto).

5. LIMITACION EXPLICITA: La muestra NO es estadisticamente representativa
   de la poblacion total de contratos de obra publica en Colombia.
   Las pruebas Chi-cuadrado y KS confirman diferencias significativas
   en las 4 dimensiones analizadas.

6. IMPLICACION PARA LA TESIS: Esto NO invalida el estudio, pero debe
   reconocerse como una limitacion. El objetivo no es inferir estadisticas
   poblacionales, sino demostrar la VIABILIDAD del enfoque: predecir
   sobrecosto a partir de matrices de riesgo usando ML. La muestra es
   adecuada para este proposito porque:
   a) Cubre 348 contratos reales con datos autenticos de SECOP
   b) Incluye variedad de tipos de riesgo, entidades y regiones
   c) El modelo SVR se evalua con validacion cruzada interna (5-fold)
   d) La generalizacion externa requeriria recoleccion adicional de datos

7. RECOMENDACION: Incluir en la tesis un parrafo que reconozca
   explicitamente esta limitacion y justifique la muestra como un
   "convenience sample" para un prototipo/de prueba de concepto.
""")
