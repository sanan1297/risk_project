"""
Consolidar matriz.csv → 577 contratos, extraer 38 features (30 riesgo + 5 macro + 3 mitigacion).
GUARDAR SOLO. No entrenar.
"""
import csv as csv_mod
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, str(Path.cwd()))
from estudio_data.features import engineer_features

ROOT = Path.cwd()

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

def compute_range_features(anio_inicio, anio_fin, max_duracion=5):
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

def extract_year(date_val):
    if pd.isna(date_val):
        return None
    s = str(date_val).strip()
    try:
        return int(s.split("-")[0])
    except (ValueError, IndexError):
        return None

# 1. Load matriz.csv (handle malformed rows with extra columns)
import csv as csv_mod
raw_path = ROOT / "docs" / "matriz.csv"
with open(raw_path, encoding="utf-8-sig", newline="") as f:
    reader = csv_mod.reader(f)
    raw_rows = [row[:20] for row in reader]
header = raw_rows[0]
data = raw_rows[1:]
matriz = pd.DataFrame(data, columns=header)
print(f"Raw rows leidas: {len(raw_rows)-1}, columnas fijadas a 20")
# Convert numeric columns
for c in ["valor_inicial", "valor_final", "sobrecosto", "probabilidad", "impacto", "valoracion"]:
    matriz[c] = pd.to_numeric(matriz[c], errors="coerce")
print(f"matriz.csv: {len(matriz)} rows, {matriz['id_contrato'].nunique()} contratos")

# 2. Consolidate: one row per contract
df = engineer_features(matriz)
outlier_mask = df["sobrecosto"] < 200
print(f"Outliers >200%: {(~outlier_mask).sum()} contratos")
df = df[outlier_mask].copy()
print(f"Consolidado: {len(df)} contratos")

# 3. Load existing macro data
macro_path = ROOT / "docs" / "contratos_macro.csv"
macro_existing = pd.read_csv(macro_path)
print(f"Macro existente: {len(macro_existing)} contratos")

# 4. Find contracts missing macro data
df_ids = set(df["id_contrato"])
macro_ids = set(macro_existing["id_contrato"])
missing_ids = sorted(df_ids - macro_ids)
print(f"Contratos sin macro: {len(missing_ids)}")

all_macro_rows = []

# Build lookup from existing macro
macro_lookup = {}
for _, r in macro_existing.iterrows():
    macro_lookup[r["id_contrato"]] = {
        "anio_inicio": r["anio_inicio"],
        "anio_fin": r["anio_fin"],
        "duracion": r["duracion"],
        "ipc_acumulado": r["ipc_acumulado"],
        "trm_promedio": r["trm_promedio"],
    }

# Fill from existing
for cid in df_ids & macro_ids:
    all_macro_rows.append({"id_contrato": cid, **macro_lookup[cid]})

# Fill missing from secop1_cache
if missing_ids:
    cache = pd.read_csv(ROOT / "contratos" / "secop1_cache.csv", encoding="utf-8", low_memory=False)
    cache = cache[["cuantia_contrato", "valor_contrato_con_adiciones",
                   "fecha_ini_ejec_contrato", "fecha_fin_ejec_contrato"]].drop_duplicates(
                       subset=["valor_contrato_con_adiciones"], keep="first")
    cache["valor_contrato_con_adiciones"] = pd.to_numeric(cache["valor_contrato_con_adiciones"], errors="coerce")
    cache["cuantia_contrato"] = pd.to_numeric(cache["cuantia_contrato"], errors="coerce")

    missing_agg = matriz[matriz["id_contrato"].isin(missing_ids)].groupby("id_contrato").agg(
        valor_inicial=("valor_inicial", "first"),
        valor_final=("valor_final", "first"),
    ).reset_index()

    matched_count = 0
    for _, row in missing_agg.iterrows():
        match = cache[cache["valor_contrato_con_adiciones"] == row["valor_final"]]
        if len(match) == 0:
            match = cache[cache["cuantia_contrato"] == row["valor_inicial"]]
        if len(match) > 0:
            matched_count += 1
            r = match.iloc[0]
            anio_ini = extract_year(r["fecha_ini_ejec_contrato"])
            anio_fin = extract_year(r["fecha_fin_ejec_contrato"])
            if anio_ini is None or anio_fin is None:
                anio_ini = anio_fin = 2019
            feat = compute_range_features(anio_ini, anio_fin)
            feat["id_contrato"] = row["id_contrato"]
            all_macro_rows.append(feat)
        else:
            # No match: default to 2019
            feat = compute_range_features(2019, 2020)
            feat["id_contrato"] = row["id_contrato"]
            all_macro_rows.append(feat)

    print(f"Matched en secop1_cache: {matched_count}/{len(missing_ids)}")

macro = pd.DataFrame(all_macro_rows)
print(f"Macro total: {len(macro)} contratos")
print(f"Rango anios: {macro['anio_inicio'].min()}-{macro['anio_fin'].max()}")
macro.to_csv(ROOT / "docs" / "contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"contratos_macro.csv actualizado: {len(macro)} contratos")

# 5. Merge macro
macro_vars = ["anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]
mitigacion_vars = ["pct_riesgos_con_mitigacion", "avg_longitud_mitigacion", "n_distinct_codes_mitigacion"]
df = df.merge(macro, on="id_contrato", how="left")

# 6. Find top 30 features (exclude macro, mitigacion, id, fuente, sobrecosto)
exclude = {"id_contrato", "fuente", "sobrecosto"} | set(macro_vars) | set(mitigacion_vars)
risk_cols = [c for c in df.columns if c not in exclude]
print(f"Total risk features disponibles para RF: {len(risk_cols)}")

X_rf = df[risk_cols].fillna(0).values.astype(np.float64)
y_rf = df["sobrecosto"].values.astype(np.float64)

rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
top30_idx = np.argsort(rf.feature_importances_)[::-1][:30]
top30 = [risk_cols[i] for i in top30_idx]

print(f"\nTop 30 features (riesgo):")
for i, f in enumerate(top30):
    print(f"  {i+1:2d}. {f:30s} {rf.feature_importances_[top30_idx[i]]:.4f}")

# 7. Build 38-feature dataset
FEATURES_38 = top30 + macro_vars + mitigacion_vars
df_38 = df[["id_contrato", "sobrecosto"] + FEATURES_38].copy()

# Check NaN
for v in macro_vars + mitigacion_vars:
    nan_count = df_38[v].isna().sum()
    if nan_count > 0:
        print(f"{v}: {nan_count} NaN")

# 8. Save
df_38.to_csv(ROOT / "docs" / "consolidado_38_features.csv", index=False, encoding="utf-8-sig")
print(f"\nGuardado: docs/consolidado_38_features.csv ({len(df_38)} rows x {len(FEATURES_38)} features)")
print("Listo.")
