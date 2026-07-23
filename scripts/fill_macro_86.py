"""
Fill macro data for 86 contracts missing anio_inicio, anio_fin, duracion, ipc_acumulado, trm_promedio.
Matches by valor_final from secop1_cache.
"""
import pandas as pd
import numpy as np
from pathlib import Path

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

# Load
clean = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
cache = pd.read_csv(ROOT / "contratos" / "secop1_cache.csv", encoding="utf-8", low_memory=False)

# Get 86 missing contracts aggregated
clean_ids = set(clean["id_contrato"].unique())
macro_ids = set(macro["id_contrato"].unique())
missing_ids = sorted(clean_ids - macro_ids)

missing_df = clean[clean["id_contrato"].isin(missing_ids)].groupby("id_contrato").agg(
    valor_inicial=("valor_inicial", "first"),
    valor_final=("valor_final", "first"),
).reset_index()

print(f"Missing: {len(missing_df)} contracts")

# Match by valor_final from secop1_cache
cache_unique = cache[["cuantia_contrato", "valor_contrato_con_adiciones",
                       "fecha_ini_ejec_contrato", "fecha_fin_ejec_contrato"]].drop_duplicates(
                           subset=["valor_contrato_con_adiciones"], keep="first")

new_rows = []
matched = 0
for _, row in missing_df.iterrows():
    match = cache_unique[cache_unique["valor_contrato_con_adiciones"] == row["valor_final"]]
    if len(match) == 0:
        match = cache_unique[cache_unique["cuantia_contrato"] == row["valor_inicial"]]
    if len(match) > 0:
        matched += 1
        r = match.iloc[0]
        try:
            anio_ini = int(str(r["fecha_ini_ejec_contrato"]).split("-")[0])
            anio_fin = int(str(r["fecha_fin_ejec_contrato"]).split("-")[0])
        except:
            anio_ini = anio_fin = 2019
        feat = compute_range_features(anio_ini, anio_fin)
        feat["id_contrato"] = row["id_contrato"]
        new_rows.append(feat)

print(f"Matched: {matched}/{len(missing_df)}")

# Append to macro
new_df = pd.DataFrame(new_rows)
macro = pd.concat([macro, new_df], ignore_index=True)
macro.to_csv(ROOT / "docs" / "contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"Macro actualizado: {len(macro)} contratos")
print(f"Anios: {macro['anio_inicio'].min()}-{macro['anio_fin'].max()}")
print("Listo.")
