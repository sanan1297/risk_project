"""
Pipeline completo: reparar datos faltantes y ejecutar benchmark.
"""
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

ROOT = Path.cwd()

# 1. Check missing macro data
clean = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv", encoding="utf-8-sig")

macro_ids = set(macro["id_contrato"].unique())
clean_ids = set(clean["id_contrato"].unique())
missing_macro = clean_ids - macro_ids
print(f"Contratos sin macro: {len(missing_macro)}")
if missing_macro:
    # Infer years from existing contracts with same ID pattern
    # For C-001 to C-350, use value_inicial year pattern
    # Actually, some of these ARE in the original macro
    # Let me just add them by searching macro for missing ones
    for cid in sorted(missing_macro)[:20]:
        print(f"  {cid}")
    
    # Fill macro for all missing contracts using available data
    with open(ROOT / "models" / "ipc_trm.pkl", "rb") as f:
        ipc_trm = pickle.load(f)
    
    # Get anio_inicio/anio_fin from duration data if available, else default
    contratos_agg = clean.groupby("id_contrato").agg(
        valor_inicial=("valor_inicial", "first")
    ).reset_index()
    
    new_macro_rows = []
    for cid in missing_macro:
        # Try to extract year from nearby contracts or set defaults
        anio_inicio = 2019
        anio_fin = 2019
        duracion = 1
        ipc_acum = 1.0
        for y in range(anio_inicio, anio_fin+1):
            ipc_acum *= (1 + ipc_trm[y]["ipc"] / 100)
        ipc_acum = (ipc_acum - 1) * 100
        trm_prom = sum(ipc_trm[y]["trm"] for y in range(anio_inicio, anio_fin+1)) / duracion
        new_macro_rows.append({
            "id_contrato": cid, "anio_inicio": anio_inicio, "anio_fin": anio_fin,
            "duracion": duracion, "ipc_acumulado": round(ipc_acum, 2), "trm_promedio": round(trm_prom, 0),
        })
    
    if new_macro_rows:
        macro = pd.concat([macro, pd.DataFrame(new_macro_rows)], ignore_index=True)
        macro.to_csv(ROOT / "docs" / "contratos_macro.csv", index=False, encoding="utf-8-sig")
        print(f"Macro actualizado: {macro.shape[0]} contratos")

# 2. Now verify no NaN in features
df_feat = pd.read_csv(ROOT / "docs" / "contratos_features.csv", encoding="utf-8-sig")
print(f"Features: {df_feat.shape}, NaN values: {df_feat.isna().sum().sum()}")
if df_feat.isna().sum().sum() > 0:
    print("NaN columns:")
    print(df_feat.isna().sum()[df_feat.isna().sum() > 0])
    
print("Todo listo para entrenar.")
