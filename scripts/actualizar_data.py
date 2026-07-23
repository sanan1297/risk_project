"""
Paso 1: Actualizar matriz_clean.csv y contratos_macro.csv
con C-360 a C-365, respetando gap (C-352 a C-359).
"""
import pandas as pd
import pickle
import numpy as np

# =========================================
# 1. Cargar matriz_clean existente
# =========================================
clean = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"matriz_clean actual: {clean.shape[0]} rows, {clean['id_contrato'].nunique()} contratos")
ids_actual = sorted(clean["id_contrato"].unique())
print(f"IDs: {ids_actual[0]} ... {ids_actual[-1]}")
print(f"Total contratos: {len(ids_actual)}")

# =========================================
# 2. Cargar nuevos riesgos y mapear
# =========================================
macro_new = {
    "C-360": {"anio_inicio": 2019, "anio_fin": 2019, "valor_inicial": 1888738443, "valor_final": 2080327120, "sobrecosto": 10.14},
    "C-361": {"anio_inicio": 2022, "anio_fin": 2022, "valor_inicial": 1885591244, "valor_final": 2245590311, "sobrecosto": 19.09},
    "C-362": {"anio_inicio": 2021, "anio_fin": 2021, "valor_inicial": 1877707543, "valor_final": 1959999799, "sobrecosto": 4.38},
    "C-363": {"anio_inicio": 2022, "anio_fin": 2022, "valor_inicial": 1869551299, "valor_final": 2004076428, "sobrecosto": 7.2},
    "C-364": {"anio_inicio": 2023, "anio_fin": 2023, "valor_inicial": 1868945401, "valor_final": 2258302331, "sobrecosto": 20.83},
    "C-365": {"anio_inicio": 2023, "anio_fin": 2027, "valor_inicial": 477834784322, "valor_final": 477834784322, "sobrecosto": float("nan")},
}

nuevas_filas = []
for cid in ["C-360", "C-361", "C-362", "C-363", "C-364", "C-365"]:
    path = f"tests/data/{cid.lower()}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    mn = macro_new[cid]
    for i, row in df.iterrows():
        prob = int(row["probabilidad"])
        imp = int(row["impacto"])
        nuevas_filas.append({
            "id_contrato": cid,
            "valor_inicial": mn["valor_inicial"],
            "valor_final": mn["valor_final"],
            "sobrecosto": mn["sobrecosto"],
            "url": "",
            "objeto": f"CONTRATO {cid}",
            "fuente": "nuevo",
            "id_riesgo": i + 1,
            "clase": "general",
            "fuente_riesgo": "externo",
            "etapa": "ejecucion",
            "tipo": row["tipo"],
            "descripcion_riesgo": row["descripcion_riesgo"],
            "consecuencia": "",
            "probabilidad": str(prob),
            "impacto": str(imp),
            "valoracion": prob * imp,
            "categoria": row["categoria"],
            "asignacion": "contratista",
            "plan_mitigacion": "",
        })
    print(f"  {cid}: {len(df)} riesgos | anios {mn['anio_inicio']}-{mn['anio_fin']} | sc={mn['sobrecosto']}")

nuevos_df = pd.DataFrame(nuevas_filas)
clean_final = pd.concat([clean, nuevos_df], ignore_index=True)

# Verificar gap
ids_final = sorted(clean_final["id_contrato"].unique())
print(f"\nIDs finales: {len(ids_final)} contratos")
print(f"IDs entre C-348 y C-366:")
for cid in ids_final:
    num = int(cid.split("-")[1])
    if 348 <= num <= 366:
        print(f"  {cid}")

# Guardar
clean_final.to_csv("docs/matriz_clean.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ matriz_clean.csv guardado: {clean_final.shape[0]} rows, {clean_final['id_contrato'].nunique()} contratos")

# =========================================
# 3. Actualizar contratos_macro.csv
# =========================================
macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
print(f"\ncontratos_macro actual: {macro.shape[0]} contratos")

with open("models/ipc_trm.pkl", "rb") as f:
    ipc_trm = pickle.load(f)

for cid, mn in macro_new.items():
    duracion = mn["anio_fin"] - mn["anio_inicio"] + 1
    
    ipc_data = ipc_trm.get("ipc", {})
    ipc_acum = 1.0
    for year in range(mn["anio_inicio"], mn["anio_fin"] + 1):
        ipc_acum *= (1 + ipc_data.get(str(year), 0.05))
    ipc_acum = (ipc_acum - 1) * 100
    
    trm_data = ipc_trm.get("trm", {})
    trm_sum = sum(trm_data.get(str(y), 4000) for y in range(mn["anio_inicio"], mn["anio_fin"] + 1))
    trm_count = mn["anio_fin"] - mn["anio_inicio"] + 1
    trm_prom = trm_sum / trm_count
    
    macro = pd.concat([macro, pd.DataFrame([{
        "id_contrato": cid,
        "anio_inicio": mn["anio_inicio"],
        "anio_fin": mn["anio_fin"],
        "duracion": duracion,
        "ipc_acumulado": round(ipc_acum, 2),
        "trm_promedio": round(trm_prom, 0),
    }])], ignore_index=True)
    print(f"  {cid}: {mn['anio_inicio']}-{mn['anio_fin']} dur={duracion} ipc={ipc_acum:.1f}% trm={trm_prom:.0f}")

macro.to_csv("docs/contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"✅ contratos_macro.csv guardado: {macro.shape[0]} contratos")
