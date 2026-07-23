"""Agregar C-360 a C-365 a matriz_clean.csv y contratos_macro.csv"""
import pandas as pd
import pickle
import numpy as np

clean = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"matriz_clean antes: {clean.shape[0]} rows, {clean['id_contrato'].nunique()} contratos")

macro_new = {
    "C-360": {"anio_inicio": 2019, "anio_fin": 2019, "sobrecosto": 10.14},
    "C-361": {"anio_inicio": 2022, "anio_fin": 2022, "sobrecosto": 19.09},
    "C-362": {"anio_inicio": 2021, "anio_fin": 2021, "sobrecosto": 4.38},
    "C-363": {"anio_inicio": 2022, "anio_fin": 2022, "sobrecosto": 7.2},
    "C-364": {"anio_inicio": 2023, "anio_fin": 2023, "sobrecosto": 20.83},
    "C-365": {"anio_inicio": 2023, "anio_fin": 2027, "sobrecosto": float("nan")},
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
            "valor_inicial": 0, "valor_final": 0, "sobrecosto": mn["sobrecosto"],
            "url": "", "objeto": f"CONTRATO {cid}", "fuente": "nuevo",
            "id_riesgo": i + 1, "clase": "general", "fuente_riesgo": "externo",
            "etapa": "ejecucion", "tipo": row["tipo"],
            "descripcion_riesgo": row["descripcion_riesgo"],
            "consecuencia": "", "probabilidad": str(prob), "impacto": str(imp),
            "valoracion": prob * imp, "categoria": row["categoria"],
            "asignacion": "contratista", "plan_mitigacion": "",
        })
    print(f"  {cid}: {len(df)} riesgos agregados")

nuevos_df = pd.DataFrame(nuevas_filas)
clean_final = pd.concat([clean, nuevos_df], ignore_index=True)

# Ensure column order matches
cols = clean.columns.tolist()
clean_final = clean_final[cols]

clean_final.to_csv("docs/matriz_clean.csv", index=False, encoding="utf-8-sig")
print(f"matriz_clean despues: {clean_final.shape[0]} rows, {clean_final['id_contrato'].nunique()} contratos")

# Now update contratos_macro.csv
macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
with open("models/ipc_trm.pkl", "rb") as f:
    ipc_trm = pickle.load(f)

for cid, mn in macro_new.items():
    if cid in macro["id_contrato"].values:
        continue
    duracion = mn["anio_fin"] - mn["anio_inicio"] + 1
    ipc_acum = 1.0
    for y in range(mn["anio_inicio"], mn["anio_fin"]+1):
        ipc_acum *= (1 + ipc_trm[y]["ipc"] / 100)
    ipc_acum = (ipc_acum - 1) * 100
    trm_prom = sum(ipc_trm[y]["trm"] for y in range(mn["anio_inicio"], mn["anio_fin"]+1)) / duracion
    macro = pd.concat([macro, pd.DataFrame([{
        "id_contrato": cid, "anio_inicio": mn["anio_inicio"], "anio_fin": mn["anio_fin"],
        "duracion": duracion, "ipc_acumulado": round(ipc_acum,2), "trm_promedio": round(trm_prom,0),
    }])], ignore_index=True)
    print(f"  Macro {cid}: {mn['anio_inicio']}-{mn['anio_fin']} dur={duracion} ipc={ipc_acum:.1f}% trm={trm_prom:.0f}")

macro.to_csv("docs/contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"macro: {macro.shape[0]} contratos")
print("Listo.")
