"""
Pipeline completo:
1. Agregar nuevos contratos (C-360 a C-365) a matriz_clean.csv y contratos_macro.csv
2. Re-generar features
3. Re-entrenar modelo
4. Ejecutar notebooks de evaluacion
"""
import os, sys, subprocess, pandas as pd
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
print(f"Working dir: {PROJECT}")

def run_cmd(cmd, label):
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: Exit code {result.returncode}")
    return result

# =============================================================
# PASO 1: Cargar matriz_clean existente + nuevos riesgos
# =============================================================
print(f"\n{'='*60}")
print("PASO 1: Agregando nuevos contratos a matriz_clean.csv")
print(f"{'='*60}")

clean = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"matriz_clean actual: {clean.shape[0]} rows, {clean['id_contrato'].nunique()} contratos")
print(f"ids actuales: {sorted(clean['id_contrato'].unique())[:5]} ... {sorted(clean['id_contrato'].unique())[-5:]}")

# Cargar nuevos riesgos
nuevos_riesgos = []
for cid in ["C-360", "C-361", "C-362", "C-363", "C-364", "C-365"]:
    path = f"tests/data/{cid.lower()}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    for i, row in df.iterrows():
        nuevos_riesgos.append({
            "id_contrato": cid,
            "id_riesgo": i + 1,
            "descripcion_riesgo": row["descripcion_riesgo"],
            "probabilidad": str(row["probabilidad"]),
            "impacto": str(row["impacto"]),
            "tipo": row["tipo"],
            "categoria": row["categoria"],
        })
    print(f"  {cid}: {len(df)} riesgos")

# Macro data
macro_new = {
    "C-360": {"anio_inicio": 2019, "anio_fin": 2019, "valor_inicial": 1888738443.0, "valor_final": 2080327120.0, "sobrecosto": 10.14},
    "C-361": {"anio_inicio": 2022, "anio_fin": 2022, "valor_inicial": 1885591244.0, "valor_final": 2245590311.0, "sobrecosto": 19.09},
    "C-362": {"anio_inicio": 2021, "anio_fin": 2021, "valor_inicial": 1877707543.0, "valor_final": 1959999799.0, "sobrecosto": 4.38},
    "C-363": {"anio_inicio": 2022, "anio_fin": 2022, "valor_inicial": 1869551299.0, "valor_final": 2004076428.0, "sobrecosto": 7.2},
    "C-364": {"anio_inicio": 2023, "anio_fin": 2023, "valor_inicial": 1868945401.0, "valor_final": 2258302331.0, "sobrecosto": 20.83},
    "C-365": {"anio_inicio": 2023, "anio_fin": 2027, "valor_inicial": 477834784322.0, "valor_final": 477834784322.0, "sobrecosto": 0.0},
}

# Agregar a matriz_clean
nuevas_filas = []
for nr in nuevos_riesgos:
    cid = nr["id_contrato"]
    mn = macro_new[cid]
    nuevas_filas.append({
        "id_contrato": cid,
        "valor_inicial": int(mn["valor_inicial"]),
        "valor_final": int(mn["valor_final"]),
        "sobrecosto": mn["sobrecosto"],
        "url": "",
        "objeto": f"CONTRATO {cid}",
        "fuente": "nuevo",
        "id_riesgo": nr["id_riesgo"],
        "clase": "general",
        "fuente_riesgo": "externo",
        "etapa": "ejecucion",
        "tipo": nr["tipo"],
        "descripcion_riesgo": nr["descripcion_riesgo"],
        "consecuencia": "",
        "probabilidad": nr["probabilidad"],
        "impacto": nr["impacto"],
        "valoracion": int(nr["probabilidad"]) * int(nr["impacto"]),
        "categoria": nr["categoria"],
        "asignacion": "contratista",
        "plan_mitigacion": "",
    })

nuevos_df = pd.DataFrame(nuevas_filas)
clean_con_nuevos = pd.concat([clean, nuevos_df], ignore_index=True)

print(f"\nmatriz_clean con nuevos: {clean_con_nuevos.shape[0]} rows, {clean_con_nuevos['id_contrato'].nunique()} contratos")
print(f"ids: {sorted(clean_con_nuevos['id_contrato'].unique())}")

# Verificar gap
ids = sorted(clean_con_nuevos["id_contrato"].unique())
print(f"\nIDs entre C-350 y C-366:")
for cid in ids:
    num = int(cid.split("-")[1])
    if 348 <= num <= 366:
        print(f"  {cid}")

# Guardar
clean_con_nuevos.to_csv("docs/matriz_clean.csv", index=False, encoding="utf-8-sig")
print(f"\nGuardado: docs/matriz_clean.csv ({clean_con_nuevos.shape[0]} rows)")

# =============================================================
# PASO 2: Actualizar contratos_macro.csv
# =============================================================
print(f"\n{'='*60}")
print("PASO 2: Actualizando contratos_macro.csv")
print(f"{'='*60}")

macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
print(f"macro actual: {macro.shape[0]} contratos")

# IPC/TRM data from existing model
import pickle
with open("models/ipc_trm.pkl", "rb") as f:
    ipc_trm = pickle.load(f)
print(f"IPC/TRM loaded: {len(ipc_trm.get('ipc',{}))} years")

for cid, mn in macro_new.items():
    # Compute duration
    duracion = mn["anio_fin"] - mn["anio_inicio"] + 1
    
    # Compute compounded IPC
    ipc_data = ipc_trm.get("ipc", {})
    ipc_acum = 1.0
    for year in range(mn["anio_inicio"], mn["anio_fin"] + 1):
        ipc_acum *= (1 + ipc_data.get(str(year), 0.05))
    ipc_acum = (ipc_acum - 1) * 100
    
    # Average TRM
    trm_data = ipc_trm.get("trm", {})
    trm_sum = 0
    trm_count = 0
    for year in range(mn["anio_inicio"], mn["anio_fin"] + 1):
        if str(year) in trm_data:
            trm_sum += trm_data[str(year)]
            trm_count += 1
    trm_prom = trm_sum / trm_count if trm_count > 0 else 4000
    
    macro = pd.concat([macro, pd.DataFrame([{
        "id_contrato": cid,
        "anio_inicio": mn["anio_inicio"],
        "anio_fin": mn["anio_fin"],
        "duracion": duracion,
        "ipc_acumulado": round(ipc_acum, 2),
        "trm_promedio": round(trm_prom, 0),
    }])], ignore_index=True)
    print(f"  {cid}: {mn['anio_inicio']}-{mn['anio_fin']}, dur={duracion}, ipc={ipc_acum:.1f}%, trm={trm_prom:.0f}")

macro.to_csv("docs/contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"macro guardado: {macro.shape[0]} contratos")

# =============================================================
# PASO 3: Re-generar features
# =============================================================
print(f"\n{'='*60}")
print("PASO 3: Re-generando features con estudio_data/features.py")
print(f"{'='*60}")
run_cmd(f"uv run python estudio_data/features.py", "Features")

# =============================================================
# PASO 4: Re-entrenar modelo
# =============================================================
print(f"\n{'='*60}")
print("PASO 4: Re-entrenando modelo con scripts/train_final_model.py")
print(f"{'='*60}")
run_cmd(f"uv run python scripts/train_final_model.py", "Training")

print(f"\n{'='*60}")
print("PIPELINE COMPLETO - Nuevos datos agregados y modelo re-entrenado")
print(f"{'='*60}")
print(f"Contratos totales: {clean_con_nuevos['id_contrato'].nunique()}")
print(f"Riesgos totales: {clean_con_nuevos.shape[0]}")
