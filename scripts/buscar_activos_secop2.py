"""
Descarga data FRESCA de SECOP II - Contratos NO terminados (en ejecución)
Criterios: Obra + >= $1,000M + Inversión + URL + NO terminado
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

CARPETA = "contratos"
MAX_POR_PAGINA = 50000

SECOP2_URL = "https://www.datos.gov.co/resource/jbjy-vk9h.json"

def descargar(where: str, max_registros: int = 100000) -> pd.DataFrame | None:
    todos = []
    for offset in range(0, max_registros, MAX_POR_PAGINA):
        pagina = offset // MAX_POR_PAGINA + 1
        params = {
            "$limit": MAX_POR_PAGINA,
            "$offset": offset,
            "$where": where,
            "$order": "valor_del_contrato DESC",
        }
        try:
            r = requests.get(SECOP2_URL, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                print(f"  Pag {pagina} - Error API: {data.get('message','')}")
                break
            if not data:
                print(f"  Pag {pagina} - Fin.")
                break
            todos.extend(data)
            print(f"  Pag {pagina}: +{len(data)} = {len(todos)}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Pag {pagina} - Error: {e}")
            time.sleep(2)
            continue

    if todos:
        df = pd.DataFrame(todos)
        return df
    return None


def main():
    print("=" * 60)
    print("DESCARGAR CONTRATOS ACTIVOS - SECOP II")
    print("No terminados + Obra + >= $1,000M")
    print(fecha := datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 60)

    # Estados EXCLUIDOS (finalizados)
    # Filtro: Obra, NO terminado, NO Cerrado
    where = (
        "tipo_de_contrato = 'Obra'"
        " AND estado_contrato != 'terminado'"
        " AND estado_contrato != 'Cerrado'"
    )

    print(f"\nQuery API:")
    print(f"  URL: {SECOP2_URL}")
    print(f"  Where: {where}")
    print(f"\nDescargando...")

    df = descargar(where, max_registros=200000)

    if df is None or len(df) == 0:
        print("\nNo se encontraron contratos activos.")
        return

    print(f"\nDescargados: {len(df)} registros, {len(df.columns)} columnas")

    # ---- NORMALIZAR ----
    # Rename key columns
    col_map = {
        "objeto_del_contrato": "objeto",
        "nombre_de_la_entidad": "entidad",
        "fecha_de_inicio_del_contrato": "fecha_inicio",
        "fecha_de_fin_del_contrato": "fecha_fin",
        "valor_del_contrato": "valor_inicial",
        "valor_pagado": "valor_final",
        "proveedor_adjudicado": "contratista",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Numeric
    for col in ["valor_inicial", "valor_final", "dias_adicionados"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fechas
    for col in ["fecha_inicio", "fecha_fin"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---- FILTROS ----
    antes = len(df)

    # 1. Valor >= $1,000M
    if "valor_inicial" in df.columns:
        df = df[df["valor_inicial"] >= 1_000_000_000].copy()
        print(f"Filtro >= $1,000M: {antes} -> {len(df)}")

    # 2. Destino = Inversión
    antes = len(df)
    if "destino_gasto" in df.columns:
        df["destino_gasto_norm"] = df["destino_gasto"].str.strip().str.lower()
        df = df[df["destino_gasto_norm"] == "inversión"].copy()
        print(f"Filtro Inversión: {antes} -> {len(df)}")

    # 3. URL válida
    antes = len(df)
    if "urlproceso" in df.columns:
        df["url"] = df["urlproceso"].apply(
            lambda v: v.get("url", "") if isinstance(v, dict) else (v if isinstance(v, str) and v.strip() not in ("", "nan", "None") else "")
        )
    df = df[df["url"].notna() & (df["url"] != "")].copy()
    print(f"Filtro URL: {antes} -> {len(df)}")

    if len(df) == 0:
        print("\nNo hay contratos después de aplicar filtros.")
        # Show available estados for debugging
        print(f"\nEstados disponibles en la descarga (>= $1,000M):")
        print(requests.get(SECOP2_URL, params={
            "$limit": 50000,
            "$where": "tipo_de_contrato = 'Obra' AND estado_contrato != 'terminado' AND estado_contrato != 'Cerrado' AND valor_del_contrato >= 1000000000",
            "$select": "estado_contrato, COUNT(*) as cnt",
            "$group": "estado_contrato"
        }).json())
        return

    # ---- ORDENAR ----
    df = df.sort_values("valor_inicial", ascending=False).reset_index(drop=True)

    # ---- SELECCIONAR COLUMNAS ----
    cols_salida = [
        "entidad", "departamento", "ciudad",
        "valor_inicial", "valor_final", "dias_adicionados",
        "destino_gasto", "estado_contrato", "objeto",
        "url", "contratista",
        "fecha_inicio", "fecha_fin",
    ]
    cols_presentes = [c for c in cols_salida if c in df.columns]

    # ---- EXPORTAR ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = os.path.join(CARPETA, f"secop2_activos_{timestamp}.csv")
    df[cols_presentes].to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\nGuardado: {archivo}")
    print(f"  {len(df)} registros, {len(cols_presentes)} columnas")

    # ---- MOSTRAR TOP 50 ----
    print(f"\n--- Top {min(50, len(df))} contratos activos ---")
    for i, (_, r) in enumerate(df.iterrows()):
        if i >= 50:
            break
        ini = r["fecha_inicio"].date() if pd.notna(r.get("fecha_inicio")) else "?"
        fin = r["fecha_fin"].date() if pd.notna(r.get("fecha_fin")) else "?"
        print(f"  ${r['valor_inicial']:>14,.0f} | {ini} → {fin} | {r.get('estado_contrato', '?'):30s} | {str(r.get('objeto', ''))[:70]}")

    print(f"\nResumen:")
    print(f"  Total: {len(df)}")
    print(f"  Rango valores: ${df['valor_inicial'].min():,.0f} - ${df['valor_inicial'].max():,.0f}")


if __name__ == "__main__":
    main()
