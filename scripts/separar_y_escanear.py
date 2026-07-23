"""Paso 1: Separa barrido 480-750 en dos CSVs.
Paso 2: Escanea registros 751-1560 de secop1_lite.csv.
Paso 3: Separa nuevo barrido en matriz_riesgos + estudios_previos.
"""
import asyncio
import csv
import os
import re
import sys
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = os.path.join("estudio_data")
BARRIDO_PREVIO = os.path.join(OUT_DIR, "barrido_480_750.csv")

# Archivos de salida
MATRIZ_CSV = os.path.join(OUT_DIR, "matriz_riesgos.csv")
ESTUDIOS_CSV = os.path.join(OUT_DIR, "estudios_previos.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def contar_registros():
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        total = sum(1 for _ in reader)
    return total

# ============================================================
# PASO 1: Separar barrido previo (480-750)
# ============================================================
def separar_barrido_previo():
    print("=" * 60)
    print("PASO 1: Separando barrido 480-750")
    print("=" * 60)

    matriz = []
    estudios = []

    with open(BARRIDO_PREVIO, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tipo = row["tipo_documento"].strip()
            if tipo == "MATRIZ_RIESGOS":
                matriz.append(row)
            elif tipo == "ESTUDIOS_PREVIOS":
                estudios.append(row)

    guardar_csv(MATRIZ_CSV, matriz)
    guardar_csv(ESTUDIOS_CSV, estudios)
    print(f"  Matriz Riesgos: {len(matriz)} contratos -> {MATRIZ_CSV}")
    print(f"  Estudios Previos: {len(estudios)} contratos -> {ESTUDIOS_CSV}")
    return len(matriz), len(estudios)

def guardar_csv(path, rows, fieldnames=None):
    if not rows:
        # Escribir header aunque este vacio
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return
    if not fieldnames:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# PASO 2: Escanear registros 751 en adelante
# ============================================================
def cargar_registros_desde(inicio):
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            fila = i + 1
            if fila < inicio:
                continue
            registros.append((fila, row))
    return registros

def debe_omitir(row):
    mun = (row.get("municipio", "") or "").strip().upper()
    return "BOGOTA" in mun and ("D.C" in mun or "DC" in mun)

async def analizar_pagina(page, url):
    try:
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(2000)
    except PwTimeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:80]}

    content = await page.content()
    result = {
        "tiene_matriz_riesgos": False,
        "tiene_estudios_previos": False,
    }

    # Buscar labels en la tabla de documentos
    rows = await page.query_selector_all("tr")
    for tr in rows:
        try:
            text = (await tr.inner_text()).strip()
        except:
            continue
        if not text:
            continue
        if "MATRIZ DE RIESGOS" in text.upper() and "MATRIZ" in text.upper():
            result["tiene_matriz_riesgos"] = True
        if "ESTUDIOS PREVIOS" in text.upper() or "ESTUDIO PREVIO" in text.upper():
            result["tiene_estudios_previos"] = True

    return result

async def main_scan():
    total_registros = contar_registros()
    print(f"\nTotal registros en secop1_lite.csv: {total_registros}")

    registros = cargar_registros_desde(751)
    print(f"Cargados {len(registros)} registros (fila 751-{total_registros})")

    omitidos_bogota = [r for r in registros if debe_omitir(r[1])]
    registros = [r for r in registros if not debe_omitir(r[1])]
    print(f"Omitidos Bogota D.C.: {len(omitidos_bogota)}")
    print(f"A procesar: {len(registros)} contratos")

    filas_matriz = []
    filas_estudios = []
    sin_doc = 0
    errores = 0
    sin_url = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="es-CO",
        )
        page = await context.new_page()

        for idx, (fila, row) in enumerate(registros):
            url = row.get("url", "").strip()
            entidad = (row.get("entidad", "") or "").strip()
            municipio = (row.get("municipio", "") or "").strip()
            depto = (row.get("departamento", "") or "").strip()

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"\n[{fila}/{total_registros}] Procesados: {idx+1}  |  Matriz: {len(filas_matriz)}  |  Estudios: {len(filas_estudios)}  |  SinDoc: {sin_doc}")

            if not url:
                sin_url += 1
                continue

            info = await analizar_pagina(page, url)

            if info.get("error"):
                errores += 1
                continue

            tipo = "SIN_DOCUMENTO"
            if info["tiene_matriz_riesgos"]:
                tipo = "MATRIZ_RIESGOS"
            elif info["tiene_estudios_previos"]:
                tipo = "ESTUDIOS_PREVIOS"

            if tipo == "SIN_DOCUMENTO":
                sin_doc += 1
            else:
                fila_data = {
                    "fila_secop1_lite": fila,
                    "url": url,
                    "entidad": entidad,
                    "municipio": municipio,
                    "departamento": depto,
                    "tipo_documento": tipo,
                }
                if tipo == "MATRIZ_RIESGOS":
                    filas_matriz.append(fila_data)
                    print(f"    [+] MATRIZ: fila {fila} | {entidad[:50]}")
                else:
                    filas_estudios.append(fila_data)

            await page.wait_for_timeout(1000)

        await browser.close()

    # Guardar CSVs
    fieldnames = ["fila_secop1_lite", "url", "entidad", "municipio", "departamento", "tipo_documento"]

    # Merge con datos previos
    matriz_path = os.path.join(OUT_DIR, "matriz_riesgos.csv")
    estudios_path = os.path.join(OUT_DIR, "estudios_previos.csv")

    # Cargar previos si existen
    matriz_existente = []
    estudios_existentes = []
    if os.path.exists(matriz_path):
        with open(matriz_path, encoding="utf-8-sig", newline="") as f:
            matriz_existente = list(csv.DictReader(f))
    if os.path.exists(estudios_path):
        with open(estudios_path, encoding="utf-8-sig", newline="") as f:
            estudios_existentes = list(csv.DictReader(f))

    todas_matriz = matriz_existente + filas_matriz
    todas_estudios = estudios_existentes + filas_estudios

    guardar_csv(matriz_path, todas_matriz, fieldnames)
    guardar_csv(estudios_path, todas_estudios, fieldnames)

    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL DEL ESCANEO COMPLETO")
    print(f"{'='*60}")
    print(f"  Rango escaneado: filas 480-{total_registros}")
    print(f"  Total procesados: {len(registros)} (nuevos 751-{total_registros})")
    print(f"  MATRIZ DE RIESGOS: {len(todas_matriz)} total ({len(matriz_existente)} previos + {len(filas_matriz)} nuevos)")
    print(f"  ESTUDIOS PREVIOS:  {len(todas_estudios)} total ({len(estudios_existentes)} previos + {len(filas_estudios)} nuevos)")
    print(f"  Sin documento: {sin_doc}")
    print(f"  Errores: {errores}")
    print(f"  Sin URL: {sin_url}")
    print(f"  Omitidos Bogota: {len(omitidos_bogota)}")
    print(f"\nArchivos guardados:")
    print(f"  {matriz_path} ({len(todas_matriz)} registros)")
    print(f"  {estudios_path} ({len(todas_estudios)} registros)")

    # Mostrar contratos con matriz
    if todas_matriz:
        print(f"\n--- CONTRATOS CON MATRIZ DE RIESGOS ---")
        for r in todas_matriz:
            print(f"  Fila {r['fila_secop1_lite']} | {r['entidad'][:50]} | {r['municipio']}")

if __name__ == "__main__":
    separar_barrido_previo()
    asyncio.run(main_scan())
