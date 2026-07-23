"""Escanea registros 480-750 de secop1_lite.csv buscando si tienen
MATRIZ DE RIESGOS o ESTUDIOS PREVIOS. Genera CSV con resultados.
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
REPORTE_CSV = os.path.join("estudio_data", "barrido_480_750.csv")
START_INDEX = 479
END_INDEX = 749

os.makedirs(OUT_DIR, exist_ok=True)

def cargar_registros():
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < START_INDEX:
                continue
            if i > END_INDEX:
                break
            registros.append((i + 1, row))  # 1-based row number
    return registros

def debe_omitir(row):
    mun = (row.get("municipio", "") or "").strip().upper()
    return "BOGOTA" in mun and ("D.C" in mun or "DC" in mun)

async def analizar_pagina(page, url):
    """Navega a la pagina SECOP y extrae info de documentos."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(2000)
    except PwTimeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:80]}

    content = await page.content()

    # Buscar labels en la tabla de documentos
    resultado = {
        "tiene_matriz_riesgos": False,
        "tiene_estudios_previos": False,
        "n_documentos": 0,
        "documentos": [],
        "error": None,
    }

    # Extraer todos los links consultaProceso con su label asociado
    rows = await page.query_selector_all("tr")
    for tr in rows:
        try:
            text = (await tr.inner_text()).strip()
        except:
            continue

        if "MATRIZ DE RIESGOS" in text.upper() and "MATRIZ" in text.upper():
            resultado["tiene_matriz_riesgos"] = True

        if "ESTUDIOS PREVIOS" in text.upper() or "ESTUDIO PREVIO" in text.upper():
            resultado["tiene_estudios_previos"] = True

        # Contar documentos (links que terminan en .pdf)
        pdfs = re.findall(r'consultaProceso\([^)]+\.pdf', text)
        if pdfs:
            resultado["n_documentos"] += len(pdfs)
            # Extraer nombre corto del PDF
            for p in pdfs:
                nombre_match = re.search(r'/([^/]+\.pdf)', p)
                if nombre_match:
                    resultado["documentos"].append(nombre_match.group(1))

    return resultado

async def main():
    registros = cargar_registros()
    print(f"Cargados {len(registros)} registros (filas {START_INDEX+1}-{END_INDEX+1})")

    omitidos = [r for r in registros if debe_omitir(r[1])]
    registros = [r for r in registros if not debe_omitir(r[1])]
    print(f"Omitidos Bogota D.C.: {len(omitidos)}")

    filas_reporte = []

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

            print(f"\n[{fila}] {entidad[:45]} | {municipio}")

            if not url:
                print(f"  -> Sin URL")
                filas_reporte.append([fila, "", entidad, municipio, depto, "SIN_URL", "", ""])
                continue

            print(f"  URL: {url[:75]}...")
            info = await analizar_pagina(page, url)

            if info.get("error"):
                print(f"  -> Error: {info['error']}")
                filas_reporte.append([fila, url, entidad, municipio, depto, "ERROR", info["error"], ""])
                continue

            docs = "; ".join(info["documentos"][:5])
            print(f"  -> MatrizRiesgos={'SI' if info['tiene_matriz_riesgos'] else 'no'} | EstudiosPrevios={'SI' if info['tiene_estudios_previos'] else 'no'} | Docs={info['n_documentos']}")

            # Determinar tipo
            if info["tiene_matriz_riesgos"]:
                tipo = "MATRIZ_RIESGOS"
            elif info["tiene_estudios_previos"]:
                tipo = "ESTUDIOS_PREVIOS"
            else:
                tipo = "SIN_DOCUMENTO"

            filas_reporte.append([fila, url, entidad, municipio, depto, tipo, info["n_documentos"], docs])

            # Esperar entre paginas
            await page.wait_for_timeout(1500)

        await browser.close()

    # Guardar CSV
    with open(REPORTE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fila_secop1_lite", "url", "entidad", "municipio", "departamento",
                         "tipo_documento", "n_documentos", "nombres_pdf"])
        writer.writerows(filas_reporte)

    print(f"\n{'='*60}")
    print(f"REPORTE GUARDADO EN: {REPORTE_CSV}")
    print(f"{'='*60}")

    # Resumen
    total = len(filas_reporte)
    con_matriz = sum(1 for r in filas_reporte if r[5] == "MATRIZ_RIESGOS")
    con_estudios = sum(1 for r in filas_reporte if r[5] == "ESTUDIOS_PREVIOS")
    sin_doc = sum(1 for r in filas_reporte if r[5] == "SIN_DOCUMENTO")
    errores = sum(1 for r in filas_reporte if r[5] == "ERROR")
    sin_url = sum(1 for r in filas_reporte if r[5] == "SIN_URL")

    print(f"\nRESUMEN:")
    print(f"  Total escaneados: {total}")
    print(f"  Con MATRIZ DE RIESGOS: {con_matriz}")
    print(f"  Con ESTUDIOS PREVIOS: {con_estudios}")
    print(f"  Sin documento relevante: {sin_doc}")
    print(f"  Errores: {errores}")
    print(f"  Sin URL: {sin_url}")

    # Mostrar los que tienen matriz de riesgos
    if con_matriz:
        print(f"\nCONTRATOS CON MATRIZ DE RIESGOS (para descargar primero):")
        for r in filas_reporte:
            if r[5] == "MATRIZ_RIESGOS":
                print(f"  Fila {r[0]} | {r[2][:45]} | {r[3]}")

if __name__ == "__main__":
    asyncio.run(main())
