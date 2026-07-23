"""Escaneo rapido 751-1560, solo detecta MATRIZ RIESGOS o ESTUDIOS PREVIOS.
Sin esperas largas entre paginas.
"""
import asyncio
import csv
import os
import sys
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = "estudio_data"
FIELD_NAMES = ["fila_secop1_lite", "url", "entidad", "municipio", "departamento", "tipo_documento"]

os.makedirs(OUT_DIR, exist_ok=True)

async def escanear():
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            fila = i + 1
            if fila >= 751:
                registros.append((fila, row))

    total = len(registros)
    print(f"Escaneando {total} registros (751-1560)...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="es-CO",
        )
        page = await context.new_page()

        resultados = []

        for idx, (fila, row) in enumerate(registros):
            url = (row.get("url") or "").strip()
            entidad = (row.get("entidad") or "").strip()
            municipio = (row.get("municipio") or "").strip()
            depto = (row.get("departamento") or "").strip()

            if not url:
                continue

            mun_up = municipio.upper()
            if "BOGOTA" in mun_up and ("D.C" in mun_up or "DC" in mun_up):
                continue

            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(500)
            except:
                continue

            try:
                text = (await page.inner_text("body")).upper()
            except:
                continue

            tipo = None
            if "MATRIZ DE RIESGOS" in text and "MATRIZ" in text:
                tipo = "MATRIZ_RIESGOS"
            elif "ESTUDIOS PREVIOS" in text or "ESTUDIO PREVIO" in text:
                tipo = "ESTUDIOS_PREVIOS"

            if tipo:
                resultados.append({
                    "fila_secop1_lite": str(fila),
                    "url": url,
                    "entidad": entidad,
                    "municipio": municipio,
                    "departamento": depto,
                    "tipo_documento": tipo,
                })

            if (idx + 1) % 30 == 0:
                m = sum(1 for r in resultados if r["tipo_documento"] == "MATRIZ_RIESGOS")
                e = sum(1 for r in resultados if r["tipo_documento"] == "ESTUDIOS_PREVIOS")
                print(f"[{fila}/1560] {idx+1}/{total} | M={m} E={e}")

        await browser.close()

    return resultados

def merge_con_previos(nuevos):
    # Cargar previos desde barrido_480_750.csv
    prev_matriz, prev_estudios = [], []
    barrido_path = os.path.join(OUT_DIR, "barrido_480_750.csv")
    if os.path.exists(barrido_path):
        with open(barrido_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                t = row["tipo_documento"].strip()
                entry = {k: row.get(k, "") for k in FIELD_NAMES}
                if t == "MATRIZ_RIESGOS":
                    prev_matriz.append(entry)
                elif t == "ESTUDIOS_PREVIOS":
                    prev_estudios.append(entry)

    # Clasificar nuevos
    nuevos_m = [r for r in nuevos if r["tipo_documento"] == "MATRIZ_RIESGOS"]
    nuevos_e = [r for r in nuevos if r["tipo_documento"] == "ESTUDIOS_PREVIOS"]

    todas_m = prev_matriz + nuevos_m
    todas_e = prev_estudios + nuevos_e

    with open(os.path.join(OUT_DIR, "matriz_riesgos.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(todas_m)

    with open(os.path.join(OUT_DIR, "estudios_previos.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(todas_e)

    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"MATRIZ_RIESGOS: {len(todas_m)} total ({len(prev_matriz)} prev 480-750 + {len(nuevos_m)} nuevo 751-1560)")
    print(f"ESTUDIOS_PREVIOS: {len(todas_e)} total ({len(prev_estudios)} prev 480-750 + {len(nuevos_e)} nuevo 751-1560)")
    print(f"\n{os.path.join(OUT_DIR, 'matriz_riesgos.csv')}")
    print(f"{os.path.join(OUT_DIR, 'estudios_previos.csv')}")

    if todas_m:
        print(f"\n--- MATRIZ DE RIESGOS ---")
        for r in todas_m:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:45]}")

if __name__ == "__main__":
    print("Paso 1: Separando barrido 480-750 en temporales...")
    print("Paso 2: Escaneando 751-1560...")
    nuevos = asyncio.run(escanear())
    merge_con_previos(nuevos)
