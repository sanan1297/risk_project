"""Escanea registros 751-1560 de secop1_lite.csv.
Guarda resultados en matriz_riesgos.csv y estudios_previos.csv
(mergeando con los 10+60 del barrido 480-750).
"""
import asyncio
import csv
import os
import sys
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = os.path.join("estudio_data")
BARRIDO_PREVIO = os.path.join(OUT_DIR, "barrido_480_750.csv")
MATRIZ_CSV = os.path.join(OUT_DIR, "matriz_riesgos.csv")
ESTUDIOS_CSV = os.path.join(OUT_DIR, "estudios_previos.csv")

FIELD_NAMES = ["fila_secop1_lite", "url", "entidad", "municipio", "departamento", "tipo_documento"]

os.makedirs(OUT_DIR, exist_ok=True)

# --- PASO 1: Cargar previos ---
def cargar_previos():
    matriz, estudios = [], []
    with open(BARRIDO_PREVIO, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = row["tipo_documento"].strip()
            entry = {k: row.get(k, "") for k in FIELD_NAMES}
            if t == "MATRIZ_RIESGOS":
                matriz.append(entry)
            elif t == "ESTUDIOS_PREVIOS":
                estudios.append(entry)
    return matriz, estudios

# --- PASO 2: Escanear 751-1560 ---
async def escanear():
    total = 1560
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            fila = i + 1
            if fila >= 751:
                registros.append((fila, row))

    print(f"Escaneando {len(registros)} registros (751-{total})...")

    matriz_nuevos = []
    estudios_nuevos = []
    sin_doc = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="es-CO",
        )
        page = await context.new_page()

        for idx, (fila, row) in enumerate(registros):
            url = (row.get("url") or "").strip()
            entidad = (row.get("entidad") or "").strip()
            municipio = (row.get("municipio") or "").strip()
            depto = (row.get("departamento") or "").strip()

            if idx % 20 == 0:
                print(f"[{fila}/{total}] M={len(matriz_nuevos)} E={len(estudios_nuevos)} S={sin_doc}")

            if not url:
                sin_doc += 1
                continue

            # Skip Bogota
            mun_up = municipio.upper()
            if "BOGOTA" in mun_up and ("D.C" in mun_up or "DC" in mun_up):
                sin_doc += 1
                continue

            try:
                await page.goto(url, wait_until="networkidle", timeout=45000)
                await page.wait_for_timeout(1500)
            except:
                sin_doc += 1
                continue

            # Check page content for document types
            try:
                rows_tr = await page.query_selector_all("tr")
                tiene_matriz = False
                tiene_estudios = False
                for tr in rows_tr:
                    text = (await tr.inner_text()).strip().upper()
                    if not text:
                        continue
                    if "MATRIZ" in text and "RIESGOS" in text:
                        tiene_matriz = True
                    if "ESTUDIOS PREVIOS" in text or "ESTUDIO PREVIO" in text:
                        tiene_estudios = True
            except:
                sin_doc += 1
                continue

            entry = {
                "fila_secop1_lite": str(fila),
                "url": url,
                "entidad": entidad,
                "municipio": municipio,
                "departamento": depto,
            }

            if tiene_matriz:
                e = dict(entry, tipo_documento="MATRIZ_RIESGOS")
                matriz_nuevos.append(e)
                print(f"  [+] MATRIZ fila {fila} | {entidad[:45]}")
            elif tiene_estudios:
                e = dict(entry, tipo_documento="ESTUDIOS_PREVIOS")
                estudios_nuevos.append(e)
            else:
                sin_doc += 1

            await page.wait_for_timeout(800)

        await browser.close()

    return matriz_nuevos, estudios_nuevos, sin_doc

# --- MAIN ---
def main():
    print("Cargando datos previos (480-750)...")
    matriz_prev, estudios_prev = cargar_previos()
    print(f"  Previos: Matriz={len(matriz_prev)}, Estudios={len(estudios_prev)}")

    print("\nEscaneando registros 751-1560...")
    matriz_nuevos, estudios_nuevos, sin_doc = asyncio.run(escanear())

    # Merge
    todas_matriz = matriz_prev + matriz_nuevos
    todas_estudios = estudios_prev + estudios_nuevos

    # Guardar
    with open(MATRIZ_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(todas_matriz)

    with open(ESTUDIOS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(todas_estudios)

    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"  MATRIZ DE RIESGOS: {len(todas_matriz)} ({len(matriz_prev)} previos + {len(matriz_nuevos)} nuevos)")
    print(f"  ESTUDIOS PREVIOS:  {len(todas_estudios)} ({len(estudios_prev)} previos + {len(estudios_nuevos)} nuevos)")
    print(f"  Sin documento: {sin_doc}")
    print(f"\nArchivos:")
    print(f"  {MATRIZ_CSV}")
    print(f"  {ESTUDIOS_CSV}")

    if todas_matriz:
        print(f"\n=== MATRIZ DE RIESGOS ===")
        for r in todas_matriz:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:45]}")

if __name__ == "__main__":
    main()
