"""Escaneo 751-1560 con Playwright optimizado (domcontentloaded + 1.5s).
El scan anterior (slow) encontro 23 Matriz + 164 Estudios nuevos.
Este script solo procesa los no encontrados aun y mergea con previos.
"""
import asyncio
import csv
import os
import sys
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

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
    total_global = 1560
    print(f"Escaneando {total} registros (751-{total_global})...")

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
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1500)
            except:
                continue

            try:
                rows_tr = await page.query_selector_all("tr")
                tiene_m = False
                tiene_e = False
                for tr in rows_tr:
                    try:
                        t = (await tr.inner_text()).strip().upper()
                    except:
                        continue
                    if not t:
                        continue
                    if "MATRIZ DE RIESGOS" in t:
                        tiene_m = True
                    if "ESTUDIOS PREVIOS" in t or "ESTUDIO PREVIO" in t:
                        tiene_e = True
            except:
                continue

            tipo = None
            if tiene_m:
                tipo = "MATRIZ_RIESGOS"
            elif tiene_e:
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

            if (idx + 1) % 10 == 0:
                m = sum(1 for r in resultados if r["tipo_documento"] == "MATRIZ_RIESGOS")
                e = sum(1 for r in resultados if r["tipo_documento"] == "ESTUDIOS_PREVIOS")
                print(f"[{fila}/{total_global}] {idx+1}/{total} M={m} E={e}")

        await browser.close()
    return resultados

def merge(nuevos):
    prev_m, prev_e = [], []
    bp = os.path.join(OUT_DIR, "barrido_480_750.csv")
    if os.path.exists(bp):
        with open(bp, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                t = row["tipo_documento"].strip()
                e = {k: row.get(k, "") for k in FIELD_NAMES}
                if t == "MATRIZ_RIESGOS":
                    prev_m.append(e)
                elif t == "ESTUDIOS_PREVIOS":
                    prev_e.append(e)

    # Previos guardados de ejecuciones anteriores (evitar duplicados)
    prev_matriz_path = os.path.join(OUT_DIR, "matriz_riesgos.csv")
    prev_estudios_path = os.path.join(OUT_DIR, "estudios_previos.csv")
    
    prev_m_path = []
    prev_e_path = []
    if os.path.exists(prev_matriz_path):
        with open(prev_matriz_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                prev_m_path.append(row["url"])
    if os.path.exists(prev_estudios_path):
        with open(prev_estudios_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                prev_e_path.append(row["url"])
    
    all_prev_m_urls = set(prev_m_path)
    all_prev_e_urls = set(prev_e_path)

    nm = [r for r in nuevos if r["tipo_documento"] == "MATRIZ_RIESGOS" and r["url"] not in all_prev_m_urls]
    ne = [r for r in nuevos if r["tipo_documento"] == "ESTUDIOS_PREVIOS" and r["url"] not in all_prev_e_urls]

    tm = prev_m + nm
    te = prev_e + ne

    with open(os.path.join(OUT_DIR, "matriz_riesgos.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(tm)

    with open(os.path.join(OUT_DIR, "estudios_previos.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(te)

    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"MATRIZ_RIESGOS: {len(tm)} total ({len(prev_m)} prev 480-750 + {len(nm)} nuevo 751-1560)")
    print(f"ESTUDIOS_PREVIOS: {len(te)} total ({len(prev_e)} prev 480-750 + {len(ne)} nuevo 751-1560)")
    print(f"\n{os.path.join(OUT_DIR, 'matriz_riesgos.csv')}")
    print(f"{os.path.join(OUT_DIR, 'estudios_previos.csv')}")

    if tm:
        print(f"\n--- MATRIZ DE RIESGOS ---")
        for r in tm:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:45]} | {r['municipio']}")

if __name__ == "__main__":
    nuevos = asyncio.run(escanear())
    merge(nuevos)
