"""Verify what the scanner actually detects on SECOP pages.
Tests 3 known-MATRIZ pages + 3 random from 480-1560 range."""
import asyncio
from playwright.async_api import async_playwright
import csv
import os

# Known contracts from matriz_clean that have MATRIZ DE RIESGOS label
known_matriz = [
    ("C-002", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-7462"),
    ("C-004", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-4526"),
    ("C-009", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-5903"),
]

# Random from 480-1560 range
SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
random_test = []
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if 480 <= fila <= 1560:
            url = (row.get("url") or "").strip()
            mun = (row.get("municipio") or "").upper()
            depto = (row.get("departamento") or "").strip()
            if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                continue
            random_test.append((fila, url, row.get("entidad",""), depto))
            if len(random_test) >= 10:
                break

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        print("=" * 60)
        print("TEST 1: Paginas CONOCIDAS con MATRIZ DE RIESGOS")
        print("=" * 60)
        for cid, url in known_matriz:
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                text = await page.inner_text("body")
                upper = text.upper()
                
                m1 = "MATRIZ DE RIESGOS" in upper
                m2 = "MATRIZ" in upper
                m3 = "RIESGOS" in upper
                e1 = "ESTUDIOS PREVIOS" in upper
                e2 = "ESTUDIO PREVIO" in upper
                
                print(f"\n{cid}:")
                print(f"  MATRIZ DE RIESGOS: {m1}")
                print(f"  MATRIZ: {m2} ({upper.count('MATRIZ')} veces)")
                print(f"  RIESGOS: {m3} ({upper.count('RIESGOS')} veces)")
                print(f"  ESTUDIOS PREVIOS: {e1}")
                print(f"  ESTUDIO PREVIO: {e2}")
            except Exception as ex:
                print(f"\n{cid}: ERROR {ex}")
        
        print("\n" + "=" * 60)
        print("TEST 2: Muestreo filas 480-1560")
        print("=" * 60)
        for fila, url, entidad, depto in random_test:
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                text = await page.inner_text("body")
                upper = text.upper()
                
                m1 = "MATRIZ DE RIESGOS" in upper
                m2 = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
                m_count = upper.count("MATRIZ")
                r_count = upper.count("RIESGOS")
                
                print(f"\nFila {fila} | {entidad[:40]} | {depto}")
                print(f"  MATRIZ DE RIESGOS: {m1}")
                print(f"  ESTUDIOS PREVIOS: {m2}")
                print(f"  'MATRIZ' aparece {m_count} veces")
                print(f"  'RIESGOS' aparece {r_count} veces")
                
                # Find context for MATRIZ/RIESGOS
                if m_count > 0 or r_count > 0:
                    import re
                    for kw in ["MATRIZ", "RIESGOS"]:
                        for m in re.finditer(r'.{0,40}' + kw + r'.{0,80}', text, re.I):
                            print(f"    [{kw}] ...{m.group()[:100]}...")
                            
            except Exception as ex:
                print(f"\nFila {fila}: ERROR {ex}")
        
        await browser.close()

asyncio.run(main())
