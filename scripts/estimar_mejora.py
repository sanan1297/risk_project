"""Estimar cuantos mas contratos se detectarian con la deteccion amplia.
Toma una muestra de 50 contratos de 480-1560 y verifica cada uno con ambos criterios."""
import asyncio, csv, os, random
from playwright.async_api import async_playwright

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")

random.seed(42)

# Cargar todos los registros
todos = []
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if 480 <= fila <= 1560:
            mun = (row.get("municipio") or "").upper()
            if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                continue
            todos.append((fila, row))

# Muestra de 50
muestra = random.sample(todos, min(50, len(todos)))

print(f"Total registros: {len(todos)}")
print(f"Tamano muestra: {len(muestra)}")
print()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        resultados = []
        errores = 0
        
        for idx, (fila, row) in enumerate(muestra):
            url = (row.get("url") or "").strip()
            entidad = (row.get("entidad") or "").strip()
            depto = (row.get("departamento") or "").strip()
            
            try:
                await page.goto(url, timeout=25000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                body = await page.inner_text("body")
            except:
                errores += 1
                continue
            
            upper = body.upper()
            
            # Criterio ANTIGUO
            ant_matriz = "MATRIZ DE RIESGOS" in upper
            ant_estudios = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
            ant_tipo = None
            if ant_matriz:
                ant_tipo = "MATRIZ_RIESGOS"
            elif ant_estudios:
                ant_tipo = "ESTUDIOS_PREVIOS"
            
            # Criterio NUEVO
            tiene_matriz = "MATRIZ" in upper
            tiene_riesgo = "RIESGO" in upper
            tiene_ep = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
            
            nuevo_tipo = None
            if tiene_matriz and tiene_riesgo:
                nuevo_tipo = "MATRIZ_RIESGO"
            elif tiene_matriz:
                nuevo_tipo = "MATRIZ"
            elif tiene_ep:
                nuevo_tipo = "ESTUDIOS_PREVIOS"
            
            resultados.append({
                "fila": fila,
                "entidad": entidad,
                "depto": depto,
                "antiguo": ant_tipo or "NINGUNO",
                "nuevo": nuevo_tipo or "NINGUNO",
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(muestra)}] err={errores}")
        
        await browser.close()
        return resultados, errores

resultados, errores = asyncio.run(main())

# Resumen
print(f"\n{'='*60}")
print(f"RESULTADOS MUESTRA ({len(resultados)} contratos)")
print(f"{'='*60}")

ant_m = sum(1 for r in resultados if r["antiguo"] in ("MATRIZ_RIESGOS",))
ant_e = sum(1 for r in resultados if r["antiguo"] == "ESTUDIOS_PREVIOS")
ant_total = sum(1 for r in resultados if r["antiguo"] != "NINGUNO")

nue_m = sum(1 for r in resultados if r["nuevo"] in ("MATRIZ", "MATRIZ_RIESGO"))
nue_mr = sum(1 for r in resultados if r["nuevo"] == "MATRIZ_RIESGO")
nue_e = sum(1 for r in resultados if r["nuevo"] == "ESTUDIOS_PREVIOS")
nue_total = sum(1 for r in resultados if r["nuevo"] != "NINGUNO")

print(f"Deteccion ANTIGUA (exacto 'MATRIZ DE RIESGOS'):")
print(f"  MATRIZ_RIESGOS={ant_m} ESTUDIOS_PREVIOS={ant_e} TOTAL={ant_total}")
print(f"Deteccion NUEVA (cualquier 'MATRIZ' o 'ESTUDIO PREVIO'):")
print(f"  MATRIZ={nue_m} (con_RIESGO={nue_mr}) ESTUDIOS_PREVIOS={nue_e} TOTAL={nue_total}")
print(f"")
print(f"Mejora: {ant_total} -> {nue_total} ({(nue_total/ant_total*100) if ant_total else 'N/A'}%)")

print(f"\n--- DETALLES (nuevo detecta pero antiguo no) ---")
for r in resultados:
    if r["antiguo"] == "NINGUNO" and r["nuevo"] != "NINGUNO":
        print(f"  Fila {r['fila']:>4} | ant={r['antiguo']:15} -> nue={r['nuevo']:15} | {r['entidad'][:40]}")

print(f"\n--- DETALLES (ambos detectan) ---")
for r in resultados:
    if r["antiguo"] != "NINGUNO" and r["nuevo"] != "NINGUNO":
        print(f"  Fila {r['fila']:>4} | ant={r['antiguo']:15} -> nue={r['nuevo']:15} | {r['entidad'][:40]}")
