"""Scanner definitivo con detección de bloqueos y retry.
Estrategia:
1. Primera pasada: visita todas las URLs, clasifica como OK/BLOQUEADA/ERROR
2. Pasadas de retry: solo URLs bloqueadas, con fresh context y delay
3. Detección: MATRIZ (cualquiera), MATRIZ+RIESGO, ESTUDIOS PREVIOS
"""
import asyncio, csv, json, os, sys, time
from datetime import datetime
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = "estudio_data"
STATE_FILE = os.path.join(OUT_DIR, "estado_scan.json")
RESULT_FILE = os.path.join(OUT_DIR, "resultados_finales.csv")
MAX_PASSES = 3
PASS_DELAY = 5  # segundos entre pasadas

FIELD_NAMES = [
    "fila", "url", "entidad", "municipio", "departamento",
    "tipo", "pass", "body_len", "timestamp"
]

def clasificar(texto):
    upper = texto.upper()
    m = "MATRIZ" in upper
    r = "RIESGO" in upper
    e = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
    if m and r:
        return "MATRIZ_RIESGO"
    elif m:
        return "MATRIZ"
    elif e:
        return "ESTUDIOS_PREVIOS"
    return None

def cargar_estado():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"pendientes": [], "resultados": []}

def guardar_estado(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

async def escanear_pagina(page, url):
    try:
        await page.goto(url, timeout=25000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        body = await page.inner_text("body")
        blen = len(body)
        if blen < 500:
            return {"estado": "BLOQUEADA", "body_len": blen}
        tipo = clasificar(body)
        return {
            "estado": tipo if tipo else "SIN_DOCUMENTO",
            "body_len": blen,
        }
    except Exception as e:
        return {"estado": "ERROR", "error": str(e)[:80], "body_len": 0}

async def pasada(browser, pendientes, num_pass):
    nuevos_pendientes = []
    resultados_pasada = []
    creados = 0
    
    for item in pendientes:
        fila = item["fila"]
        row = item["row"]
        url = item["url"]
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="es-CO",
            timezone_id="America/Bogota",
            viewport={"width": 1920, "height": 1080},
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
            }
        )
        page = await context.new_page()
        
        try:
            result = await escanear_pagina(page, url)
            result["fila"] = fila
            result["url"] = url
            result["entidad"] = row.get("entidad", "").strip()
            result["municipio"] = row.get("municipio", "").strip()
            result["departamento"] = row.get("departamento", "").strip()
            result["pass"] = num_pass
            
            registros = []
            for r in (resultados_pasada + state.get("resultados", [])):
                if r["fila"] == fila:
                    continue
            
            if result["estado"] == "BLOQUEADA" and num_pass < MAX_PASSES:
                nuevos_pendientes.append(item)
                print(f"  P{num_pass} Fila {fila}: BLOQUEADA -> retry P{num_pass+1}")
            elif result["estado"] == "ERROR" and num_pass < MAX_PASSES:
                nuevos_pendientes.append(item)
                print(f"  P{num_pass} Fila {fila}: ERROR ({result.get('error','')[:30]}) -> retry")
            else:
                # Ya no hay mas retry para esta URL
                state["resultados"].append({
                    "fila": str(fila),
                    "url": url,
                    "entidad": result["entidad"],
                    "municipio": result["municipio"],
                    "departamento": result["departamento"],
                    "tipo": result["estado"] if result["estado"] in ("MATRIZ_RIESGO", "MATRIZ", "ESTUDIOS_PREVIOS", "SIN_DOCUMENTO", "BLOQUEADA", "ERROR") else result["estado"],
                    "pass": str(num_pass),
                    "body_len": str(result.get("body_len", 0)),
                    "timestamp": datetime.now().isoformat(),
                })
                creados += 1
                if result["estado"] in ("MATRIZ_RIESGO", "MATRIZ", "ESTUDIOS_PREVIOS"):
                    print(f"  P{num_pass} Fila {fila}: {result['estado']} ({result.get('body_len',0)} chars) ✅")
        except Exception as e:
            print(f"  P{num_pass} Fila {fila}: EXCEPTION {e}")
            if num_pass < MAX_PASSES:
                nuevos_pendientes.append(item)
        finally:
            await context.close()
    
    return nuevos_pendientes, resultados_pasada

async def main():
    state = cargar_estado()
    
    # Cargar registros de secop1_lite si es primera vez
    if not state["pendientes"] and not state["resultados"]:
        with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                fila = i + 1
                if fila < 480 or fila > 1560:
                    continue
                mun = (row.get("municipio") or "").upper()
                if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                    continue
                url = (row.get("url") or "").strip()
                if not url:
                    continue
                state["pendientes"].append({
                    "fila": fila,
                    "url": url,
                    "row": {
                        "entidad": (row.get("entidad") or "").strip(),
                        "municipio": (row.get("municipio") or "").strip(),
                        "departamento": (row.get("departamento") or "").strip(),
                    }
                })
        print(f"Cargados {len(state['pendientes'])} registros (480-1560, sin Bogota).")
        guardar_estado(state)
    
    total_pendientes = len(state["pendientes"])
    resultados_existentes = len(state["resultados"])
    print(f"Pendientes: {total_pendientes}")
    print(f"Resultados existentes: {resultados_existentes}")
    print(f"Pasadas maximas: {MAX_PASSES}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        
        for num_pass in range(1, MAX_PASSES + 1):
            pendientes = state["pendientes"]
            if not pendientes:
                print(f"\nNo hay pendientes. Completado en pasada {num_pass-1}.")
                break
            
            print(f"\n{'='*60}")
            print(f"PASADA {num_pass}/{MAX_PASSES} - {len(pendientes)} URLs pendientes")
            print(f"{'='*60}")
            start = time.time()
            
            nuevos_pendientes = []
            for idx, item in enumerate(pendientes):
                fila = item["fila"]
                url = item["url"]
                
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    locale="es-CO",
                    timezone_id="America/Bogota",
                    viewport={"width": 1920, "height": 1080},
                )
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=25000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)
                    body = await page.inner_text("body")
                    blen = len(body)
                    
                    if blen < 500:
                        print(f"  P{num_pass} Fila {fila}: BLOQUEADA ({blen}c)")
                        if num_pass < MAX_PASSES:
                            nuevos_pendientes.append(item)
                        else:
                            state["resultados"].append({
                                "fila": str(fila), "url": url,
                                "entidad": item["row"]["entidad"],
                                "municipio": item["row"]["municipio"],
                                "departamento": item["row"]["departamento"],
                                "tipo": "BLOQUEADA", "pass": str(num_pass),
                                "body_len": str(blen),
                                "timestamp": datetime.now().isoformat(),
                            })
                    else:
                        tipo = clasificar(body)
                        tipo_final = tipo if tipo else "SIN_DOCUMENTO"
                        state["resultados"].append({
                            "fila": str(fila), "url": url,
                            "entidad": item["row"]["entidad"],
                            "municipio": item["row"]["municipio"],
                            "departamento": item["row"]["departamento"],
                            "tipo": tipo_final, "pass": str(num_pass),
                            "body_len": str(blen),
                            "timestamp": datetime.now().isoformat(),
                        })
                        if tipo:
                            print(f"  P{num_pass} Fila {fila}: {tipo} ({blen}c)")
                except Exception as e:
                    print(f"  P{num_pass} Fila {fila}: ERROR: {str(e)[:40]}")
                    if num_pass < MAX_PASSES:
                        nuevos_pendientes.append(item)
                    else:
                        state["resultados"].append({
                            "fila": str(fila), "url": url,
                            "entidad": item["row"]["entidad"],
                            "municipio": item["row"]["municipio"],
                            "departamento": item["row"]["departamento"],
                            "tipo": "ERROR", "pass": str(num_pass),
                            "body_len": "0",
                            "timestamp": datetime.now().isoformat(),
                        })
                finally:
                    await context.close()
                
                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - start
                    done = sum(1 for r in state["resultados"] if r["pass"] == str(num_pass))
                    m = sum(1 for r in state["resultados"] if r["tipo"] in ("MATRIZ", "MATRIZ_RIESGO"))
                    e = sum(1 for r in state["resultados"] if r["tipo"] == "ESTUDIOS_PREVIOS")
                    b = sum(1 for r in state["resultados"] if r["tipo"] == "BLOQUEADA")
                    print(f"  P{num_pass} [{idx+1}/{len(pendientes)}] M={m} E={e} B={b} | {elapsed:.0f}s")
            
            state["pendientes"] = nuevos_pendientes
            guardar_estado(state)
            
            elapsed = time.time() - start
            if nuevos_pendientes and num_pass < MAX_PASSES:
                print(f"\n  Pasada {num_pass} completa en {elapsed:.0f}s.")
                print(f"  {len(nuevos_pendientes)} URLs bloqueadas. Esperando {PASS_DELAY}s antes de siguiente pasada...")
                await asyncio.sleep(PASS_DELAY)
        
        await browser.close()
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES")
    print(f"{'='*60}")
    
    results = state["resultados"]
    total = len(results)
    m = sum(1 for r in results if r["tipo"] in ("MATRIZ", "MATRIZ_RIESGO"))
    mr = sum(1 for r in results if r["tipo"] == "MATRIZ_RIESGO")
    e = sum(1 for r in results if r["tipo"] == "ESTUDIOS_PREVIOS")
    s = sum(1 for r in results if r["tipo"] == "SIN_DOCUMENTO")
    b = sum(1 for r in results if r["tipo"] == "BLOQUEADA")
    err = sum(1 for r in results if r["tipo"] == "ERROR")
    
    print(f"Total: {total}")
    print(f"MATRIZ (cualquiera):     {m}")
    print(f"  - Con RIESGO:         {mr}")
    print(f"ESTUDIOS PREVIOS:        {e}")
    print(f"SIN DOCUMENTO:           {s}")
    print(f"BLOQUEADA (sin retry):   {b}")
    print(f"ERROR:                   {err}")
    
    # Guardar CSV final
    with open(RESULT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in FIELD_NAMES})
    print(f"\nResultados guardados en: {RESULT_FILE}")
    
    # Mostrar matrices
    matrices = [r for r in results if r["tipo"] in ("MATRIZ", "MATRIZ_RIESGO")]
    if matrices:
        print(f"\n--- MATRICES DETECTADAS ({len(matrices)}) ---")
        for r in matrices[:30]:
            print(f"  Fila {r['fila']:>4} | {r['tipo']:13} | {r.get('entidad','')[:40]}")
        if len(matrices) > 30:
            print(f"  ... y {len(matrices)-30} mas")
    
    guardar_estado(state)

if __name__ == "__main__":
    asyncio.run(main())
