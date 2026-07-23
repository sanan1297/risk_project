"""Escaneo CORREGIDO con parallel requests.
Deteccion amplia en body text:
- MATRIZ: cualquier texto que contenga 'MATRIZ'
- MATRIZ_RIESGO: 'MATRIZ' + 'RIESGO'
- ESTUDIOS_PREVIOS: 'ESTUDIOS PREVIOS' o 'ESTUDIO PREVIO'
"""
import csv, math, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = "estudio_data"
FIELD_NAMES = ["fila_secop1_lite", "url", "entidad", "municipio", "departamento", "tipo"]
os.makedirs(OUT_DIR, exist_ok=True)

def clasificar(texto):
    upper = texto.upper()
    tiene_matriz = "MATRIZ" in upper
    tiene_riesgo = "RIESGO" in upper
    tiene_estudio_previo = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
    
    if tiene_matriz and tiene_riesgo:
        return "MATRIZ_RIESGO"
    elif tiene_matriz:
        return "MATRIZ"
    elif tiene_estudio_previo:
        return "ESTUDIOS_PREVIOS"
    return None

def fetch_page(url, timeout=20):
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                      "Accept-Language": "es-CO,es;q=0.9"})
    try:
        r = s.get(url, timeout=timeout)
        for _ in range(3):
            if "__zjc" not in r.text:
                return r.text
            m = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if not m:
                return None
            v = math.floor(int(m.group(1)) * float(m.group(2)))
            ck = re.search(r'cookie = "(\w+)="', r.text)
            if not ck:
                return None
            s.cookies.set(ck.group(1), str(v), domain="www.contratos.gov.co")
            rd = re.search(r"window\.location='([^']+)'", r.text)
            if not rd:
                return None
            ru = rd.group(1)
            if not ru.startswith("http"):
                ru = "https://www.contratos.gov.co" + ru
            time.sleep(0.2)
            r = s.get(ru, timeout=timeout)
        return r.text
    except Exception:
        return None

def procesar_fila(fila, row):
    url = (row.get("url") or "").strip()
    if not url:
        return None
    mun_up = (row.get("municipio") or "").upper()
    if "BOGOTA" in mun_up and ("D.C" in mun_up or "DC" in mun_up):
        return None
    
    html = fetch_page(url)
    if not html or len(html) < 500:
        return None
    
    tipo = clasificar(html)
    if not tipo:
        return None
    
    return {
        "fila_secop1_lite": str(fila),
        "url": url,
        "entidad": (row.get("entidad") or "").strip(),
        "municipio": (row.get("municipio") or "").strip(),
        "departamento": (row.get("departamento") or "").strip(),
        "tipo": tipo,
    }

def main():
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            fila = i + 1
            if 480 <= fila <= 1560:
                registros.append((fila, row))
    
    total_filas = 0
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            fila = i + 1
            if 480 <= fila <= 1560:
                mun = (row.get("municipio") or "").upper()
                if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                    continue
                total_filas += 1
    
    total = len(registros)
    print(f"Escaneando {total} registros con {os.cpu_count()} workers (CORREGIDO)...")
    print(f"Busqueda: MATRIZ (cualquiera), ESTUDIOS PREVIOS / ESTUDIO PREVIO\n")
    
    resultados = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=15) as ex:
        futuros = {ex.submit(procesar_fila, fila, row): fila for fila, row in registros}
        
        for i, futuro in enumerate(as_completed(futuros)):
            fila = futuros[futuro]
            try:
                r = futuro.result()
                if r:
                    resultados.append(r)
            except:
                pass
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                m = sum(1 for rr in resultados if rr["tipo"] in ("MATRIZ", "MATRIZ_RIESGO"))
                e = sum(1 for rr in resultados if rr["tipo"] == "ESTUDIOS_PREVIOS")
                print(f"  [{i+1}/{total}] M={m} E={e} | {elapsed:.0f}s")
    
    elapsed = time.time() - start
    
    # Remove duplicates (same fila + same tipo)
    vistos = set()
    unicos = []
    for r in resultados:
        key = (r["fila_secop1_lite"], r["tipo"])
        if key not in vistos:
            vistos.add(key)
            unicos.append(r)
    
    matrices = [r for r in unicos if r["tipo"] in ("MATRIZ", "MATRIZ_RIESGO")]
    estudios = [r for r in unicos if r["tipo"] == "ESTUDIOS_PREVIOS"]
    
    out_m = os.path.join(OUT_DIR, "matriz_detectados_corregido.csv")
    out_e = os.path.join(OUT_DIR, "estudios_detectados_corregido.csv")
    
    with open(out_m, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(matrices)
    
    with open(out_e, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(estudios)
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES (CORREGIDO)")
    print(f"{'='*60}")
    print(f"Tiempo: {elapsed:.0f}s")
    print(f"Total filas procesables: {total_filas}")
    print(f"")
    print(f"MATRIZ (cualquier label con MATRIZ):     {len(matrices)}")
    print(f"  - Con 'RIESGO' en el texto:           {sum(1 for r in matrices if r['tipo']=='MATRIZ_RIESGO')}")
    print(f"ESTUDIOS PREVIOS o ESTUDIO PREVIO:       {len(estudios)}")
    print(f"")
    print(f"Archivos:")
    print(f"  {out_m}")
    print(f"  {out_e}")
    
    if matrices:
        print(f"\n--- PRIMERAS 30 MATRICES ---")
        for r in matrices[:30]:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:40]} | {r['tipo']}")
        if len(matrices) > 30:
            print(f"  ... y {len(matrices)-30} mas")
    
    if estudios:
        print(f"\n--- PRIMEROS 20 ESTUDIOS ---")
        for r in estudios[:20]:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:40]}")

if __name__ == "__main__":
    main()
