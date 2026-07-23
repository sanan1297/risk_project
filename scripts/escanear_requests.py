"""Escaneo ultra-rapido 751-1560 usando requests (sin Playwright).
Maneja Zenedge cookie challenge.
"""
import csv
import math
import os
import re
import sys
import time
from datetime import datetime

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = "estudio_data"
FIELD_NAMES = ["fila_secop1_lite", "url", "entidad", "municipio", "departamento", "tipo_documento"]
os.makedirs(OUT_DIR, exist_ok=True)

def crear_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    })
    return s

def fetch_page(session, url, max_retry=3):
    """Fetch SECOP page with Zenedge handling."""
    for attempt in range(max_retry):
        try:
            r = session.get(url, timeout=30)
            
            # Handle Zenedge cookie challenge
            if "__zjc" in r.text and "cookie" in r.text:
                match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
                if match:
                    num = int(match.group(1))
                    factor = float(match.group(2))
                    v = math.floor(num * factor)
                    ck_match = re.search(r'cookie = "(\w+)="', r.text)
                    if ck_match:
                        session.cookies.set(ck_match.group(1), str(v), domain="www.contratos.gov.co")
                        rd_match = re.search(r"window\.location='([^']+)'", r.text)
                        if rd_match:
                            redirect_url = rd_match.group(1)
                            if not redirect_url.startswith("http"):
                                redirect_url = "https://www.contratos.gov.co" + redirect_url
                            time.sleep(0.5)
                            r = session.get(redirect_url, timeout=30)
                            return r.text
            else:
                return r.text
        except Exception as e:
            if attempt == max_retry - 1:
                return None
            time.sleep(1)
    return None

def detectar_tipo(html):
    text = html.upper()
    tiene_m = "MATRIZ DE RIESGOS" in text
    tiene_e = "ESTUDIOS PREVIOS" in text or "ESTUDIO PREVIO" in text
    if tiene_m:
        return "MATRIZ_RIESGOS"
    elif tiene_e:
        return "ESTUDIOS_PREVIOS"
    return None

def escanear():
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            fila = i + 1
            if fila >= 751:
                registros.append((fila, row))

    total = len(registros)
    print(f"Escaneando {total} registros con requests...")

    session = crear_session()
    resultados = []
    errores = 0
    start = time.time()

    for idx, (fila, row) in enumerate(registros):
        url = (row.get("url") or "").strip()
        entidad = (row.get("entidad") or "").strip()
        municipio = (row.get("municipio") or "").strip()
        depto = (row.get("departamento") or "").strip()

        if not url:
            errores += 1
            continue
        mun_up = municipio.upper()
        if "BOGOTA" in mun_up and ("D.C" in mun_up or "DC" in mun_up):
            errores += 1
            continue

        html = fetch_page(session, url)
        if html is None:
            errores += 1
            continue

        tipo = detectar_tipo(html)
        if tipo:
            resultados.append({
                "fila_secop1_lite": str(fila),
                "url": url,
                "entidad": entidad,
                "municipio": municipio,
                "departamento": depto,
                "tipo_documento": tipo,
            })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            m = sum(1 for r in resultados if r["tipo_documento"] == "MATRIZ_RIESGOS")
            e = sum(1 for r in resultados if r["tipo_documento"] == "ESTUDIOS_PREVIOS")
            rate = (idx + 1) / elapsed
            print(f"[{fila}/1560] {idx+1}/{total} err={errores} | M={m} E={e} | {rate:.1f} pag/s")

    return resultados, errores

def merge(nuevos, errores):
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

    nm = [r for r in nuevos if r["tipo_documento"] == "MATRIZ_RIESGOS"]
    ne = [r for r in nuevos if r["tipo_documento"] == "ESTUDIOS_PREVIOS"]

    # Merge: evitar duplicados por URL
    urls_m = {r["url"] for r in prev_m}
    urls_e = {r["url"] for r in prev_e}
    
    nm_filtrados = [r for r in nm if r["url"] not in urls_m]
    ne_filtrados = [r for r in ne if r["url"] not in urls_e]

    tm = prev_m + nm_filtrados
    te = prev_e + ne_filtrados

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
    print(f"Errores totales: {errores}")
    print(f"MATRIZ_RIESGOS: {len(tm)} ({len(prev_m)} prev + {len(nm_filtrados)} nuevo)")
    print(f"ESTUDIOS_PREVIOS: {len(te)} ({len(prev_e)} prev + {len(ne_filtrados)} nuevo)")
    print(f"\n{os.path.join(OUT_DIR, 'matriz_riesgos.csv')}")
    print(f"{os.path.join(OUT_DIR, 'estudios_previos.csv')}")

    if tm:
        print(f"\n--- MATRIZ DE RIESGOS ---")
        for r in tm:
            print(f"  Fila {r['fila_secop1_lite']:>4} | {r['entidad'][:45]} | {r['municipio']}")

if __name__ == "__main__":
    start = time.time()
    resultados, errores = escanear()
    elapsed = time.time() - start
    print(f"\nTiempo total: {elapsed:.1f}s ({len(resultados)} resultados, {errores} errores)")
    merge(resultados, errores)
