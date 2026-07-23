"""Compare requests vs Playwright on C-009 and a few others."""
import asyncio
import csv
import os
import re
import math
import time
import requests
from playwright.async_api import async_playwright

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = "estudio_data"

def fetch_requests(url):
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    r = s.get(url, timeout=30)
    for _ in range(3):
        if "__zjc" in r.text:
            m = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if m:
                v = math.floor(int(m.group(1)) * float(m.group(2)))
                ck = re.search(r'cookie = "(\w+)="', r.text)
                if ck:
                    s.cookies.set(ck.group(1), str(v), domain="www.contratos.gov.co")
                    rd = re.search(r"window\.location='([^']+)'", r.text)
                    if rd:
                        ru = rd.group(1)
                        if not ru.startswith("http"):
                            ru = "https://www.contratos.gov.co" + ru
                        time.sleep(0.5)
                        r = s.get(ru, timeout=30)
                        continue
        break
    return r.text

# Get C-009 URL from matriz_clean
import pandas as pd
df = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
urls = df[["id_contrato", "url"]].drop_duplicates()
c009_url = urls[urls["id_contrato"] == "C-009"]["url"].values[0]

# Also get some from secop1_lite
test_urls = [
    ("C-009 (desde matriz_clean)", c009_url),
    ("C-002 (desde matriz_clean)", urls[urls["id_contrato"] == "C-002"]["url"].values[0]),
]

# Add 8 random from 480-1560
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if 480 <= fila <= 1560:
            url = (row.get("url") or "").strip()
            mun = (row.get("municipio") or "").upper()
            if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                continue
            test_urls.append((f"Fila {fila} ({row.get('entidad','')[:30]})", url))
            if len(test_urls) >= 12:
                break

async def analyze_with_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        
        # Method A: inner_text of body
        body_text = await page.inner_text("body")
        
        # Method B: text_content of body
        body_tc = await page.evaluate("() => document.body.textContent")
        
        # Method C: Raw HTML via page.content
        html = await page.content()
        
        # Method D: Evaluate all tr text
        tr_texts = await page.evaluate("""() => {
            const rows = document.querySelectorAll('tr');
            return Array.from(rows).map(r => r.innerText.trim()).filter(t => t.length > 0);
        }""")
        
        await browser.close()
        return body_text, body_tc, html, tr_texts

print("=" * 80)
print("COMPARACION REQUESTS vs PLAYWRIGHT")
print("=" * 80)

for label, url in test_urls:
    if not url:
        continue
    print(f"\n{'='*60}")
    print(f"URL: {label}")
    print(f"     {url}")
    
    # 1. Requests method
    html_r = fetch_requests(url)
    upper_r = html_r.upper()
    r_matriz = "MATRIZ DE RIESGOS" in upper_r
    r_estudios = "ESTUDIOS PREVIOS" in upper_r or "ESTUDIO PREVIO" in upper_r
    r_matriz_any = "MATRIZ" in upper_r
    r_riesgos = "RIESGOS" in upper_r
    r_labels = re.findall(r'consultaProceso\([^)]+\)[^<]*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)', html_r)
    r_doc_labels = [l.strip() for l in r_labels if "MATRIZ" in l.upper() or "RIESGO" in l.upper() or "ESTUDIO" in l.upper()]
    
    print(f"  [REQUESTS] MATRIZ_RIESGOS={r_matriz} ESTUDIOS_PREVIOS={r_estudios} MATRIZ_ANY={r_matriz_any} RIESGOS={r_riesgos}")
    if r_doc_labels:
        for l in r_doc_labels:
            print(f"    -> '{l}'")
    
    # 2. Playwright methods
    try:
        body_text, body_tc, html_pw, tr_texts = asyncio.run(analyze_with_playwright(url))
        upper_pw = body_text.upper()
        upper_tc = body_tc.upper()
        upper_html = html_pw.upper()
        
        pw_m = "MATRIZ DE RIESGOS" in upper_pw
        pw_e = "ESTUDIOS PREVIOS" in upper_pw or "ESTUDIO PREVIO" in upper_pw
        tc_m = "MATRIZ DE RIESGOS" in upper_tc
        tc_e = "ESTUDIOS PREVIOS" in upper_tc or "ESTUDIO PREVIO" in upper_tc
        html_m = "MATRIZ DE RIESGOS" in upper_html
        html_e = "ESTUDIOS PREVIOS" in upper_html or "ESTUDIO PREVIO" in upper_html
        
        # Check tr texts for MATRIZ
        tr_matches = [t for t in tr_texts if "MATRIZ" in t.upper() or "RIESGO" in t.upper()]
        
        print(f"  [PW-innerText] MATRIZ_RIESGOS={pw_m} ESTUDIOS_PREVIOS={pw_e}")
        print(f"  [PW-textContent] MATRIZ_RIESGOS={tc_m} ESTUDIOS_PREVIOS={tc_e}")
        print(f"  [PW-pageHTML]   MATRIZ_RIESGOS={html_m} ESTUDIOS_PREVIOS={html_e}")
        if tr_matches:
            for t in tr_matches:
                print(f"    TR: {t[:120]}")
        
        # C: Summary
        print(f"  => {label}: requests_M={r_matriz} pw_M={pw_m} html_M={html_m}")
    except Exception as ex:
        print(f"  [PW] ERROR: {ex}")
