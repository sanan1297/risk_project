"""Debug: Check SECOP pages that we KNOW have matriz de riesgos
(from the existing 351 contracts in matriz_clean.csv).
"""
import pandas as pd
import requests
import math
import re
import time

# Load existing contracts
df = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
urls = df[["id_contrato", "url"]].drop_duplicates()

print(f"Total contracts in matriz_clean: {len(urls)}")
print()

# Check what the URLs look like
print("Sample URLs from existing contracts:")
for _, r in urls.head(5).iterrows():
    print(f"  {r['id_contrato']}: {r['url']}")

# Now check the SECOP1 lite URLs for the same contracts
s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})

print("\n--- Checking contract C-001 page ---")
url = urls.iloc[0]["url"]
print(f"URL: {url}")

try:
    r = s.get(url, timeout=30)
    print(f"Status: {r.status_code}, Length: {len(r.text)}")
    
    # Handle Zenedge
    for attempt in range(3):
        if "__zjc" in r.text:
            match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if match:
                num, factor = int(match.group(1)), float(match.group(2))
                v = math.floor(num * factor)
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
    
    text = r.text
    print(f"After Zenedge: {len(text)} bytes")
    
    # Check for document indicators
    upper = text.upper()
    keywords = ["MATRIZ DE RIESGOS", "MATRIZ", "RIESGOS", "ESTUDIOS PREVIOS", "ESTUDIO PREVIO",
                "PDF", "Documento", "documento", "archivo"]
    for kw in keywords:
        count = upper.count(kw.upper())
        if count > 0:
            # Find context around keyword
            idx = upper.find(kw.upper())
            ctx = text[max(0,idx-100):idx+200]
            print(f"  '{kw}': {count} veces | ctx: ...{ctx[:150]}...")
    
    # Check for document links
    links = re.findall(r'consultaProceso\([^)]+\)', text)
    print(f"\nTotal consultaProceso links: {len(links)}")
    
    # Check for specific document types in the labels
    doc_labels = re.findall(r'<td[^>]*>\s*(MATRIZ\s*DE\s*RIESGOS|ESTUDIOS\s*PREVIOS|ESTUDIO\s*PREVIO)\s*</td>', text, re.I)
    print(f"Document section labels found: {doc_labels}")
    
    # Also check for href-based document links
    pdfs = re.findall(r'consultaProceso\([^)]*\.pdf', text, re.I)
    print(f"PDF links in consultaProceso: {len(pdfs)}")
    
except Exception as e:
    print(f"Error: {e}")
