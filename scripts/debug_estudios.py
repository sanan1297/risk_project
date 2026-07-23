"""Debug: How do ESTUDIOS PREVIOS labels appear in SECOP HTML?
Check multiple contracts from the existing 351."""
import pandas as pd
import requests
import math
import re
import time
from collections import Counter

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

df = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
urls = df[["id_contrato", "url"]].drop_duplicates()

def fetch_page(url):
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

# Check first 20 contracts from the existing dataset
all_labels = Counter()
estudios_count = 0
matriz_count = 0
total = 0

for _, row in urls.head(50).iterrows():
    cid = row["id_contrato"]
    url = row["url"]
    total += 1
    
    html = fetch_page(url)
    text_upper = html.upper()
    
    # Check for document labels near consultaProceso
    labels = re.findall(r'consultaProceso\([^)]+\)[^<]*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)', html)
    
    has_estudios = False
    has_matriz_label = False
    has_estudios_label = False
    
    for label in labels:
        lbl = label.strip()
        all_labels[lbl] += 1
        if "MATRIZ" in lbl.upper() and "RIESGO" in lbl.upper():
            has_matriz_label = True
        if "ESTUDIO" in lbl.upper() and "PREVIO" in lbl.upper():
            has_estudios_label = True
        if "ESTUDIO" in lbl.upper() and "PREVIO" in lbl.upper():
            has_estudios = True
    
    if has_matriz_label:
        matriz_count += 1
    if has_estudios_label:
        estudios_count += 1

    if total <= 10:
        print(f"\n{cid}:")
        print(f"  Matriz label: {has_matriz_label}, Estudios label: {has_estudios_label}")
        for lbl in labels:
            l = lbl.strip()
            if "MATRIZ" in l.upper() or "RIESGO" in l.upper() or "ESTUDIO" in l.upper():
                print(f"    -> '{l}'")

print(f"\n\n=== RESUMEN (primeros {total}) ===")
print(f"Contratos con label MATRIZ: {matriz_count}")
print(f"Contratos con label ESTUDIOS PREVIOS: {estudios_count}")

print(f"\n=== TOP 30 LABELS ===")
for label, count in all_labels.most_common(30):
    print(f"  {count:4d}x | {label}")
