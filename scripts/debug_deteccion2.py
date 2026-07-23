"""Deep debug: find how MATRIZ DE RIESGOS appears in SECOP HTML."""
import requests
import math
import re
import time

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

# C-001 from existing contracts
url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-1-185635"

r = s.get(url, timeout=30)
for attempt in range(3):
    if "__zjc" in r.text:
        match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
        if match:
            v = math.floor(int(match.group(1)) * float(match.group(2)))
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

# Search for MATRIZ in raw HTML
print("=== Searching for MATRIZ in HTML ===")
for m in re.finditer(r'.{0,100}MATRIZ.{0,200}', text, re.I):
    print(f"  Found at {m.start()}:")
    print(f"  {m.group()[:300]}")
    print("  ---")

print("\n=== Searching for RIESGOS in HTML ===")  
for m in re.finditer(r'.{0,100}RIESGOS.{0,200}', text, re.I):
    print(f"  Found at {m.start()}:")
    print(f"  {m.group()[:300]}")
    print("  ---")
    if m.start() > 50000:
        break

# Also check the document table structure
print("\n=== Document table section ===")
# Find rows that contain PDF links
pdf_rows = re.findall(r'<tr[^>]*>.*?consultaProceso.*?</tr>', text, re.I | re.DOTALL)
print(f"TR rows with consultaProceso: {len(pdf_rows)}")

# Look at labels near consultaProceso links
print("\n=== Labels near consultaProceso ===")
for m in re.finditer(r'consultaProceso\([^)]+\)[^<]*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)', text[:200000]):
    label = m.group(1).strip()
    print(f"  Label: '{label}'")
