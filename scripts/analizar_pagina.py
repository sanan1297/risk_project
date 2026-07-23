"""Analyze SECOP page for download URLs."""
import requests
import re
import math

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

constancia = "18-1-197178"

# Get detail page
resp = s.get(
    f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}",
    timeout=30
)

# Handle Zenedge
if "__zjc" in resp.text:
    match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', resp.text)
    if match:
        num = int(match.group(1))
        factor = float(match.group(2))
        v = math.floor(num * factor)
        cookie_match = re.search(r'cookie = "(\w+)="', resp.text)
        if cookie_match:
            cookie_name = cookie_match.group(1)
            s.cookies.set(cookie_name, str(v), domain="www.contratos.gov.co")
            redirect_match = re.search(r"window\.location='([^']+)'", resp.text)
            if redirect_match:
                redirect_url = redirect_match.group(1)
                if not redirect_url.startswith("http"):
                    redirect_url = "https://www.contratos.gov.co" + redirect_url
                resp = s.get(redirect_url, timeout=30)

text = resp.text
print(f"Page size: {len(text)}")

# Find the document table rows
# Pattern: MATRIZ DE RIESGOS followed by documents
sections = re.findall(
    r'<td[^>]*>(MATRIZ\s*DE\s*RIESGOS|ESTUDIOS\s*PREVIOS|ESTUDIO\s*PREVIO|ESTUDIOS\s*Y\s*DOCUMENTOS\s*PREVIOS)</td>',
    text, re.I
)
print(f"\nSecciones encontradas: {sections}")

# Find ALL consultaProceso calls with their surrounding context
pattern = r'consultaProceso\([^)]+\)'
for i, m in enumerate(re.finditer(pattern, text)):
    start = max(0, m.start() - 200)
    end = min(len(text), m.end() + 50)
    context = text[start:end]
    # Extract the label before this call
    label_match = re.search(r'<td[^>]*>\s*([^<]+?)\s*</td>', context[:200])
    label = label_match.group(1).strip() if label_match else "NO LABEL"
    
    # Check if it's a MATRIZ DE RIESGOS section by looking backward
    is_matriz = 'MATRIZ' in context[:200].upper()
    is_estudio = 'ESTUDIO' in context[:200].upper()
    
    if is_matriz or is_estudio or i < 5:
        print(f"\n--- Documento {i} ---")
        print(f"  Label cercano: {label[:80]}")
        print(f"  Es Matriz: {is_matriz}, Es Estudio: {is_estudio}")
        print(f"  Link: {m.group()[:200]}")
