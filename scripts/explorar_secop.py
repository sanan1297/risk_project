"""Explorar estructura HTML de SECOP para encontrar enlaces a documentos."""
import requests
import re
from html.parser import HTMLParser

url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-1-197178"
r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
r.encoding = "utf-8"
text = r.text

# Search for sections
sections = re.findall(
    r'(MATRIZ\s*DE\s*RIESGOS|ESTUDIOS\s*PREVIOS|ESTUDIO\s*PREVIO)',
    text, re.I
)
print("Secciones encontradas:", sections[:20])

# Find links near those sections
for i, match in enumerate(re.finditer(
    r'(MATRIZ\s*DE\s*RIESGOS|ESTUDIOS\s*PREVIOS|ESTUDIO\s*PREVIO)',
    text, re.I
)):
    start = max(0, match.start() - 500)
    end = min(len(text), match.end() + 1500)
    chunk = text[start:end]
    links = re.findall(r'href=["\']([^"\']+)["\']', chunk, re.I)
    print(f"\n--- Sección '{match.group()}' (i={i}) ---")
    print("Contexto:", chunk[:200])
    print("Links encontrados:")
    for link in links[:10]:
        print(f"  {link}")
    if i >= 3:
        break
