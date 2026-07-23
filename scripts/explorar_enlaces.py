"""Explorar enlaces directos a PDFs en SECOP."""
import requests
import re

url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-1-197178"
r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
r.encoding = "utf-8"
text = r.text

# Find ALL links
links = re.findall(r'href="([^"]+)"', text)
print("=== TODOS LOS LINKS ===")
for link in links:
    print(link)

# Find form actions
forms = re.findall(r'<form[^>]*action="([^"]+)"', text)
print("\n=== FORM ACTIONS ===")
for f in forms:
    print(f)

# Find file links
file_links = re.findall(r'href="([^"]+\.(pdf|zip|doc|docx|xls|xlsx))"', text, re.I)
print("\n=== FILE LINKS ===")
for link in file_links:
    print(link)

# Find the documentos section
prox_rows = re.findall(
    r'(MATRIZ\s*DE\s*RIESGOS|ESTUDIOS\s*PREVIOS|ESTUDIO\s*PREVIO).*?</tr>',
    text[:100000], re.I | re.DOTALL
)
print("\n=== SECCIONES DOCUMENTOS ===")
for i, sec in enumerate(prox_rows[:5]):
    links_in_sec = re.findall(r'href="([^"]+)"', sec)
    print(f"Seccion {i}: {sec[:200]}")
    print(f"Links: {links_in_sec}")
    print("---")
