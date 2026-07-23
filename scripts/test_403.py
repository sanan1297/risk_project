"""Check what 403 page says."""
import requests

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
url = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"
r = s.get(url, timeout=15)
print(r.text[:1500])
