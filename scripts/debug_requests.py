"""Debug: Check if requests is getting full page or Zenedge challenge."""
import requests
import math
import re

urls_to_check = [
    ("Fila 767 (Matriz en PW)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=21-1-221074"),
    ("Fila 794 (Matriz en PW)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-4662"),
    ("Fila 536 (Matriz confirmada)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=21-21-25255"),
]

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})

for label, url in urls_to_check:
    print(f"\n--- {label} ---")
    print(f"URL: {url}")
    try:
        r = s.get(url, timeout=30)
        print(f"Status: {r.status_code}, Length: {len(r.text)}")
        print(f"Zenedge challenge: {'__zjc' in r.text}")

        text = r.text.upper()
        has_m = "MATRIZ DE RIESGOS" in text
        has_e = "ESTUDIOS PREVIOS" in text or "ESTUDIO PREVIO" in text
        print(f"MATRIZ: {has_m}, ESTUDIOS: {has_e}")

        # Handle Zenedge
        if "__zjc" in r.text:
            match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if match:
                num = int(match.group(1))
                factor = float(match.group(2))
                v = math.floor(num * factor)
                ck = re.search(r'cookie = "(\w+)="', r.text)
                if ck:
                    s.cookies.set(ck.group(1), str(v), domain="www.contratos.gov.co")
                    rd = re.search(r"window\.location='([^']+)'", r.text)
                    if rd:
                        ru = rd.group(1)
                        if not ru.startswith("http"):
                            ru = "https://www.contratos.gov.co" + ru
                        import time
                        time.sleep(1)
                        r2 = s.get(ru, timeout=30)
                        print(f"After redirect: Status={r2.status_code}, Length={len(r2.text)}")
                        text2 = r2.text.upper()
                        print(f"MATRIZ after: {'MATRIZ DE RIESGOS' in text2}, ESTUDIOS after: {'ESTUDIOS PREVIOS' in text2 or 'ESTUDIO PREVIO' in text2}")
                        if len(r2.text) < 1000:
                            print(f"Response was short: {r2.text[:200]}")
        else:
            print("No Zenedge challenge - page looks complete")
            if len(r.text) < 1000:
                print(f"Response was short: {r.text[:200]}")
            
    except Exception as e:
        print(f"Error: {e}")
