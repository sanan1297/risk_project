"""Test if we can use requests instead of Playwright for SECOP pages."""
import requests
import re
import math
import time

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
})

# Test with a known working URL
test_urls = [
    "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=21-21-25255",
    "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-31692",
]

for url in test_urls:
    print(f"\nFetching: {url[:80]}...")
    try:
        r = s.get(url, timeout=30)
        print(f"Status: {r.status_code}, Length: {len(r.text)}")
        
        # Handle Zenedge
        if "__zjc" in r.text and "cookie" in r.text:
            match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if match:
                num = int(match.group(1))
                factor = float(match.group(2))
                v = math.floor(num * factor)
                cookie_match = re.search(r'cookie = "(\w+)="', r.text)
                if cookie_match:
                    s.cookies.set(cookie_match.group(1), str(v), domain="www.contratos.gov.co")
                    redirect_match = re.search(r"window\.location='([^']+)'", r.text)
                    if redirect_match:
                        redirect_url = redirect_match.group(1)
                        if not redirect_url.startswith("http"):
                            redirect_url = "https://www.contratos.gov.co" + redirect_url
                        time.sleep(1)
                        r = s.get(redirect_url, timeout=30)
                        print(f"After redirect: Status: {r.status_code}, Length: {len(r.text)}")
        
        # Check content
        text = r.text.upper()
        has_matriz = "MATRIZ DE RIESGOS" in text
        has_estudios = "ESTUDIOS PREVIOS" in text or "ESTUDIO PREVIO" in text
        print(f"MATRIZ_RIESGOS: {has_matriz}, ESTUDIOS_PREVIOS: {has_estudios}")
        print(f"First 200 chars: {r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
