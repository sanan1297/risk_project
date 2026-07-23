"""Bypass Zenedge WAF cookie challenge."""
import requests
import re
import math

constancia = "18-1-197178"
ruta = "/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf"
token_global = "EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx"

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

# First visit detail page 
resp = s.get(
    f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}",
    timeout=30
)
print(f"Detail page: status={resp.status_code}, len={len(resp.text)}")

# Check for Zenedge challenge
if "__zjc" in resp.text:
    # Extract the cookie calculation
    match = re.search(r'var v = (\d+)\s*\*\s*(\d+(?:\.\d+)?)', resp.text)
    if match:
        num = int(match.group(1))
        factor = float(match.group(2))
        v = math.floor(num * factor)
        # Find the cookie name
        cookie_match = re.search(r'cookie = "(\w+)="', resp.text)
        if cookie_match:
            cookie_name = cookie_match.group(1)
            s.cookies.set(cookie_name, str(v))
            print(f"Set Zenedge cookie: {cookie_name}={v}")
    
    # Follow redirect
    redirect_match = re.search(r"window\.location='([^']+)'", resp.text)
    if redirect_match:
        redirect_url = redirect_match.group(1)
        print(f"Following redirect: {redirect_url}")
        resp2 = s.get(redirect_url, timeout=30)
        print(f"After redirect: status={resp2.status_code}, len={len(resp2.text)}")
        
        # Try download URL
        url_dl = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token_global}&constancia={constancia}"
        r = s.get(url_dl, timeout=30, headers={"Referer": redirect_url})
        print(f"Download: status={r.status_code}, type={r.headers.get('Content-Type', 'N/A')}, len={len(r.content)}")
        
        if r.headers.get('Content-Type', '').startswith('application'):
            print("PDF DOWNLOADED SUCCESSFULLY!")
        else:
            print(r.text[:500])
    else:
        print("No redirect found")
else:
    print("No Zenedge challenge - trying download directly")
    url_dl = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token_global}&constancia={constancia}"
    r = s.get(url_dl, timeout=30)
    print(f"Download: status={r.status_code}, type={r.headers.get('Content-Type', 'N/A')}")
    if r.headers.get('Content-Type', '').startswith('application'):
        print("PDF DOWNLOADED SUCCESSFULLY!")
    else:
        print(r.text[:500])
