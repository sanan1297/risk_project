"""Robust download with automatic Zenedge challenge handling."""
import requests
import re
import math
import time

constancia = "18-1-197178"
ruta = "/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf"
token_global = "EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx"

def resolve_zenedge(session, url, max_retries=5):
    """Resolve Zenedge cookie challenge."""
    for attempt in range(max_retries):
        resp = session.get(url, timeout=30)
        
        if "__zjc" in resp.text and "cookie" in resp.text:
            match = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', resp.text)
            if match:
                num = int(match.group(1))
                factor = float(match.group(2))
                v = math.floor(num * factor)
                cookie_match = re.search(r'cookie = "(\w+)="', resp.text)
                if cookie_match:
                    cookie_name = cookie_match.group(1)
                    session.cookies.set(cookie_name, str(v), domain="www.contratos.gov.co")
                    print(f"  Set Zenedge cookie: {cookie_name}={v}")
                    
                    redirect_match = re.search(r"window\.location='([^']+)'", resp.text)
                    if redirect_match:
                        redirect_url = redirect_match.group(1)
                        if not redirect_url.startswith("http"):
                            redirect_url = "https://www.contratos.gov.co" + redirect_url
                        print(f"  Following redirect: {redirect_url}")
                        resp = session.get(redirect_url, timeout=30)
                        print(f"  After redirect: status={resp.status_code}, len={len(resp.content)}")
                        continue
        break
    return resp

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

# Step 1: Get the detail page with Zenedge handling
print("Step 1: Getting detail page...")
detail_url = f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}"
detail_resp = resolve_zenedge(s, detail_url)
print(f"  status={detail_resp.status_code}, len={len(detail_resp.content)}")

# Step 2: Try download
print("Step 2: Downloading PDF...")
time.sleep(2)
dl_url = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token_global}&constancia={constancia}"
dl_resp = resolve_zenedge(s, dl_url)
print(f"  status={dl_resp.status_code}, type={dl_resp.headers.get('Content-Type', 'N/A')}, len={len(dl_resp.content)}")

if dl_resp.headers.get('Content-Type', '').startswith('application'):
    outpath = "C:\\Users\\Santa\\Documents\\risk_project\\estudio_data\\test_download.pdf"
    with open(outpath, "wb") as f:
        f.write(dl_resp.content)
    print(f"  PDF saved to {outpath} ({len(dl_resp.content)} bytes)")
else:
    print("  Response content:")
    print(dl_resp.text[:800])
