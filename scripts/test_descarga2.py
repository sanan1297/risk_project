"""Test direct PDF download with various approaches."""
import requests

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

constancia = "18-1-197178"
ruta = "/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf"
token = "EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx"

# Approach 1: With a dummy captcha token
url = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token}&constancia={constancia}&g-recaptcha-response=bypass"
r = s.get(url, timeout=30)
print(f"Approach 1 - Status: {r.status_code}, Type: {r.headers.get('Content-Type', '')[:50]}")
print(f"Response length: {len(r.content)}")
if len(r.content) < 500:
    print(r.text[:500])

# Approach 2: Try with requests session, visit main page first to get cookies
s.get("https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-1-197178", timeout=15)
url2 = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token}&constancia={constancia}"
r2 = s.get(url2, timeout=30)
print(f"\nApproach 2 - Status: {r2.status_code}, Type: {r2.headers.get('Content-Type', '')[:50]}")
print(f"Response length: {len(r2.content)}")

# Approach 3: Try with referer
url3 = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token}&constancia={constancia}"
r3 = s.get(url3, timeout=30, headers={"Referer": f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}"})
print(f"\nApproach 3 - Status: {r3.status_code}, Type: {r3.headers.get('Content-Type', '')[:50]}")
print(f"Response length: {len(r3.content)}")
if r3.headers.get('Content-Type', '').startswith('application'):
    print("PDF downloaded successfully!")
