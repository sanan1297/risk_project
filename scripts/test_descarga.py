"""Test direct PDF download from SECOP."""
import requests

url_dl = "https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta=/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf&token=EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx&constancia=18-1-197178"

r = requests.get(url_dl, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type', 'N/A')}")
print(f"Content-Length: {r.headers.get('Content-Length', 'N/A')}")
print(f"First 100 bytes: {r.content[:100]}")
