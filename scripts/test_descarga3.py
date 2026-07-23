"""Check the error response HTML."""
import requests

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

constancia = "18-1-197178"
ruta = "/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf"
token = "EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx"

# Get the detail page first
s.get(f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}", timeout=15)

url = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token}&constancia={constancia}"
r = s.get(url, timeout=30)
print(r.text[:1000])
