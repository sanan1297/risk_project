"""Test download with cloudscraper to bypass Zenedge."""
import cloudscraper
import requests

scraper = cloudscraper.create_scraper()

constancia = "18-1-197178"
ruta = "/2020/2020Q2/2021/ALUMA/285440011/18-1-197178/ALUMA_PROCESO_18-1-197178_285440011_85757311.pdf"
token_global = "EKL2sF3rYa_Jer648xvkk_8VZEsoM1qQ_tAQSCPgSrx"

# First visit the detail page
resp = scraper.get(
    f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}",
    timeout=30
)
print(f"Detail page status: {resp.status_code}, length: {len(resp.text)}")

# Now try the download
url_dl = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token_global}&constancia={constancia}"
r = scraper.get(url_dl, timeout=30, headers={"Referer": resp.url})
print(f"Download status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type', 'N/A')}")
print(f"Content-Length: {r.headers.get('Content-Length', 'N/A')}")
print(f"Size: {len(r.content)} bytes")

if r.headers.get('Content-Type', '').startswith('application'):
    with open("C:\\Users\\Santa\\Documents\\risk_project\\estudio_data\\test_download.pdf", "wb") as f:
        f.write(r.content)
    print("PDF SAVED!")
else:
    print(r.text[:800])
