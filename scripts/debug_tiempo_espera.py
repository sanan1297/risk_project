"""Check if SECOP document table loads with different wait times."""
import asyncio
from playwright.async_api import async_playwright

# Test URLs: one known to have MATRIZ (C-002, 2018), one 2021, one 2022
test_urls = [
    ("C-002 (2018)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-7462"),
    ("Fila 488 (2022)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"),
    ("Fila 489 (2021)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=21-21-29054"),
    ("Fila 480 (2019)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=19-4-9587254"),
]

wait_times = [0.5, 2, 5]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        
        for label, url in test_urls:
            print(f"\n{'='*60}")
            print(f"URL: {label}")
            
            for wt in wait_times:
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(int(wt * 1000))
                    
                    body_text = await page.inner_text("body")
                    upper = body_text.upper()
                    
                    m = "MATRIZ DE RIESGOS" in upper
                    e = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
                    matriz_count = upper.count("MATRIZ")
                    riesgo_count = upper.count("RIESGO")
                    estudio_count = upper.count("ESTUDIO")
                    
                    print(f"  wait={wt}s: M={m} E={e} | MATRIZ={matriz_count} RIESGO={riesgo_count} ESTUDIO={estudio_count} | len={len(body_text)}")
                except Exception as ex:
                    print(f"  wait={wt}s: ERROR {ex}")
                finally:
                    await page.close()
        
        await browser.close()

asyncio.run(main())
