"""Quick test: Can Playwright still access SECOP?"""
import asyncio
from playwright.async_api import async_playwright

urls = [
    ("Fila 488 (2022)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"),
    ("Fila 480 (2019)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=19-4-9587254"),
    ("C-002 (2018)", "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=18-21-7462"),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        for label, url in urls:
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                body = await page.inner_text("body")
                upper = body.upper()
                print(f"\n{label}: len={len(body)}")
                print(f"  MATRIZ: {'MATRIZ' in upper}")
                print(f"  MATRIZ_RIESGOS: {'MATRIZ DE RIESGOS' in upper}")
                print(f"  RIESGO: {'RIESGO' in upper}")
                print(f"  ESTUDIOS PREVIOS: {'ESTUDIOS PREVIOS' in upper}")
                print(f"  Bloqueada: {len(body) < 500}")
            except Exception as e:
                print(f"\n{label}: ERROR {e}")
        
        await browser.close()

asyncio.run(main())
