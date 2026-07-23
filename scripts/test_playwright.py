"""Use Playwright to download from SECOP."""
import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    constancia = "18-1-197178"
    out_dir = "C:\\Users\\Santa\\Documents\\risk_project\\estudio_data"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()
        
        # Navigate to detail page
        url = f"https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia={constancia}"
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle", timeout=60000)
        print(f"Page loaded: {page.url}")
        
        # Look for MATRIZ DE RIESGOS section
        content = await page.content()
        print(f"Page contains 'MATRIZ': {'MATRIZ' in content}")
        print(f"Page contains 'RIESGOS': {'RIESGOS' in content}")
        
        # Try to find download links
        # Look for links near Matriz de Riesgos text
        links = await page.query_selector_all('a[href*="consultaProceso"]')
        print(f"Found {len(links)} consultaProceso links")
        
        # Try clicking links near MATRIZ DE RIESGOS
        matriz_tds = await page.query_selector_all('text=MATRIZ DE RIESGOS')
        print(f"Found {len(matriz_tds)} MATRIZ DE RIESGOS elements")
        
        # Take a screenshot
        await page.screenshot(path=os.path.join(out_dir, "secop_page.png"))
        print("Screenshot saved")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
