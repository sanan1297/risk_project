"""Try to bypass SECOP blocking with retries and fresh contexts."""
import asyncio
from playwright.async_api import async_playwright

# Use a blocked URL from the previous test
URL = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=21-4-11670334"

async def try_with_context(browser, label):
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        locale="es-CO",
        timezone_id="America/Bogota",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()
    try:
        await page.goto(URL, timeout=20000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        body = await page.inner_text("body")
        print(f"  {label}: len={len(body)} {'BLOQUEADA' if len(body)<500 else 'OK'}")
        if len(body) >= 500:
            upper = body.upper()
            print(f"    MATRIZ={'MATRIZ' in upper} RIESGO={'RIESGO' in upper} ESTUDIOS_PREV={'ESTUDIOS PREVIOS' in upper or 'ESTUDIO PREVIO' in upper}")
        # Save screenshot
        await page.screenshot(path=f"scripts/debug_bypass_{label.replace(' ','_')}.png", full_page=True)
    except Exception as e:
        print(f"  {label}: ERROR {e}")
    finally:
        await context.close()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        
        # Try 3 times with different approaches
        attempts = [
            ("fresh_default", {}),
            ("with_extra_headers", {
                "extra_http_headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                }
            }),
            ("no_js", {
                "java_script_enabled": False,
            }),
            ("mobile_viewport", {
                "viewport": {"width": 375, "height": 812},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            }),
            ("headful_like", {
                "no_viewport": True,
            }),
        ]
        
        for label, opts in attempts:
            await try_with_context(browser, label)
            await asyncio.sleep(1)  # Delay between attempts
        
        await browser.close()

asyncio.run(main())
