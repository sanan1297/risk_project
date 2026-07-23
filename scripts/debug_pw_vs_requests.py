"""Debug: Compare what Playwright vs Requests actually see on a 2021 contract."""
import asyncio, math, os, re, time, requests
from playwright.async_api import async_playwright

URL_2021 = "https://www.contratos.gov.co/consultas/detalleProceso.do?numConstancia=22-21-30043"

async def main():
    # PLAYWRIGHT
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        print("=== PLAYWRIGHT ===")
        await page.goto(URL_2021, timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        
        # Screenshot to see what's visible
        await page.screenshot(path="scripts/debug_pw.png", full_page=True)
        
        url_final = page.url
        title = await page.title()
        print(f"URL final: {url_final}")
        print(f"Title: {title}")
        
        # Check body length
        body_text = await page.inner_text("body")
        print(f"Body length: {len(body_text)} chars")
        
        # Check for Zenedge/gateway messages
        if "Zenedge" in body_text or "captcha" in body_text.lower():
            print("!!! ZENEDGE/CAPTCHA BLOCKING PAGE !!!")
        
        # Find what text exists
        for kw in ["MATRIZ", "RIESGO", "ESTUDIO", "DOCUMENTO", "ARCHIVO", "ADJUNTO"]:
            count = body_text.upper().count(kw)
            if count:
                print(f"  '{kw}': {count}")
        
        # Save HTML
        html = await page.content()
        with open("scripts/debug_pw.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML saved to scripts/debug_pw.html ({len(html)} bytes)")
        
        await browser.close()
    
    # REQUESTS (standalone)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    r = s.get(URL_2021, timeout=30)
    for _ in range(3):
        if "__zjc" in r.text:
            m = re.search(r'var v = (\d+)\s*\*\s*([\d.]+)', r.text)
            if m:
                v = math.floor(int(m.group(1)) * float(m.group(2)))
                ck = re.search(r'cookie = "(\w+)="', r.text)
                if ck:
                    s.cookies.set(ck.group(1), str(v), domain="www.contratos.gov.co")
                    rd = re.search(r"window\.location='([^']+)'", r.text)
                    if rd:
                        ru = rd.group(1)
                        if not ru.startswith("http"):
                            ru = "https://www.contratos.gov.co" + ru
                        time.sleep(0.5)
                        r = s.get(ru, timeout=30)
                        continue
        break
    
    with open("scripts/debug_requests.html", "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"\n=== REQUESTS ===")
    print(f"HTML saved to scripts/debug_requests.html ({len(r.text)} bytes)")
    print(f"Status: {r.status_code}, Length: {len(r.text)}")
    
    # Check labels
    labels = re.findall(r'consultaProceso\([^)]+\)[^<]*</a>\s*</td>\s*<td[^>]*>\s*([^<]+)', r.text)
    for l in labels:
        if any(kw in l.upper() for kw in ["MATRIZ", "RIESGO", "ESTUDIO"]):
            print(f"  -> '{l.strip()}'")

asyncio.run(main())
