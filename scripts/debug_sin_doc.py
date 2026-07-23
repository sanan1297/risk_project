"""Check SIN_DOCUMENTO contracts: do they have ANY document labels on the SECOP page?"""
import asyncio, csv, os
from playwright.async_api import async_playwright

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")

# Get 5 SIN_DOCUMENTO contracts from barrido
sin_doc_filas = []
with open("estudio_data/barrido_480_750.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["tipo_documento"] == "SIN_DOCUMENTO" and len(sin_doc_filas) < 6:
            sin_doc_filas.append(int(row["fila_secop1_lite"]))

# Load their URLs from secop1_lite
registros = []
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if fila in sin_doc_filas:
            registros.append((fila, row))

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        for fila, row in registros:
            url = (row.get("url") or "").strip()
            entidad = (row.get("entidad") or "").strip()[:35]
            
            print(f"\n{'='*60}")
            print(f"Fila {fila} | {entidad}")
            print(f"URL: {url}")
            
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            
            body_text = await page.inner_text("body")
            print(f"  Body length: {len(body_text)}")
            
            if len(body_text) < 500:
                print(f"  ** BLOQUEADA **")
                continue
            
            # Extract ALL document labels from the page
            doc_info = await page.evaluate("""() => {
                const links = document.querySelectorAll('a[href*="consultaProceso"], a[href*="VerDocumento"]');
                return Array.from(links).map(a => ({
                    text: a.textContent.trim(),
                    href: a.getAttribute('href') || ''
                }));
            }""")
            
            if doc_info:
                print(f"  Documentos ({len(doc_info)}):")
                for d in doc_info:
                    print(f"    -> '{d['text'][:80]}'")
            else:
                print(f"  ** NO HAY DOCUMENTOS visibles en la pagina **")
                # Check if there's a document table at all
                tables = await page.query_selector_all("table")
                print(f"  Total tablas en pagina: {len(tables)}")
                # Check all links
                all_links = await page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a')).map(a => ({
                        text: a.textContent.trim().substring(0, 100),
                        href: (a.getAttribute('href') || '').substring(0, 100)
                    })).filter(x => x.text.length > 0);
                }""")
                for l in all_links[:15]:
                    print(f"    Link: '{l['text']}' -> {l['href']}")
            
        await browser.close()

asyncio.run(main())
