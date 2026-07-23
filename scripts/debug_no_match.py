"""Check contracts that had NO detection - what documents do they actually have?"""
import asyncio, csv, os
from playwright.async_api import async_playwright

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")

# Take a sample of NON-MATCHING contracts (from earlier)
# Filas 482, 483, 484, 485, 486, 487 from the earlier test
test_filas = [482, 483, 484, 485, 486, 487]

registros = []
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if fila in test_filas:
            registros.append((fila, row))

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        for fila, row in registros:
            url = (row.get("url") or "").strip()
            entidad = (row.get("entidad") or "").strip()[:30]
            depto = (row.get("departamento") or "").strip()
            
            print(f"\n{'='*60}")
            print(f"Fila {fila} | {entidad} | {depto}")
            print(f"URL: {url}")
            
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  ERROR loading: {e}")
                continue
            
            body = await page.inner_text("body")
            upper = body.upper()
            
            print(f"  Body length: {len(body)}")
            
            # What keywords appear?
            for kw in ["MATRIZ", "RIESGO", "ESTUDIO", "DOCUMENTO", "ARCHIVO",
                       "ADJUNTO", "PDF", "CONTRATO", "PLIEGO", "ADENDA",
                       "ACTA", "PROPUESTA", "INFORME", "CERTIFICADO",
                       "FACTURA", "LIQUIDACION", "INICIO", "ADJUDICACION"]:
                count = upper.count(kw)
                if count:
                    print(f"  '{kw}': {count} veces")
            
            # Extract document labels via evaluate
            try:
                doc_labels = await page.evaluate("""() => {
                    const links = document.querySelectorAll('a');
                    const labels = [];
                    for (const a of links) {
                        const txt = a.textContent.trim();
                        const href = a.getAttribute('href') || '';
                        if (txt.length > 3 && (href.includes('consultaProceso') || href.includes('VerDocumento'))) {
                            labels.push(txt);
                        }
                    }
                    return labels;
                }""")
                
                if doc_labels:
                    print(f"  Documentos encontrados ({len(doc_labels)}):")
                    for l in doc_labels:
                        print(f"    -> '{l[:80]}'")
                else:
                    print(f"  NO DOCUMENTOS ENCONTRADOS")
                    
                    # Maybe the page structure is different
                    all_texts = await page.evaluate("""() => {
                        // Get ALL td and th content
                        const cells = document.querySelectorAll('td, th, span, div');
                        return Array.from(cells).map(c => c.textContent.trim()).filter(t => t.length > 3);
                    }""")
                    print(f"  Total texto en celdas: {len(all_texts)}")
                    # Show some potentially interesting ones
                    interesting = [t for t in all_texts if any(k in t.upper() for k in ['DOCUMENTO', 'ARCHIVO', 'ADJUNTO', 'NOMBRE', 'TIPO', 'DESCRIPCION'])]
                    if interesting:
                        for t in interesting[:10]:
                            print(f"    Celda: '{t[:100]}'")
            except Exception as e:
                print(f"  Error en evaluate: {e}")
            
        await browser.close()

asyncio.run(main())
