"""Determine how many contracts are blocked vs actually have no matrix.
Scan a batch and report: success, blocked, no-docs."""
import asyncio, csv, os, random
from playwright.async_api import async_playwright

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")

random.seed(123)

todos = []
with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if 480 <= fila <= 1560:
            mun = (row.get("municipio") or "").upper()
            if "BOGOTA" in mun and ("D.C" in mun or "DC" in mun):
                continue
            todos.append((fila, row))

muestra = random.sample(todos, 30)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()
        
        stats = {"ok": 0, "blocked": 0, "dns_err": 0, "nav_err": 0}
        con_matriz = 0
        con_estudios = 0
        sin_docs = 0
        
        for idx, (fila, row) in enumerate(muestra):
            url = (row.get("url") or "").strip()
            
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
            except Exception as e:
                err = str(e)
                if "ERR_NAME_NOT_RESOLVED" in err or "ERR_DNS" in err:
                    stats["dns_err"] += 1
                else:
                    stats["nav_err"] += 1
                continue
            
            try:
                body = await page.inner_text("body")
            except:
                stats["nav_err"] += 1
                continue
            
            if len(body) < 500:
                stats["blocked"] += 1
                continue
            
            stats["ok"] += 1
            upper = body.upper()
            tiene_m = "MATRIZ" in upper
            tiene_e = "ESTUDIOS PREVIOS" in upper or "ESTUDIO PREVIO" in upper
            
            if tiene_m:
                con_matriz += 1
            if tiene_e:
                con_estudios += 1
            if not tiene_m and not tiene_e:
                sin_docs += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(muestra)}] ok={stats['ok']} block={stats['blocked']} dns={stats['dns_err']} nav={stats['nav_err']} | M={con_matriz} E={con_estudios}")
        
        await browser.close()
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS ({len(muestra)} contratos)")
        print(f"{'='*60}")
        print(f"  OK (pagina cargada):       {stats['ok']} ({stats['ok']/len(muestra)*100:.0f}%)")
        print(f"  Bloqueada (<500 chars):     {stats['blocked']} ({stats['blocked']/len(muestra)*100:.0f}%)")
        print(f"  Error DNS:                  {stats['dns_err']} ({stats['dns_err']/len(muestra)*100:.0f}%)")
        print(f"  Error navegacion:           {stats['nav_err']} ({stats['nav_err']/len(muestra)*100:.0f}%)")
        print(f"  ---")
        total_ok = stats['ok']
        if total_ok:
            print(f"  Con MATRIZ:                 {con_matriz} ({con_matriz/total_ok*100:.0f}% de OK)")
            print(f"  Con ESTUDIOS PREVIOS:       {con_estudios} ({con_estudios/total_ok*100:.0f}% de OK)")
            print(f"  Sin docs detectables:       {sin_docs} ({sin_docs/total_ok*100:.0f}% de OK)")

asyncio.run(main())
