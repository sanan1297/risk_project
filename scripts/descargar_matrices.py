"""Descargar Matrices de Riesgo de SECOP I desde secop1_lite.csv

Rango: registros 480-750 (índice 0-based: 479-749)
Salta: municipio BOGOTA D.C.
Nombres: C-366 en adelante (estudio_data/)
CSV de mapeo: estudio_data/mapeo_descargas.csv

Estrategia:
1. Busca MATRIZ DE RIESGOS → descarga primer PDF
2. Si no hay, busca ESTUDIOS PREVIOS / ESTUDIO PREVIO → descarga primer PDF
"""
import asyncio
import csv
import os
import re
import sys
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")
OUT_DIR = os.path.join("estudio_data")
MAPPING_CSV = os.path.join("estudio_data", "mapeo_descargas.csv")
START_INDEX = 479  # 0-based = fila 480
END_INDEX = 749    # 0-based = fila 750 inclusive
START_ID = 366

os.makedirs(OUT_DIR, exist_ok=True)

def cargar_registros():
    """Carga registros desde secop1_lite.csv."""
    registros = []
    with open(SECOP1_LITE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < START_INDEX:
                continue
            if i > END_INDEX:
                break
            registros.append((i, row))
    return registros

def debe_omitir(row):
    """True si el municipio es Bogotá D.C."""
    mun = (row.get("municipio", "") or "").strip().upper()
    return "BOGOTA" in mun and "D.C" in mun.upper()

async def visitar_pagina(page, url):
    """Navega a la página SECOP y espera a que cargue."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(2000)
        return True
    except PwTimeout:
        print(f"    ⚠ Timeout en {url}")
        return False
    except Exception as e:
        print(f"    ⚠ Error navegando: {e}")
        return False

async def _click_primer_link_en_tr(page, tr_elem, max_rows=5):
    """Busca el primer link con consultaProceso en la misma fila o siguientes."""
    for offset in range(max_rows):
        current = tr_elem
        if offset > 0:
            current = await page.evaluate(f"""
                (tr) => {{
                    const rows = Array.from(document.querySelectorAll('tr'));
                    const idx = rows.indexOf(tr);
                    return idx >= 0 && idx + {offset} < rows.length ? rows[idx + {offset}] : null;
                }}
            """, tr_elem)
            if not current:
                break
        links = await page.evaluate("""
            (tr) => {
                const anchors = tr.querySelectorAll('a');
                const results = [];
                anchors.forEach(a => {
                    const href = a.getAttribute('href');
                    if (href && href.includes('consultaProceso')) {
                        results.push(href);
                    }
                });
                return results;
            }
        """, current)
        if links:
            # Return the first link href - we'll find it by href
            return links[0]
    return None

async def encontrar_boton_por_texto(page, texto_buscar):
    """Busca en el page el texto y extrae links de documento cercanos."""
    try:
        # Find the cell containing our text
        cell = await page.query_selector(f"td:has-text('{texto_buscar}')")
        if not cell:
            return None

        # Get parent tr
        tr = await cell.evaluate_handle("el => el.closest('tr')")
        if not tr:
            return None
        tr_elem = tr.as_element()
        if not tr_elem:
            return None

        href = await _click_primer_link_en_tr(page, tr_elem)
        return href
    except Exception as e:
        print(f"    [Error buscando {texto_buscar}]: {e}")
        return None

async def encontrar_boton_matriz(page):
    """Busca MATRIZ DE RIESGOS y devuelve href del primer documento."""
    return await encontrar_boton_por_texto(page, "MATRIZ DE RIESGOS")

async def encontrar_boton_estudios(page):
    """Busca ESTUDIOS PREVIOS / ESTUDIO PREVIO y devuelve href."""
    for term in ["ESTUDIOS PREVIOS", "ESTUDIO PREVIO"]:
        href = await encontrar_boton_por_texto(page, term)
        if href:
            return href
    return None

async def descargar_documento(page, href, out_path):
    """Navega directamente a VerDocumentoPublic y descarga el PDF."""
    try:
        # Extract params from consultaProceso href
        # Format: javascript: consultaProceso('/path/...pdf','token','constancia')
        match = re.search(r"consultaProceso\('([^']+)','([^']+)','([^']+)'\)", href)
        if not match:
            print(f"    [Error: no se pudo parsear href]")
            return False
        ruta = match.group(1)
        token_from = match.group(2)
        constancia = match.group(3)

        dl_url = f"https://www.contratos.gov.co/consultas/VerDocumentoPublic?ruta={ruta}&token={token_from}&constancia={constancia}"

        # Navigate to the download URL with timeout - this triggers the PDF
        async with page.expect_download(timeout=60000) as download_info:
            await page.goto(dl_url, wait_until="domcontentloaded", timeout=30000)
        download = await download_info.value
        await download.save_as(out_path)
        return True
    except PwTimeout:
        print(f"    [Timeout en descarga]")
        return False
    except Exception as e:
        print(f"    [Error en descarga: {e}]")
        return False

async def procesar_un_contrato(page, idx, row, codigo):
    """Procesa un contrato: visita URL, busca y descarga."""
    url = row.get("url", "").strip()
    entidad = row.get("entidad", "").strip()
    municipio = row.get("municipio", "").strip()
    
    print(f"\n[{codigo}] Fila {idx+1} | {entidad[:50]} | {municipio}")
    
    if not url:
        print(f"    [Sin URL]")
        return None
    
    print(f"    URL: {url[:80]}...")
    
    ok = await visitar_pagina(page, url)
    if not ok:
        return None
    
    # Buscar MATRIZ DE RIESGOS
    href = await encontrar_boton_matriz(page)
    tipo = "MATRIZ_RIESGOS"
    
    if not href:
        # Fallback: ESTUDIOS PREVIOS
        href = await encontrar_boton_estudios(page)
        tipo = "ESTUDIOS_PREVIOS"
    
    if not href:
        print(f"    [No se encontró Matriz Riesgos ni Estudios Previos]")
        return None
    
    print(f"    Encontrado: {tipo}")
    
    # Descargar
    out_path = os.path.join(OUT_DIR, f"{codigo}.pdf")
    ok = await descargar_documento(page, href, out_path)
    if ok:
        size = os.path.getsize(out_path)
        print(f"    Descargado: {codigo}.pdf ({size:,} bytes)")
        return {"url": url, "codigo": codigo, "tipo": tipo, "tamano": size, "entidad": entidad}
    else:
        print(f"    Fallo descarga")
        return None

async def main():
    registros = cargar_registros()
    total = len(registros)
    print(f"Cargados {total} registros (filas {START_INDEX+1}-{END_INDEX+1})")
    
    # Filtrar Bogotá D.C.
    omitidos = [r for r in registros if debe_omitir(r[1])]
    registros = [r for r in registros if not debe_omitir(r[1])]
    print(f"Omitidos por Bogotá D.C.: {len(omitidos)}")
    print(f"A procesar: {len(registros)} contratos")
    print(f"Códigos desde: C-{START_ID}")
    
    resultados = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="es-CO",
            accept_downloads=True,
        )
        page = await context.new_page()
        
        for i, (idx, row) in enumerate(registros):
            codigo = f"C-{START_ID + i}"
            resultado = await procesar_un_contrato(page, idx, row, codigo)
            if resultado:
                resultados.append(resultado)
            
            # Pequeña pausa entre contratos
            await page.wait_for_timeout(1000)
        
        await browser.close()
    
    # Guardar CSV de mapeo
    print(f"\n\n===== RESUMEN =====")
    print(f"Procesados: {len(registros)}")
    print(f"Descargados: {len(resultados)}")
    print(f"Fallos: {len(registros) - len(resultados)}")
    
    # Mapeo CSV
    with open(MAPPING_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url_secop", "codigo_asignado", "tipo_documento", "tamano_bytes", "entidad", "fecha_descarga"])
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in resultados:
            writer.writerow([r["url"], r["codigo"], r["tipo"], r["tamano"], r["entidad"], fecha])
    
    print(f"Mapeo guardado en: {MAPPING_CSV}")
    
    # Listar archivos descargados
    print(f"\nArchivos en {OUT_DIR}/:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".pdf"):
            fpath = os.path.join(OUT_DIR, f)
            print(f"  {f} ({os.path.getsize(fpath):,} bytes)")

if __name__ == "__main__":
    asyncio.run(main())
