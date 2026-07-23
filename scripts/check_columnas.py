"""Check secop1_lite columns and sample data, and check what contracts WITHOUT
detection actually look like on SECOP."""
import csv
import os

SECOP1_LITE = os.path.join("contratos", "proyectos_secop1_lite.csv")

with open(SECOP1_LITE, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    print("Columnas:", reader.fieldnames)
    print()
    for i, row in enumerate(reader):
        if i == 0:
            for k, v in row.items():
                val = v[:80] if v else "(empty)"
                print(f"  {k}: {val}")
        if i >= 2:
            break

# Check: does the CSV have any indicator of "tiene matriz"?
# Also check if some contracts are from SECOP II
print("\n\nContando tipos de URL...")
total = 0
secop1 = 0
other = 0
with open(SECOP1_LITE, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        total += 1
        url = (row.get("url") or "").strip()
        if "secop" in url.lower() and "i" in url.lower():
            secop1 += 1
        else:
            other += 1
    
print(f"Total registros: {total}")
print(f"Con 'secop' en URL: {secop1}")
print(f"Otros: {other}")

# Check how many of the first 480 contracts (the ones in matriz_clean) 
# also appear in secop1_lite  
print("\n\nVerificando cuantos de los primeros 480 estan en matriz_clean...")
with open(SECOP1_LITE, encoding="utf-8-sig") as f:
    urls_secop = set()
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if fila <= 480:
            url = (row.get("url") or "").strip()
            urls_secop.add(url)
            
print(f"URLs unicas en filas 1-480 de secop1_lite: {len(urls_secop)}")
