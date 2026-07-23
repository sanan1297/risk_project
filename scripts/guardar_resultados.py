"""Guardar resultados finales limpios: separar MATRIZ y ESTUDIOS."""
import csv, os, json

RESULT_FILE = os.path.join("estudio_data", "resultados_finales.csv")
STATE_FILE = os.path.join("estudio_data", "estado_scan.json")
OUT_DIR = "estudio_data"

FIELD_NAMES = ["fila", "url", "entidad", "municipio", "departamento", "tipo"]

# Cargar resultados
results = []
with open(RESULT_FILE, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        results.append(row)

print(f"Total resultados: {len(results)}")

# Separar por tipo
matriz = [r for r in results if r["tipo"] in ("MATRIZ", "MATRIZ_RIESGO")]
estudios = [r for r in results if r["tipo"] == "ESTUDIOS_PREVIOS"]
sin_doc = [r for r in results if r["tipo"] == "SIN_DOCUMENTO"]
bloq = [r for r in results if r["tipo"] == "BLOQUEADA"]

print(f"MATRIZ: {len(matriz)}")
print(f"ESTUDIOS PREVIOS: {len(estudios)}")
print(f"SIN DOCUMENTO: {len(sin_doc)}")
print(f"BLOQUEADA: {len(bloq)}")

# Guardar MATRIZ
with open(os.path.join(OUT_DIR, "matriz_detectados.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    w.writeheader()
    for r in matriz:
        w.writerow({k: r.get(k, "") for k in FIELD_NAMES})
print(f"\nmatriz_detectados.csv: {len(matriz)} contratos")

# Guardar ESTUDIOS
with open(os.path.join(OUT_DIR, "estudios_detectados.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    w.writeheader()
    for r in estudios:
        w.writerow({k: r.get(k, "") for k in FIELD_NAMES})
print(f"estudios_detectados.csv: {len(estudios)} contratos")

# Guardar BLOQUEADAS (para reintentar manual)
with open(os.path.join(OUT_DIR, "bloqueadas.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    w.writeheader()
    for r in bloq:
        w.writerow({k: r.get(k, "") for k in FIELD_NAMES})
print(f"bloqueadas.csv: {len(bloq)} contratos (para reintentar)")

# Resumen
print(f"\n{'='*60}")
print(f"RESUMEN FINAL")
print(f"{'='*60}")
print(f"MATRIZ detectados:          {len(matriz)}")
print(f"  - Con RIESGO en texto:   {sum(1 for r in matriz if r['tipo']=='MATRIZ_RIESGO')}")
print(f"ESTUDIOS PREVIOS:           {len(estudios)}")
print(f"SIN DOCUMENTO (cargaron):   {len(sin_doc)}")
print(f"BLOQUEADAS (3 intentos):    {len(bloq)}")
print(f"")
print(f"Total con documentos:       {len(matriz) + len(estudios)}")
print(f"Total procesados:           {len(results)}")

# Comparacion con barrido original
print(f"\n{'='*60}")
print(f"COMPARACION con deteccion original")
print(f"{'='*60}")

# Si existe barrido original, comparar
barrido_path = os.path.join(OUT_DIR, "barrido_480_750.csv")
if os.path.exists(barrido_path):
    original_m = 0
    original_e = 0
    with open(barrido_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t = row["tipo_documento"].strip()
            if t == "MATRIZ_RIESGOS":
                original_m += 1
            elif t == "ESTUDIOS_PREVIOS":
                original_e += 1
    print(f"Original: MATRIZ_RIESGOS={original_m} ESTUDIOS_PREVIOS={original_e} TOTAL={original_m+original_e}")
    print(f"Corregido: MATRIZ={len(matriz)} ESTUDIOS_PREVIOS={len(estudios)} TOTAL={len(matriz)+len(estudios)}")
    mejora = (len(matriz)+len(estudios)) / (original_m+original_e) * 100
    print(f"Mejora: {mejora:.0f}%")

# Mostrar primeras matrices
if matriz:
    print(f"\n--- PRIMERAS 20 MATRICES ---")
    for r in matriz[:20]:
        print(f"  Fila {r['fila']:>4} | {r['tipo']:13} | {r.get('entidad','')[:40]}")
