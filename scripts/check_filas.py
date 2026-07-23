import csv, os
with open(os.path.join('contratos', 'proyectos_secop1_lite.csv'), encoding='utf-8-sig') as f:
    for i, row in enumerate(csv.DictReader(f)):
        fila = i + 1
        if 485 <= fila <= 490:
            url = row.get("url","")[:90]
            ent = row.get("entidad","")[:30]
            print(f"Fila {fila}: url={url} | entidad={ent}")
