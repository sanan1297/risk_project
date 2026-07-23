import json

with open('estudio_modelos/modelo_final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Count cells with GRUPO 3
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell['source'])
        if 'GRUPO 3' in src:
            print(f"Cell {i} (id={cell.get('id')}): {src[:80]}...")
        if 'MODELOS' in src and 'SVR' in src and 'm__kernel' in src:
            print(f"Cell {i} (id={cell.get('id')}): MODELOS definition line")
            for line in cell['source']:
                if 'SVR' in line:
                    print(f"  {line[:100]}")
