import json

for nb_path in [
    "estudio_modelos/modelado_v2.ipynb",
    "estudio_modelos/modelado.ipynb",
    "estudio_modelos/modelo_final.ipynb",
]:
    nb = json.load(open(nb_path, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed = [c for c in code_cells if c.get("execution_count")]
    print(f"{nb_path}:")
    print(f"  Code cells: {len(code_cells)}, Executed: {len(executed)}")
