import json, sys

def extract_text_outputs(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    texts = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and cell.get("outputs"):
            for out in cell["outputs"]:
                if out.get("output_type") == "stream" and out.get("text"):
                    t = "".join(out["text"])
                    t = t.encode("ascii", errors="replace").decode()
                    texts.append(f"--- Cell {i} ---\n{t}")
                elif out.get("output_type") == "execute_result" and out.get("data", {}).get("text/plain"):
                    t = "".join(out["data"]["text/plain"])
                    t = t.encode("ascii", errors="replace").decode()
                    texts.append(f"--- Cell {i} (result) ---\n{t}")
    return "\n".join(texts)

out = []
out.append("=" * 60)
out.append("modelado_v2.ipynb - ROC & Classifier")
out.append("=" * 60)
out.append(extract_text_outputs("estudio_modelos/modelado_v2.ipynb"))

out.append("\n" + "=" * 60)
out.append("modelo_final.ipynb - Benchmark v2")
out.append("=" * 60)
out.append(extract_text_outputs("estudio_modelos/modelo_final.ipynb"))

sys.stdout.reconfigure(encoding="utf-8")
with open("notebooks_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("OK")
