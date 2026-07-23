"""Check current scan state."""
import json, os
from collections import Counter

state = {}
if os.path.exists("estudio_data/estado_scan.json"):
    with open("estudio_data/estado_scan.json", encoding="utf-8") as f:
        state = json.load(f)

print(f"Total resultados: {len(state.get('resultados',[]))}")
print(f"Pendientes: {len(state.get('pendientes',[]))}")

tipos = Counter(r["tipo"] for r in state.get("resultados",[]))
for t, c in tipos.most_common():
    print(f"  {t}: {c}")
