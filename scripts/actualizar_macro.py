import pandas as pd
import pickle

macro_new = {
    "C-360": {"anio_inicio":2019,"anio_fin":2019,"sobrecosto":10.14},
    "C-361": {"anio_inicio":2022,"anio_fin":2022,"sobrecosto":19.09},
    "C-362": {"anio_inicio":2021,"anio_fin":2021,"sobrecosto":4.38},
    "C-363": {"anio_inicio":2022,"anio_fin":2022,"sobrecosto":7.2},
    "C-364": {"anio_inicio":2023,"anio_fin":2023,"sobrecosto":20.83},
    "C-365": {"anio_inicio":2023,"anio_fin":2027,"sobrecosto":float("nan")},
}

macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
with open("models/ipc_trm.pkl", "rb") as f:
    ipc_trm = pickle.load(f)

for cid, mn in macro_new.items():
    duracion = mn["anio_fin"] - mn["anio_inicio"] + 1
    ipc_acum = 1.0
    for y in range(mn["anio_inicio"], mn["anio_fin"]+1):
        ipc_acum *= (1 + ipc_trm[y]["ipc"] / 100)
    ipc_acum = (ipc_acum - 1) * 100
    trm_prom = sum(ipc_trm[y]["trm"] for y in range(mn["anio_inicio"], mn["anio_fin"]+1)) / duracion
    macro = pd.concat([macro, pd.DataFrame([{
        "id_contrato": cid, "anio_inicio": mn["anio_inicio"], "anio_fin": mn["anio_fin"],
        "duracion": duracion, "ipc_acumulado": round(ipc_acum,2), "trm_promedio": round(trm_prom,0),
    }])], ignore_index=True)

macro.to_csv("docs/contratos_macro.csv", index=False, encoding="utf-8-sig")
print(f"Macro guardado: {macro.shape[0]} contratos")
