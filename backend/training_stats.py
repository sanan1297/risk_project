from pathlib import Path

import joblib
import pandas as pd

from .predictor import MODEL_META

ROOT = Path(__file__).resolve().parent.parent


def _load_contratos() -> pd.DataFrame:
    path = ROOT / "contratos" / "proyectos_secop1_lite.csv"
    return pd.read_csv(path)


def _load_matriz() -> pd.DataFrame:
    path = ROOT / "docs" / "matriz.csv"
    return pd.read_csv(path, on_bad_lines="skip")


def _load_coeficientes() -> pd.DataFrame:
    path = ROOT / "models" / "coeficientes_ridge.csv"
    return pd.read_csv(path)


def _load_ipc_trm() -> dict:
    path = ROOT / "models" / "ipc_trm.pkl"
    return joblib.load(path)


def compute() -> dict:
    contratos = _load_contratos()
    matriz = _load_matriz()
    coefs = _load_coeficientes()
    ipc_trm_data = _load_ipc_trm()

    sc = contratos["sobrecosto_pct"]
    bins = [0, 10, 25, 50, 100, sc.max()]
    labels = ["0-10%", "10-25%", "25-50%", "50-100%", "100%+"]
    dist_bins = pd.cut(sc, bins=bins, labels=labels, right=True, include_lowest=True)
    dist_sobrecosto = (
        dist_bins.value_counts().reindex(labels, fill_value=0).reset_index()
    )
    dist_sobrecosto.columns = ["rango", "cantidad"]
    dist_sobrecosto["cantidad"] = dist_sobrecosto["cantidad"].astype(int)

    top_aumentan = (
        coefs.sort_values("coef", ascending=False)
        .head(5)
        .assign(tipo="aumenta")
        .to_dict(orient="records")
    )
    top_disminuyen = (
        coefs.sort_values("coef", ascending=True)
        .head(5)
        .assign(tipo="disminuye")
        .to_dict(orient="records")
    )

    cat_counts = matriz["categoria"].value_counts().reset_index()
    cat_counts.columns = ["categoria", "cantidad"]

    ipc_trm_series = [
        {"anio": int(k), "ipc": v["ipc"], "trm": v["trm"]}
        for k, v in sorted(ipc_trm_data.items())
    ]

    tipo_counts = matriz["tipo"].value_counts().reset_index()
    tipo_counts.columns = ["tipo", "cantidad"]

    return {
        "modelo": MODEL_META,
        "total_contratos": len(contratos),
        "sobrecosto_promedio": round(sc.mean(), 1),
        "sobrecosto_mediana": round(sc.median(), 1),
        "porcentaje_alto_riesgo": round((sc > 25).mean(), 4),
        "total_riesgos_matriz": len(matriz),
        "contratos_en_matriz": int(matriz["id_contrato"].nunique()),
        "distribucion_sobrecosto": dist_sobrecosto.to_dict(orient="records"),
        "top_coeficientes": top_aumentan + top_disminuyen,
        "categorias_riesgo": cat_counts.to_dict(orient="records"),
        "tipos_riesgo": tipo_counts.to_dict(orient="records"),
        "ipc_trm": ipc_trm_series,
    }
