from pathlib import Path

import joblib
import pandas as pd

from .predictor import MODEL_META

ROOT = Path(__file__).resolve().parent.parent


def _load_consolidado() -> pd.DataFrame:
    path = ROOT / "docs" / "consolidado_38_features.csv"
    return pd.read_csv(path)


def _load_matriz() -> pd.DataFrame:
    import csv as csv_mod
    path = ROOT / "docs" / "matriz.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv_mod.reader(f)
        rows = [row[:20] for row in reader]
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    for c in ["valor_inicial", "valor_final", "sobrecosto", "probabilidad", "impacto", "valoracion"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_feature_importances() -> pd.DataFrame:
    path = ROOT / "models" / "feature_importances_rf.csv"
    return pd.read_csv(path)


def _load_ipc_trm() -> dict:
    path = ROOT / "models" / "ipc_trm.pkl"
    return joblib.load(path)


def compute() -> dict:
    df = _load_consolidado()
    matriz = _load_matriz()
    coefs = _load_feature_importances()
    ipc_trm_data = _load_ipc_trm()

    sc = df["sobrecosto"]
    n_contratos = len(df)

    bins = [0, 10, 25, 50, 100, sc.max()]
    labels = ["0-10%", "10-25%", "25-50%", "50-100%", "100%+"]
    dist_bins = pd.cut(sc, bins=bins, labels=labels, right=True, include_lowest=True)
    dist_sobrecosto = (
        dist_bins.value_counts().reindex(labels, fill_value=0).reset_index()
    )
    dist_sobrecosto.columns = ["rango", "cantidad"]
    dist_sobrecosto["cantidad"] = dist_sobrecosto["cantidad"].astype(int)

    top_aumentan = (
        coefs.sort_values("importance", ascending=False)
        .head(5)
        .rename(columns={"importance": "coef"})
        .assign(tipo="aumenta")
        .to_dict(orient="records")
    )
    top_disminuyen = []

    cat_counts = matriz["categoria"].value_counts().reset_index()
    cat_counts.columns = ["categoria", "cantidad"]

    tipo_counts = matriz["tipo"].value_counts().reset_index()
    tipo_counts.columns = ["tipo", "cantidad"]

    ipc_trm_series = [
        {"anio": int(k), "ipc": v["ipc"], "trm": v["trm"]}
        for k, v in sorted(ipc_trm_data.items())
    ]

    pool_path = ROOT / "contratos" / "proyectos_secop1_lite.csv"
    n_pool = len(pd.read_csv(pool_path)) if pool_path.exists() else 0

    n_riesgos_raw = len(matriz)
    n_contratos_raw = int(matriz["id_contrato"].nunique())

    contratos_riesgos = matriz.groupby("id_contrato", sort=False).agg(
        sobrecosto=("sobrecosto", "first"),
        n_riesgos=("id_riesgo", "nunique"),
    ).reset_index()
    contratos_riesgos = contratos_riesgos[contratos_riesgos["sobrecosto"] < 200].copy()
    n_riesgos_filtro = int(contratos_riesgos["n_riesgos"].sum())

    riesgos_pc = contratos_riesgos["n_riesgos"]
    dist_n_riesgos = {
        "min": int(riesgos_pc.min()),
        "max": int(riesgos_pc.max()),
        "media": round(riesgos_pc.mean(), 1),
        "mediana": int(riesgos_pc.median()),
    }

    top_n_riesgos = (
        contratos_riesgos.sort_values("n_riesgos", ascending=False)
        .head(5)
        .assign(sobrecosto=lambda x: x["sobrecosto"].round(1))
        .to_dict(orient="records")
    )

    top_sobrecosto = (
        df.nlargest(5, "sobrecosto")[["id_contrato", "sobrecosto"]]
        .merge(contratos_riesgos[["id_contrato", "n_riesgos"]], on="id_contrato", how="left")
        .assign(sobrecosto=lambda x: x["sobrecosto"].round(1))
        .fillna({"n_riesgos": 0})
        .astype({"n_riesgos": int})
        .to_dict(orient="records")
    )

    return {
        "modelo": MODEL_META,
        "total_contratos": n_contratos,
        "contratos_raw": n_contratos,
        "contratos_pool_secop1": n_pool,
        "contratos_secop2_incluidos": 5,
        "total_riesgos_matriz": n_riesgos_raw,
        "total_riesgos_filtro": n_riesgos_filtro,
        "contratos_en_matriz": n_contratos,
        "sobrecosto_promedio": round(sc.mean(), 1),
        "sobrecosto_mediana": round(sc.median(), 1),
        "porcentaje_alto_riesgo": round((sc > 25).mean(), 4),
        "distribucion_sobrecosto": dist_sobrecosto.to_dict(orient="records"),
        "top_coeficientes": top_aumentan + top_disminuyen,
        "categorias_riesgo": cat_counts.to_dict(orient="records"),
        "tipos_riesgo": tipo_counts.to_dict(orient="records"),
        "ipc_trm": ipc_trm_series,
        "dist_n_riesgos": dist_n_riesgos,
        "top_n_riesgos": top_n_riesgos,
        "top_sobrecosto": top_sobrecosto,
    }
