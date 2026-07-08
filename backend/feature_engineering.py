from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CAT_COLS = ["tipo", "clase", "asignacion", "fuente_riesgo", "etapa", "categoria"]
NUM_COLS = ["probabilidad", "impacto", "valoracion"]
REQUIRED_COLS = ["id_contrato", "descripcion_riesgo", "probabilidad", "impacto", "tipo", "categoria"]


@lru_cache(maxsize=1)
def _load_features():
    return joblib.load(MODELS_DIR / "feature_names.pkl")


@lru_cache(maxsize=1)
def _load_ipc_trm():
    return joblib.load(MODELS_DIR / "ipc_trm.pkl")


@lru_cache(maxsize=1)
def _load_vectorizer():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")


def get_feature_names() -> list[str]:
    return _load_features()


def validate_input(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return [f"Columna requerida faltante: {c}" for c in missing]
    if df["descripcion_riesgo"].isna().all():
        return ["Todas las descripciones de riesgo estan vacias"]
    return []


def _lookup_ipc_trm(anio: int | float) -> tuple[float | None, float | None]:
    data = _load_ipc_trm().get(int(anio))
    if data is None:
        return None, None
    return (data["ipc"], data["trm"]) if isinstance(data, dict) else (data[0], data[1])


def resolve_ipc_trm(
    anio_override: int | None,
    ipc_override: float | None,
    trm_override: float | None,
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    anios = df["anio"] if "anio" in df.columns else pd.Series([anio_override or 2022] * len(df))

    if ipc_override is not None and trm_override is not None:
        ipc = pd.Series([ipc_override] * len(df))
        trm = pd.Series([trm_override] * len(df))
    elif anio_override is not None and anio_override in _load_ipc_trm():
        ipc_v, trm_v = _lookup_ipc_trm(anio_override)
        ipc = pd.Series([ipc_v] * len(df))
        trm = pd.Series([trm_v] * len(df))
    else:
        pairs = anios.map(_lookup_ipc_trm)
        ipc = pairs.map(lambda x: x[0])
        trm = pairs.map(lambda x: x[1])
        bad = anios[ipc.isna()].unique()
        if len(bad):
            raise ValueError(f"Año(s) fuera de rango: {list(bad)}. IPC/TRM desconocidos.")
    return anios, ipc, trm


def _aggregate_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    grupos = df.groupby("id_contrato", sort=False)
    rows = []
    for cid, g in grupos:
        row = {"id_contrato": cid}
        row["valor_inicial"] = pd.to_numeric(g["valor_inicial"].iloc[0], errors="coerce") if "valor_inicial" in g.columns else 0
        row["n_riesgos"] = len(g)
        for col, name in [("probabilidad", "prob"), ("impacto", "imp"), ("valoracion", "val")]:
            vals = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            row[f"{name}_promedio"] = vals.mean() if len(vals) else 0
            row[f"{name}_std"] = vals.std(ddof=0) if len(vals) else 0
        row["interaccion_prob_x_impacto"] = row.get("prob_promedio", 0) * row.get("imp_promedio", 0)
        row["suma_valoracion"] = g["valoracion"].sum() if "valoracion" in g.columns else 0
        for cat in ["bajo", "medio", "alto", "extremo", "no especificado"]:
            row[f"n_categoria_{cat}"] = (g["categoria"] == cat).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_cat_props(df: pd.DataFrame) -> pd.DataFrame:
    partes = [df[["id_contrato"]]]
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col[:4])
        partes.append(dummies)
    merged = pd.concat(partes, axis=1)
    cat_agg = merged.groupby("id_contrato", sort=False).mean()
    cat_agg.columns = [f"prop_{c}" for c in cat_agg.columns if c != "id_contrato"]
    return cat_agg.reset_index()


def _aggregate_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    textos = df.groupby("id_contrato", sort=False)["descripcion_riesgo"].apply(
        lambda g: " ".join(g.dropna().astype(str))
    )
    vectorizer = _load_vectorizer()
    tfidf_matrix = vectorizer.transform(textos)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
        index=textos.index,
    )
    return tfidf_df.reset_index()


def aggregate_risks(
    df: pd.DataFrame,
    anio: int | None = None,
    ipc: float | None = None,
    trm: float | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "no especificado")

    FEATURES_33 = _load_features()

    out = _aggregate_basic_stats(df)

    cat_agg = _aggregate_cat_props(df)
    out = out.merge(cat_agg, on="id_contrato", how="left")

    tfidf_df = _aggregate_tfidf(df)
    out = out.merge(tfidf_df, on="id_contrato", how="left")

    anios, ipc_s, trm_s = resolve_ipc_trm(anio, ipc, trm, df)
    out["anio"] = anios
    out["ipc"] = ipc_s
    out["trm"] = trm_s

    for c in out.columns:
        if c.startswith("prop_") or c.startswith("tfidf_"):
            out[c] = out[c].fillna(0.0)

    for c in FEATURES_33:
        if c not in out.columns:
            out[c] = 0.0

    keep = list(dict.fromkeys(FEATURES_33 + ["id_contrato", "n_riesgos"]))
    keep = [c for c in keep if c in out.columns]
    return out[keep]
