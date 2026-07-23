from pathlib import Path
from functools import lru_cache

import numpy as np
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CAT_COLS = ["tipo", "clase", "asignacion", "fuente_riesgo", "etapa", "categoria"]
NUM_COLS = ["probabilidad", "impacto", "valoracion"]
REQUIRED_COLS = ["id_contrato", "descripcion_riesgo", "probabilidad", "impacto", "tipo", "categoria"]

MAX_DURACION = 5


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


def compute_range_features(anio_inicio: int, anio_fin: int) -> dict:
    if anio_fin < anio_inicio:
        anio_fin = anio_inicio
    duracion = anio_fin - anio_inicio
    if duracion > MAX_DURACION:
        anio_fin = anio_inicio + MAX_DURACION
        duracion = MAX_DURACION
    ipc_trm_data = _load_ipc_trm()
    ipc_acum = 1.0
    trm_vals = []
    for y in range(anio_inicio, anio_fin + 1):
        d = ipc_trm_data.get(y, {"ipc": 3.0, "trm": 4000})
        ipc_acum *= (1 + d["ipc"] / 100)
        trm_vals.append(d["trm"])
    ipc_acum = (ipc_acum - 1) * 100
    trm_prom = float(np.mean(trm_vals))
    return {
        "anio_inicio": anio_inicio,
        "anio_fin": anio_fin,
        "duracion": duracion,
        "ipc_acumulado": round(ipc_acum, 2),
        "trm_promedio": round(trm_prom, 2),
    }


def resolve_macro_range(
    anio_inicio: int | None,
    anio_fin: int | None,
    ipc_override: float | None,
    trm_override: float | None,
) -> dict:
    if anio_inicio is None:
        anio_inicio = 2022
    if anio_fin is None:
        anio_fin = anio_inicio
    feat = compute_range_features(anio_inicio, anio_fin)
    if ipc_override is not None:
        feat["ipc_acumulado"] = ipc_override
    if trm_override is not None:
        feat["trm_promedio"] = trm_override
    return feat


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

        # Mitigation plan features
        if "plan_mitigacion" in g.columns:
            pm = g["plan_mitigacion"].astype(str).str.strip().replace("nan", "")
            has_pm = pm.ne("")
            row["pct_riesgos_con_mitigacion"] = has_pm.mean()
            non_empty = pm[has_pm]
            row["avg_longitud_mitigacion"] = non_empty.str.len().mean() if len(non_empty) > 0 else 0
            codes = non_empty[non_empty.str.len() < 30]
            row["n_distinct_codes_mitigacion"] = codes.nunique()
        else:
            row["pct_riesgos_con_mitigacion"] = 0.0
            row["avg_longitud_mitigacion"] = 0.0
            row["n_distinct_codes_mitigacion"] = 0.0

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
    anio_inicio: int | None = None,
    anio_fin: int | None = None,
    ipc_override: float | None = None,
    trm_override: float | None = None,
) -> pd.DataFrame:
    df = df.copy()

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "no especificado")

    feature_names_target = _load_features()

    out = _aggregate_basic_stats(df)

    cat_agg = _aggregate_cat_props(df)
    out = out.merge(cat_agg, on="id_contrato", how="left")

    tfidf_df = _aggregate_tfidf(df)
    out = out.merge(tfidf_df, on="id_contrato", how="left")

    macro = resolve_macro_range(anio_inicio, anio_fin, ipc_override, trm_override)
    out["anio_inicio"] = macro["anio_inicio"]
    out["anio_fin"] = macro["anio_fin"]
    out["duracion"] = macro["duracion"]
    out["ipc_acumulado"] = macro["ipc_acumulado"]
    out["trm_promedio"] = macro["trm_promedio"]

    for c in out.columns:
        if c.startswith("prop_") or c.startswith("tfidf_"):
            out[c] = out[c].fillna(0.0)

    missing = [c for c in feature_names_target if c not in out.columns]
    if missing:
        out = out.assign(**{c: 0.0 for c in missing})

    keep = list(dict.fromkeys(feature_names_target + ["id_contrato", "n_riesgos"]))
    keep = [c for c in keep if c in out.columns]
    return out[keep]
