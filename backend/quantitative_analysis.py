import re
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from .predictor import load as load_model
from .feature_engineering import aggregate_risks

ROOT = Path(__file__).resolve().parent.parent

VAR_FEATURE_NAMES = [
    "prob_promedio",
    "prob_std",
    "imp_promedio",
    "interaccion_prob_x_impacto",
]


def _rmse_por_contrato(n_riesgos: int) -> float:
    if n_riesgos <= 10:
        return 12.0
    elif n_riesgos <= 20:
        return 16.0
    elif n_riesgos <= 30:
        return 20.0
    else:
        return 24.0


@lru_cache(maxsize=1)
def _load_rmse_predictor():
    """Cargar modelo de predicción de error (RMSE dinámico)."""
    path = ROOT / "models" / "rmse_predictor.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _build_x(df_feat, feature_names, probs, imps, idx_var):
    x = df_feat[feature_names].values.copy()
    pm = probs.mean()
    ps = probs.std(ddof=0)
    im = imps.mean()
    inter = pm * im
    x[0, idx_var] = [pm, ps, im, inter]
    return x


def _valor_inicial_contrato(df_contrato: pd.DataFrame, valor_inicial: float | None = None) -> float | None:
    if valor_inicial is not None:
        return valor_inicial
    if "valor_inicial" in df_contrato.columns:
        vals = df_contrato["valor_inicial"].dropna().unique()
        if len(vals) > 0:
            return float(vals[0])
    return None


def _cop(valor, vi):
    if vi is None:
        return None
    return round(valor * vi / 100, 2)


@lru_cache(maxsize=1)
def _load_shap_background(n_samples: int = 100) -> pd.DataFrame:
    """Load SHAP background data: 100 random contracts from training set."""
    cache_path = ROOT / "models" / "shap_background.pkl"
    if cache_path.exists():
        return joblib.load(cache_path)

    matriz = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
    macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")

    rng = np.random.default_rng(42)
    contratos = matriz["id_contrato"].unique()
    sampled = list(rng.choice(contratos, size=min(n_samples, len(contratos)), replace=False))

    macro_map = macro.set_index("id_contrato")[["anio_inicio", "anio_fin"]].to_dict("index")
    feat_names: list[str] = joblib.load(ROOT / "models" / "feature_names.pkl")

    rows: list[pd.DataFrame] = []
    for cid in sampled:
        sub = matriz[matriz["id_contrato"] == cid].copy()
        feats = macro_map.get(cid, {"anio_inicio": 2022, "anio_fin": 2022})
        feat = aggregate_risks(sub, anio_inicio=feats["anio_inicio"], anio_fin=feats["anio_fin"])
        rows.append(feat)

    bg = pd.concat(rows, ignore_index=True)
    for c in feat_names:
        if c not in bg.columns:
            bg[c] = 0.0
    bg = bg[feat_names]

    joblib.dump(bg, cache_path)
    return bg


@lru_cache(maxsize=1)
def _get_shap_explainer():
    """Get or create cached SHAP KernelExplainer for the SVR model."""
    regressor, _, scaler, feature_names = load_model()
    background = _load_shap_background()
    bg_raw = background[feature_names].values.astype(np.float64)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return regressor.predict(scaler.transform(X))

    return shap.KernelExplainer(predict_fn, bg_raw)


_PROP_PREFIX_TO_COL = {
    "tipo": "tipo",
    "clas": "clase",
    "asig": "asignacion",
    "fuen": "fuente_riesgo",
    "etap": "etapa",
    "cate": "categoria",
}

_SHARED_WEIGHTED = {
    "prob_promedio", "prob_std", "imp_promedio",
    "interaccion_prob_x_impacto", "prob_min", "prob_max",
    "imp_min", "imp_max", "prob_rango", "imp_rango",
    "val_promedio", "suma_valoracion", "suma_probabilidad",
    "suma_impacto", "val_std", "val_min", "val_max", "val_rango",
    "val_p25", "val_p75", "prob_p25", "prob_p75", "imp_p25", "imp_p75",
}

_EQUAL_SPLIT = {
    "n_riesgos", "anio_inicio", "anio_fin", "duracion",
    "ipc_acumulado", "trm_promedio", "valor_inicial",
}


def _map_shap_to_risks(
    shap_vals: np.ndarray,
    df_contrato: pd.DataFrame,
    feature_names: list[str],
    probs: np.ndarray,
    imps: np.ndarray,
) -> np.ndarray:
    n = len(df_contrato)
    contributions = np.zeros(n)
    prod = probs * imps
    prod_total = prod.sum()
    if prod_total <= 0:
        prod_total = float(n)
        weq = 1.0
    else:
        weq = prod / prod_total

    for idx, feat_name in enumerate(feature_names):
        sv = shap_vals[idx]
        if sv == 0.0:
            continue

        if feat_name in _EQUAL_SPLIT:
            contributions[:] += sv / n

        elif feat_name in _SHARED_WEIGHTED:
            contributions += sv * weq

        elif feat_name.startswith("prop_"):
            rest = feat_name[5:]
            parts = rest.split("_", 1)
            if len(parts) != 2:
                contributions[:] += sv / n
                continue
            prefix, value = parts
            col_name = _PROP_PREFIX_TO_COL.get(prefix)
            if col_name is None or col_name not in df_contrato.columns:
                contributions[:] += sv / n
                continue
            mask = (df_contrato[col_name].astype(str).str.strip().str.lower() == value).values
            n_mask = mask.sum()
            if n_mask == 0:
                contributions[:] += sv / n
            else:
                prod_mask = prod[mask]
                ptotal = prod_mask.sum()
                if ptotal > 0:
                    contributions[mask] += sv * (prod_mask / ptotal)
                else:
                    contributions[mask] += sv / n_mask

        elif feat_name.startswith("tfidf_"):
            word = feat_name[6:]
            descs = df_contrato["descripcion_riesgo"].astype(str).str.lower()
            contains = np.array([word in d for d in descs])
            n_contains = contains.sum()
            if n_contains == 0:
                contributions[:] += sv / n
            else:
                prod_cont = prod[contains]
                ptotal = prod_cont.sum()
                if ptotal > 0:
                    contributions[contains] += sv * (prod_cont / ptotal)
                else:
                    contributions[contains] += sv / n_contains

        else:
            contributions[:] += sv / n

    return contributions


def _desglose_por_riesgo_shap(
    df_contrato: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_names: list[str],
    idx_var: list[int],
    scaler,
    regressor,
    pred_base: float,
) -> list[dict]:
    n = len(df_contrato)
    probs = df_contrato["probabilidad"].values.astype(float)
    imps = df_contrato["impacto"].values.astype(float)

    explainer = _get_shap_explainer()

    X_raw = df_feat[feature_names].values.astype(np.float64).reshape(1, -1)
    shap_values = explainer.shap_values(X_raw, nsamples=1000)

    if shap_values.ndim == 2:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    per_risk = _map_shap_to_risks(shap_vals, df_contrato, feature_names, probs, imps)
    per_risk += explainer.expected_value / n

    prod_total = (probs * imps).sum() or 1.0
    items = []
    for i in range(n):
        peso = (probs[i] * imps[i]) / prod_total
        items.append({
            "riesgo": str(df_contrato.iloc[i].get("descripcion_riesgo", f"Riesgo {i + 1}")),
            "tipo": str(df_contrato.iloc[i].get("tipo", "")),
            "categoria": str(df_contrato.iloc[i].get("categoria", "")),
            "probabilidad": int(probs[i]),
            "impacto": int(imps[i]),
            "peso_contribucion": round(peso, 4),
            "contribucion_porcentaje": round(per_risk[i], 2),
        })
    items.sort(key=lambda x: x["contribucion_porcentaje"], reverse=True)
    return items


def compute(
    df_contrato: pd.DataFrame,
    anio_inicio: int | None = None,
    anio_fin: int | None = None,
    ipc_override: float | None = None,
    trm_override: float | None = None,
    n_iteraciones: int = 1000,
    seed: int = 42,
    incluir_ruido: bool = True,
    valor_inicial: float | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    regressor, _classifier, scaler, feature_names = load_model()

    vi = _valor_inicial_contrato(df_contrato, valor_inicial)

    idx_var = [list(feature_names).index(n) for n in VAR_FEATURE_NAMES]

    df_feat = aggregate_risks(df_contrato, anio_inicio=anio_inicio, anio_fin=anio_fin, ipc_override=ipc_override, trm_override=trm_override)
    n_riesgos = len(df_contrato)
    probs_orig = df_contrato["probabilidad"].values.astype(float)
    imps_orig = df_contrato["impacto"].values.astype(float)

    X_base = _build_x(df_feat, feature_names, probs_orig, imps_orig, idx_var)

    # RMSE dinámico: predecir error esperado del SVR (con fallback a heurística)
    rmse_predictor = _load_rmse_predictor()
    if rmse_predictor is not None:
        X_s = scaler.transform(X_base)
        rmse_pred = float(rmse_predictor.predict(X_s)[0])
        rmse_heur = _rmse_por_contrato(n_riesgos)
        rmse = max(rmse_pred, rmse_heur * 0.85, 2.0)  # piso 2pp, mínimo 85% de heurística
    else:
        rmse = _rmse_por_contrato(n_riesgos)

    pred_base = float(regressor.predict(scaler.transform(X_base))[0])

    muestras = np.empty(n_iteraciones)
    for i in range(n_iteraciones):
        delta_prob = rng.integers(-1, 2, size=n_riesgos)
        delta_imp = rng.integers(-1, 2, size=n_riesgos)
        probs = np.clip(probs_orig + delta_prob, 1, 5)
        imps = np.clip(imps_orig + delta_imp, 1, 5)
        X = _build_x(df_feat, feature_names, probs, imps, idx_var)
        X_s = scaler.transform(X)
        pred = regressor.predict(X_s)[0]
        if incluir_ruido:
            pred += rng.normal(0, rmse)
        muestras[i] = pred

    percentiles = {}
    percentiles_cop = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = round(float(np.percentile(muestras, p)), 2)
        key = f"P{p:02d}"
        percentiles[key] = val
        if vi is not None:
            percentiles_cop[key] = _cop(val, vi)

    stats = {
        "media": round(float(np.mean(muestras)), 2),
        "std": round(float(np.std(muestras)), 2),
        "min": round(float(np.min(muestras)), 2),
        "max": round(float(np.max(muestras)), 2),
        "prediccion_central": round(pred_base, 2),
    }
    stats_cop = None
    if vi is not None:
        stats_cop = {k: _cop(v, vi) for k, v in stats.items()}

    dist_bins = np.linspace(muestras.min(), muestras.max(), 21)
    dist_hist = np.digitize(muestras, dist_bins[:-1]) - 1
    histograma = [
        {
            "bin_inicio": round(float(dist_bins[i]), 2),
            "bin_fin": round(float(dist_bins[i + 1]), 2),
            "frecuencia": int((dist_hist == i).sum()),
        }
        for i in range(len(dist_bins) - 1)
    ]
    histograma_cop = None
    if vi is not None:
        histograma_cop = [
            {"bin_inicio": _cop(b["bin_inicio"], vi), "bin_fin": _cop(b["bin_fin"], vi), "frecuencia": b["frecuencia"]}
            for b in histograma
        ]

    tornado = _tornado(
        df_contrato, df_feat, feature_names, idx_var,
        probs_orig, imps_orig, scaler, regressor, pred_base,
    )
    tornado_cop = None
    if vi is not None:
        tornado_cop = []
        for t in tornado:
            t2 = dict(t)
            t2["prediccion_alta"] = _cop(t["prediccion_alta"], vi)
            t2["prediccion_baja"] = _cop(t["prediccion_baja"], vi)
            t2["swing"] = _cop(t["swing"], vi)
            tornado_cop.append(t2)

    riesgo_cuantitativo = _desglose_por_riesgo_shap(
        df_contrato, df_feat, feature_names, idx_var,
        scaler, regressor, pred_base,
    )
    riesgo_cuantitativo_cop = None
    if vi is not None:
        riesgo_cuantitativo_cop = []
        for r in riesgo_cuantitativo:
            r2 = dict(r)
            r2["contribucion_porcentaje"] = _cop(r["contribucion_porcentaje"], vi)
            riesgo_cuantitativo_cop.append(r2)

    result: dict = {
        "prediccion_central": round(pred_base, 2),
        "percentiles": percentiles,
        "stats": stats,
        "histograma": histograma,
        "tornado": tornado,
        "riesgos": riesgo_cuantitativo,
        "n_simulaciones": n_iteraciones,
        "rmse": rmse,
        "ruido_incluido": incluir_ruido,
    }

    if vi is not None:
        result["valor_inicial"] = vi
        result["percentiles_cop"] = percentiles_cop
        result["stats_cop"] = stats_cop
        result["histograma_cop"] = histograma_cop
        result["tornado_cop"] = tornado_cop
        result["riesgos_cop"] = riesgo_cuantitativo_cop

    return result


def _tornado(
    df_contrato, df_feat, feature_names, idx_var,
    probs_orig, imps_orig, scaler, regressor, pred_base,
):
    tipos = df_contrato["tipo"].unique()
    items = []
    for tipo in tipos:
        mask = (df_contrato["tipo"].values == tipo).astype(int)

        probs = np.clip(probs_orig + mask, 1, 5)
        imps = np.clip(imps_orig + mask, 1, 5)
        X = _build_x(df_feat, feature_names, probs, imps, idx_var)
        pred_alta = float(regressor.predict(scaler.transform(X))[0])

        probs = np.clip(probs_orig - mask, 1, 5)
        imps = np.clip(imps_orig - mask, 1, 5)
        X = _build_x(df_feat, feature_names, probs, imps, idx_var)
        pred_baja = float(regressor.predict(scaler.transform(X))[0])

        items.append({
            "riesgo": f"Tipo: {tipo}",
            "tipo": str(tipo),
            "categoria": "",
            "probabilidad_original": int(round(probs_orig[mask.astype(bool)].mean())),
            "impacto_original": int(round(imps_orig[mask.astype(bool)].mean())),
            "prediccion_alta": round(pred_alta, 2),
            "prediccion_baja": round(pred_baja, 2),
            "swing": round(abs(pred_alta - pred_baja), 2),
            "direccion": "aumenta" if pred_alta > pred_baja else "disminuye",
            "n_riesgos": int(mask.sum()),
        })

    items.sort(key=lambda x: x["swing"], reverse=True)
    return items



