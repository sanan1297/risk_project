import numpy as np
import pandas as pd

from .predictor import load as load_model
from .feature_engineering import aggregate_risks

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
    rmse = _rmse_por_contrato(n_riesgos)
    probs_orig = df_contrato["probabilidad"].values.astype(float)
    imps_orig = df_contrato["impacto"].values.astype(float)

    X_base = _build_x(df_feat, feature_names, probs_orig, imps_orig, idx_var)
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

    riesgo_cuantitativo = _desglose_por_riesgo(
        df_contrato, probs_orig, imps_orig,
        df_feat, feature_names, idx_var,
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


def _desglose_por_riesgo(
    df_contrato, probs_orig, imps_orig,
    df_feat, feature_names, idx_var,
    scaler, regressor, pred_base,
):
    n = len(df_contrato)
    items = []
    prod_total = (probs_orig * imps_orig).sum()
    for i in range(n):
        peso = (probs_orig[i] * imps_orig[i]) / prod_total if prod_total > 0 else 0
        contribucion = pred_base * peso
        items.append({
            "riesgo": str(df_contrato.iloc[i].get("descripcion_riesgo", f"Riesgo {i + 1}")),
            "tipo": str(df_contrato.iloc[i].get("tipo", "")),
            "categoria": str(df_contrato.iloc[i].get("categoria", "")),
            "probabilidad": int(probs_orig[i]),
            "impacto": int(imps_orig[i]),
            "peso_contribucion": round(peso, 4),
            "contribucion_porcentaje": round(contribucion, 2),
        })
    items.sort(key=lambda x: x["contribucion_porcentaje"], reverse=True)
    return items
