"""Enriquece contratos_prueba.csv con datos cuantitativos (Monte Carlo percentiles, stats, etc.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from backend.predictor import predict as run_prediction, load as load_model
from backend.feature_engineering import aggregate_risks
from backend.quantitative_analysis import compute as run_mc


TEST_DATA = Path(__file__).resolve().parent / "data"
CONTRATOS_CSV = TEST_DATA / "contratos_prueba.csv"


def _get_anios(df_contrato: pd.DataFrame) -> tuple[int | None, int | None]:
    if "anio_inicio" in df_contrato.columns:
        vals = df_contrato["anio_inicio"].dropna().unique()
        if len(vals) > 0:
            return int(vals[0]), None
    return None, None


def _get_valor_inicial(df_contrato: pd.DataFrame) -> float | None:
    if "valor_inicial" in df_contrato.columns:
        vals = df_contrato["valor_inicial"].dropna().unique()
        if len(vals) > 0:
            return float(vals[0])
    return None


def main():
    df_meta = pd.read_csv(CONTRATOS_CSV)

    rows = []

    for _, row in df_meta.iterrows():
        cid = row["id_contrato"]
        anio_inicio = int(row["anio_inicio"])
        anio_fin = int(row["anio_fin"])
        vi = float(row["valor_inicial"])

        csv_path = TEST_DATA / f"{cid.lower()}.csv"
        if not csv_path.exists():
            print(f"[SKIP] No se encontró {csv_path}")
            continue

        df_contrato = pd.read_csv(csv_path)
        print(f"[{cid}] {len(df_contrato)} riesgos, año {anio_inicio}-{anio_fin}")

        # Basic prediction
        df_feat = aggregate_risks(df_contrato, anio_inicio=anio_inicio, anio_fin=anio_fin)
        pred_result = run_prediction(df_feat)

        svr = pred_result["predicciones"][0]
        prob = pred_result["probabilidades"][0]
        alerta = pred_result["alertas"][0]
        n_riesgos = pred_result["n_riesgos"][0]

        # Monte Carlo
        mc = run_mc(
            df_contrato,
            anio_inicio=anio_inicio,
            anio_fin=anio_fin,
            n_iteraciones=1000,
            seed=42,
            incluir_ruido=True,
            valor_inicial=vi,
        )

        percentiles = mc["percentiles"]
        stats = mc["stats"]
        percentiles_cop = mc.get("percentiles_cop", {})
        stats_cop = mc.get("stats_cop", {})

        rows.append({
            "grupo": row["grupo"],
            "id_contrato": cid,
            "anio_inicio": anio_inicio,
            "anio_fin": anio_fin,
            "valor_inicial": vi,
            "valor_final": row["valor_final"],
            "sobrecosto_real": row["sobrecosto_real"],
            "perfil": row.get("perfil", ""),
            "n_riesgos": n_riesgos,
            "svr_prediccion": svr,
            "probabilidad": round(prob, 4),
            "alerta": alerta,
            "rmse": mc["rmse"],
            "mc_iteraciones": mc["n_simulaciones"],
            "prediccion_central": mc["prediccion_central"],
            "P05": percentiles.get("P05"),
            "P10": percentiles.get("P10"),
            "P25": percentiles.get("P25"),
            "P50": percentiles.get("P50"),
            "P75": percentiles.get("P75"),
            "P90": percentiles.get("P90"),
            "P95": percentiles.get("P95"),
            "media": stats.get("media"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "P90_minus_P10": round(percentiles.get("P90", 0) - percentiles.get("P10", 0), 2),
            "P90_cop": percentiles_cop.get("P90"),
            "P50_cop": percentiles_cop.get("P50"),
            "P10_cop": percentiles_cop.get("P10"),
            "media_cop": stats_cop.get("media"),
        })

    df_out = pd.DataFrame(rows)
    out_path = TEST_DATA / "contratos_prueba_enriquecido.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Archivo generado: {out_path}")
    print(f"Filas: {len(df_out)}, Columnas: {len(df_out.columns)}")
    print("\nColumnas:", ", ".join(df_out.columns))

    summary = df_out[["id_contrato", "sobrecosto_real", "svr_prediccion", "alerta", "rmse", "P10", "P50", "P90", "P90_minus_P10"]]
    print("\nResumen:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
