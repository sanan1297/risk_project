"""
Enrich test contracts in tests/data/ with mitigations from matriz_clean.csv
and generate predictions using the new RF model.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.feature_engineering import aggregate_risks
from backend.predictor import predict as run_prediction

TEST_DIR = ROOT / "tests" / "data"
MATRIZ_PATH = ROOT / "docs" / "matriz_clean.csv"
MACRO_PATH = ROOT / "docs" / "contratos_macro.csv"

matriz = pd.read_csv(MATRIZ_PATH, encoding="utf-8-sig")
print(f"Matriz clean: {len(matriz)} riesgos, {matriz['id_contrato'].nunique()} contratos")

# Macro data for year ranges
macro = pd.read_csv(MACRO_PATH)
macro_map = macro.set_index("id_contrato")[["anio_inicio", "anio_fin"]].to_dict("index")

test_files = sorted(TEST_DIR.glob("c-*.csv"))
print(f"Test contracts: {len(test_files)}")

results = []

for fpath in test_files:
    cid = fpath.stem.upper().replace("_TEMPORAL", "")
    df_test = pd.read_csv(fpath, encoding="utf-8-sig")

    if "id_contrato" not in df_test.columns:
        print(f"\n--- {fpath.name} (SKIP - no id_contrato) ---")
        continue

    n_orig = len(df_test)
    df_merged = df_test.merge(
        matriz[["id_contrato", "descripcion_riesgo", "plan_mitigacion"]],
        on=["id_contrato", "descripcion_riesgo"],
        how="left"
    )
    n_match = df_merged["plan_mitigacion"].notna().sum()
    df_merged["plan_mitigacion"] = df_merged["plan_mitigacion"].fillna("")

    # Get year range from macro data
    years = macro_map.get(cid, {"anio_inicio": 2022, "anio_fin": 2023})
    anio_inicio = years["anio_inicio"]
    anio_fin = years["anio_fin"]

    # Compute features
    df_feat = aggregate_risks(df_merged, anio_inicio=anio_inicio, anio_fin=anio_fin)

    # Predict
    pred = run_prediction(df_feat)

    p = pred["predicciones"][0]
    prob = pred["probabilidades"][0]
    alert = pred["alertas"][0]

    print(f"\n--- {cid} ({n_orig} riesgos, {n_match} mitig) ---")
    print(f"  Pred: {p:.1f}% | Prob AR: {prob:.3f} | {alert}")

    results.append({
        "id_contrato": cid,
        "n_riesgos": n_orig,
        "n_mitigaciones": n_match,
        "prediccion": round(p, 2),
        "prob_alto_riesgo": round(prob, 4),
        "alerta": alert,
    })

# Summary
print("\n" + "=" * 90)
print(f"{'Contrato':12s} {'Riesgos':8s} {'Mitig':6s} {'Prediccion':10s} {'Prob AR':8s} {'Alerta'}")
print("-" * 90)
for r in results:
    print(f"{r['id_contrato']:12s} {r['n_riesgos']:8d} {r['n_mitigaciones']:6d} {r['prediccion']:8.2f}%  {r['prob_alto_riesgo']:.3f}     {r['alerta']}")

# Save enriched CSVs
enriched_dir = TEST_DIR / "enriched"
enriched_dir.mkdir(exist_ok=True)
for fpath in test_files:
    cid_raw = fpath.stem.upper().replace("_TEMPORAL", "")
    df_test = pd.read_csv(fpath, encoding="utf-8-sig")
    if "id_contrato" not in df_test.columns:
        continue
    df_merged = df_test.merge(
        matriz[["id_contrato", "descripcion_riesgo", "plan_mitigacion"]],
        on=["id_contrato", "descripcion_riesgo"],
        how="left"
    )
    df_merged["plan_mitigacion"] = df_merged["plan_mitigacion"].fillna("")
    out_path = enriched_dir / fpath.name
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nEnriquecidos guardados en: {enriched_dir}")
