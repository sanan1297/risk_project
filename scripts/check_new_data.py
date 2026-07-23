"""Check new contracts data."""
import pandas as pd

# Check risk files
for cid in ["C-360", "C-361", "C-362", "C-363", "C-364", "C-365"]:
    path = f"tests/data/{cid.lower()}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"{cid}: {df.shape[0]} riesgos, cols={list(df.columns)}")
    print(f"  prob vals={sorted(df['probabilidad'].unique())}")
    print(f"  imp vals={sorted(df['impacto'].unique())}")
    print(f"  tipo vals={df['tipo'].unique()}")
    print()

# Check macro
macro = pd.read_csv("tests/data/contratos_prueba.csv", encoding="utf-8-sig")
print("Macro data for new contracts:")
for _, r in macro.iterrows():
    cid = r["id_contrato"]
    if cid in ["C-360","C-361","C-362","C-363","C-364","C-365"]:
        print(f"  {cid}: anios={r['anio_inicio']}-{r['anio_fin']}, "
              f"v_ini={r['valor_inicial']}, v_fin={r['valor_final']}, "
              f"sc={r['sobrecosto_real']}")

# Check existing matriz_clean columns
clean = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"\nmatriz_clean columns: {list(clean.columns)}")
print(f"matriz_clean dtypes:")
for c in clean.columns:
    print(f"  {c}: {clean[c].dtype}")
