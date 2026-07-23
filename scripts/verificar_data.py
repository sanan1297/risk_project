"""Verify updated data files."""
import pandas as pd

clean = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"matriz_clean: {clean.shape[0]} rows, {clean['id_contrato'].nunique()} contratos")

ids = sorted(clean["id_contrato"].unique())
print("IDs entre C-348 y C-366:")
for c in ids:
    n = int(c.split("-")[1])
    if 348 <= n <= 366:
        print(f"  {c}")

macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
print(f"\nmacro: {macro.shape[0]} contratos")
print("Ultimos 10:")
print(macro.tail(10).to_string())
