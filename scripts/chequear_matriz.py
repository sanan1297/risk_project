import pandas as pd

df = pd.read_csv("docs/matriz_clean.csv", encoding="utf-8-sig")
print(f"matriz_clean: {len(df)} filas, {df['id_contrato'].nunique()} contratos")
ids = sorted(df["id_contrato"].unique())
print(f"IDs: {ids[0]} ... {ids[-1]}")
print(f"Total: {len(ids)}")
# Check gap
for c in ids:
    n = int(c.split("-")[1])
    if 340 <= n <= 370 or n == 21621:
        print(f"  {c}")
