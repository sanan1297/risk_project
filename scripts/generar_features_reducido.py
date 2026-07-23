"""Regenerar contratos_features_reducido.csv con 357 contratos."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Cargar features completos
df = pd.read_csv("docs/contratos_features.csv", encoding="utf-8-sig")
macro = pd.read_csv("docs/contratos_macro.csv", encoding="utf-8-sig")
print(f"Features completas: {len(df)} contratos")

# Filtrar NaN + outliers > 200%
nan_mask = df["sobrecosto"].isna()
df = df[~nan_mask].copy()
print(f"NaN sobrecosto: {nan_mask.sum()} eliminados")
outliers = df["sobrecosto"] > 200
df = df[~outliers].copy()
print(f"Outliers >200%: {outliers.sum()} eliminados")
print(f"Dataset: {len(df)} contratos")

# Features (excluir id, fuente, sobrecosto)
exclude = {"id_contrato", "fuente", "sobrecosto"}
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].fillna(0)
y = df["sobrecosto"].values

# RandomForest importance
rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X, y)
imp = pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_})
imp = imp.sort_values("importance", ascending=False)

# Top 30
top30 = imp.head(30)["feature"].tolist()
print(f"Top 30 features:")
for i, (_, r) in enumerate(imp.head(30).iterrows()):
    print(f"  {i+1:2d}. {r['feature']:30s} {r['importance']:.4f}")

# Control variables from macro
macro_control = macro[["id_contrato", "anio_inicio", "ipc_acumulado", "trm_promedio"]].copy()
macro_control.columns = ["id_contrato", "anio", "ipc", "trm"]

# Merge
reducido = df[["id_contrato", "sobrecosto"] + top30].merge(macro_control, on="id_contrato", how="left")
print(f"Reducido: {len(reducido)} contratos, {reducido.shape[1]} columnas")

reducido.to_csv("docs/contratos_features_reducido.csv", index=False, encoding="utf-8-sig")
print("Guardado: docs/contratos_features_reducido.csv")
