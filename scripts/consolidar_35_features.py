"""
Consolidar matriz_clean → 429 registros, extraer 35 features (30 riesgo + 5 macro).
GUARDAR SOLO. No entrenar.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, str(Path.cwd()))
from estudio_data.features import engineer_features

ROOT = Path.cwd()

# 1. Load
clean = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
print(f"matriz_clean: {len(clean)} rows, {clean['id_contrato'].nunique()} contratos")

# 2. Consolidate: one row per contract
df = engineer_features(clean)
df = df[df["sobrecosto"] < 200].copy()  # remove outliers
print(f"Consolidado: {len(df)} contratos (outliers >200% removidos)")

# 3. Merge macro
df = df.merge(macro, on="id_contrato", how="left")
macro_vars = ["anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]

# 4. Find top 30 features from risk data (exclude macro vars, id, fuente, sobrecosto)
exclude = {"id_contrato", "fuente", "sobrecosto"}
risk_cols = [c for c in df.columns if c not in exclude and c not in macro_vars]

# Fill NaN for RF
X_rf = df[risk_cols].fillna(0).values.astype(np.float64)
y_rf = df["sobrecosto"].values.astype(np.float64)

rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
top30_idx = np.argsort(rf.feature_importances_)[::-1][:30]
top30 = [risk_cols[i] for i in top30_idx]

print(f"\nTop 30 features:")
for i, f in enumerate(top30):
    print(f"  {i+1:2d}. {f:30s} {rf.feature_importances_[top30_idx[i]]:.4f}")

# 5. Build 35-feature dataset
FEATURES_35 = top30 + macro_vars
df_35 = df[["id_contrato", "sobrecosto"] + FEATURES_35].copy()

# Show NaN count per macro var
for v in macro_vars:
    nan_count = df_35[v].isna().sum()
    if nan_count > 0:
        print(f"\n{v}: {nan_count} NaN (contratos sin macro data)")

# 6. Save
df_35.to_csv(ROOT / "docs" / "consolidado_35_features.csv", index=False, encoding="utf-8-sig")
print(f"\nGuardado: docs/consolidado_35_features.csv ({len(df_35)} rows x {len(FEATURES_35)} features)")
