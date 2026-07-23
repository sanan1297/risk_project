"""
Entrenamiento con contratos C-001 a C-351 que existen en macro y clean.
Skip: C-352 a C-365 (para testing).
35 features: 30 TF-IDF/proporciones + 5 macro.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, str(Path.cwd()))
from estudio_data.features import engineer_features, STOP_WORDS
import warnings
warnings.filterwarnings('ignore')

ROOT = Path.cwd()

# Load data
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv")
clean = pd.read_csv(ROOT / "docs" / "matriz_clean.csv", encoding="utf-8-sig")

# Filter to training contracts: those in macro only (C-001 to C-351)
train_ids = set(macro["id_contrato"].unique())
clean = clean[clean["id_contrato"].isin(train_ids)].copy()
print(f"Training: {clean['id_contrato'].nunique()} contratos, {len(clean)} rows")

# Feature engineering
df_feat = engineer_features(clean)
n_antes = len(df_feat)
df_feat = df_feat[df_feat["sobrecosto"] < 200].copy()
print(f"Outliers >200%: {n_antes - len(df_feat)}")

# Merge macro
df_feat = df_feat.merge(macro, on="id_contrato", how="left")

# Collect all feature columns (everything except id, fuente, sobrecosto)
exclude = {"id_contrato", "fuente", "sobrecosto"}
macro_vars = ["anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]
feat_cols = [c for c in df_feat.columns if c not in exclude and c not in macro_vars]

# Fill NaN
for c in feat_cols:
    df_feat[c] = df_feat[c].fillna(0)
for c in macro_vars:
    df_feat[c] = df_feat[c].fillna(2022 if c in ["anio_inicio", "anio_fin", "duracion"] else 0)

# RandomForest to find top 30 features
X_all = df_feat[feat_cols].values.astype(np.float64)
y_all = df_feat["sobrecosto"].values.astype(np.float64)

rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_all, y_all)
top30_idx = np.argsort(rf.feature_importances_)[::-1][:30]
top30 = [feat_cols[i] for i in top30_idx]

print(f"\nTop 30 features (n={len(df_feat)}):")
for i, f in enumerate(top30):
    print(f"  {i+1:2d}. {f:30s} {rf.feature_importances_[top30_idx[i]]:.4f}")

# Build 35-feature set
FEATURES_35 = top30 + macro_vars
X = df_feat[FEATURES_35].values.astype(np.float64)
y = df_feat["sobrecosto"].values.astype(np.float64)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVR
svr = SVR(kernel="rbf", C=10, gamma="scale")
svr_scores = cross_val_score(svr, X_scaled, y, cv=5, scoring="r2")
svr.fit(X_scaled, y)
y_pred = svr.predict(X_scaled)
rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
print(f"\nSVR: R2 CV={svr_scores.mean():.4f}+/-{svr_scores.std():.4f} | RMSE={rmse:.2f} | R2 full={r2_score(y, y_pred):.4f}")

# Ridge
ridge = Ridge(alpha=244.0)
ridge_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")
ridge.fit(X_scaled, y)
print(f"Ridge: R2 CV={ridge_scores.mean():.4f}+/-{ridge_scores.std():.4f}")

# Logistic Regression
y_bin = (y > 25).astype(int)
lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
lr_scores = cross_val_score(lr, X_scaled, y_bin, cv=5, scoring="roc_auc")
lr.fit(X_scaled, y_bin)
print(f"LogisticRegression: AUC CV={lr_scores.mean():.4f}+/-{lr_scores.std():.4f}")
print(f"Prevalence >25%: {y_bin.mean():.1%} ({y_bin.sum()}/{len(y_bin)})")

# Save model and features for later use
import joblib
joblib.dump(svr, ROOT / "models" / "svr_343.pkl")
joblib.dump(scaler, ROOT / "models" / "scaler_343.pkl")
joblib.dump(FEATURES_35, ROOT / "models" / "features_35_343.pkl")
print("\nSaved: models/svr_343.pkl, scaler_343.pkl, features_35_343.pkl")
print("Done.")
