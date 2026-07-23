"""
Generate model artifacts for RandomForest champion:
- feature_names.pkl (38 feature names)
- permutation_importance.csv (for RF)
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent

# Load data
df = pd.read_csv(ROOT / 'docs' / 'consolidado_38_features.csv')
target = 'sobrecosto'
feature_cols = [c for c in df.columns if c not in ['id_contrato', target]]
X = df[feature_cols].copy()
y = df[target].copy()
X = X.fillna(X.median())

# Load trained model
rf = joblib.load(ROOT / 'models' / 'modelo_campeon.pkl')
scaler = joblib.load(ROOT / 'models' / 'scaler.pkl')

# 1. Save feature_names.pkl
print(f'Generating feature_names.pkl ({len(feature_cols)} features)...')
joblib.dump(feature_cols, ROOT / 'models' / 'feature_names.pkl')
print(f'[OK] feature_names.pkl')

# 2. Compute permutation importance
print('Computing permutation importance (may take a moment)...')
X_scaled = scaler.transform(X)
r = permutation_importance(
    rf, X_scaled, y,
    n_repeats=20, random_state=42, n_jobs=-1,
    scoring='neg_mean_absolute_error'
)
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': r.importances_mean,
    'std': r.importances_std,
}).sort_values('importance', ascending=False)
imp_df.to_csv(ROOT / 'models' / 'permutation_importance.csv', index=False)
print(f'[OK] permutation_importance.csv')
print(f'\nTop 10 features:')
print(imp_df.head(10).to_string(index=False))
print(f'\nBottom 5 features:')
print(imp_df.tail(5).to_string(index=False))

# 3. Generate feature_importances_rf.csv (for training_stats.py)
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_,
}).sort_values('importance', ascending=False)
fi_df.to_csv(ROOT / 'models' / 'feature_importances_rf.csv', index=False)
print(f'\n[OK] feature_importances_rf.csv (RF built-in importances)')
print(f'Top 5 by RF importance:')
print(fi_df.head(5).to_string(index=False))
