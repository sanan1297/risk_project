"""
Entrenamiento final de modelos para produccion (100% datos)
575 contratos, 38 features
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)

ROOT = Path(__file__).resolve().parent.parent

# Cargar datos
df = pd.read_csv(ROOT / 'docs' / 'consolidado_38_features.csv')
target = 'sobrecosto'
feature_cols = [c for c in df.columns if c not in ['id_contrato', target]]
X_full = df[feature_cols].copy()
y_full = df[target].copy()
X_full = X_full.fillna(X_full.median())

print(f'Dataset: {len(df)} contratos x {df.shape[1]} columnas')
print(f'Features: {len(feature_cols)} | Target: {target}')
print('=' * 60)

# ====== 1. PREDICTOR RandomForest ======
print('\n--- 1. RandomForest (predictor de sobrecosto) ---')
rf_final = RandomForestRegressor(
    n_estimators=390, max_depth=19, max_features='log2',
    min_samples_leaf=3, min_samples_split=8,
    random_state=42, n_jobs=-1
)
scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(X_full)
rf_final.fit(X_full_scaled, y_full)
y_pred_full = rf_final.predict(X_full_scaled)
print(f'R2 en 100% datos: {r2_score(y_full, y_pred_full):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_full, y_pred_full)):.2f} pp')
print(f'MAE: {mean_absolute_error(y_full, y_pred_full):.2f} pp')
print(f'Predicciones - Min: {y_pred_full.min():.1f}% Max: {y_pred_full.max():.1f}% Media: {y_pred_full.mean():.2f}%')
print(f'Reales     - Min: {y_full.min():.1f}% Max: {y_full.max():.1f}% Media: {y_full.mean():.2f}%')

joblib.dump(rf_final, ROOT / 'models' / 'modelo_campeon.pkl')
joblib.dump(scaler_final, ROOT / 'models' / 'scaler.pkl')
print('[OK] modelo_campeon.pkl + scaler.pkl guardados')

# ====== 2. CLASIFICADOR ALTO RIESGO ======
print('\n--- 2. Clasificador de alto riesgo (>25%) ---')
y_full_bin = (y_full > 25).astype(int)
print(f'Alto riesgo: {y_full_bin.sum()}/{len(y_full_bin)} ({y_full_bin.mean()*100:.1f}%)')

clf_final = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_full)
clf_final.fit(X_clf_scaled, y_full_bin)
y_clf_pred = clf_final.predict(X_clf_scaled)
y_clf_proba = clf_final.predict_proba(X_clf_scaled)[:, 1]
print(f'AUC: {roc_auc_score(y_full_bin, y_clf_proba):.3f}')
print(f'Accuracy: {accuracy_score(y_full_bin, y_clf_pred):.3f}')
print(f'Precision: {precision_score(y_full_bin, y_clf_pred):.3f}')
print(f'Recall: {recall_score(y_full_bin, y_clf_pred):.3f}')
print(f'F1: {f1_score(y_full_bin, y_clf_pred):.3f}')

joblib.dump(clf_final, ROOT / 'models' / 'classifier.pkl')
print('[OK] classifier.pkl guardado (RandomForestClassifier)')

# ====== 3. RMSE PREDICTOR (benchmark + best) ======
print('\n--- 3. RMSE Predictor (benchmark CV) ---')
y_err_full = np.abs(y_full - rf_final.predict(X_full_scaled))
print(f'Error absoluto medio del predictor: {y_err_full.mean():.2f} pp')

RMSE_MODELS = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'SVR Linear': SVR(kernel='linear', C=1.0),
    'SVR RBF': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}
cv = KFold(5, shuffle=True, random_state=42)
print(f"{'Modelo':20s} {'MAE':>8s} {'R2':>8s}")
print('-' * 40)
resultados = {}
for nombre, md in RMSE_MODELS.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('m', md)])
    cv_mae = cross_val_score(pipe, X_full, y_err_full, cv=cv, scoring='neg_mean_absolute_error')
    cv_r2 = cross_val_score(pipe, X_full, y_err_full, cv=cv, scoring='r2')
    resultados[nombre] = {'MAE': -cv_mae, 'R2': cv_r2}
    print(f'{nombre:20s} {-cv_mae.mean():7.3f}~{cv_mae.std():.3f} {cv_r2.mean():7.3f}~{cv_r2.std():.3f}')

mejor = max(resultados, key=lambda k: resultados[k]['R2'].mean())
print(f'\n>>> Mejor RMSE model: {mejor}')
print(f'    R2={resultados[mejor]["R2"].mean():.3f} MAE={resultados[mejor]["MAE"].mean():.2f}')

rmse_final = RMSE_MODELS[mejor]
rmse_final.fit(X_full_scaled, y_err_full)
y_rmse_pred = rmse_final.predict(X_full_scaled)
print(f'RMSE Predictor (entrenado 100%) - MAE: {mean_absolute_error(y_err_full, y_rmse_pred):.3f} pp | R2: {r2_score(y_err_full, y_rmse_pred):.3f}')

joblib.dump(rmse_final, ROOT / 'models' / 'rmse_predictor.pkl')
print(f'[OK] rmse_predictor.pkl guardado -> {mejor}')

# ====== RESUMEN ======
print('\n' + '=' * 60)
print('  MODELOS PRODUCCION (100% DATOS)')
print('=' * 60)
print(f'  Predictor:    RandomForest (390 trees)')
print(f'  Clasificador: RandomForestClassifier')
print(f'  RMSE Dyn:     {mejor}')
print(f'  Data:         {len(X_full)} contratos, {X_full.shape[1]} features')
print(f'  Guardado en:  models/')
print('=' * 60)
