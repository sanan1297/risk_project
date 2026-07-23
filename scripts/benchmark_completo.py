"""Benchmark completo: 10 modelos x 35 features, 3 flujos."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

ROOT = Path.cwd()
np.random.seed(42)

MODELS = [
    ("Ridge", Ridge(alpha=1.0)),
    ("Lasso", Lasso(alpha=0.01, max_iter=10000)),
    ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
    ("KNN", KNeighborsRegressor(n_neighbors=7)),
    ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=42)),
    ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
    ("XGBoost", None),  # will import conditionally
    ("SVR", SVR(kernel="rbf", C=10, gamma="scale")),
    ("MLP", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
]

try:
    import xgboost as xgb
    MODELS[7] = ("XGBoost", xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
except ImportError:
    MODELS.pop(7)

# ============ Load data ============
print("=" * 70)
print("BENCHMARK COMPLETO: 10 MODELOS x 35 FEATURES x 3 FLUJOS")
print("=" * 70)

# 35 features: 30 TF-IDF + proporciones + 5 macro
TOP_30_FEATURES = [
    "tfidf_desarrollo", "interaccion_prob_x_impacto", "tfidf_insumos",
    "prob_std", "tfidf_expedicion", "tfidf_materiales", "imp_promedio",
    "tfidf_obra", "tfidf_ejecucion", "tfidf_contrato",
    "prop_tipo_operacional", "prob_promedio", "prop_cate_bajo",
    "valor_inicial", "tfidf_riesgo", "tfidf_tecnicas", "tfidf_municipio",
    "tfidf_obras", "tfidf_informacion", "prop_fuen_externo", "tfidf_cuando",
    "tfidf_disenos", "tfidf_ejecucion contrato", "tfidf_calidad",
    "tfidf_manejo", "prop_cate_alto", "tfidf_pago", "prop_tipo_economico",
    "prop_asig_entidad", "tfidf_falta",
]
CONTROL_VARS = ["anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]
FEATURES_35 = TOP_30_FEATURES + CONTROL_VARS

# Load features + macro
df_feat = pd.read_csv(ROOT / "docs" / "contratos_features.csv", encoding="utf-8-sig")
macro = pd.read_csv(ROOT / "docs" / "contratos_macro.csv", encoding="utf-8-sig")

# Merge: 30 features from contratos_features + 5 macro vars
data = df_feat[["id_contrato", "sobrecosto"] + TOP_30_FEATURES].merge(
    macro[["id_contrato"] + CONTROL_VARS], on="id_contrato"
)

# Drop NaN sobrecosto (C-365) and fill any missing features
data = data.dropna(subset=["sobrecosto"])
for f in FEATURES_35:
    if f not in data.columns:
        data[f] = 0.0

# Remove outliers > 200%
data = data[data["sobrecosto"] <= 200].copy()

y = data["sobrecosto"].values
y_bin = (y > 25).astype(int)
X = data[FEATURES_35].fillna(0).values

print(f"Dataset: {len(data)} contratos, {len(FEATURES_35)} features")
print(f"Sobrecosto >25%: {y_bin.sum()} ({y_bin.mean()*100:.1f}%)")
print()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ============ FLUJO 1: REGRESSION (Predictor) ============
print("=" * 70)
print("FLUJO 1: PREDICTOR — Regresion (R2, RMSE, MAE)")
print("=" * 70)

reg_results = []
for name, model in MODELS:
    t0 = time.time()
    r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
    neg_rmse = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")
    neg_mae = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2_full = r2_score(y, y_pred)
    elapsed = time.time() - t0
    reg_results.append({
        "modelo": name,
        "R2 CV": f"{r2_scores.mean():.4f} +/- {r2_scores.std():.4f}",
        "R2 full": f"{r2_full:.4f}",
        "RMSE CV": f"{-neg_rmse.mean():.2f} +/- {neg_rmse.std():.2f}",
        "MAE CV": f"{-neg_mae.mean():.2f} +/- {neg_mae.std():.2f}",
        "tiempo": f"{elapsed:.1f}s",
    })
    print(f"{name:20s} | R2 CV={r2_scores.mean():.4f}+/-{r2_scores.std():.4f} | RMSE={-neg_rmse.mean():.2f} | MAE={-neg_mae.mean():.2f} | R2 full={r2_full:.4f} | {elapsed:.1f}s")

reg_df = pd.DataFrame(reg_results)
print()

# Identify best regressor (by R2 CV mean)
best_reg_idx = np.argmax([float(r["R2 CV"].split(" ")[0]) for r in reg_results])
best_reg_name = MODELS[best_reg_idx][0]
best_reg_model = MODELS[best_reg_idx][1]
print(f"Mejor predictor: {best_reg_name} (R2 CV = {reg_results[best_reg_idx]['R2 CV']})")

# ============ FLUJO 2: CLASSIFIER (ROC Alert) ============
print()
print("=" * 70)
print("FLUJO 2: ALERTA — Clasificacion Binaria (>25%)")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

CLASSIFIERS = [
    ("Ridge", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
    ("Lasso", LogisticRegression(penalty="l1", C=0.01, solver="saga", max_iter=1000, class_weight="balanced", random_state=42)),
    ("ElasticNet", LogisticRegression(penalty="elasticnet", C=0.01, l1_ratio=0.5, solver="saga", max_iter=1000, class_weight="balanced", random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=7)),
    ("DecisionTree", DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced")),
    ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced")),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
    ("XGBoost", None),
    ("SVC", LinearSVC(C=10, max_iter=10000, class_weight="balanced", random_state=42)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
]

try:
    import xgboost as xgb
    CLASSIFIERS[7] = ("XGBoost", xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, scale_pos_weight=(~y_bin.astype(bool)).sum() / y_bin.sum()))
except ImportError:
    CLASSIFIERS.pop(7)

clf_results = []
for name, model in CLASSIFIERS:
    t0 = time.time()
    try:
        auc_scores = cross_val_score(model, X_scaled, y_bin, cv=cv, scoring="roc_auc")
        acc_scores = cross_val_score(model, X_scaled, y_bin, cv=cv, scoring="accuracy")
        model.fit(X_scaled, y_bin)
        y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_scaled)
        if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
            y_prob = model.decision_function(X_scaled)
        auc_full = roc_auc_score(y_bin, y_prob)
        y_pred = model.predict(X_scaled)
        elapsed = time.time() - t0
        clf_results.append({
            "modelo": name,
            "AUC CV": f"{auc_scores.mean():.4f} +/- {auc_scores.std():.4f}",
            "AUC full": f"{auc_full:.4f}",
            "Acc CV": f"{acc_scores.mean():.4f} +/- {acc_scores.std():.4f}",
            "Precision": f"{precision_score(y_bin, y_pred):.3f}",
            "Recall": f"{recall_score(y_bin, y_pred):.3f}",
            "F1": f"{f1_score(y_bin, y_pred):.3f}",
            "tiempo": f"{elapsed:.1f}s",
        })
        print(f"{name:20s} | AUC CV={auc_scores.mean():.4f}+/-{auc_scores.std():.4f} | AUC full={auc_full:.4f} | Prec={precision_score(y_bin, y_pred):.3f} Rec={recall_score(y_bin, y_pred):.3f} | {elapsed:.1f}s")
    except Exception as e:
        print(f"{name:20s} | ERROR: {e}")
        clf_results.append({"modelo": name, "AUC CV": "ERROR", "AUC full": "ERROR", "Acc CV": "ERROR", "Precision": "ERROR", "Recall": "ERROR", "F1": "ERROR", "tiempo": "ERROR"})

clf_df = pd.DataFrame(clf_results)
print()

# Identify best classifier
best_clf_idx = None
for i, r in enumerate(clf_results):
    if r["AUC CV"] != "ERROR":
        if best_clf_idx is None or float(r["AUC CV"].split(" ")[0]) > float(clf_results[best_clf_idx]["AUC CV"].split(" ")[0]):
            best_clf_idx = i

if best_clf_idx is not None:
    print(f"Mejor clasificador: {clf_results[best_clf_idx]['modelo']} (AUC CV = {clf_results[best_clf_idx]['AUC CV']})")

# Best classifier for alert
best_clf_model = CLASSIFIERS[best_clf_idx][1]
best_clf_model.fit(X_scaled, y_bin)

# ============ FLUJO 3: RMSE DINAMICO (Residuals) ============
print()
print("=" * 70)
print("FLUJO 3: RMSE DINAMICO — Prediccion del Error Absoluto")
print("=" * 70)

# Use the best regressor's predictions
best_reg_model.fit(X_scaled, y)
y_pred_best = best_reg_model.predict(X_scaled)
abs_errors = np.abs(y - y_pred_best)

print(f"Target: error absoluto del {best_reg_name}")
print(f"Media error: {abs_errors.mean():.2f} pp | Mediana: {np.median(abs_errors):.2f} pp")
print(f"RMSE del predictor: {np.sqrt(mean_squared_error(y, y_pred_best)):.2f} pp")

# Train meta-models on residuals
META_MODELS = [
    ("Ridge", Ridge(alpha=1.0)),
    ("Lasso", Lasso(alpha=0.01, max_iter=10000)),
    ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
    ("KNN", KNeighborsRegressor(n_neighbors=7)),
    ("DecisionTree", DecisionTreeRegressor(max_depth=10, random_state=42)),
    ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
    ("SVR", SVR(kernel="rbf", C=10, gamma="scale")),
    ("MLP", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
]

meta_results = []
for name, model in META_MODELS:
    t0 = time.time()
    r2_scores = cross_val_score(model, X_scaled, abs_errors, cv=cv, scoring="r2")
    neg_mae = cross_val_score(model, X_scaled, abs_errors, cv=cv, scoring="neg_mean_absolute_error")
    model.fit(X_scaled, abs_errors)
    y_pred_err = model.predict(X_scaled)
    r2_full = r2_score(abs_errors, y_pred_err)
    elapsed = time.time() - t0
    meta_results.append({
        "modelo": name,
        "R2 CV": f"{r2_scores.mean():.4f} +/- {r2_scores.std():.4f}",
        "R2 full": f"{r2_full:.4f}",
        "MAE CV": f"{-neg_mae.mean():.2f} +/- {neg_mae.std():.2f}",
        "tiempo": f"{elapsed:.1f}s",
    })
    print(f"{name:20s} | R2 CV={r2_scores.mean():.4f}+/-{r2_scores.std():.4f} | MAE CV={-neg_mae.mean():.2f} | R2 full={r2_full:.4f} | {elapsed:.1f}s")

meta_df = pd.DataFrame(meta_results)

# Best meta-model
best_meta_idx = np.argmax([float(r["R2 CV"].split(" ")[0]) for r in meta_results if r["R2 CV"] != "ERROR"])
print(f"\nMejor meta-model para RMSE dinamico: {meta_results[best_meta_idx]['modelo']} (R2 CV = {meta_results[best_meta_idx]['R2 CV']})")

# ============ SAVE & PRINT SUMMARY ============
print()
print("=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print()
print("1. MEJOR PREDICTOR (Regresion):")
for i, r in enumerate(reg_results):
    mark = " <<<" if i == best_reg_idx else ""
    print(f"   {r['modelo']:20s} | R2 CV={r['R2 CV']:20s} | RMSE={r['RMSE CV']:15s} | MAE={r['MAE CV']:15s}{mark}")
print()
print("2. MEJOR ALERTA (Clasificador ROC):")
for i, r in enumerate(clf_results):
    mark = " <<<" if i == best_clf_idx else ""
    print(f"   {r['modelo']:20s} | AUC CV={r['AUC CV']:20s} | AUC full={r['AUC full']:8s} | Prec={r['Precision']:8s} Rec={r['Recall']:8s}{mark}")
print()
print("3. RMSE DINAMICO (Meta-model sobre errores del predictor):")
for i, r in enumerate(meta_results):
    mark = " <<<" if i == best_meta_idx else ""
    print(f"   {r['modelo']:20s} | R2 CV={r['R2 CV']:20s} | MAE CV={r['MAE CV']:15s}{mark}")

# Save all results
reg_df.to_csv(ROOT / "docs" / "benchmark_regresion.csv", index=False, encoding="utf-8-sig")
clf_df.to_csv(ROOT / "docs" / "benchmark_clasificacion.csv", index=False, encoding="utf-8-sig")
meta_df.to_csv(ROOT / "docs" / "benchmark_rmse_dinamico.csv", index=False, encoding="utf-8-sig")

print()
print("Resultados guardados en docs/benchmark_*.csv")
print("Listo.")
