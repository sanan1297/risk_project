"""Version rapida del RMSE dinamico - solo estadisticas basicas."""
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = "."

# Cargar artefactos
svr = joblib.load(f"{ROOT}/models/svr_regressor.pkl")
scaler = joblib.load(f"{ROOT}/models/scaler.pkl")
feature_names = joblib.load(f"{ROOT}/models/feature_names.pkl")
tfidf_vec = joblib.load(f"{ROOT}/models/tfidf_vectorizer.pkl")

# Cargar datos
matriz = pd.read_csv(f"{ROOT}/docs/matriz_clean.csv")
macro = pd.read_csv(f"{ROOT}/docs/contratos_macro.csv")
print(f"Matriz: {len(matriz)} filas, {matriz['id_contrato'].nunique()} contratos")
print(f"Features: {len(feature_names)}")

# Coerce numeric columns
for c in ["valor_inicial", "probabilidad", "impacto", "valoracion"]:
    matriz[c] = pd.to_numeric(matriz[c], errors="coerce").fillna(0)

# Replicar feature engineering combinando macro
contratos_df = matriz.groupby("id_contrato", sort=False).agg(
    valor_inicial=("valor_inicial", "first"),
    n_riesgos=("id_riesgo", "nunique"),
    prob_promedio=("probabilidad", "mean"), prob_std=("probabilidad", "std"),
    imp_promedio=("impacto", "mean"), imp_std=("impacto", "std"),
    val_promedio=("valoracion", "mean"), val_std=("valoracion", "std"),
).reset_index()
contratos_df["interaccion_prob_x_impacto"] = contratos_df["prob_promedio"] * contratos_df["imp_promedio"]
contratos_df = contratos_df.fillna(0)

# Macro
macro_vars = macro[["id_contrato", "anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]]
data = contratos_df.merge(macro_vars, on="id_contrato")

# Real sobrecosto
real_df = matriz.groupby("id_contrato")["sobrecosto"].first().reset_index()
data = data.merge(real_df, on="id_contrato")
data = data.dropna(subset=["sobrecosto"])
print(f"Dataset: {len(data)} contratos (con sobrecosto real)")

# Alinear features
for f in feature_names:
    if f not in data.columns:
        data[f] = 0.0
X = data[feature_names].fillna(0).values

# Predecir
X_scaled = scaler.transform(X)
preds = svr.predict(X_scaled)

data["svr_pred"] = np.round(preds, 2)
data["abs_error"] = np.abs(data["sobrecosto"] - data["svr_pred"])

print("\nEstadisticas del error absoluto (|real - svr|):")
print(data["abs_error"].describe())

rmse = np.sqrt(mean_squared_error(data["sobrecosto"], data["svr_pred"]))
mae = mean_absolute_error(data["sobrecosto"], data["svr_pred"])
print(f"\nRMSE global: {rmse:.2f} pp")
print(f"MAE global: {mae:.2f} pp")

# Heuristica actual
def heuristic_rmse(n):
    if n <= 10: return 12.0
    elif n <= 20: return 16.0
    elif n <= 30: return 20.0
    else: return 24.0

data["rmse_heuristic"] = data["n_riesgos"].apply(heuristic_rmse)
data["bucket"] = pd.cut(data["n_riesgos"], bins=[0, 10, 20, 30, 200], labels=["1-10", "11-20", "21-30", ">30"])

print("\nError real por bucket vs RMSE heuristico:")
print(data.groupby("bucket", observed=True).agg(
    n=("abs_error", "count"),
    error_prom=("abs_error", "mean"),
    error_std=("abs_error", "std"),
    error_p90=("abs_error", lambda x: np.percentile(x, 90)),
    rmse_asignado=("rmse_heuristic", "first"),
))

# Meta-model: Ridge rapido
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

y = data["abs_error"].values
cv = KFold(5, shuffle=True, random_state=42)

ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")
r2 = cross_val_score(ridge, X_scaled, y, cv=cv, scoring="r2")
ridge.fit(X_scaled, y)

print(f"\nMeta-model Ridge para RMSE dinamico:")
print(f"  MAE CV: {-scores.mean():.2f} +/- {scores.std():.2f}")
print(f"  R2 CV: {r2.mean():.3f} +/- {r2.std():.3f}")

# Top features del meta-model
coefs = pd.DataFrame({"feature": feature_names, "coef": ridge.coef_})
print("\nTop 10 features que MAS aumentan el error esperado:")
print(coefs.sort_values("coef", ascending=False).head(10).to_string(index=False))
print("\nTop 10 features que MAS disminuyen el error esperado:")
print(coefs.sort_values("coef").head(10).to_string(index=False))

# Correlacion entre error real y heuristico
from scipy.stats import pearsonr
corr, p = pearsonr(data["abs_error"], data["rmse_heuristic"])
print(f"\nCorrelacion error real vs heuristico: r={corr:.3f} (p={p:.4f})")

# Proporcion donde heuristica subestima
subestima = (data["abs_error"] > data["rmse_heuristic"]).mean()
print(f"Proporcion donde heuristica SUBESTIMA el error: {subestima:.1%}")

# Guardar output
data.to_csv("docs/rmse_dinamico_resultados.csv", index=False, encoding="utf-8-sig")
print("\nResultados guardados en docs/rmse_dinamico_resultados.csv")
