import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# --- Load data (same as PoC) ---
matriz = pd.read_csv("tests/data/matriz_riesgos.csv", encoding="latin1", low_memory=False)
matriz["descripcion_riesgo"] = matriz["descripcion_riesgo"].fillna("").astype(str)
text_series = matriz["descripcion_riesgo"]
texts = text_series.groupby(matriz["id_contrato"]).apply(lambda x: " ".join(x)).reset_index()
texts.columns = ["id_contrato", "descripcion_riesgo"]

tfidf = TfidfVectorizer(max_features=20, stop_words="spanish")
tfidf_mat = tfidf.fit_transform(texts["descripcion_riesgo"])
tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()])
tfidf_df["id_contrato"] = texts["id_contrato"].values

macro = pd.read_csv("tests/data/macro_colombia.csv", encoding="latin1")
macro["anio_inicio"] = macro["anio"].astype(int)
macro_cols = macro[["anio_inicio", "ipc", "trm"]].copy()
macro_cols.columns = ["anio_inicio", "ipc_acumulado", "trm_promedio"]

# Build contract-level features
def agg_riesgos(grp):
    probs = grp["probabilidad"].values.astype(float)
    impacts = grp["impacto"].values.astype(float)
    return pd.Series({
        "prob_promedio": np.mean(probs),
        "prob_std": np.std(probs),
        "imp_promedio": np.mean(impacts),
        "interaccion_prob_x_impacto": np.mean(probs * impacts),
        "prop_fuen_externo": np.mean(grp["fuente"].str.lower().str.contains("externo|entidad", na=False)),
        "prop_tipo_operacional": np.mean(grp["tipo"].str.lower().str.contains("operacional", na=False)),
        "prop_tipo_economico": np.mean(grp["tipo"].str.lower().str.contains("economico|econ\u00f3mico", na=False)),
        "prop_asig_entidad": np.mean(grp["asignacion"].str.lower().str.contains("entidad|contratante", na=False)),
        "prop_cate_alto": np.mean(grp["categoria"].str.lower().str.contains("alto", na=False)),
        "prop_cate_bajo": np.mean(grp["categoria"].str.lower().str.contains("bajo", na=False)),
        "valor_inicial": grp["valor_inicial"].iloc[0] if "valor_inicial" in grp.columns else 0,
        "n_riesgos": len(grp),
    })

basic = matriz.groupby("id_contrato").apply(agg_riesgos).reset_index()

merged = basic.merge(tfidf_df, on="id_contrato", how="left")
merged = merged.merge(macro_cols, on="anio_inicio", how="left")

# Get SVR predictions
from joblib import load, dump
quant = load("backend/quantitative_analysis.pkl")
preds = quant.predict(matriz)
# Actually let's just compute dummy predictions for testing
# Since we can't run SVR on raw data easily here, let's use logistic-like

# For the purposes of this analysis, let's load a small trained model
# or use the fact that the PoC already ran - let's just load from CSV
print("Cargando datos desde PoC...")
# Simple test: simulate what the PoC produced
# Load actual predictions from contracts
contratos = pd.read_csv("tests/data/contratos_prueba.csv", encoding="latin1")
print(f"Contratos de prueba: {len(contratos)}")

# Merge with risk matrix to get n_riesgos per contract
n_riesgos_per = matriz.groupby("id_contrato").size().reset_index(name="n_riesgos")
data = merged.merge(n_riesgos_per, on="id_contrato", how="left")

# Since we can't easily replicate SVR predictions, use contratos_prueba
# as a reference for real errors (they have real MC results)
has_real = contratos[contratos["p50"].notna()].copy()
print(f"Contratos con P50 real: {len(has_real)}")

# For the full dataset analysis, train SVR on the PoC's features
feature_names = [c for c in merged.columns if c not in ["id_contrato", "anio_inicio"]]
print(f"Features: {len(feature_names)}")

# We need actual SVR predictions. Let me check if contratos_prueba has SVR
if "svr" in contratos.columns:
    contratos_svr = contratos[["id_contrato", "svr", "p50"]].dropna()
    data_svr = data.merge(contratos_svr, on="id_contrato", how="inner")
    print(f"Contratos con SVR+P50: {len(data_svr)}")
    real_error = np.abs(data_svr["p50"].values - data_svr["svr"].values)
    print(f"MAE real: {np.mean(real_error):.2f} pp")
    print(f"RMSE real: {np.sqrt(np.mean(real_error**2)):.2f} pp")
else:
    print("No SVR column in contratos_prueba.csv")

print("\nPara analizar a profundidad necesitamos ejecutar el PoC completo.")
print("Ejecuta: python estudio_modelos/rmse_dinamico.py (timeout ~5 min)")
