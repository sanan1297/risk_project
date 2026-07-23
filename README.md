# Risk Predictor — ML de Sobrecostos en Contratos Públicos

**Tesis de Maestría** — José Luis Santamaría Andrade
Predicción de sobrecostos en contratos públicos colombianos usando matrices de riesgo y modelos de Machine Learning.

## Overview

```
SECOP I/II API → Extracción → Matrices de Riesgo (PDF → LLM → CSV)
                                        ↓
                        Feature Engineering (575 contratos, 38 features)
                                        ↓
              RandomForest (R²=0.622 full) + RandomForestClassifier (AUC=0.591)
                                        ↓
                              Streamlit App (Dashboard + Predicción + Historial)
```

El pipeline descarga contratos de obra pública desde las APIs de SECOP, extrae sus matrices de riesgo, entrena un stack de modelos (RandomForest + RandomForestClassifier + RMSE Predictor) para estimar el sobrecosto porcentual usando **38 features (30 RF-selected + 5 de rango de fechas + 3 de mitigación)**, y despliega un dashboard interactivo con análisis cualitativo y cuantitativo.

## Dataset

| Métrica | Valor |
|---|---|
| Contratos entrenados | 575 |
| Riesgos en matriz (`docs/matriz.csv`) | 11,029 |
| Riesgos filtrados (outliers >200% eliminados) | 10,987 |
| Features | 38 (30 RF-selected + 5 macro rango + 3 mitigación) |
| Sobrecosto promedio | 24.3% |
| Sobrecosto mediana | 19.0% |
| Alto riesgo (>25%) | 40.4% |

## Resultados del Modelo

| Componente | Modelo | Métrica clave | Archivo |
|---|---|---|---|
| **Regresor** | **RandomForest** (390 trees, max_depth=19, max_features='log2') | R² nested CV=0.235, R² full=0.622, RMSE=11.4 pp | `modelo_campeon.pkl` |
| **Clasificador** (>25%) | **RandomForestClassifier** (100 trees, max_depth=10, class_weight='balanced') | AUC nested CV=0.591 | `classifier.pkl` |
| **RMSE Predictor** | **SVR RBF** (C=1.0, gamma=scale) | MAE=7.98 pp (benchmark CV) | `rmse_predictor.pkl` |
| Features | 38 (30 RF-selected + 5 rango fechas + 3 mitigación) | | |
| Interpretabilidad | SHAP (TreeExplainer) + Permutation importance | | |

### Arquitectura de tres capas

1. **RandomForest** — predicción central del sobrecosto % usando 38 features agregadas
2. **Análisis cuantitativo** — Monte Carlo (iteraciones configurables, ruido Gaussiano con **RMSE dinámico** SVR RBF), Tornado por tipo, Desglose SHAP individual
3. **Interpretabilidad local (SHAP)** — cada riesgo recibe su contribución real vía valores Shapley sobre el RandomForest (TreeExplainer)

### Traza de la Investigación

| Fase | Modelo | Features | R² | ¿Qué pasó? |
|---|---|---|---|---|
| v1–v3 | **Ridge** | 33 (año único) | 0.149 | Funcionaba con año único |
| v4a | **Ridge** (rango) | 35 (rango fechas) | 0.066 CV | No capturó no-linealidades |
| v4b | **SVR RBF** | 35 (rango fechas) | 0.068 CV | Campeón anterior |
| v5 | **Ridge** | 35 (rango fechas) | 0.086 hold-out | Campeón v5 |
| **v6** | **RandomForest** | 38 (30 RF-selected + 5 rango + 3 mitigación) | **0.235 nested CV** | **Campeón actual** |

### RMSE Predictor (Error ML)

El ruido del Monte Carlo se predice con **SVR RBF** entrenado sobre residuales del RandomForest (`|real - rf_pred|`):

| Método | MAE |
|---|---|
| Heurística (bucket n_riesgos) | 12.0–24.0 pp |
| **SVR RBF** | **7.98 pp** |

Benchmark de 6 meta-modelos: SVR RBF ganó (R²=−0.145, MAE=7.98). Safety factor: `rmse = max(rmse_pred, rmse_heur * 0.85, 2.0)`.

## Benchmark Completo (10 modelos, nested CV 5-fold, 575 contratos)

### Regresión

| Modelo | R² CV | RMSE CV | MAE CV |
|---|---|---|---|
| Ridge | 0.026 ± 0.078 | 18.34 ± 2.74 | 14.25 |
| Lasso | 0.026 ± 0.078 | 18.35 ± 2.74 | 14.25 |
| ElasticNet | 0.029 ± 0.077 | 18.32 ± 2.74 | 14.24 |
| KNN | −0.030 ± 0.081 | 18.76 ± 1.97 | 15.03 |
| DecisionTree | −0.742 ± 0.369 | 24.23 ± 2.92 | 17.80 |
| **RandomForest** | **0.045 ± 0.056** | **18.15 ± 2.49** | **14.18** |
| GradientBoosting | −0.092 ± 0.129 | 19.32 ± 2.40 | 14.71 |
| XGBoost | −0.008 ± 0.115 | 18.61 ± 2.64 | 14.16 |
| SVR | 0.048 ± 0.080 | 18.15 ± 2.84 | 13.74 |
| MLP | −0.782 ± 0.289 | 24.68 ± 3.68 | 18.18 |

Selección final: RandomForest por mejor R² CV (0.045) y R² en 100% datos (0.622).

### Clasificación (>25% sobrecosto)

| Modelo | AUC CV | Acc CV |
|---|---|---|
| Ridge (Logistic) | 0.665 ± 0.046 | 0.619 |
| KNN | 0.600 ± 0.066 | 0.587 |
| DecisionTree | 0.576 ± 0.049 | 0.586 |
| **RandomForest** | **0.685 ± 0.034** | **0.630** |
| GradientBoosting | 0.629 ± 0.048 | 0.610 |
| XGBoost | 0.654 ± 0.021 | 0.633 |
| SVC | 0.661 ± 0.048 | 0.617 |
| MLP | 0.609 ± 0.032 | 0.601 |

Selección: RandomForestClassifier (AUC CV=0.685, mejor ranking).

### RMSE Predictor (meta-modelo de error)

| Modelo | R² CV | MAE CV |
|---|---|---|
| Ridge | −0.084 ± 0.094 | 8.26 |
| Lasso | −0.082 ± 0.094 | 8.25 |
| ElasticNet | −0.078 ± 0.095 | 8.24 |
| KNN | −0.283 ± 0.239 | 9.03 |
| DecisionTree | −1.368 ± 0.993 | 10.62 |
| RandomForest | −0.177 ± 0.141 | 8.43 |
| GradientBoosting | −0.230 ± 0.152 | 8.61 |
| **SVR** | **−0.145 ± 0.059** | **7.98** |
| MLP | −2.767 ± 1.494 | 15.33 |

Selección: SVR RBF (mejor MAE 7.98 pp).

## Top 10 Features (RandomForest)

| # | Feature | Importancia | Tipo |
|---|---|---|---|
| 1 | `val_std` | 0.048 | Desviación valoración |
| 2 | `valor_inicial` | 0.044 | Valor del contrato |
| 3 | `val_promedio` | 0.044 | Valoración promedio |
| 4 | `imp_promedio` | 0.043 | Impacto promedio |
| 5 | `suma_impacto` | 0.041 | Suma impactos |
| 6 | `tfidf_ejecución` | 0.038 | TF-IDF ejecución |
| 7 | `tfidf_municipio` | 0.034 | TF-IDF municipio |
| 8 | `avg_longitud_mitigacion` | 0.034 | Longitud mitigación prom. |
| 9 | `tfidf_demoras` | 0.033 | TF-IDF demoras |
| 10 | `imp_std` | 0.032 | Desviación impacto |

## Arquitectura

```
risk_project/
├── backend/
│   ├── main.py                   # FastAPI REST API (11 endpoints)
│   ├── schemas.py                # Pydantic models
│   ├── predictor.py              # RandomForest (38 features, permutation importance)
│   ├── feature_engineering.py    # Agregación de riesgos → 38 features (rango fechas + mitigación)
│   ├── quantitative_analysis.py  # Monte Carlo + RMSE dinámico + Tornado + SHAP TreeExplainer
│   ├── history.py                # SQLite CRUD + stats + resultado_json MC
│   ├── training_stats.py         # Estadísticas de entrenamiento (575 contratos)
│   ├── mlflow_tracker.py         # Carga modelo_campeon.pkl desde MLflow (fallback local)
│   └── feature_labels.py         # Labels legibles para features técnicas
├── frontend/
│   └── streamlit_app.py          # App Streamlit (Dashboard + Predicción + Historial)
├── models/
│   ├── modelo_campeon.pkl        # RandomForest (390 trees, R²=0.622)
│   ├── classifier.pkl            # RandomForestClassifier (AUC=0.685)
│   ├── rmse_predictor.pkl        # SVR RBF para RMSE dinámico (MAE=7.98 pp)
│   ├── scaler.pkl                # StandardScaler
│   ├── feature_names.pkl         # 38 feature names
│   ├── feature_importances_rf.csv# Importancia de features (RandomForest)
│   ├── permutation_importance.csv# Importancia global por permutación
│   ├── tfidf_vectorizer.pkl      # Vectorizador TF-IDF
│   └── ipc_trm.pkl               # IPC/TRM por año
├── scripts/
│   ├── train_production.py       # Entrenamiento final (100% datos, 3 modelos)
│   ├── consolidar_38_features.py # Genera consolidado_38_features.csv
│   ├── generate_model_artifacts.py# feature_names, permutation_importance, feature_importances_rf
│   └── enrich_test_contracts.py  # Enriquece test contracts con mitigaciones
├── docs/
│   ├── proceso.md                # Documentación completa del proceso
│   ├── modelo.md                 # Traza de investigación del modelo
│   ├── matriz.csv                # Matrices de riesgo originales (11,029 riesgos)
│   ├── matriz_clean.csv          # 7,914 riesgos normalizados, 429 contratos
│   ├── consolidado_38_features.csv # 575 contratos × 38 features — listo para ML
│   └── contratos_macro.csv       # Features de rango por contrato
├── Dockerfile                    # Imagen Python 3.12-slim + uv
├── docker-compose.yml            # 3 servicios: mlflow, backend, frontend
└── tests/
    ├── plan_de_pruebas.md        # Plan y resultados de validación
    └── data/enriched/            # CSVs de prueba con mitigaciones
```

### API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/predict` | Predecir sobrecosto desde CSV o texto (RF + alerta + factores) |
| `POST` | `/predict/montecarlo` | Simulación Monte Carlo + tornado por tipo + desglose por riesgo |
| `GET` | `/history` | Historial paginado (15/page) de predicciones |
| `GET` | `/history/{id}` | Predicción individual con resultado_json del MC |
| `GET` | `/history/{id}/resultados` | Solo resultados cuantitativos del MC |
| `PUT` | `/history/{id}` | Guardar validación (sobrecosto real) |
| `DELETE` | `/history/{id}` | Eliminar predicción |
| `DELETE` | `/history` | Limpiar todo el historial |
| `GET` | `/stats/usage` | Estadísticas de uso (predicciones, MC, factores) |
| `GET` | `/model/info` | Metadatos del modelo (versión, experimento, MLflow) |
| `GET` | `/stats/training` | Estadísticas del dataset de entrenamiento |

### Vistas Frontend

1. **Dashboard** — Dos pestañas: *Uso del Modelo* (KPIs, evolución, distribución alertas) y *Entrenamiento* (575 contratos, 38 features, distribución, categorías)
2. **Predicción** — Subir CSV o pegar texto → RF estima sobrecosto % + clasificador para alerta. Opcional: MC con RMSE variable, histograma, tornado, desglose en % y COP
3. **Historial** — Lista paginada con "Ver análisis completo" inline

## Setup

### Local (desarrollo)

```bash
# 1. Clonar
git clone https://github.com/sanan1297/risk_project.git
cd risk_project

# 2. Crear entorno virtual
uv venv
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
uv sync

# 4. Iniciar backend
uv run uvicorn backend.main:app --reload --port 8000

# 5. En otra terminal, iniciar frontend
uv run streamlit run frontend/streamlit_app.py --server.port 8502
```

### Docker (producción)

```bash
# Levantar los 3 servicios
docker compose up -d

# Ver estado
docker compose ps

# Reconstruir imagen
docker compose build backend

# Ver logs
docker compose logs -f
```

| Servicio | Puerto | URL |
|---|---|---|
| MLflow (experimentos) | `:5000` | http://localhost:5000 |
| Backend API | `:8003` | http://localhost:8003/docs |
| Frontend | `:8501` | http://localhost:8501 |

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.12+, FastAPI, Uvicorn |
| Frontend | Streamlit 1.59+, Plotly |
| ML | scikit-learn (RandomForest, SVR, TF-IDF, StandardScaler), SHAP (TreeExplainer) |
| Trazabilidad | **MLflow** (experimentos + model registry + artifact store) |
| Contenerización | **Docker Compose** (3 servicios: mlflow, backend, frontend) |
| Almacenamiento | SQLite (predicciones + resultados MC), joblib/pickle (modelos) |
| Extracción datos | Pandas, requests (API datos.gov.co) |

## MLflow Traceability

El modelo y sus experimentos se registran en **MLflow** para trazabilidad completa:

- **Servidor:** `http://localhost:5000` (UI web para explorar experimentos)
- **Experiment:** `risk-predictor` — todos los runs de entrenamiento
- **Model Registry:** `risk-predictor-svr` — modelo promovido a producción
- **Fallback local:** si MLflow no está disponible, el backend carga artefactos desde `models/`

Cada entrenamiento (`uv run train`) registra:
- Parámetros: `n_estimators`, `max_depth`, `max_features`, feature count
- Métricas: `r2_cv`, `auc_cv`, `rmse`, `r2_full`
- Artefactos: `.pkl` del modelo, `permutation_importance.csv`, scaler, feature names
- Model Registry: versión auto-incremental con run_id vinculado

El endpoint `GET /model/info` expone los metadatos del modelo activo (versión, experimento, run_id, estado de conexión MLflow).

## Source Data

- **SECOP I** (`f789-7hwg`): Pool de 1,560 contratos de obra pública con sobrecosto > 0
- **SECOP II** (`jbjy-vk9h`): 5 contratos complementarios
- **Matriz de Riesgos** (`docs/matriz.csv`): 11,029 riesgos de **577 contratos**
- **Feature engineering**: 38 features (30 RF-selected + 5 de rango + 3 mitigación)
