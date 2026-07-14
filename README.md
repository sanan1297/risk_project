# Risk Predictor — ML de Sobrecostos en Contratos Públicos

**Tesis de Maestría** — José Luis Santamaría Andrade
Predicción de sobrecostos en contratos públicos colombianos usando matrices de riesgo y modelos de Machine Learning.

## Overview

```
SECOP I/II API → Extracción → Matrices de Riesgo (PDF → LLM → CSV)
                                        ↓
                        Feature Engineering (351 contratos, 35 features)
                                        ↓
                           SVR RBF (R² CV 0.068, AUC 0.673) + Monte Carlo
                                        ↓
                              Streamlit App (Dashboard + Predicción + Historial)
```

El pipeline descarga contratos de obra pública desde las APIs de SECOP, extrae sus matrices de riesgo, entrena un modelo **SVR con kernel RBF** para estimar el sobrecosto porcentual usando **35 features (30 TF-IDF + 5 de rango de fechas)**, y despliega un dashboard interactivo con análisis cualitativo y cuantitativo. El R² CV (5-fold) es de **0.068**, y el R² in-sample (full training) de **0.417**.

## Resultados del Modelo

| Métrica | Valor | Nota |
|---|---|---|---|
| Modelo | **SVR (kernel RBF, C=10, gamma=scale)** | |
| **R² CV (5-fold)** | **0.068** | **Métrica real de generalización** |
| R² (full training) | 0.417 | In-sample (entrena y predice sobre los mismos datos) |
| AUC (LogisticRegression CV) | **0.673** | |
| RMSE | **17.1 pp** | |
| Features | **35 (30 TF-IDF + 5 rango fechas)** | |
| Interpretabilidad | **Permutation importance (10 reps) + Ridge referencia** | |

### Arquitectura de dos capas

1. **SVR** — predicción central del sobrecosto % usando 35 features agregadas (promedios de prob/impacto, proporciones por tipo, TF-IDF, IPC acumulado compuesto, TRM promedio, duración)
2. **Análisis cuantitativo** — Monte Carlo (1000 iteraciones, perturbación discreta ±1, ruido Gaussiano con **RMSE variable según complejidad del contrato**), Tornado por tipo, Desglose individual

### Traza de la Investigación

El modelo pasó por 3 fases antes de llegar a SVR:

| Fase | Modelo | Features | R² | ¿Qué pasó? |
|---|---|---|---|---|
| v1–v3 | **Ridge** | 33 (año único) | **0.149** | Funcionaba con año único, pero forzaba todos los contratos a un solo año |
| v4a | **Ridge** (rango) | 35 (rango fechas) | 0.066 CV | **No funcionó** — Ridge no captura relaciones no lineales entre duración, IPC compuesto y TRM |
| **v4b** | **SVR RBF** | 35 (rango fechas) | **0.068 CV** | **Campeón final** — kernel RBF sí modela las no-linealidades del rango temporal |

### Mejora: RMSE Variable por Complejidad

La incertidumbre del Monte Carlo ahora varía según la cantidad de riesgos del contrato:

| # Riesgos | RMSE | Ejemplo |
|---|---|---|
| 1–10 | 12 pp | Contratos simples |
| 11–20 | 16 pp | Contratos típicos (C-001, C-017) |
| 21–30 | 20 pp | Contratos complejos (C-043, C-361) |
| >30 | 24 pp | Contratos muy complejos (C-364) |

### Pruebas de Validación (Julio 2026)

**10 contratos de prueba — MAE global: 10.5 pp — Aciertos de alerta: 7/10**

| Contrato | Real | SVR | Error | P50 | P90-P10 | RMSE | Alerta |
|---|---|---|---|---|---|---|---|---|
| C-001 | 28.6% | 25.01% | −3.6 | 23.6% | 42.7 pp | 16 | ALTO RIESGO ✅ |
| C-010 | 37.3% | 16.84% | −20.5 | 16.4% | 40.4 pp | 16 | RIESGO MODERADO ❌ |
| C-017 | 53.1% | 33.16% | −19.9 | 32.6% | 41.3 pp | 16 | ALTO RIESGO ✅ |
| C-043 | 2.2% | 28.54% | +26.3 | 28.9% | 52.3 pp | 20 | ALTO RIESGO ❌ |
| C-128 | 30.4% | 26.31% | −4.1 | 25.8% | 41.3 pp | 16 | ALTO RIESGO ✅ |
| C-360 | 10.1% | 15.55% | +5.4 | 16.5% | 38.9 pp | 16 | RIESGO MODERADO ✅ |
| C-361 | 19.1% | 16.99% | −2.1 | 18.5% | 51.3 pp | 20 | ALTO RIESGO ❌ |
| C-362 | 4.4% | 9.54% | +5.2 | 10.5% | 50.9 pp | 20 | RIESGO MODERADO ✅ |
| C-363 | 7.2% | 15.13% | +7.9 | 16.2% | 39.0 pp | 16 | RIESGO MODERADO ✅ |
| C-364 | 20.8% | 10.85% | −10.0 | 12.0% | 65.4 pp | 24 | RIESGO MODERADO ✅ |

## Arquitectura

```
risk_project/
├── backend/
│   ├── main.py                   # FastAPI REST API (10 endpoints)
│   ├── schemas.py                # Pydantic models
│   ├── predictor.py              # SVR + LogisticRegression (permutation importance)
│   ├── feature_engineering.py    # Agregación de riesgos → 35 features (rango fechas)
│   ├── quantitative_analysis.py  # Monte Carlo + Tornado + RMSE variable por n_riesgos
│   ├── history.py                # SQLite CRUD + stats + resultado_json MC
│   ├── training_stats.py         # Estadísticas de entrenamiento (351 contratos)
│   └── feature_labels.py         # Labels legibles para features
├── frontend/
│   └── streamlit_app.py          # App Streamlit (Dashboard + Predicción + Historial)
├── models/                       # Artefactos del modelo SVR
│   ├── svr_regressor.pkl         # SVR (R² CV 0.068, C=10, gamma=scale)
│   ├── classifier.pkl            # LogisticRegression (AUC 0.673)
│   ├── ridge_reference.pkl       # Ridge de referencia (coeficientes)
│   ├── permutation_importance.csv# Importancia global (10 reps)
│   ├── scaler.pkl                # StandardScaler
│   ├── feature_names.pkl         # 35 feature names
│   ├── tfidf_vectorizer.pkl      # Vectorizador TF-IDF
│   └── ipc_trm.pkl               # IPC/TRM por año
├── scripts/
│   ├── train_final_model.py      # Entrenamiento SVR reproducible
│   └── compute_ipc_range.py      # Cálculo IPC acumulado + TRM promedio por rango
├── estudio_data/                 # Normalización y EDA
├── notebooks/
│   ├── evaluacion_modelos.py     # Benchmark Ridge vs SVR vs RF vs MLP
│   └── test_svr_shap.py          # Evaluación SVR + permutation importance
├── docs/
│   ├── proceso.md                # Documentación completa del proceso
│   ├── modelo.md                 # Traza de investigación del modelo (Ridge → SVR)
│   ├── contratos_macro.csv       # Features de rango por contrato
│   ├── matriz.csv                # Matrices de riesgo originales
│   └── matriz_clean.csv          # 6,525 riesgos normalizados, 351 contratos
├── tests/
│   ├── plan_de_pruebas.md        # Plan y resultados de validación (4 grupos: A-D)
│   └── data/                     # CSVs de prueba (Grupos A, B, C, D)
└── contratos/                    # Datos SECOP cache y depurados (excluidos de git)
```

### API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/predict` | Predecir sobrecosto desde CSV o texto (SVR + alerta + factores) |
| `POST` | `/predict/montecarlo` | Simulación Monte Carlo + tornado por tipo + desglose por riesgo |
| `GET` | `/history` | Historial paginado (15/page) de predicciones |
| `GET` | `/history/{id}` | Predicción individual con resultado_json del MC |
| `GET` | `/history/{id}/resultados` | Solo resultados cuantitativos del MC |
| `PUT` | `/history/{id}` | Guardar validación (sobrecosto real) |
| `DELETE` | `/history/{id}` | Eliminar predicción |
| `DELETE` | `/history` | Limpiar todo el historial |
| `GET` | `/stats/usage` | Estadísticas de uso (predicciones, MC, factores) |
| `GET` | `/stats/training` | Estadísticas del dataset de entrenamiento |

### Vistas Frontend

1. **Dashboard** — Dos pestañas: *Uso del Modelo* (KPIs) y *Entrenamiento* (35 features, SVR, distribución)
2. **Predicción** — Subir CSV o pegar texto → SVR estima sobrecosto % + LogisticRegression para alerta. Opcional: MC con RMSE variable, histograma, tornado, desglose en % y COP
3. **Historial** — Lista paginada con "Ver análisis completo" inline

## Setup

```bash
# 1. Clonar
git clone https://github.com/sanan1297/risk_project.git
cd risk_project

# 2. Crear entorno virtual
uv venv
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
uv sync

# 4. Entrenar modelo SVR (opcional, ya hay artefactos)
uv run train

# 5. Iniciar backend
uv run uvicorn backend.main:app --reload

# 6. En otra terminal, iniciar frontend
uv run streamlit run frontend/streamlit_app.py
```

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.14+, FastAPI, Uvicorn |
| Frontend | Streamlit 1.59+, Plotly |
| ML | scikit-learn (SVR, LogisticRegression, TF-IDF, Ridge referencia) |
| Almacenamiento | SQLite (predicciones + resultados MC), joblib/pickle (modelos) |
| Extracción datos | Pandas, requests (API datos.gov.co) |
| Notebooks | Jupytext, scikit-learn |

## Source Data

- **SECOP I** (`f789-7hwg`): Pool de 1,560 contratos de obra pública con sobrecosto > 0
- **SECOP II** (`jbjy-vk9h`): 5 contratos complementarios
- **Matriz de Riesgos** (`docs/matriz_clean.csv`): 6,525 riesgos de **351 contratos**
- **Feature engineering**: 35 features (30 top TF-IDF + 5 de rango: anio_inicio, anio_fin, duracion, ipc_acumulado, trm_promedio)
