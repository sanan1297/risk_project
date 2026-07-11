# Risk Predictor — ML de Sobrecostos en Contratos Públicos

**Tesis de Maestría** — José Luis Santamaría Andrade
Predicción de sobrecostos en contratos públicos colombianos usando matrices de riesgo y modelos de Machine Learning.

## Overview

```
SECOP I/II API → Extracción → Matrices de Riesgo (PDF → LLM → CSV)
                                        ↓
                              Feature Engineering (351 contratos)
                                        ↓
                         Ridge (R² 0.149) + LogisticRegression (AUC 0.662)
                                        ↓
                              Análisis Cuantitativo: Monte Carlo + Tornado + Desglose
                                        ↓
                              Streamlit App (Dashboard + Predicción + Historial)
```

El pipeline descarga contratos de obra pública desde las APIs de SECOP, extrae sus matrices de riesgo, entrena un modelo Ridge para estimar el sobrecosto porcentual, y despliega un dashboard interactivo con análisis cualitativo y cuantitativo.

## Resultados del Modelo

| Métrica | Valor |
|---|---|
| R² (33 features, v3) | **0.149** |
| RMSE | **16.0 pp** |
| AUC (LogisticRegression CV) | **0.662 ± 0.026** |
| MAE (Grupo B — datos no vistos) | **11.4 pp** |

### Arquitectura de dos capas

1. **Ridge** — predicción central del sobrecosto % usando 33 features agregadas (promedios de prob/impacto, proporciones por tipo, TF-IDF, IPC, TRM)
2. **Análisis cuantitativo** — Monte Carlo (1000 iteraciones, perturbación discreta ±1, ruido Gaussiano), Tornado por tipo (swing real del modelo), Desglose individual (peso = prob×imp / Σ)

> El evaluador validó que esta arquitectura es defendible académicamente: Ridge no puede diferenciar riesgos individuales (usa features agregadas), por lo que la heurística de pesos está debidamente documentada como capa complementaria.

### Pruebas de Validación (Julio 2026)

**Grupo A — Sanidad (5 contratos del dataset, con valores inicial/final):**

| Contrato | Año | Valor Inicial | Valor Final | Real | Ridge | Error |
|---|---|---|---|---|---|---|
| C-001 | 2018 | $16,148M | $20,760M | 28.6% | 28.07% | −0.5 pp |
| C-010 | 2018 | $31,074M | $31,639M | 37.3% | 17.31% | −20.0 pp |
| C-017 | 2019 | $23,880M | $36,561M | 53.1% | 31.25% | −21.9 pp |
| C-043 | 2021 | $13,586M | $13,886M | 2.2% | 28.31% | +26.1 pp |
| C-128 | 2019 | $5,217M | $6,802M | 30.4% | 27.15% | −3.3 pp |

**MAE: 14.3 pp — Aciertos de alerta: 3/5**

**Grupo B — Generalización (5 contratos no vistos, 2019-2023):** MAE: 11.4 pp (< 20 pp ✅)

## Arquitectura

```
risk_project/
├── backend/
│   ├── main.py                   # FastAPI REST API (10 endpoints)
│   ├── schemas.py                # Pydantic models
│   ├── predictor.py              # Ridge + LogisticRegression (R² 0.149, AUC 0.662)
│   ├── feature_engineering.py    # Agregación de riesgos → 33 features
│   ├── quantitative_analysis.py  # Monte Carlo + Tornado + Desglose por riesgo
│   ├── history.py                # SQLite CRUD + stats + resultado_json MC
│   ├── training_stats.py         # Estadísticas de entrenamiento (351 contratos)
│   └── feature_labels.py         # Labels legibles para features
├── frontend/
│   └── streamlit_app.py          # App Streamlit (Dashboard + Predicción + Historial)
├── models/                       # Artefactos del modelo
│   ├── ridge_regressor.pkl       # Ridge (R² 0.149)
│   ├── ridge_classifier.pkl      # Logistic (AUC 0.662)
│   ├── scaler.pkl                # StandardScaler
│   ├── feature_names.pkl         # 33 feature names
│   ├── tfidf_vectorizer.pkl      # Vectorizador TF-IDF
│   ├── coeficientes_ridge.csv    # Coeficientes para dashboard
│   └── ipc_trm.pkl               # IPC/TRM por año
├── scripts/
│   └── train_final_model.py      # Entrenamiento reproducible
├── estudio_data/                 # Normalización y EDA
├── estudio_modelos/              # Notebooks de benchmark
├── docs/
│   ├── proceso.md                # Documentación completa del proceso
│   ├── modelo.md                 # Resultados del modelo
│   ├── matriz.csv                # Matrices de riesgo originales
│   └── matriz_clean.csv          # 6,525 riesgos normalizados, 351 contratos
├── tests/
│   ├── plan_de_pruebas.md        # Plan y resultados de validación
│   └── data/                     # CSVs de prueba (Grupos A y B)
└── contratos/                    # Datos SECOP cache y depurados (excluidos de git)
```

### API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/predict` | Predecir sobrecosto desde CSV o texto (cualitativo: Ridge + alerta + factores) |
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

1. **Dashboard** — Dos pestañas: *Uso del Modelo* (KPIs de predicciones/MC, evolución temporal, alertas, factores) y *Entrenamiento* (351 contratos, R²=0.149, coeficientes, distribución sobrecosto, IPC/TRM)
2. **Predicción** — Subir CSV o pegar texto → Ridge estima sobrecosto % + LogisticRegression para alerta. Opcional: Monte Carlo (histograma, percentiles, tornado por tipo, desglose individual en % y COP)
3. **Historial** — Lista paginada (15/pág) con "Ver análisis completo" inline (expande MC guardado sin modal)

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

# 4. Entrenar modelo (opcional, ya hay artefactos)
uv run train

# 5. Iniciar backend (puerto 8000, o 8003 si hay zombie)
uv run uvicorn backend.main:app --reload

# 6. En otra terminal, iniciar frontend
uv run streamlit run frontend/streamlit_app.py
```

> **Nota:** El frontend apunta a `localhost:8003` por un zombie PID 12248 en puerto 8000. Para usar puerto 8000, matar el proceso o reiniciar la máquina.

## Uso

1. Abrir `http://localhost:8501` en el navegador
2. **Dashboard**: KPIs de uso del modelo y estadísticas de entrenamiento con gráficos interactivos Plotly
3. **Predicción**: Subir CSV o pegar texto con riesgos → análisis cualitativo (Ridge % + alerta + factores) + cuantitativo (Monte Carlo 1000 iter, tornado por tipo, desglose individual, histograma)
4. **Historial**: Ver predicciones guardadas, expandir análisis completo inline, guardar validaciones con sobrecosto real

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.13+, FastAPI, Uvicorn |
| Frontend | Streamlit 1.59+, Plotly |
| ML | scikit-learn (Ridge, LogisticRegression, TF-IDF) |
| Almacenamiento | SQLite (predicciones + resultados MC), joblib/pickle (modelos) |
| Extracción datos | Pandas, requests (API datos.gov.co) |
| Notebooks | Jupyter, XGBoost (GPU), SHAP |

## Source Data

- **SECOP I** (`f789-7hwg`): Pool de 1,560 contratos de obra pública con sobrecosto > 0 (de 4,723 filtrados)
- **SECOP II** (`jbjy-vk9h`): 5 contratos complementarios dentro del entrenamiento
- **Matriz de Riesgos** (`docs/matriz_clean.csv`): 6,525 riesgos de **351 contratos** con 20 columnas normalizadas
- **Feature engineering**: 219 features → selección de 33 (30 top TF-IDF/estadísticas + 3 variables macro)
