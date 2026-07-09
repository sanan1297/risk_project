# Risk Predictor — ML de Sobrecostos en Contratos Públicos

**Tesis de Maestría** — José Luis Santamaría Andrade  
Predicción de sobrecostos en contratos públicos colombianos usando matrices de riesgo y modelos de Machine Learning.

## Overview

```
SECOP I/II API → Extracción → Matrices de Riesgo (PDF → LLM → CSV)
                                        ↓
                              Feature Engineering (351 contratos)
                                        ↓
                              Modelo Ridge + LogisticRegression
                                        ↓
                              Streamlit App (Dashboard + Predicción)
```

El pipeline descarga contratos de obra pública desde las APIs de SECOP, extrae sus matrices de riesgo, entrena un modelo Ridge para estimar el sobrecosto porcentual, y despliega un dashboard interactivo para predicción y seguimiento.

## Datos del Proyecto

| Conjunto | Contratos | Descripción |
|---|---|---|
| Pool SECOP I (`proyectos_secop1_lite.csv`) | 1,560 | Todos los contratos de obra pública con sobrecosto > 0 y filtros aplicados |
| **Entrenamiento real** (`matriz_clean.csv`) | **350** | Los que tienen matriz de riesgo extraída (excluye 1 outlier con 808%) |
| SECOP II incluidos | 5 | Contratos complementarios dentro de los 350 (C-110 a C-114) |

> El modelo solo puede entrenarse con los contratos que tienen matriz de riesgo (necesita descripciones, probabilidades, impactos, etc. como features). Las matrices se extrajeron manualmente: descarga de PDF desde contratos.gov.co → OCR → LLM.

## Resultados del Modelo

| Métrica | Valor |
|---|---|
| R² (CV anidada, 33 features) | 0.103 ± 0.080 |
| RMSE (CV) | 15.6 ± 1.1 pp |
| MAE (contratos de entrenamiento) | ~11 pp |
| AUC (LogisticRegression CV) | ~0.70 |
| Accuracy (clasificación binaria) | ~65% |

### Pruebas de Validación (Julio 2026)

**Grupo A — Sanidad (5 contratos del dataset):**
- MAE: 14.2 pp
- Aciertos de alerta: 3/5
- El modelo tiende a regresión a la media (subestima sobrecostos altos, sobreestima bajos)

**Grupo B — Generalización (5 contratos no vistos, 2019-2023):**
- MAE: 11.3 pp (< umbral 20 pp ✅)
- Tiempo de respuesta: < 2s por contrato
- Conclusión: El modelo generaliza aceptablemente. Tiende a sobreestimar en contratos con sobrecosto real bajo.

> Resultados detallados en [`tests/plan_de_pruebas.md`](tests/plan_de_pruebas.md) y [`docs/proceso.md`](docs/proceso.md#12-pruebas-de-validación).

## Arquitectura

```
risk_project/
├── backend/
│   ├── main.py                   # FastAPI REST API (7 endpoints)
│   ├── schemas.py                # Pydantic models
│   ├── predictor.py              # Ridge + LogisticRegression
│   ├── feature_engineering.py    # Agregación de riesgos → features
│   ├── history.py                # SQLite CRUD
│   ├── training_stats.py         # Estadísticas de entrenamiento
│   └── feature_labels.py         # Labels legibles para features
├── frontend/
│   └── streamlit_app.py          # App Streamlit (Dashboard + Predicción + Historial)
├── models/                       # Artefactos del modelo
│   ├── ridge_regressor.pkl
│   ├── ridge_classifier.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── ipc_trm.pkl
│   └── coeficientes_ridge.csv
├── scripts/
│   └── train_final_model.py      # Entrenamiento final (reproducible)
├── estudio_data/
│   ├── features.py               # Feature engineering pipeline
│   └── normalizar.py             # Normalización de matriz de riesgos
├── estudio_modelos/              # Notebooks de exploración y benchmark
│   ├── modelo_final.ipynb
│   ├── modelado.ipynb / modelo_final.ipynb
│   └── matriz_inicial.ipynb
├── docs/
│   ├── proceso.md                # Documentación completa del proceso
│   └── modelo.md                 # Resultados del modelo
├── tests/
│   ├── plan_de_pruebas.md        # Plan y resultados de pruebas
│   └── data/                     # CSVs de prueba (Grupos A y B)
├── backend/history.db            # SQLite con predicciones y validaciones
├── unificar_secop.py             # Descarga SECOP I + II
├── depurar.py                    # Depuración de datos descargados
├── separar_fuentes.py            # Separación SECOP I / II
└── excel_lite.py                 # Versión reducida del dataset
```

### API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/predict` | Predecir sobrecosto desde CSV o texto |
| `GET` | `/history` | Historial paginado de predicciones |
| `PUT` | `/history/{id}` | Guardar validación (sobrecosto real) |
| `DELETE` | `/history/{id}` | Eliminar predicción |
| `DELETE` | `/history` | Limpiar todo el historial |
| `GET` | `/stats/usage` | Estadísticas de uso del modelo |
| `GET` | `/stats/training` | Estadísticas del dataset de entrenamiento |

## Setup

```bash
# 1. Clonar
git clone https://github.com/sanan1297/risk_project.git
cd risk_project

# 2. Crear entorno virtual
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependencias
uv sync

# 4. Entrenar modelo (opcional, ya hay artefactos)
uv run train

# 5. Iniciar backend
uv run uvicorn backend.main:app --reload

# 6. En otra terminal, iniciar frontend
uv run streamlit run frontend/streamlit_app.py
```

## Uso

1. Abrir `http://localhost:8501` en el navegador
2. **Dashboard**: KPIs de uso del modelo y estadísticas de entrenamiento
3. **Predicción**: Subir CSV o pegar texto con riesgos → obtener sobrecosto estimado, probabilidad de alto riesgo y factores explicativos
4. **Historial**: Ver predicciones anteriores, guardar validaciones con el valor real observado

### Parámetros IPC / TRM

En la sidebar se configuran año, IPC y TRM del contrato. Solo editables en la vista de predicción; en Dashboard e Historial se muestran bloqueados.

## Source Data

- **SECOP I** (`f789-7hwg`): Pool de 1,560 contratos de obra pública con sobrecosto > 0 (de 4,723 filtrados)
- **SECOP II** (`jbjy-vk9h`): 5 contratos complementarios dentro del entrenamiento (C-110 a C-114)
- **Matriz de Riesgos** (`docs/matriz_clean.csv`): 6,525 riesgos de **350 contratos** (entrenamiento real) con 20 columnas normalizadas
- **Feature engineering**: ~150 features → selección de 33 (30 top TF-IDF/estadísticas + 3 variables macro)

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.13+, FastAPI, Uvicorn |
| Frontend | Streamlit 1.59, Plotly |
| ML | scikit-learn (Ridge, LogisticRegression, TF-IDF) |
| Almacenamiento | SQLite (predicciones), joblib + pickle (modelos) |
| Extracción datos | Pandas, requests (API datos.gov.co) |
| Notebooks | Jupyter, XGBoost (GPU), SHAP |
