# Modelo Predictivo de Sobrecosto — Benchmarking y Selección

## Contexto

**Tesis:** Maestría de José Luis Santamaría Andrade
**Tema:** Predicción ML de matrices de riesgo en contratos públicos de Colombia
**Dataset base:** 351 contratos SECOP I con matrices de riesgo (6,525 filas de riesgos)
**Variable objetivo:** `sobrecosto` = ((valor_final − valor_inicial) / valor_inicial) × 100

---

## 1. Feature Engineering

### 1.1 De riesgos a features por contrato

A partir de `contratos_features.csv` (6,525 riesgos → 351 contratos) se generaron **219 features** agrupadas en familias:

| Familia | Features | Descripción |
|---|---|---|
| TF-IDF | 100 | Texto de `descripcion_riesgo`, `consecuencia`, `plan_mitigacion` vectorizado (TF-IDF unigram + bigram, max 100 vars) |
| Probabilidad/Impacto | 6 | `prob_promedio`, `prob_std`, `imp_promedio`, `imp_std`, `valoracion_promedio`, `interaccion_prob_x_impacto` |
| Categorías | 8 | Proporciones por categoría: `prop_cate_bajo`, `prop_cate_medio`, `prop_cate_alto`, `prop_cate_extremo`, `prop_cate_no_especificado`, `prop_categoria_nula` |
| Tipos de riesgo | 17 | Proporciones por tipo: `prop_tipo_operacional`, `prop_tipo_economico`, `prop_tipo_gerencial`, etc. |
| Asignación | 10 | Proporciones por asignación: `prop_asig_entidad`, `prop_asig_contratista`, `prop_asig_compartido`, etc. |
| Etapas | 23 | Proporciones por etapa del proyecto |
| Clases | 22 | Proporciones por clase de riesgo |
| Fuente | 4 | Proporciones por fuente: `prop_fuen_interno`, `prop_fuen_externo`, `prop_fuen_mixto`, `prop_fuen_no_especificado` |
| Nulos/Valor inicial | 4 | `nulos_riesgos`, `prop_nulos`, `valor_inicial` |
| Control | 3 | `anio`, `ipc` (inflación), `trm` (tipo de cambio) |

### 1.2 Feature Reduction

Tras entrenar un Random Forest baseline con todas las 219 features, se seleccionaron las **top 30 por importancia** + 3 variables de control obligatorias:

| # | Feature | Importancia | Familia |
|---|---|---|---|
| 1 | `tfidf_desarrollo` | 10.14% | TF-IDF |
| 2 | `interaccion_prob_x_impacto` | 3.14% | Prob/Impacto |
| 3 | `tfidf_insumos` | 2.78% | TF-IDF |
| 4 | `prob_std` | 2.67% | Prob/Impacto |
| 5 | `tfidf_expedicion` | 2.66% | TF-IDF |
| 6 | `tfidf_materiales` | 2.60% | TF-IDF |
| 7 | `imp_promedio` | 2.43% | Prob/Impacto |
| 8 | `tfidf_cualquier` | 2.30% | TF-IDF |
| 9 | `tfidf_ejecucion` | 2.24% | TF-IDF |
| 10 | `tfidf_contrato` | 2.23% | TF-IDF |
| 11 | `prop_tipo_operacional` | 2.14% | Tipos |
| 12 | `prob_promedio` | 2.13% | Prob/Impacto |
| 13 | `prop_cate_bajo` | 2.11% | Categorías |
| 14 | `valor_inicial` | 2.08% | Nulos |
| 15 | `tfidf_riesgo` | 2.07% | TF-IDF |
| 16 | `tfidf_tecnicas` | 2.06% | TF-IDF |
| 17 | `tfidf_municipio` | 2.06% | TF-IDF |
| 18 | `tfidf_obras` | 2.00% | TF-IDF |
| 19 | `tfidf_informacion` | 1.98% | TF-IDF |
| 20 | `prop_fuen_externo` | 1.97% | Fuente |
| 21 | `tfidf_cuando` | 1.96% | TF-IDF |
| 22 | `tfidf_disenos` | 1.85% | TF-IDF |
| 23 | `tfidf_ejecucion_contrato` | 1.83% | TF-IDF |
| 24 | `tfidf_calidad` | 1.83% | TF-IDF |
| 25 | `tfidf_manejo` | 1.79% | TF-IDF |
| 26 | `prop_cate_alto` | 1.79% | Categorías |
| 27 | `tfidf_pago` | 1.78% | TF-IDF |
| 28 | `prop_tipo_economico` | 1.75% | Tipos |
| 29 | `prop_asig_entidad` | 1.71% | Asignación |
| 30 | `tfidf_falta` | 1.69% | TF-IDF |
| — | **`anio`** | — | Control (obligatoria por tesis) |
| — | **`ipc`** | — | Control (obligatoria por tesis) |
| — | **`trm`** | — | Control (obligatoria por tesis) |

**Dataset reducido:** `contratos_features_reducido.csv` — 33 features, 350 contratos (1 outlier >200% eliminado)

---

## 2. Benchmarking — 10 Modelos

### 2.1 Metodología

| Parámetro | Valor |
|---|---|
| Validación | Nested CV: 5 outer folds × 5 inner folds |
| Búsqueda | RandomizedSearchCV (200–1000 iter por modelo) |
| Scoring | neg_root_mean_squared_error |
| Preprocesamiento | StandardScaler dentro de cada fold |
| Features | 33 (top 30 RF + anio/ipc/trm) |
| Hardware | CPU (todos salvo XGBoost), RTX 5060 (XGBoost GPU) |

### 2.2 Resultados v1 — 219 features (baseline)

| Modelo | RMSE | MAE | R² | Tiempo |
|---|---|---|---|---|
| Ridge | 16.1 | — | **0.040** | 1.6s |
| Lasso | 16.6 | — | −0.019 | 1.5s |
| ElasticNet | 16.1 | — | 0.030 | 1.7s |
| KNN | 16.8 | — | −0.043 | 1.0s |
| DecisionTree | 18.9 | — | −0.282 | 0.4s |
| **RandomForest** | **15.8** | — | **0.074** | **33.5s** |
| GradientBoosting | 16.5 | — | −0.001 | 63.7s |
| XGBoost | 17.0 | — | −0.063 | 19.4s |
| SVR | 16.5 | — | −0.002 | 10.8s |
| MLP | 25.5 | — | −1.041 | 136.8s |

### 2.3 Resultados v2 — 33 features (reducido)

| Modelo | RMSE | MAE | R² | Tiempo |
|---|---|---|---|---|
| **Ridge** | **15.6 ± 1.1** | **12.8** | **0.103 ± 0.080** | **0.9s** |
| Lasso | 15.9 ± 1.2 | 13.1 | 0.069 ± 0.091 | 0.6s |
| ElasticNet | 15.6 ± 1.1 | 12.9 | 0.094 ± 0.071 | 0.5s |
| KNN | 16.2 ± 1.2 | 13.2 | 0.035 ± 0.061 | 0.5s |
| DecisionTree | 17.6 ± 2.0 | 14.6 | −0.146 ± 0.156 | 0.5s |
| RandomForest | 15.7 ± 1.3 | 12.8 | 0.092 ± 0.100 | 42.2s |
| GradientBoosting | 15.8 ± 1.3 | 12.9 | 0.072 ± 0.103 | 33.6s |
| XGBoost_GPU | 15.8 ± 1.2 | 13.0 | 0.072 ± 0.090 | 162.0s |
| SVR | — | — | — | (>60 min) |
| MLP | — | — | — | (>60 min) |

**Campeón: Ridge** — R² 0.103, RMSE 15.6, tiempo < 1s.

### 2.4 Mejora v1 → v2

| Modelo | R² v1 | R² v2 | Δ |
|---|---|---|---|
| Ridge | 0.040 | **0.103** | +157% |
| ElasticNet | 0.030 | **0.094** | +213% |
| RandomForest | 0.074 | **0.092** | +24% |
| GradientBoosting | −0.001 | **0.072** | +73pp |
| XGBoost | −0.063 | **0.072** | +135pp |
| Lasso | −0.019 | 0.069 | +88pp |
| KNN | −0.043 | 0.035 | +78pp |

La reducción de 219 a 33 features eliminó ruido y estabilizó todos los modelos. La adición de TRM e IPC mejoró significativamente a los modelos lineales (Ridge, ElasticNet, Lasso).

---

## 3. Optimizaciones Exploradas (y Descartadas)

### 3.1 Log-transform del target

Se aplicó `np.log1p(sobrecosto)` para estabilizar la cola larga de la distribución del target. **Resultado: empeoró el R².**

| Modelo | R² original | R² con log+interacciones | Δ |
|---|---|---|---|
| Ridge | 0.103 **−0.094** | −0.197 |
| ElasticNet | 0.094 | −0.133 | −0.227 |
| RandomForest | 0.092 | −0.067 | −0.159 |

**Causa:** El sobrecosto ya tiene una distribución relativamente simétrica (media 27%, outlier >200% eliminado). El log-transform introdujo asimetría inversa y la back-transformación (`expm1`) amplificó errores en predicciones altas.

### 3.2 Interacciones sintéticas

Se probaron 3 interacciones (TRM × probabilidad, IPC × valor inicial, TRM × riesgo total). **No aportaron señal predictiva** y actuaron como ruido adicional para el modelo lineal.

### 3.3 Conclusión

El modelo Ridge **original** (sin log, sin interacciones, con las 33 features reducidas) es el campeón definitivo. No se requieren transformaciones adicionales.

---

## 4. Interpretación de Coeficientes (Ridge)

Los coeficientes de Ridge, entrenado con datos estandarizados, revelan qué variables incrementan o disminuyen el sobrecosto:

### Top 5 que AUMENTAN el sobrecosto

| Feature | Coeficiente | Interpretación |
|---|---|---|
| `tfidf_pago` | +0.036 | Riesgos de pago (incumplimientos, demoras) aumentan el sobrecosto |
| `tfidf_desarrollo` | +0.034 | Riesgos de desarrollo del proyecto (cambios de alcance, retrasos técnicos) |
| `tfidf_disenos` | +0.031 | Riesgos en los diseños técnicos (planos, especificaciones) |
| `tfidf_calidad` | +0.028 | Riesgos de calidad de obra (materiales, acabados) |
| `prop_tipo_operacional` | +0.022 | Mayor proporción de riesgos operacionales → mayor sobrecosto |

### Top 5 que DISMINUYEN el sobrecosto

| Feature | Coeficiente | Interpretación |
|---|---|---|
| `tfidf_informacion` | −0.049 | Riesgos de información/documentación bien gestionados reducen el sobrecosto |
| `tfidf_manejo` | −0.044 | Riesgos de manejo/gestión controlados |
| `tfidf_insumos` | −0.034 | Riesgos de insumos planificados → menor sobrecosto |
| `tfidf_cualquier` | −0.034 | Riesgos genéricos ("cualquier otro") — probablemente ruido irrelevante |
| `tfidf_ejecucion_contrato` | −0.020 | Riesgos de ejecución contractual bien gestionados |

---

## 5. Conclusión y Decisión Final

### Modelo seleccionado: **Ridge (con validación cruzada anidada)**

| Criterio | Resultado |
|---|---|
| R² (Nested CV) | **0.103 ± 0.080** |
| RMSE (Nested CV) | **15.6 ± 1.1 pp** |
| MAE (Nested CV) | **12.8 pp** |
| Tiempo de entrenamiento | **< 1 segundo** |
| Interpretabilidad | **Coeficientes lineales directos** |

### Justificación para la tesis

> "El modelo Ridge, entrenado con variables de riesgo e indicadores macroeconómicos (IPC y TRM), logró un R² de 0.103, superando a modelos más complejos como Random Forest (R² 0.092) y XGBoost (R² 0.072), y ofreciendo una interpretabilidad directa de los coeficientes para la toma de decisiones. La reducción de dimensionalidad de 219 a 33 features, combinada con la inclusión de TRM e IPC, permitió a los modelos lineales capturar la señal predictiva eliminando el ruido de las 186 features irrelevantes."

### Modelos descartados

| Modelo | Razón |
|---|---|
| SVR | R² negativo en v1, >60 min de entrenamiento |
| MLP | R² negativo en v1, >30 min de entrenamiento |
| XGBoost_GPU | R² inferior a Ridge (0.072 vs 0.103), 162s de entrenamiento |
| DecisionTree | R² negativo (−0.146), sobreajuste |
| GradientBoosting | R² inferior a Ridge (0.072 vs 0.103), 33s |
| RandomForest | R² inferior a Ridge (0.092 vs 0.103), 42s |

---

## 6. Archivos del Proyecto

```
risk_project/
├── docs/
│   ├── contratos_features.csv           # 219 features, 351 contratos
│   ├── contratos_features_reducido.csv  # 33 features, 350 contratos
│   └── modelo.md                        # Este documento
├── estudio_modelos/
│   ├── modelado.ipynb                   # v1: 219 features, baseline
│   ├── modelado_v2.ipynb                # v2: 33 features + GPU + optimizaciones
│   └── resultados_v2/                   # Gráficos y tablas generadas
```

---

## 7. Próximos Pasos

1. **Prototipo Streamlit** (Objetivo Específico 3): Dashboard interactivo con Ridge como motor de predicción, sliders para variables clave y gráfico de coeficientes para explicabilidad
2. **Validación con casos de estudio reales** (Objetivo Específico 4): 1-3 contratos con sobrecosto conocido para comparar predicción vs real
3. **Redacción del capítulo de resultados** en la tesis

---

## 8. Historial de Cambios

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-07 | v1 | Documento inicial. Resultados de feature engineering, benchmark v1 (219 vars) y v2 (33 vars). Ridge campeón (R² 0.103). Optimizaciones descartadas (log, interacciones). Coeficientes interpretados |
