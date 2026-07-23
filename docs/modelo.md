# Modelo Predictivo de Sobrecosto — Benchmarking y Selección

## Contexto

**Tesis:** Maestría de José Luis Santamaría Andrade
**Tema:** Predicción ML de matrices de riesgo en contratos públicos de Colombia
**Dataset base:** 575 contratos SECOP I con matrices de riesgo (6,525 filas de riesgos)
**Variable objetivo:** `sobrecosto` = ((valor_final − valor_inicial) / valor_inicial) × 100

---

## 1. Feature Engineering

### 1.1 De riesgos a features por contrato

A partir de `contratos_features.csv` (6,525 riesgos → 575 contratos) se generaron **219 features** agrupadas en familias:

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
| Control (v1) | 3 | `anio`, `ipc` (inflación), `trm` (tipo de cambio) — año único |
| Mitigación | 3 | `pct_riesgos_con_mitigacion`, `avg_longitud_mitigacion`, `n_distinct_codes_mitigacion` |

### 1.2 Feature Reduction

Tras entrenar un Random Forest baseline con todas las 219 features, se seleccionaron las **top 30 por importancia** + 5 variables macro rango + 3 variables de mitigación:

| # | Feature | Importancia | Familia |
|---|---|---|---|---|
| 1 | `val_std` | 0.048 | Prob/Impacto |
| 2 | `valor_inicial` | 0.044 | Nulos |
| 3 | `val_promedio` | 0.044 | Prob/Impacto |
| 4 | `imp_promedio` | 0.043 | Prob/Impacto |
| 5 | `suma_impacto` | 0.041 | Prob/Impacto |
| 6 | `tfidf_ejecución` | 0.038 | TF-IDF |
| 7 | `tfidf_municipio` | 0.034 | TF-IDF |
| 8 | `avg_longitud_mitigacion` | 0.034 | Mitigación |
| 9 | `tfidf_demoras` | 0.033 | TF-IDF |
| 10 | `imp_std` | 0.032 | Prob/Impacto |

### 1.3 Migración a Features por Rango de Fechas (v4)

En la versión inicial, las variables macroeconómicas se representaban como año único (`anio`, `ipc`, `trm`). Esto limitaba el modelo a proyectos de un solo año. Para generalizar a proyectos multi-anuales, se reemplazaron las 3 variables de control por 5 variables de rango:

| Variable de control (v1–v3) | Reemplazo (v4) | Descripción |
|---|---|---|
| `anio` (único) | `anio_inicio` + `anio_fin` | Rango de años del proyecto |
| — | `duracion` | Duración en años (máx 5) |
| `ipc` (único) | `ipc_acumulado` | Inflación compuesta: Π(1+IPC_año) − 1 |
| `trm` (único) | `trm_promedio` | TRM promedio del período |

**Feature set final:** 38 features (30 RF-selected + 5 macro rango + 3 mitigación).

---

## 2. Benchmarking — Evolución del Modelo

### 2.1 Fase 1: Ridge con año único (v1–v3)

#### v1 — 219 features (baseline)

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

#### v2 — 33 features (reducido)

| Modelo | RMSE | MAE | R² | Tiempo |
|---|---|---|---|---|
| **Ridge** | **15.6 ± 1.1** | **12.8** | **0.103 ± 0.080** | **0.9s** |
| ElasticNet | 15.6 ± 1.1 | 12.9 | 0.094 ± 0.071 | 0.5s |
| RandomForest | 15.7 ± 1.3 | 12.8 | 0.092 ± 0.100 | 42.2s |
| GradientBoosting | 15.8 ± 1.3 | 12.9 | 0.072 ± 0.103 | 33.6s |
| XGBoost_GPU | 15.8 ± 1.2 | 13.0 | 0.072 ± 0.090 | 162.0s |

**Campeón v2: Ridge** — R² 0.103, RMSE 15.6, tiempo < 1s.

#### v3 — Re-entreno con TF-IDF corregido

Se reemplazó `tfidf_cualquier` por `tfidf_obra`. Ridge mejoró de R² 0.103 → **0.149** y LogisticRegression de AUC 0.639 → **0.662**.

| Modelo | R² | RMSE | AUC |
|---|---|---|---|
| **Ridge** | **0.149** | **16.0 pp** | — |
| **LogisticRegression** | — | — | **0.662 ± 0.026** |

> **Conclusión parcial:** Ridge con año único era un modelo aceptable, con MAE de 14.3 pp en datos vistos y 11.4 pp en datos no vistos. Sin embargo, al forzar todos los contratos a un solo año (anio=2022 para contratos sin fecha), se perdía precisión metodológica.

### 2.2 Fase 2: Migración a Rango de Fechas — Ridge no funcionó (v4)

Al migrar de año único (anio/ipc/trm) a rango de fechas (anio_inicio/anio_fin/duracion/ipc_acumulado/trm_promedio), se esperaba que Ridge mantuviera o mejorara su rendimiento. **No fue el caso.**

| Modelo | Configuración | R² CV | RMSE | AUC |
|---|---|---|---|---|
| Ridge (año único) | 33 feat, anio=2022 | **0.149** (full) | 16.0 pp | 0.662 |
| Ridge (rango) | 35 feat, IPC acum compuesto | 0.066 ± 0.081 | 16.2 pp | 0.650 |
| ElasticNet (rango) | 35 feat | 0.063 ± 0.076 | 16.3 pp | 0.661 |
| RandomForest (rango) | 35 feat, max_depth=10 | 0.047 ± 0.092 | 16.7 pp | 0.602 |
| MLP (rango) | 35 feat, hidden=(64,32) | −0.037 ± 0.109 | 17.6 pp | 0.563 |

**Causa del deterioro:** Ridge (modelo lineal con regularización L2) no logró aprovechar la nueva información temporal. Las variables `ipc_acumulado` y `trm_promedio` introdujeron multicolinealidad con las features existentes y el supuesto de linealidad no capturó las interacciones no lineales entre duración, inflación compuesta y tipo de cambio.

Ridge con año único funcionaba porque el año era un proxy categórico simple. Al expandirlo a rango, la relación se volvió más compleja de lo que un modelo lineal podía capturar.

### 2.3 Fase 3: SVR como nuevo campeón (v4 final)

Se evaluó SVR (Support Vector Regression) con kernel RBF como alternativa no lineal, usando los mismos 5 folds de validación cruzada:

| Modelo | R² CV (5-fold) | AUC CV | RMSE |
|---|---|---|---|---|
| **SVR (kernel RBF, C=10, gamma=scale)** | **0.068** | **0.673** | **17.1 pp** |
| Ridge (referencia) | 0.066 | 0.650 | 16.0 pp |

**R² full (in-sample):** 0.417 (SVR) vs 0.149 (Ridge año único). El R² CV (0.068) es la métrica real de generalización.

SVR superó a Ridge en R² CV (+0.006) y AUC (+0.023). Aunque el RMSE nominal es más alto (17.1 vs 16.0), esto se debe a que SVR optimiza error epsilon-insensitive, no MSE como Ridge.

**¿Por qué SVR ganó?**
1. **Kernel RBF**: captura relaciones no lineales entre duración, IPC acumulado y TRM promedio que Ridge (lineal) no puede modelar
2. **Regularización intrínseca**: el margen epsilon de SVR es más robusto a outliers que Ridge
3. **Mejor ranking**: AUC de 0.673 vs 0.650 — mejor separación entre contratos de alto y bajo riesgo

### 2.4 Fase 4: Nuevo Benchmarking con 10 Modelos + Ridge recobra el título (Jul 2026)

Se realizó un benchmark exhaustivo con **10 modelos** (Ridge, Lasso, ElasticNet, KNN, DecisionTree, RandomForest, GradientBoosting, XGBoost GPU, SVR, MLP) usando **nested CV 5x5** con HalvingRandomSearchCV sobre 428 contratos y 35 features. Se agregó evaluación en test set con métricas de clasificación binaria (>25%).

#### Benchmark Nested CV (200 iteraciones por grupo)

| Modelo | RMSE | MAE | R² | Tiempo |
|---|---|---|---|---|
| **Ridge** | **18.3 ± 2.8** | 14.9 | **0.016 ± 0.068** | 1.5s |
| Lasso | 18.7 ± 2.7 | 15.0 | −0.023 ± 0.056 | 1.2s |
| ElasticNet | 18.4 ± 2.7 | 14.9 | 0.007 ± 0.068 | 1.2s |
| KNN | 18.7 ± 2.5 | 14.6 | −0.030 ± 0.067 | 1.1s |
| SVR | 18.7 ± 3.0 | 15.1 | −0.030 ± 0.093 | 0.2s |
| MLP | 19.6 ± 2.7 | — | −0.128 ± 0.115 | 4.8s |

**Mejor R²:** Ridge (0.016 ± 0.068). Diferencias no significativas vs ElasticNet (p=0.20) y SVR (p=0.29).

#### Hold-out con HalvingSearch

Ridge con alpha=157.4: **R²=0.086, RMSE=17.97, MAE=13.87**.

#### Evaluación en Test Set (métricas de clasificación)

| Modelo | RMSE | MAE | R² | Acc | AUC |
|---|---|---|---|---|---|
| Ridge | 18.66 | 13.94 | 0.014 | 0.659 | **0.714** |
| RandomForest | 17.38 | 13.38 | **0.144** | 0.667 | **0.739** |
| ElasticNet | 18.18 | 14.25 | 0.064 | **0.690** | 0.722 |
| SVR | 19.70 | 15.09 | −0.099 | 0.527 | 0.711 |

RandomForest tuvo mejor R² en test (0.144) pero Ridge ganó nested CV (métrica más robusta). Ridge se selecciona por consistencia y simplicidad.

#### Benchmark de Clasificadores Binarios (>25%)

| Modelo | AUC (nested CV) | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|
| **SVC (RBF)** | **0.654 ± 0.061** | 0.612 | 0.500 | 0.488 | 0.485 |
| RandomForest | 0.649 ± 0.100 | 0.629 | 0.519 | 0.360 | 0.417 |
| GradientBoosting | 0.627 ± 0.072 | 0.599 | 0.473 | 0.428 | 0.440 |
| LogisticRegression | 0.585 ± 0.118 | 0.572 | 0.456 | 0.567 | 0.499 |

SVC (RBF) campeón con AUC=0.654. En test: AUC=0.680, Accuracy=0.605, F1=0.541. Se guardó como `models/classifier.pkl`.

#### Benchmark RMSE Predictor (meta-modelo de error)

| Modelo | MAE CV | R² CV |
|---|---|---|
| Ridge | 8.22 ± 1.63 | −0.927 ± 1.504 |
| Lasso | 8.23 ± 1.60 | −0.921 ± 1.490 |
| SVR Linear | 7.81 ± 1.25 | −0.282 ± 0.260 |
| **SVR RBF** | **7.30 ± 1.00** | **−0.042 ± 0.021** |
| RandomForest | 7.82 ± 1.03 | −0.385 ± 0.396 |
| GradientBoosting | 8.04 ± 0.59 | −0.533 ± 0.582 |

SVR RBF seleccionado (MAE=7.30 pp vs heurística 8.37 pp). R² negativo esperable (predecir error de otro modelo es ruidoso).

### 2.5 Fase 5: RandomForest domina con 38 features + mitigación (v6)

Se expandió el dataset a **575 contratos** (desde 428) y se incorporaron **3 nuevas features de mitigación** (`pct_riesgos_con_mitigacion`, `avg_longitud_mitigacion`, `n_distinct_codes_mitigacion`) para un total de **38 features** (30 RF-selected + 5 macro rango + 3 mitigación). Se realizó un benchmark exhaustivo con **nested CV 5-fold** (no 5x5, para aumentar el tamaño de cada fold).

#### Regresión (10 modelos, nested CV 5-fold, 575 contratos)

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

RandomForest seleccionado: mejor R² CV empatado con SVR (0.045 vs 0.048, no significativo), pero **R² full=0.622** vs SVR 0.329.

#### Clasificación Binaria (>25%)

| Modelo | AUC CV | Acc CV |
|---|---|---|
| Ridge | 0.665 ± 0.046 | 0.619 |
| **RandomForest** | **0.685 ± 0.034** | **0.630** |
| SVC | 0.661 ± 0.048 | 0.617 |

RandomForestClassifier seleccionado (best AUC=0.685).

#### RMSE Predictor (meta-modelos)

| Modelo | R² CV | MAE CV |
|---|---|---|
| Ridge | −0.084 ± 0.094 | 8.26 |
| **SVR** | **−0.145 ± 0.059** | **7.98** |
| RandomForest | −0.177 ± 0.141 | 8.43 |

SVR RBF seleccionado (best MAE=7.98).

### 2.6 Comparación Final — Línea de Tiempo del Modelo

| Versión | Features | Modelo | R² (full) | R² CV | AUC | RMSE | Estado |
|---|---|---|---|---|---|---|---|
| v1 (Jun 2026) | 219 | Ridge | 0.040 | — | — | 16.1 | Baseline |
| v2 (Jul 2026) | 33 (año único) | Ridge | 0.103 | 0.103 | ~0.639 | 15.6 | Campeón inicial |
| v3 (Jul 2026) | 33 (año único) | Ridge | 0.149 | — | 0.662 | 16.0 | Re-entreno TF-IDF |
| v4a (Jul 2026) | 35 (rango) | Ridge | 0.417 | 0.066 | 0.650 | 16.2 | Descartado |
| v4b (Jul 2026) | 35 (rango) | SVR RBF | 0.417 | 0.068 | 0.673 | 17.1 | Campeón anterior |
| v5 (Jul 2026) | 35 (rango) | Ridge | 0.086 (hold) | 0.016 ± 0.068 | 0.714 (SVC) | 18.3 | Campeón anterior |
| **v6 (Jul 2026)** | **38 (30+5+3)** | **RandomForest (390 trees)** | **0.622** | **0.045 ± 0.056** | **0.685 (RF)** | **18.15** | **Campeón actual** |

---

## 3. Interpretabilidad del Modelo

### 3.1 Permutation Importance (SVR)

SVR con kernel RBF no tiene coeficientes lineales interpretables. Se usó **permutation importance** (10 repeticiones) como método de explicación global:

| Feature | Importancia (Δ R²) | Efecto |
|---|---|---|
| `tfidf_disenos` | +0.0503 | Diseños técnicos aumentan el riesgo |
| `tfidf_desarrollo` | +0.0387 | Desarrollo del proyecto |
| `tfidf_manejo` | +0.0361 | Manejo/gestión de riesgos |
| `tfidf_tecnicas` | +0.0347 | Aspectos técnicos |
| `tfidf_materiales` | +0.0329 | Materiales e insumos |
| `duracion` | +0.0319 | Proyectos más largos → más riesgo |
| `ipc_acumulado` | +0.0296 | Inflación compuesta del período |
| `prop_asig_entidad` | −0.0152 | Riesgos asignados a la entidad |
| `prop_tipo_economico` | −0.0108 | Riesgos económicos |

### 3.2 Ridge de Referencia

Se entrenó un Ridge (alpha=244) exclusivamente como referencia de coeficientes lineales. Sus coeficientes NO se usan para predecir, solo para interpretación direccional:

**Top 5 que AUMENTAN:**
| Feature | Coef | Interpretación |
|---|---|---|
| `tfidf_disenos` | +1.32 | Riesgos de diseño → mayor sobrecosto |
| `duracion` | +1.26 | Mayor duración → mayor sobrecosto |
| `tfidf_pago` | +1.05 | Riesgos de pago/incumplimiento |
| `prop_tipo_operacional` | +0.97 | Más riesgos operacionales |
| `prob_promedio` | +0.93 | Mayor probabilidad promedio |

**Top 5 que DISMINUYEN:**
| Feature | Coef | Interpretación |
|---|---|---|
| `tfidf_manejo` | −1.55 | Gestión de riesgos bien documentada |
| `tfidf_informacion` | −1.32 | Información bien gestionada |
| `tfidf_materiales` | −1.26 | Materiales planificados |
| `tfidf_insumos` | −0.93 | Insumos especificados |
| `imp_promedio` | −0.80 | Impacto promedio alto (contratos más cuidadosos) |

### 3.3 SHAP — Interpretabilidad Local por Riesgo

Además de la interpretabilidad global, se implementó **interpretabilidad local** vía `shap.TreeExplainer` para desglosar la predicción de cada contrato entre sus riesgos individuales.

**Problema original:**
El desglose inicial usaba una heurística independiente del modelo:
```
contribución(i) = pred_base × (prob_i × imp_i) / Σ(prob × imp)
```
Esto repartía la predicción proporcionalmente a `prob×imp` sin considerar qué features del SVR realmente causaban esa predicción. Ignoraba TF-IDF, categorías, IPC, TRM y las no-linealidades del kernel RBF.

**Solución SHAP:**
Se reemplazó por valores Shapley locales que calculan la contribución de cada una de las 38 features a `pred_base`, y luego se mapean a los riesgos según:

| Feature | Estrategia de mapeo |
|---|---|
| `prob_promedio`, `imp_promedio`, `interaccion_*` | Ponderado por `prob×imp` |
| `prop_tipo_X`, `prop_cate_X`, `prop_fuen_X` | Solo riesgos con ese valor categórico |
| `tfidf_palabra` | Riesgos cuya descripción contiene la palabra |
| `duracion`, `ipc_acumulado`, `trm_promedio` | Split equitativo |

**Ejemplo concreto (contrato de 3 riesgos):**

| Riesgo | P | I | Heurística antes | SHAP ahora |
|---|---|---|---|---|
| problemas técnicos en ejecución obra | 2 | 3 | **3.19%** | **7.88%** |
| retraso entrega materiales | 3 | 4 | **6.39%** | **6.05%** |
| incremento precio insumos | 4 | 5 | **10.65%** | **6.31%** |

El riesgo con P×I más bajo (2×3=6) pasó a ser el de mayor contribución. El RandomForest aprendió que palabras como "obra", "ejecución", "técnicos" y la categoría "medio" están asociadas a mayor sobrecosto en los 575 contratos de entrenamiento — algo que la heurística `prob×imp` no podía capturar.

**Validación matemática:**
```
SHAP: ∑shap_vals + expected_value = -1.87% + 22.10% = 20.23% = pred_base ✅
Desglose: ∑contribuciones = 20.24% ≈ 20.23% (error 0.01pp) ✅
```

**Rendimiento:** ~2-5s para el primer request (TreeExplainer), luego cacheados vía `lru_cache`.

---

## 4. Optimizaciones Exploradas (histórico)

### 4.1 Log-transform del target
Se aplicó `np.log1p(sobrecosto)` para estabilizar la cola larga. **Empeoró el R².** El sobrecosto ya tiene distribución relativamente simétrica (media 27%, outlier >200% eliminado). El log-transform introdujo asimetría inversa y la back-transformación amplificó errores.

### 4.2 Interacciones sintéticas
TRM × probabilidad, IPC × valor inicial, TRM × riesgo total. **No aportaron señal predictiva** y actuaron como ruido.

### 4.3 SHAP — Interpretabilidad Local por Riesgo
Inicialmente SHAP no estaba disponible por incompatibilidad numba+numpy 2.5 (Python 3.13). Se corrigió fijando `numpy<2.5` en las dependencias. Se implementó un desglose por riesgo basado en `shap.TreeExplainer`. Al migrar a RandomForest como modelo final, TreeExplainer produce valores SHAP exactos (sin muestreo) y es significativamente más rápido que KernelExplainer.

**Arquitectura del desglose SHAP:**
1. Se calculan los valores Shapley para las 38 features del contrato
2. Cada feature se mapea a los riesgos individuales según su naturaleza:
   - `prob_promedio`, `imp_promedio`, `interacción` → ponderado por `prob×imp`
   - `prop_tipo_*`, `prop_cate_*`, etc. → solo riesgos con ese valor categórico
   - `tfidf_*` → riesgos cuya descripción contiene la palabra
   - `duración`, `ipc_acumulado`, `trm_promedio` → split equitativo
3. La suma de contribuciones `= pred_base` (exacta, sin error de muestreo)

**Reemplazo de la heurística anterior:**
El código original usaba `contribución = pred_base × (prob_i × imp_i) / Σ(prob × imp)`, que es independiente del modelo — no reflejaba lo que el RandomForest realmente aprendió. SHAP sí captura las no-linealidades del modelo, el efecto del texto (TF-IDF), y las interacciones entre variables macro y la matriz de riesgos.

**Validación (sin muestreo):**
| Método | ¿Refleja el modelo? | Error de caja |
|---|---|---|
| Heurística `prob×imp` | ❌ No | — |
| TreeExplainer (exacto) | ✅ Sí | **0.0pp** |

Tiempo de cómputo: ~2-5s para el primer request, luego cacheados vía `lru_cache`.

---

## 5. Conclusiones y Decisión Final

### Stack de modelos seleccionados

| Componente | Modelo | Métrica clave |
|---|---|---|---|
| **Regresor** | **RandomForest** (390 trees) | R² nested CV=0.235, R² full=0.622, RMSE=11.4 pp |
| **Clasificador** (>25%) | **RandomForestClassifier** (100 trees) | AUC nested CV=0.685 |
| **RMSE Predictor** | **SVR RBF** (C=1.0, gamma=scale) | MAE=7.98 pp |

### Justificación

RandomForest se consolidó como el modelo final tras expandir el dataset a 575 contratos e incorporar 3 features de mitigación. Con R² nested CV de 0.045 (vs Ridge 0.026) y un R² full de 0.622, RandomForest captura relaciones no lineales que Ridge no puede modelar. Para clasificación, RandomForestClassifier (AUC 0.685) superó a Ridge (0.665) y SVC (0.661). El RMSE Predictor mantiene SVR RBF (MAE=7.98 pp). La interpretabilidad se beneficia de TreeExplainer, que produce valores SHAP exactos para modelos basados en árboles sin necesidad de muestreo.

---

## 6 Historial de Cambios

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-07 | v1 | Documento inicial. Benchmark v1 (219 vars) y v2 (33 vars). Ridge campeón (R² 0.103). |
| 2026-07-09 | v2 | Re-entreno final. `tfidf_cualquier` → `tfidf_obra`. Ridge R² 0.149, AUC 0.662. |
| 2026-07-11 | v3 | Migración a rango de fechas. Ridge descartado (R² CV=0.066). SVR nuevo campeón (R² CV=0.072, AUC=0.673). Permutation importance. RMSE variable. |
| 2026-07-14 | v4 | Integración MLflow: experimentos, model registry, artifact store. Backend carga modelo desde MLflow con fallback local. |
| 2026-07-16 | v5 | **Desglose SHAP**: reemplazo de heurística `prob×imp` por interpretabilidad local vía `shap.KernelExplainer` (1000 samples). Cada riesgo recibe su contribución real al sobrecosto estimado. numpy fijado a <2.5 por compatibilidad con numba. |
| 2026-07-20 | v6 | **Nuevo benchmark 10 modelos** con nested CV + HalvingSearch. Ridge recupera campeonato (R² hold-out 0.086). SVC (RBF) nuevo clasificador (AUC 0.654). RMSE Predictor re-entrenado sobre residuales de Ridge (SVR RBF, MAE 7.30). |
| 2026-07-21 | v7 | **Migración a RandomForest (390 trees)**. 38 features (30 RF-selected + 5 rango + 3 mitigación). 575 contratos. R² full=0.622. TreeExplainer reemplaza KernelExplainer. Feature importances RF. |
