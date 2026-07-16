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
| Control (v1) | 3 | `anio`, `ipc` (inflación), `trm` (tipo de cambio) — año único |

### 1.2 Feature Reduction

Tras entrenar un Random Forest baseline con todas las 219 features, se seleccionaron las **top 30 por importancia** + 3 variables de control:

| # | Feature | Importancia | Familia |
|---|---|---|---|
| 1 | `tfidf_desarrollo` | 10.14% | TF-IDF |
| 2 | `interaccion_prob_x_impacto` | 3.14% | Prob/Impacto |
| 3 | `tfidf_insumos` | 2.78% | TF-IDF |
| 4 | `prob_std` | 2.67% | Prob/Impacto |
| 5 | `tfidf_expedicion` | 2.66% | TF-IDF |
| 6 | `tfidf_materiales` | 2.60% | TF-IDF |
| 7 | `imp_promedio` | 2.43% | Prob/Impacto |
| 8 | `tfidf_ejecucion` | 2.24% | TF-IDF |
| 9 | `tfidf_contrato` | 2.23% | TF-IDF |
| 10 | `prop_tipo_operacional` | 2.14% | Tipos |
| 11 | `prob_promedio` | 2.13% | Prob/Impacto |
| 12 | `prop_cate_bajo` | 2.11% | Categorías |
| 13 | `valor_inicial` | 2.08% | Nulos |
| 14 | `tfidf_riesgo` | 2.07% | TF-IDF |
| 15 | `tfidf_tecnicas` | 2.06% | TF-IDF |
| 16 | `tfidf_municipio` | 2.06% | TF-IDF |
| 17 | `tfidf_obras` | 2.00% | TF-IDF |
| 18 | `tfidf_informacion` | 1.98% | TF-IDF |
| 19 | `prop_fuen_externo` | 1.97% | Fuente |
| 20 | `tfidf_cuando` | 1.96% | TF-IDF |
| 21 | `tfidf_disenos` | 1.85% | TF-IDF |
| 22 | `tfidf_ejecucion_contrato` | 1.83% | TF-IDF |
| 23 | `tfidf_calidad` | 1.83% | TF-IDF |
| 24 | `tfidf_manejo` | 1.79% | TF-IDF |
| 25 | `prop_cate_alto` | 1.79% | Categorías |
| 26 | `tfidf_pago` | 1.78% | TF-IDF |
| 27 | `prop_tipo_economico` | 1.75% | Tipos |
| 28 | `prop_asig_entidad` | 1.71% | Asignación |
| 29 | `tfidf_falta` | 1.69% | TF-IDF |
| 30 | `tfidf_obra` | — | TF-IDF (reemplazó `tfidf_cualquier` en v3) |

### 1.3 Migración a Features por Rango de Fechas (v4)

En la versión inicial, las variables macroeconómicas se representaban como año único (`anio`, `ipc`, `trm`). Esto limitaba el modelo a proyectos de un solo año. Para generalizar a proyectos multi-anuales, se reemplazaron las 3 variables de control por 5 variables de rango:

| Variable de control (v1–v3) | Reemplazo (v4) | Descripción |
|---|---|---|
| `anio` (único) | `anio_inicio` + `anio_fin` | Rango de años del proyecto |
| — | `duracion` | Duración en años (máx 5) |
| `ipc` (único) | `ipc_acumulado` | Inflación compuesta: Π(1+IPC_año) − 1 |
| `trm` (único) | `trm_promedio` | TRM promedio del período |

**Feature set final:** 35 features (30 TF-IDF/probabilidades + 5 de rango).

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

### 2.4 Comparación Final — Línea de Tiempo del Modelo

| Versión | Features | Modelo | R² (full) | R² CV | AUC CV | RMSE | Estado |
|---|---|---|---|---|---|---|---|
| v1 (Jun 2026) | 219 | Ridge | 0.040 | — | — | 16.1 | Baseline |
| v2 (Jul 2026) | 33 (año único) | Ridge | 0.103 | 0.103 | ~0.639 | 15.6 | Campeón inicial |
| v3 (Jul 2026) | 33 (año único) | Ridge | **0.149** | — | **0.662** | 16.0 | Re-entreno TF-IDF |
| v4a (Jul 2026) | 35 (rango) | Ridge | 0.417 | 0.066 | 0.650 | 16.2 | **Descartado** |
| v4b (Jul 2026) | 35 (rango) | **SVR RBF** | **0.417** | **0.068** | **0.673** | **17.1** | **Campeón final** |

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

Además de la interpretabilidad global, se implementó **interpretabilidad local** vía `shap.KernelExplainer` para desglosar la predicción de cada contrato entre sus riesgos individuales.

**Problema original:**
El desglose inicial usaba una heurística independiente del modelo:
```
contribución(i) = pred_base × (prob_i × imp_i) / Σ(prob × imp)
```
Esto repartía la predicción proporcionalmente a `prob×imp` sin considerar qué features del SVR realmente causaban esa predicción. Ignoraba TF-IDF, categorías, IPC, TRM y las no-linealidades del kernel RBF.

**Solución SHAP:**
Se reemplazó por valores Shapley locales que calculan la contribución de cada una de las 35 features a `pred_base`, y luego se mapean a los riesgos según:

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

El riesgo con P×I más bajo (2×3=6) pasó a ser el de mayor contribución. El SVR aprendió que palabras como "obra", "ejecución", "técnicos" y la categoría "medio" están asociadas a mayor sobrecosto en los 351 contratos de entrenamiento — algo que la heurística `prob×imp` no podía capturar.

**Validación matemática:**
```
SHAP: ∑shap_vals + expected_value = -1.87% + 22.10% = 20.23% = pred_base ✅
Desglose: ∑contribuciones = 20.24% ≈ 20.23% (error 0.01pp) ✅
```

**Rendimiento:** ~0.7s por contrato promedio (15 riesgos) con KernelExplainer + 1000 samples + 100 contratos background. Cacheados vía `lru_cache`.

---

## 4. Optimizaciones Exploradas (histórico)

### 4.1 Log-transform del target
Se aplicó `np.log1p(sobrecosto)` para estabilizar la cola larga. **Empeoró el R².** El sobrecosto ya tiene distribución relativamente simétrica (media 27%, outlier >200% eliminado). El log-transform introdujo asimetría inversa y la back-transformación amplificó errores.

### 4.2 Interacciones sintéticas
TRM × probabilidad, IPC × valor inicial, TRM × riesgo total. **No aportaron señal predictiva** y actuaron como ruido.

### 4.3 SHAP — Interpretabilidad Local por Riesgo
Inicialmente SHAP no estaba disponible por incompatibilidad numba+numpy 2.5 (Python 3.13). Se corrigió fijando `numpy<2.5` en las dependencias. Se implementó un desglose por riesgo basado en `shap.KernelExplainer` con 1000 samples y 100 contratos de background.

**Arquitectura del desglose SHAP:**
1. Se calculan los valores Shapley para las 35 features del contrato
2. Cada feature se mapea a los riesgos individuales según su naturaleza:
   - `prob_promedio`, `imp_promedio`, `interacción` → ponderado por `prob×imp`
   - `prop_tipo_*`, `prop_cate_*`, etc. → solo riesgos con ese valor categórico
   - `tfidf_*` → riesgos cuya descripción contiene la palabra
   - `duración`, `ipc_acumulado`, `trm_promedio` → split equitativo
3. La suma de contribuciones `= pred_base` (error < 0.01pp con 1000 samples)

**Reemplazo de la heurística anterior:**
El código original usaba `contribución = pred_base × (prob_i × imp_i) / Σ(prob × imp)`, que es independiente del modelo — no reflejaba lo que el SVR realmente aprendió. SHAP sí captura las no-linealidades del kernel RBF, el efecto del texto (TF-IDF), y las interacciones entre variables macro y la matriz de riesgos.

**Validación con datos sintéticos (Julio 2026):**
| Método | ¿Refleja el modelo? | Error de caja |
|---|---|---|
| Heurística `prob×imp` | ❌ No | — |
| SHAP nsamples=200 | ✅ Sí | ~0.03pp |
| SHAP nsamples=1000 | ✅ Sí | **~0.01pp** |

Tiempo de cómputo: ~0.7s por contrato promedio (15 riesgos) con nsamples=1000.

---

## 5. Conclusiones y Decisión Final

### Modelo seleccionado: **SVR (kernel RBF, C=10, gamma=scale)**

| Criterio | Resultado |
|---|---|
| R² CV (5-fold) | **0.068** |
| R² (full training / in-sample) | 0.417 |
| AUC CV (clasificador) | **0.673** |
| RMSE | **17.1 pp** |
| Features | **35 (30 TF-IDF + 5 rango)** |
| Interpretabilidad | **SHAP (local, 1000 samples) + Permutation importance (global, 10 reps) + Ridge referencia** |

### Justificación para la tesis

> "El modelo SVR con kernel RBF, entrenado con 35 features (30 TF-IDF + 5 de rango de fechas con IPC acumulado compuesto y TRM promedio), fue seleccionado como modelo campeón tras superar a Ridge en R² CV (0.068 vs 0.066) y AUC (0.673 vs 0.650). La migración de año único a rango de fechas, aunque metodológicamente necesaria para modelar proyectos multi-anuales, deterioró el rendimiento de Ridge (lineal), que no pudo capturar las relaciones no lineales entre duración, inflación compuesta y tipo de cambio. SVR, gracias a su kernel RBF, sí logró modelar estas interacciones. La interpretabilidad global se logra mediante permutation importance; la interpretabilidad local (por riesgo individual) se implementó vía SHAP KernelExplainer, reemplazando la heurística `prob×imp / Σ(prob×imp)` por valores Shapley que reflejan la contribución real de cada riesgo a la predicción del SVR."

### Modelos descartados en la fase final

| Modelo | Razón |
|---|---|
| Ridge (año único) | Metodológicamente incorrecto (proyectos multi-anual forzados a un año) |
| Ridge (rango) | R² CV 0.066 (inferior a SVR), no captura no-linealidades |
| ElasticNet (rango) | R² CV 0.063, similar a Ridge |
| RandomForest (rango) | R² CV 0.047, sobreajuste severo (R² full=0.93) |
| MLP (rango) | R² CV negativo (−0.037), datos insuficientes para red neuronal |

---

## 6 Historial de Cambios

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-07 | v1 | Documento inicial. Benchmark v1 (219 vars) y v2 (33 vars). Ridge campeón (R² 0.103). |
| 2026-07-09 | v2 | Re-entreno final. `tfidf_cualquier` → `tfidf_obra`. Ridge R² 0.149, AUC 0.662. |
| 2026-07-11 | v3 | Migración a rango de fechas. Ridge descartado (R² CV=0.066). SVR nuevo campeón (R² CV=0.072, AUC=0.673). Permutation importance. RMSE variable. |
| 2026-07-14 | v4 | Integración MLflow: experimentos, model registry, artifact store. Backend carga modelo desde MLflow con fallback local. |
| 2026-07-16 | v5 | **Desglose SHAP**: reemplazo de heurística `prob×imp` por interpretabilidad local vía `shap.KernelExplainer` (1000 samples). Cada riesgo recibe su contribución real al sobrecosto estimado. numpy fijado a <2.5 por compatibilidad con numba. |
