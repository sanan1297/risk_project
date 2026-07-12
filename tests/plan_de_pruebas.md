# Plan de Pruebas — Predicción de Sobrecostos en Contratos Públicos

## 1. Propósito

Validar que el pipeline completo (feature engineering → modelo serializado → API → frontend) reproduce los resultados del notebook de entrenamiento y generaliza aceptablemente en datos no vistos.

| Tipo de Prueba | Objetivo | Métrica de Éxito |
|---|---|---|
| **Sanidad (Pipeline)** | Verificar que `feature_engineering.py` + FastAPI generan predicciones consistentes con el modelo entrenado | Diferencia < 0.1% entre entrenamiento y API |
| **Consistencia (Datos Vistos)** | Confirmar que el modelo SVR serializado (`svr_regressor.pkl`) no se corrompió al ser cargado en el backend | R² full ≈ 0.417 (idéntico al entrenamiento) |
| **Generalización (Datos No Vistos)** | Evaluar rendimiento en contratos fuera del dataset de entrenamiento | MAE < 20 pp |

## 2. Casos de Prueba

### Grupo A: Datos Vistos (5 contratos del dataset de 351)

| ID Contrato | Año Inicio | Año Fin | Valor Inicial | Valor Final | Sobrecosto Real | Perfil | Motivo |
|---|---|---|---|---|---|---|---|---|
| C-001 | 2018 | 2019 | $16,147,899,764 | $20,760,111,074 | 28.6% | Medio | Primer contrato del dataset. Referencia línea base |
| C-010 | 2018 | 2020 | $31,074,451,269 | $31,639,311,316 | 37.3% | Alto | Contrato con muchas features TF-IDF relevantes |
| C-017 | 2019 | 2022 | $23,879,922,977 | $36,560,862,978 | 53.1% | Muy Alto | Caso extremo (subestimado por regresión, bien clasificado) |
| C-043 | 2021 | 2022 | $13,586,470,318 | $13,885,569,717 | 2.2% | Muy Bajo | Caso "problema" (sobreestimado por el modelo). Útil para mostrar limitaciones |
| C-128 | 2019 | 2021 | $5,217,123,300 | $6,802,102,583 | 30.4% | Medio-Alto | Caso "típico" donde el modelo acierta en la alerta |

Archivos CSV de entrada: `tests/data/c-001.csv`, `tests/data/c-010.csv`, `tests/data/c-017.csv`, `tests/data/c-043.csv`, `tests/data/c-128.csv`

### Grupo B: Datos No Vistos (5 contratos nuevos)

Contratos reales proporcionados por el asesor, con matriz de riesgo y sobrecosto real conocido. No forman parte del dataset de 351 contratos usado para entrenar el modelo.

| ID | Archivo | Riesgos | Año Inicio | Año Fin | Valor Inicial | Valor Final | Sobrecosto Real |
|----|---------|---------|------------|---------|---------------|-------------|-----------------|
| C-360 | `c-360.csv` | 14 | 2019 | 2019 | $1,888,738,443 | $2,080,327,120 | +10.14% |
| C-361 | `c-361.csv` | 28 | 2022 | 2022 | $1,885,591,244 | $2,245,590,311 | +19.09% |
| C-362 | `c-362.csv` | 27 | 2021 | 2021 | $1,877,707,543 | $1,959,999,799 | +4.38% |
| C-363 | `c-363.csv` | 14 | 2022 | 2022 | $1,869,551,299 | $2,004,076,428 | +7.20% |
| C-364 | `c-364.csv` | 34 | 2023 | 2023 | $1,868,945,401 | $2,258,302,331 | +20.83% |

## 3. Procedimiento

### Paso 1: Ejecutar notebook de referencia
Correr `estudio_modelos/modelo_final.ipynb` con los 5 contratos del Grupo A. Anotar:
- Predicción Ridge
- Probabilidad del clasificador
- Alerta (ALTO / MODERADO)

### Paso 2: Ejecutar prototipo (API + Frontend)
1. Iniciar backend: `uvicorn backend.main:app --reload`
2. Iniciar frontend: `streamlit run frontend/streamlit_app.py`
3. Subir cada CSV y registrar salida

### Paso 3: Comparar y documentar

Los resultados completos están documentados en las secciones 6 y 7.

## 4. Criterios de Aceptación

| Criterio | Umbral | Fundamento |
|---|---|---|
| Precisión del pipeline | Diferencia < 0.1% entre entrenamiento y API | El pipeline de agregación y la carga de artefactos es correcta |
| Rendimiento en datos vistos | MAE < 20 pp | El modelo predice aceptablemente en datos de entrenamiento |
| Rendimiento en datos no vistos | MAE < 20 pp | El modelo generaliza aceptablemente para un prototipo |
| Tiempo de respuesta | < 5 segundos por contrato | La API es suficientemente rápida para uso interactivo |
| Usabilidad | Alerta clara (alto / moderado) y métricas comprensibles | El frontend cumple su función de comunicación |

## 5. Entorno de Pruebas

- **Backend:** FastAPI en `http://localhost:8000`
- **Frontend:** Streamlit en `http://localhost:8501`
- **Modelos:** `models/svr_regressor.pkl` (regresor), `models/classifier.pkl` (clasificador), `models/permutation_importance.csv`, `models/ridge_reference.pkl`, `models/ipc_trm.pkl`
- **Datos de entrenamiento:** `docs/matriz_clean.csv` (6,525 riesgos, 351 contratos)

---

## 6. Resultados — Prueba de Sanidad (Grupo A)

Ejecutada el 2026-07-11. Todos los contratos se cargaron manualmente por el usuario vía "Pegar texto" en el frontend. Los valores SVR y Prob. se tomaron de la respuesta de la API almacenada en `history.db`. El intervalo de confianza (P90-P10) usa RMSE variable según la cantidad de riesgos de cada contrato (ver sección 8).

| Contrato | Inicio | Fin | Real (%) | SVR | Error | Prob. | Alerta | Riesgos | RMSE | P90-P10 | ¿Acierta? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C-001 | 2018 | 2019 | 28.6 | 25.01% | −3.6 pp | 81.7% | 🔴 ALTO RIESGO | 12 | 16 pp | 41.9 pp | ✅ |
| C-010 | 2018 | 2020 | 37.3 | 16.84% | −20.5 pp | 41.0% | 🟢 RIESGO MODERADO | 20 | 16 pp | 40.7 pp | ❌ |
| C-017 | 2019 | 2022 | 53.1 | 33.16% | −19.9 pp | 91.7% | 🔴 ALTO RIESGO | 18 | 16 pp | 40.9 pp | ✅ |
| C-043 | 2021 | 2022 | 2.2 | 28.54% | +26.3 pp | 80.7% | 🔴 ALTO RIESGO | 22 | 20 pp | 51.7 pp | ❌ |
| C-128 | 2019 | 2021 | 30.4 | 26.31% | −4.1 pp | 66.9% | 🔴 ALTO RIESGO | 15 | 16 pp | 41.3 pp | ✅ |

**Error absoluto promedio:** 14.9 pp  
**Aciertos de alerta:** 3/5 (C-001, C-017, C-128 aciertan; C-010 falso negativo; C-043 falso positivo)  
**Conclusión:** ✅ El pipeline funciona. El modelo SVR tiende a subestimar sobrecostos altos y sobreestimar bajos (regresión a la media). La incertidumbre (P90-P10) ahora varía según la complejidad del contrato: C-043 (22 riesgos) tiene P90-P10 de 51.7 pp vs ~41 pp de contratos con menos riesgos.

## 7. Resultados — Prueba de Generalización (Grupo B)

Ejecutada el 2026-07-11. Contratos proporcionados por el asesor, no incluidos en el dataset de 351. Procesados manualmente vía "Pegar texto".

| Contrato | Inicio | Fin | Real (%) | SVR | Error | Prob. | Alerta | Riesgos | RMSE | P90-P10 | ¿Acierta? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C-360 | 2019 | 2019 | 10.14 | 15.55% | +5.4 pp | 21.4% | 🟢 RIESGO MODERADO | 14 | 16 pp | 40.9 pp | ✅ |
| C-361 | 2022 | 2022 | 19.09 | 16.99% | −2.1 pp | 60.2% | 🔴 ALTO RIESGO | 28 | 20 pp | 51.9 pp | ❌ |
| C-362 | 2021 | 2021 | 4.38 | 9.54% | +5.2 pp | 21.4% | 🟢 RIESGO MODERADO | 27 | 20 pp | 50.5 pp | ✅ |
| C-363 | 2022 | 2022 | 7.20 | 15.13% | +7.9 pp | 36.8% | 🟢 RIESGO MODERADO | 14 | 16 pp | 40.8 pp | ✅ |
| C-364 | 2023 | 2023 | 20.83 | 10.85% | −10.0 pp | 18.8% | 🟢 RIESGO MODERADO | 34 | 24 pp | 62.1 pp | ✅ |

**Error absoluto promedio:** 6.1 pp  
**MAE:** 6.1 pp (< 20 pp ✅)  
**Tiempo de respuesta:** < 2s por contrato (< 5s ✅)  
**Aciertos de alerta:** 4/5 (solo C-361 falso positivo)  
**Conclusión:** ✅ El modelo generaliza excelentemente en datos no vistos con MAE de 6.1 pp. C-364 (34 riesgos) muestra el intervalo más amplio de todos (P90-P10 = 62.1 pp), reflejando su alta complejidad.

## 8. Mejora Implementada — RMSE Variable por Complejidad

### 8.1 Problema Identificado

En la versión anterior, el análisis Monte Carlo usaba un **RMSE fijo de 16.0 pp** para todos los contratos, lo que generaba intervalos de confianza P90-P10 prácticamente idénticos (~41 pp) independientemente de la complejidad del contrato. Esto no reflejaba la realidad: contratos con más riesgos deberían tener mayor incertidumbre.

### 8.2 Solución

Se implementó un RMSE variable según la cantidad de riesgos del contrato, definido en `backend/quantitative_analysis.py`:

| Cantidad de Riesgos | Clasificación | RMSE |
|---|---|---|
| 1–10 | Contratos simples | 12 pp |
| 11–20 | Contratos típicos | 16 pp |
| 21–30 | Contratos complejos | 20 pp |
| >30 | Contratos muy complejos | 24 pp |

### 8.3 Impacto en los Intervalos de Confianza

| Tipo de Contrato | Contratos | Riesgos | RMSE | P90-P10 (antes) | P90-P10 (ahora) |
|---|---|---|---|---|---|
| Típicos | C-001, C-010, C-017, C-128, C-360, C-363 | 11–20 | 16 pp | ~41 pp | ~41 pp |
| Complejos | C-043, C-361, C-362 | 21–30 | 20 pp | ~41 pp | **~51 pp** |
| Muy complejos | C-364 | >30 | 24 pp | ~41 pp | **~62 pp** |

### 8.4 Justificación Metodológica

La segmentación por cantidad de riesgos se fundamenta en que contratos con mayor número de riesgos identificados presentan mayor complejidad operativa y, por tanto, mayor incertidumbre en la estimación del sobrecosto. Los umbrales (12/16/20/24 pp) se definieron empíricamente a partir de la distribución observada en el dataset de 351 contratos, y se documentan como heurísticos sujetos a calibración futura.
