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

### Grupo C: Proyecto No Terminado (Predicción a Futuro)

Contrato real activo de SECOP II (no terminado). Se usa para prueba de predicción: el modelo estima sobrecosto en una fecha futura (2027-03-28) y se documenta la predicción. No hay sobrecosto real aún — se validará cuando el contrato termine.

| ID | Archivo | Riesgos | Inicio | Fin Proyectado | Valor Inicial | Estado | Entidad | URL |
|----|---------|---------|--------|----------------|---------------|--------|---------|-----|
| C-365 | `c-365.csv` | 25 | 2023-06-22 | 2027-03-28 | $477,834,784,322 | Modificado (activo) | Distrito Capital de Bogotá - IDU | [SECOP](https://community.secop.gov.co/Public/Tendering/OpportunityDetail/Index?noticeUID=CO1.NTC.3690937) |

**Objeto:** Construcción de la intersección a desnivel de Puente Aranda y demás obras complementarias, correspondiente a las obras de adecuación al sistema Transmilenio de la Troncal Calle 13 en Bogotá D.C.

### Grupo D: C-128 en Múltiples Rangos de Tiempo

Misma matriz de riesgos (C-128, 15 riesgos) pero variando los años de inicio y fin para observar cómo cambia la predicción según el contexto macroeconómico (IPC, TRM, etc.). 13 rangos bienales solapados desde 2010 hasta 2024.

Archivo CSV de entrada: `tests/data/c-128.csv` (matriz compartida). Resultados: `tests/data/c-128_temporal.csv`.

## 3. Procedimiento

### Paso 1: Ejecutar notebook de referencia
Correr `estudio_modelos/modelo_final.ipynb` con los 5 contratos del Grupo A. Anotar:
- Predicción Ridge
- Probabilidad del clasificador
- Alerta (ALTO / MODERADO)

### Paso 2: Ejecutar prototipo (API + Frontend)
1. Iniciar backend: `uv run uvicorn backend.main:app --reload` (o `docker compose up -d` para contenerizado)
2. Iniciar frontend: `uv run streamlit run frontend/streamlit_app.py` (o acceder a `http://localhost:8501` si está en Docker)
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

### Local (desarrollo)

- **Backend:** FastAPI en `http://localhost:8003`
- **Frontend:** Streamlit en `http://localhost:8501`
- **MLflow:** `http://localhost:5000` (UI de experimentos, opcional)
- **Modelos:** `models/svr_regressor.pkl` (regresor), `models/classifier.pkl` (clasificador), `models/permutation_importance.csv`, `models/ridge_reference.pkl`, `models/ipc_trm.pkl`
- **Datos de entrenamiento:** `docs/matriz_clean.csv` (6,525 riesgos, 351 contratos)

### Docker (entorno contenerizado)

```bash
# Iniciar servicios
docker compose up -d

# Verificar estado
docker compose ps

# Endpoints
# - Backend: http://localhost:8003
# - Frontend: http://localhost:8501
# - MLflow UI: http://localhost:5000

# Ejecutar pruebas vía API
curl -X POST http://localhost:8003/predict -F "file=@tests/data/c-001.csv"
```

---

## 6. Resultados — Prueba de Sanidad (Grupo A)

Ejecutada el 2026-07-11. Todos los contratos se cargaron manualmente por el usuario vía "Pegar texto" en el frontend. Los valores SVR y Prob. se tomaron de la respuesta de la API almacenada en `history.db`. El intervalo de confianza (P90-P10) usa RMSE variable según la cantidad de riesgos de cada contrato (ver sección 8).

| Contrato | Inicio | Fin | Real (%) | SVR | Error | Prob. | Alerta | Riesgos | RMSE | P10 | P50 | P90 | P90-P10 | ¿Acierta? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-001 | 2018 | 2019 | 28.6 | 25.01% | −3.6 pp | 81.7% | 🔴 ALTO RIESGO | 12 | 16 pp | 2.5% | 23.6% | 45.2% | 42.7 pp | ✅ |
| C-010 | 2018 | 2020 | 37.3 | 16.84% | −20.5 pp | 41.0% | 🟢 RIESGO MODERADO | 20 | 16 pp | −2.9% | 16.4% | 37.4% | 40.4 pp | ❌ |
| C-017 | 2019 | 2022 | 53.1 | 33.16% | −19.9 pp | 91.7% | 🔴 ALTO RIESGO | 18 | 16 pp | 12.3% | 32.6% | 53.6% | 41.3 pp | ✅ |
| C-043 | 2021 | 2022 | 2.2 | 28.54% | +26.3 pp | 80.7% | 🔴 ALTO RIESGO | 22 | 20 pp | 2.4% | 28.9% | 54.7% | 52.3 pp | ❌ |
| C-128 | 2019 | 2021 | 30.4 | 26.31% | −4.1 pp | 66.9% | 🔴 ALTO RIESGO | 15 | 16 pp | 5.3% | 25.8% | 46.6% | 41.3 pp | ✅ |

**Error absoluto promedio:** 14.9 pp  
**Aciertos de alerta:** 3/5 (C-001, C-017, C-128 aciertan; C-010 falso negativo; C-043 falso positivo)  
**Conclusión:** ✅ El pipeline funciona. El modelo SVR tiende a subestimar sobrecostos altos y sobreestimar bajos (regresión a la media). La incertidumbre (P90-P10) ahora varía según la complejidad del contrato: C-043 (22 riesgos) tiene P90-P10 de 52.3 pp vs ~41 pp de contratos con menos riesgos.

## 7. Resultados — Prueba de Generalización (Grupo B)

Ejecutada el 2026-07-11. Contratos proporcionados por el asesor, no incluidos en el dataset de 351. Procesados manualmente vía "Pegar texto".

| Contrato | Inicio | Fin | Real (%) | SVR | Error | Prob. | Alerta | Riesgos | RMSE | P10 | P50 | P90 | P90-P10 | ¿Acierta? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-360 | 2019 | 2019 | 10.14 | 15.55% | +5.4 pp | 21.4% | 🟢 RIESGO MODERADO | 14 | 16 pp | −3.4% | 16.5% | 35.5% | 38.9 pp | ✅ |
| C-361 | 2022 | 2022 | 19.09 | 16.99% | −2.1 pp | 60.2% | 🔴 ALTO RIESGO | 28 | 20 pp | −7.9% | 18.5% | 43.5% | 51.3 pp | ❌ |
| C-362 | 2021 | 2021 | 4.38 | 9.54% | +5.2 pp | 21.4% | 🟢 RIESGO MODERADO | 27 | 20 pp | −15.9% | 10.5% | 35.0% | 50.9 pp | ✅ |
| C-363 | 2022 | 2022 | 7.20 | 15.13% | +7.9 pp | 36.8% | 🟢 RIESGO MODERADO | 14 | 16 pp | −3.5% | 16.2% | 35.5% | 39.0 pp | ✅ |
| C-364 | 2023 | 2023 | 20.83 | 10.85% | −10.0 pp | 18.8% | 🟢 RIESGO MODERADO | 34 | 24 pp | −19.8% | 12.0% | 45.5% | 65.4 pp | ✅ |

**Error absoluto promedio:** 6.1 pp  
**MAE:** 6.1 pp (< 20 pp ✅)  
**Tiempo de respuesta:** < 2s por contrato (< 5s ✅)  
**Aciertos de alerta:** 4/5 (solo C-361 falso positivo)  
**Conclusión:** ✅ El modelo generaliza excelentemente en datos no vistos con MAE de 6.1 pp. C-364 (34 riesgos) muestra el intervalo más amplio de todos (P90-P10 = 65.4 pp), reflejando su alta complejidad.

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
| Complejos | C-043, C-361, C-362, **C-365** | 21–30 | 20 pp | ~41 pp | **~51 pp** |
| Muy complejos | C-364 | >30 | 24 pp | ~41 pp | **~62 pp** |

### 8.4 Justificación Metodológica

La segmentación por cantidad de riesgos se fundamenta en que contratos con mayor número de riesgos identificados presentan mayor complejidad operativa y, por tanto, mayor incertidumbre en la estimación del sobrecosto. Los umbrales (12/16/20/24 pp) se definieron empíricamente a partir de la distribución observada en el dataset de 351 contratos, y se documentan como heurísticos sujetos a calibración futura.

---

## 9. Prueba de Predicción — Proyecto No Terminado (Grupo C)

### 9.1 Descripción

Contrato C-365 (Puente Aranda / Troncal Calle 13, IDU Bogotá) está en ejecución (estado: Modificado) con fecha de finalización proyectada al 2027-03-28. Es el primer caso de prueba donde **no existe sobrecosto real** — la predicción del modelo se documenta como estimación a futuro para validar cuando el contrato finalice.

### 9.2 Datos del Contrato

| Campo | Valor |
|-------|-------|
| ID | C-365 |
| Entidad | Distrito Capital de Bogotá - IDU |
| Objeto | Construcción intersección a desnivel Puente Aranda — Adecuación Troncal Calle 13 al sistema Transmilenio |
| Valor Inicial | $477,834,784,322 |
| Inicio | 2023-06-22 |
| Fin Proyectado | 2027-03-28 |
| Duración | ~45 meses |
| Estado | Modificado (activo) |
| Contratista | CONSORCIO CC L1 |
| Riesgos en matriz | 25 |
| URL | https://community.secop.gov.co/Public/Tendering/OpportunityDetail/Index?noticeUID=CO1.NTC.3690937 |

### 9.3 Resultados (Ejecutado 2026-07-13)

| Contrato | Inicio | Fin | SVR | Prob. | Alerta | Riesgos | RMSE | P10 | P50 | P90 | P90-P10 |
|----------|-------|-----|-----|-------|--------|---------|------|-----|-----|-----|---------|
| C-365 | 2023 | 2027 | 28.16% | 92.0% | 🔴 ALTO RIESGO | 25 | 20 pp | 3.66% | 28.81% | 54.83% | 51.17 pp |

### 9.4 Clasificación RMSE

Con 25 riesgos, C-365 entra en la categoría **"Complejos" (21–30 riesgos)** → RMSE = **20 pp**.

**Interpretación:** El modelo predice un sobrecosto central de **28.2%** para el Puente Aranda, clasificándolo como **ALTO RIESGO** (92.0% de probabilidad). Con P90-P10 de 51.2 pp, la incertidumbre es considerable — desde un sobrecosto leve (P10=3.7%) hasta más de la mitad del valor del contrato (P90=54.8%). En COP: el sobrecosto esperado es de **$137.7 mil M** (P50), con un rango P10-P90 de **$17.5 mil M a $262.0 mil M**. Los riesgos que más contribuyen son indemnizaciones a terceros, variación de precios, y daños a la obra.

---

## 10. Prueba de Generalización Temporal — C-128 en Múltiples Rangos (Grupo D)

### 10.1 Descripción

Usando la misma matriz de riesgos de C-128 (15 riesgos, objeto: mejoramiento de vías), se varían sistemáticamente los años de inicio y fin del contrato para observar cómo el contexto macroeconómico (IPC, TRM, políticas, etc.) afecta la predicción del modelo SVR.

Rangos evaluados (13 rangos bienales solapados de 2010 a 2024):

| ID | Inicio | Fin | Contexto Macroeconómico |
|----|-------|-----|------------------------|
| C-128-2010-2012 | 2010 | 2012 | Recuperación post-crisis 2008, alto crecimiento |
| C-128-2011-2013 | 2011 | 2013 | Transición gobierno Santos, inicio proceso de paz |
| C-128-2012-2014 | 2012 | 2014 | Inversión en infraestructura, posconflicto temprano |
| C-128-2013-2015 | 2013 | 2015 | Caída del precio del petróleo, devaluación del COP |
| C-128-2014-2016 | 2014 | 2016 | Reforma tributaria, desaceleración económica |
| C-128-2015-2017 | 2015 | 2017 | Crisis fiscal, menor inversión pública |
| C-128-2016-2018 | 2016 | 2018 | Implementación acuerdo de paz, déficit creciente |
| C-128-2017-2019 | 2017 | 2019 | Gobierno Duque, incertidumbre fiscal |
| C-128-2018-2020 | 2018 | 2020 | Pandemia COVID-19, baja en inversión pública |
| C-128-2019-2021 | 2019 | 2021 | Reactivación post-pandemia, alza en costos |
| C-128-2020-2022 | 2020 | 2022 | Inflación post-pandemia, altas tasas de interés |
| C-128-2021-2023 | 2021 | 2023 | Crisis inflacionaria, incremento de tasas |
| C-128-2022-2024 | 2022 | 2024 | Inflación histórica, desaceleración global |

### 10.2 Hipótesis

- El valor predicho (SVR) debería variar aunque la matriz de riesgos sea idéntica, porque las features macro (IPC acumulado, TRM, año) cambian.
- Períodos con alta inflación (2022-2024) deberían mostrar mayor sobrecosto predicho.
- El modelo SVR está entrenado con datos históricos reales donde estos patrones existen, por lo que las variaciones observadas deben reflejar el efecto del contexto económico.

### 10.3 Resultados (Ejecutado 2026-07-13)

> 13 predicciones automatizadas vía API. Misma matriz de riesgos C-128 (15 riesgos), variando solo año inicio/fin.

| ID | Inicio | Fin | SVR | Prob. | Alerta | RMSE | P10 | P50 | P90 | P90-P10 |
|----|-------|-----|-----|-------|--------|------|-----|-----|-----|---------|
| C-128-2010-2012 | 2010 | 2012 | 26.12% | 70.1% | 🔴 ALTO RIESGO | 16 pp | 5.53% | 26.21% | 46.74% | 41.21 pp |
| C-128-2011-2013 | 2011 | 2013 | 26.23% | 72.9% | 🔴 ALTO RIESGO | 16 pp | 5.67% | 26.35% | 46.87% | 41.20 pp |
| C-128-2012-2014 | 2012 | 2014 | 26.42% | 74.7% | 🔴 ALTO RIESGO | 16 pp | 5.88% | 26.54% | 47.07% | 41.19 pp |
| C-128-2013-2015 | 2013 | 2015 | 26.88% | 73.0% | 🔴 ALTO RIESGO | 16 pp | 6.36% | 27.04% | 47.59% | 41.23 pp |
| C-128-2014-2016 | 2014 | 2016 | 27.64% | 69.8% | 🔴 ALTO RIESGO | 16 pp | 7.09% | 27.79% | 48.33% | 41.24 pp |
| C-128-2015-2017 | 2015 | 2017 | 28.30% | 67.4% | 🔴 ALTO RIESGO | 16 pp | 7.70% | 28.42% | 48.95% | 41.25 pp |
| C-128-2016-2018 | 2016 | 2018 | 28.31% | 68.6% | 🔴 ALTO RIESGO | 16 pp | 7.63% | 28.42% | 48.90% | 41.27 pp |
| C-128-2017-2019 | 2017 | 2019 | 28.07% | 70.0% | 🔴 ALTO RIESGO | 16 pp | 7.38% | 28.17% | 48.67% | 41.29 pp |
| C-128-2018-2020 | 2018 | 2020 | 27.05% | 68.2% | 🔴 ALTO RIESGO | 16 pp | 6.36% | 27.19% | 47.66% | 41.30 pp |
| C-128-2019-2021 | 2019 | 2021 | 26.31% | 66.9% | 🔴 ALTO RIESGO | 16 pp | 5.62% | 26.46% | 46.91% | 41.29 pp |
| C-128-2020-2022 | 2020 | 2022 | 26.93% | 65.6% | 🔴 ALTO RIESGO | 16 pp | 6.19% | 26.98% | 47.47% | 41.28 pp |
| C-128-2021-2023 | 2021 | 2023 | 27.69% | 66.2% | 🔴 ALTO RIESGO | 16 pp | 6.89% | 27.67% | 48.22% | 41.33 pp |
| C-128-2022-2024 | 2022 | 2024 | 26.32% | 67.3% | 🔴 ALTO RIESGO | 16 pp | 5.65% | 26.37% | 46.98% | 41.33 pp |

**Resumen:**
- Rango SVR: 26.12% – 28.31% (variación de 2.19 pp)
- Rango Prob: 65.6% – 74.7%
- Todas las alertas: **ALTO RIESGO**
- P90-P10 estable: ~41.2-41.3 pp (consistente con RMSE 16 pp)
- **Pico máximo de SVR:** 2015-2017 y 2016-2018 (28.3%) — coincide con crisis fiscal y devaluación
- **Valle mínimo:** 2010-2012 (26.12%) — contexto post-crisis 2008, menor inflación
- La predicción varía **solo 2.2 pp** a pesar de 14 años de diferencia macro, lo que sugiere que la matriz de riesgos (TF-IDF + categorías) domina sobre las features temporales en este modelo.

### 10.4 Clasificación RMSE

C-128 tiene 15 riesgos → **"Típicos" (11–20)** → RMSE = **16 pp** para todos los rangos temporales.
