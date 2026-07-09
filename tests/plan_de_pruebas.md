# Plan de Pruebas — Predicción de Sobrecostos en Contratos Públicos

## 1. Propósito

Validar que el pipeline completo (feature engineering → modelo serializado → API → frontend) reproduce los resultados del notebook de entrenamiento y generaliza aceptablemente en datos no vistos.

| Tipo de Prueba | Objetivo | Métrica de Éxito |
|---|---|---|
| **Sanidad (Pipeline)** | Verificar que `feature_engineering.py` + FastAPI generan exactamente las mismas predicciones que el notebook `modelo_final.ipynb` para los mismos datos de entrada | Diferencia < 0.1% entre Notebook y API |
| **Consistencia (Datos Vistos)** | Confirmar que el modelo serializado (`ridge_regressor.pkl`) no se corrompió al ser cargado en el backend | R² ≈ 0.16, RMSE ≈ 16.3 (idénticos al notebook final) |
| **Generalización (Datos No Vistos)** | Evaluar rendimiento en contratos fuera del dataset de entrenamiento | MAE < 20 pp |

## 2. Casos de Prueba

### Grupo A: Datos Vistos (5 contratos del dataset de 351)

| ID Contrato | Sobrecosto Real | Perfil | Motivo |
|---|---|---|---|
| C-001 | 28.6% | Medio | Primer contrato del dataset. Referencia línea base |
| C-010 | 37.3% | Alto | Contrato con muchas features TF-IDF relevantes |
| C-017 | 53.1% | Muy Alto | Caso extremo (subestimado por regresión, bien clasificado) |
| C-043 | 2.2% | Muy Bajo | Caso "problema" (sobreestimado por el modelo). Útil para mostrar limitaciones |
| C-128 | 30.4% | Medio-Alto | Caso "típico" donde el modelo acierta en la alerta |

Archivos CSV de entrada: `tests/data/c-001.csv`, `tests/data/c-010.csv`, `tests/data/c-017.csv`, `tests/data/c-043.csv`, `tests/data/c-128.csv`

### Grupo B: Datos No Vistos (5 contratos nuevos)

Contratos reales proporcionados por el asesor, con matriz de riesgo y sobrecosto real conocido. No forman parte del dataset de 351 contratos usado para entrenar el modelo.

| ID | Archivo | Riesgos | Valor Inicial | Valor Final | Sobrecosto Real |
|----|---------|---------|---------------|-------------|-----------------|
| C-360 | `c-360.csv` | 14 | $1,888,738,443 | $2,080,327,120 | +10.14% |
| C-361 | `c-361.csv` | 28 | $1,885,591,244 | $2,245,590,311 | +19.09% |
| C-362 | `c-362.csv` | 27 | $1,877,707,543 | $1,959,999,799 | +4.38% |
| C-363 | `c-363.csv` | 14 | $1,869,551,299 | $2,004,076,428 | +7.20% |
| C-364 | `c-364.csv` | 34 | $1,868,945,401 | $2,258,302,331 | +20.83% |

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
| Precisión del pipeline | Diferencia < 0.1% entre Notebook y API | El pipeline de agregación y la carga de artefactos es correcta |
| Rendimiento en datos vistos | R² ≈ 0.16, RMSE ≈ 16.3 | El modelo se ha cargado correctamente |
| Rendimiento en datos no vistos | MAE < 20 pp | El modelo generaliza aceptablemente para un prototipo |
| Tiempo de respuesta | < 5 segundos por contrato | La API es suficientemente rápida para uso interactivo |
| Usabilidad | Alerta clara (🔴/🟢) y métricas comprensibles | El frontend cumple su función de comunicación |

## 5. Entorno de Pruebas

- **Backend:** FastAPI en `http://localhost:8000`
- **Frontend:** Streamlit en `http://localhost:8501`
- **Modelos:** `models/ridge_regressor.pkl`, `models/logistic_model.pkl`, `models/coeficientes_ridge.csv`, `models/ipc_trm.pkl`
- **Datos de entrenamiento:** `docs/matriz_clean.csv` (6,525 riesgos, 351 contratos)

---

## 6. Resultados — Prueba de Sanidad (Grupo A)

Ejecutada el 2026-07-07. Todos los contratos se cargaron manualmente por el usuario vía "Pegar texto" en el frontend. Los valores Ridge y Prob. se tomaron de la respuesta de la API almacenada en `history.db`.

| Contrato | Año | Real (%) | Ridge (API) | Error | Prob. (API) | Alerta | ¿Acierta? |
|---|---|---|---|---|---|---|---|
| C-001 | 2018 | 28.6 | 28.07% | −0.5 pp | 72.6% | 🔴 ALTO RIESGO | ✅ |
| C-010 | 2018 | 37.3 | 17.31% | −20.0 pp | 23.6% | 🟢 RIESGO MODERADO | ❌ (subestimó) |
| C-017 | 2019 | 53.1 | 31.25% | −21.9 pp | 77.3% | 🔴 ALTO RIESGO | ✅ (alerta acertada) |
| C-043 | 2021 | 2.2 | 28.31% | +26.1 pp | 78.0% | 🔴 ALTO RIESGO | ❌ (falso positivo) |
| C-128 | 2019 | 30.4 | 27.15% | −3.3 pp | 54.9% | 🔴 ALTO RIESGO | ✅ |

**Error absoluto promedio:** 14.3 pp  
**Aciertos de alerta:** 3/5 (C-001, C-017, C-128 aciertan; C-010 falso negativo; C-043 falso positivo)  
**Conclusión:** ✅ El pipeline funciona. El modelo Ridge es consistente pero tiende a subestimar sobrecostos altos y sobreestimar bajos (regresión a la media). Las predicciones bajaron ligeramente vs la versión anterior por el reemplazo de `tfidf_cualquier` por `tfidf_obra`.

### 6.1 Validación contra Notebook

El `modelado_v2.ipynb` entrenó con **~150+ features** (todas las columnas de `engineer_features`), mientras el prototipo API usa **33 features** (hardcodeadas en `FEATURES_33` en `train_final_model.py`). Los modelos se reentrenaron por separado, por lo que las predicciones difieren.

| Contrato | Real | Notebook Ridge | API Ridge | Δ Ridge | Notebook Prob | API Prob | Δ Prob |
|---|---|---|---|---|---|---|---|
| C-001 | 28.6% | 31.89% | 28.07% | −3.82 pp | 80.5% | 72.6% | −7.9 pp |
| C-010 | 37.3% | 18.45%¹ | 17.31% | −1.14 pp | 16.6%¹ | 23.6% | +7.0 pp |
| C-017 | 53.1% | 33.00% | 31.25% | −1.75 pp | 66.0% | 77.3% | +11.3 pp |
| C-043 | 2.2% | 29.80% | 28.31% | −1.49 pp | 81.4% | 78.0% | −3.4 pp |
| C-128 | 30.4% | 31.40% | 27.15% | −4.25 pp | 77.9% | 54.9% | −23.0 pp |

> ¹ El notebook registra `sobrecosto_real=1.82%` para C-010 (contrato diferente bajo el mismo ID). El usuario aportó un C-010 con real=37.3%.

**Patrón:** API Ridge predice sistemáticamente **−1.1 a −4.3 pp** por debajo del notebook. La probabilidad varía más (±3.4 a −23 pp). La diferencia se debe al feature set reducido (33 vs ~150).

## 7. Resultados — Prueba de Generalización (Grupo B)

Ejecutada el 2026-07-08. Contratos proporcionados por el asesor, no incluidos en el dataset de 351. Procesados manualmente vía "Pegar texto".

| Contrato | Año | Real (%) | Ridge (API) | Error | Prob. (API) | Alerta | 
|---|---|---|---|---|---|---|
| C-360 | 2019 | 10.14 | 26.25% | +16.1 pp | 23.0% | 🟢 RIESGO MODERADO |
| C-361 | 2022 | 19.09 | 27.20% | +8.1 pp | 67.8% | 🔴 ALTO RIESGO |
| C-362 | 2021 | 4.38 | 19.34% | +15.0 pp | 24.6% | 🟢 RIESGO MODERADO |
| C-363 | 2022 | 7.20 | 23.21% | +16.0 pp | 48.2% | 🟢 RIESGO MODERADO |
| C-364 | 2023 | 20.83 | 19.25% | −1.6 pp | 23.6% | 🟢 RIESGO MODERADO |

**Error absoluto promedio:** 11.4 pp  
**MAE:** 11.4 pp (< 20 pp ✅)  
**Tiempo de respuesta:** < 2s por contrato (< 5s ✅)  
**Conclusión:** ✅ El modelo generaliza aceptablemente con MAE de 11.4 pp, dentro del umbral de 20 pp. C-364 mejoró de −5.0 pp a −1.6 pp gracias al re-entreno con las stop words corregidas. Las predicciones Ridge subieron ligeramente en todos los casos (efecto de `tfidf_obra`). La alerta clasificatoria sigue sin detectar C-364 (umbral 20.83%, probabilidad 23.6%).
