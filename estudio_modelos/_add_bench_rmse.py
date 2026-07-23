import json

NOTEB_PATH = 'estudio_modelos/modelo_final.ipynb'

with open(NOTEB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of sec12_train_rmse to insert benchmark before it
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'sec12_train_rmse':
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: sec12_train_rmse not found")
else:
    benchmark_cell = {
        "cell_type": "code",
        "id": "sec12_benchmark_rmse",
        "metadata": {},
        "source": [
            "# Benchmark de modelos para RMSE Predictor (meta-modelo de error)\n",
            "# Objetivo: predecir |real - Ridge_pred| a partir de las 35 features\n",
            "from sklearn.linear_model import Ridge, Lasso\n",
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
            "from sklearn.svm import SVR\n",
            "from sklearn.model_selection import KFold, cross_val_score\n",
            "from sklearn.metrics import mean_absolute_error, r2_score, make_scorer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "RMSE_MODELS = {\n",
            "    'Ridge': Ridge(alpha=1.0, random_state=42),\n",
            "    'Lasso': Lasso(alpha=0.01, random_state=42),\n",
            "    'SVR Linear': SVR(kernel='linear', C=1.0),\n",
            "    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),\n",
            "    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),\n",
            "}\n",
            "\n",
            "# Target: error absoluto de Ridge en train\n",
            "y_err = np.abs(y_train - ridge_pipe.predict(X_train))\n",
            "print(f\"Error absoluto medio de Ridge en train: {y_err.mean():.2f} pp\")\n",
            "print(f\"  Std: {y_err.std():.2f} pp | Min: {y_err.min():.2f} | Max: {y_err.max():.2f}\")\n",
            "\n",
            "# Nested CV para cada modelo\n",
            "print(f\"\\n{'Modelo':20s} {'MAE':>8s} {'R2':>8s}\")\n",
            "print('-' * 38)\n",
            "resultados_rmse = {}\n",
            "for nombre, md in RMSE_MODELS.items():\n",
            "    pipe = Pipeline([('scaler', StandardScaler()), ('m', md)])\n",
            "    cv_mae = cross_val_score(pipe, X_train, y_err, cv=KFold(5, shuffle=True, random_state=42),\n",
            "                            scoring='neg_mean_absolute_error')\n",
            "    cv_r2 = cross_val_score(pipe, X_train, y_err, cv=KFold(5, shuffle=True, random_state=42),\n",
            "                           scoring='r2')\n",
            "    resultados_rmse[nombre] = {'MAE': -cv_mae, 'R2': cv_r2}\n",
            "    print(f'{nombre:20s} {-cv_mae.mean():7.3f}~{cv_mae.std():.3f} {cv_r2.mean():7.3f}~{cv_r2.std():.3f}')\n",
            "\n",
            "# Mejor modelo\n",
            "mejor_rmse = max(resultados_rmse, key=lambda k: resultados_rmse[k]['R2'].mean())\n",
            "print(f\"\\nMejor modelo RMSE: {mejor_rmse} (R2={resultados_rmse[mejor_rmse]['R2'].mean():.3f})\")\n",
            "\n",
            "# Comparacion con heuristic\n",
            "from backend.quantitative_analysis import _rmse_por_contrato\n",
            "rmse_heur = np.array([_rmse_por_contrato(n) for n in X_train['n_riesgos']])\n",
            "print(f\"\\nComparacion con heuristica:\")\n",
            "print(f\"  Heuristica MAE: {mean_absolute_error(y_err, rmse_heur):.2f} pp\")\n",
            "print(f\"  Heuristica R2:  {r2_score(y_err, rmse_heur):.4f}\")\n",
            "for nombre in resultados_rmse:\n",
            "    mejora = (mean_absolute_error(y_err, rmse_heur) - resultados_rmse[nombre]['MAE'].mean())\n",
            "    print(f'  {nombre:20s} MAE={resultados_rmse[nombre][\"MAE\"].mean():.2f} (mejora vs heur: {mejora:+.2f} pp)')\n"
        ]
    }

    nb['cells'].insert(insert_idx, benchmark_cell)
    print(f"[OK] Inserted benchmark cell at index {insert_idx}")

with open(NOTEB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("[OK] Done")
