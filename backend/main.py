from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json

from .feature_engineering import aggregate_risks, validate_input
from .predictor import predict as run_prediction, MODEL_META
from .schemas import PrediccionSalida, PrediccionHistorial, FactorInfo, MonteCarloSalida
from .feature_labels import label_feature
from . import history
from . import training_stats
from . import quantitative_analysis
from . import mlflow_tracker

app = FastAPI(
    title="Risk Predictor API",
    description="Predice el sobrecosto de contratos publicos a partir de matrices de riesgo desagregadas.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RANGO_ANOS = list(range(2000, 2028))
REQUIRED_COLS = ["id_contrato", "descripcion_riesgo", "probabilidad", "impacto", "tipo", "categoria"]


@app.on_event("startup")
def startup():
    history.init()
    mlflow_tracker.init_from_mlflow()


@app.get("/health")
def health():
    registry = mlflow_tracker.get_model_registry()
    return {
        "status": "ok",
        "rango_anos": RANGO_ANOS,
        "model_version": registry.get("model_version"),
        "mlflow_available": registry.get("mlflow_available"),
        **MODEL_META,
    }


@app.get("/model/info")
def model_info():
    registry = mlflow_tracker.get_model_registry()
    return {
        "modelo": MODEL_META["modelo"],
        "features": MODEL_META["features"],
        "r2_cv": MODEL_META["r2_cv"],
        "auc_cv": MODEL_META["auc_cv"],
        "rmse": MODEL_META["rmse"],
        "mlflow_tracking_uri": mlflow_tracker.MLFLOW_TRACKING_URI,
        "mlflow_experiment": mlflow_tracker.MLFLOW_EXPERIMENT_NAME,
        "mlflow_model_name": mlflow_tracker.MLFLOW_MODEL_NAME,
        "model_version": registry.get("model_version"),
        "run_id": registry.get("run_id"),
        "experiment_id": registry.get("experiment_id"),
        "mlflow_available": registry.get("mlflow_available"),
    }


def _parse_input(file: UploadFile | None, riesgos: str | None) -> pd.DataFrame:
    if file is not None:
        if not file.filename.endswith(".csv"):
            raise HTTPException(400, "Solo se aceptan archivos CSV")
        content = file.file.read()
        try:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except Exception as e:
            raise HTTPException(400, f"Error al leer CSV: {e}")
    if riesgos is not None:
        try:
            return pd.read_csv(io.StringIO(riesgos))
        except Exception as e:
            raise HTTPException(400, f"Error al interpretar texto como CSV: {e}")
    raise HTTPException(400, "Debes enviar un archivo CSV o texto CSV en el campo 'riesgos'")


def _build_factor_info(coef_list: list[dict]) -> list[FactorInfo]:
    return [
        FactorInfo(feature=item["feature"], label=label_feature(item["feature"]), coef=item["coef"])
        for item in coef_list
    ]


@app.post("/predict", response_model=list[PrediccionSalida])
def predict(
    file: UploadFile | None = File(None),
    anio_inicio: int | None = Form(None),
    anio_fin: int | None = Form(None),
    ipc_override: float | None = Form(None),
    trm_override: float | None = Form(None),
    riesgos: str | None = Form(None),
):
    df = _parse_input(file, riesgos)

    errors = validate_input(df)
    if errors:
        raise HTTPException(422, {"errores": errors, "columnas_requeridas": REQUIRED_COLS})

    if anio_inicio is not None and anio_inicio not in RANGO_ANOS:
        raise HTTPException(400, f"Año inicio {anio_inicio} fuera de rango ({RANGO_ANOS[0]}-{RANGO_ANOS[-1]}).")

    if "id_contrato" not in df.columns:
        df["id_contrato"] = "CONTRATO_01"

    try:
        df_feat = aggregate_risks(df, anio_inicio=anio_inicio, anio_fin=anio_fin, ipc_override=ipc_override, trm_override=trm_override)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        result = run_prediction(df_feat)
    except Exception as e:
        raise HTTPException(500, f"Error en prediccion: {e}")

    output = []
    for i, pred in enumerate(result["predicciones"]):
        factores_aumentan = _build_factor_info(result["explicacion"]["aumentan"])
        factores_disminuyen = _build_factor_info(result["explicacion"]["disminuyen"])

        cid = result["contratos"][i]
        n_riesgos = result["n_riesgos"][i]

        i_macro = df_feat.iloc[i]
        hid = history.guardar(
            id_contrato=cid,
            n_riesgos=n_riesgos,
            anio_inicio=anio_inicio,
            anio_fin=anio_fin,
            ipc_override=ipc_override,
            trm_override=trm_override,
            ipc_acumulado=i_macro.get("ipc_acumulado"),
            trm_promedio=i_macro.get("trm_promedio"),
            prediccion_ridge=pred,
            probabilidad_alto_riesgo=result["probabilidades"][i],
            alerta=result["alertas"][i],
            factores_aumentan=[f.model_dump() for f in factores_aumentan],
            factores_disminuyen=[f.model_dump() for f in factores_disminuyen],
        )

        output.append(PrediccionSalida(
            id_contrato=cid,
            sobrecosto_estimado=pred,
            probabilidad_alto_riesgo=result["probabilidades"][i],
            alerta=result["alertas"][i],
            riesgos_procesados=n_riesgos,
            factores_aumentan=factores_aumentan,
            factores_disminuyen=factores_disminuyen,
            history_id=hid,
        ))

    return output


@app.post("/predict/montecarlo", response_model=MonteCarloSalida)
def predict_montecarlo(
    file: UploadFile | None = File(None),
    anio_inicio: int | None = Form(None),
    anio_fin: int | None = Form(None),
    ipc_override: float | None = Form(None),
    trm_override: float | None = Form(None),
    riesgos: str | None = Form(None),
    n_iteraciones: int = Form(1000),
    incluir_ruido: bool = Form(True),
    valor_inicial: float | None = Form(None),
    history_id: int | None = Form(None),
):
    df = _parse_input(file, riesgos)

    errors = validate_input(df)
    if errors:
        raise HTTPException(422, {"errores": errors, "columnas_requeridas": REQUIRED_COLS})

    try:
        result = quantitative_analysis.compute(
            df, anio_inicio=anio_inicio, anio_fin=anio_fin, ipc_override=ipc_override, trm_override=trm_override,
            n_iteraciones=n_iteraciones, incluir_ruido=incluir_ruido,
            valor_inicial=valor_inicial,
        )
    except Exception as e:
        raise HTTPException(500, f"Error en simulación Monte Carlo: {e}")

    if history_id is not None:
        try:
            history.actualizar_mc(history_id, n_iteraciones)
            history.guardar_resultado_completo(history_id, result)
        except Exception:
            pass

    return result


@app.get("/stats/usage")
def stats_usage():
    return history.stats()


@app.get("/stats/training")
def stats_training():
    return training_stats.compute()


@app.get("/history")
def get_history(page: int = 1, page_size: int = 20):
    result = history.listar_paginado(page=page, page_size=page_size)
    out = []
    for r in result["data"]:
        fa = json.loads(r["factores_aumentan"]) if r["factores_aumentan"] else []
        fd = json.loads(r["factores_disminuyen"]) if r["factores_disminuyen"] else []
        out.append(PrediccionHistorial(
            id=r["id"],
            created_at=r["created_at"],
            id_contrato=r["id_contrato"],
            n_riesgos=r["n_riesgos"],
            anio=r.get("anio"),
            ipc=r.get("ipc"),
            trm=r.get("trm"),
            anio_inicio=r.get("anio_inicio"),
            anio_fin=r.get("anio_fin"),
            duracion=r.get("duracion"),
            ipc_acumulado=r.get("ipc_acumulado"),
            trm_promedio=r.get("trm_promedio"),
            prediccion_ridge=r["prediccion_ridge"],
            probabilidad_alto_riesgo=r["probabilidad_alto_riesgo"],
            alerta=r["alerta"],
            sobrecosto_real=r["sobrecosto_real"],
            notas=r["notas"],
            factores_aumentan=_build_factor_info(fa),
            factores_disminuyen=_build_factor_info(fd),
        ))
    return {"data": out, "total": result["total"], "page": result["page"], "page_size": result["page_size"], "paginas": result["paginas"]}


@app.get("/history/{pred_id}")
def get_history_item(pred_id: int):
    data = history.obtener_por_id(pred_id)
    if data is None:
        raise HTTPException(404, "Predicción no encontrada")
    return data


@app.get("/history/{pred_id}/resultados")
def get_resultados_completos(pred_id: int):
    data = history.obtener_resultado_completo(pred_id)
    if data is None:
        raise HTTPException(404, "No hay resultados cuantitativos para esta predicción")
    return data


@app.put("/history/{pred_id}")
def update_history(
    pred_id: int,
    sobrecosto_real: float = Form(...),
    notas: str | None = Form(None),
):
    history.actualizar_sobrecosto_real(pred_id, sobrecosto_real, notas)
    return {"status": "ok", "id": pred_id}


@app.delete("/history/{pred_id}")
def delete_history(pred_id: int):
    history.eliminar(pred_id)
    return {"status": "ok", "id": pred_id}


@app.delete("/history")
def delete_all_history():
    history.eliminar_todos()
    return {"status": "ok"}
