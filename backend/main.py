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

RANGO_ANOS = list(range(2000, 2026))
REQUIRED_COLS = ["id_contrato", "descripcion_riesgo", "probabilidad", "impacto", "tipo", "categoria"]


@app.on_event("startup")
def startup():
    history.init()


@app.get("/health")
def health():
    return {"status": "ok", "rango_anos": RANGO_ANOS, **MODEL_META}


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
    anio: int | None = Form(None),
    ipc: float | None = Form(None),
    trm: float | None = Form(None),
    riesgos: str | None = Form(None),
):
    df = _parse_input(file, riesgos)

    errors = validate_input(df)
    if errors:
        raise HTTPException(422, {"errores": errors, "columnas_requeridas": REQUIRED_COLS})

    if anio is not None and (anio < 2000 or anio > 2025) and (ipc is None or trm is None):
        raise HTTPException(400, f"Año {anio} fuera de rango (2000-2025). Debes proporcionar ipc y trm manualmente.")

    if "id_contrato" not in df.columns:
        df["id_contrato"] = "CONTRATO_01"

    try:
        df_feat = aggregate_risks(df, anio=anio, ipc=ipc, trm=trm)
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

        hid = history.guardar(
            id_contrato=cid,
            n_riesgos=n_riesgos,
            anio=anio,
            ipc=ipc,
            trm=trm,
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
    anio: int | None = Form(None),
    ipc: float | None = Form(None),
    trm: float | None = Form(None),
    riesgos: str | None = Form(None),
    n_iteraciones: int = Form(1000),
    incluir_ruido: bool = Form(True),
    valor_inicial: float | None = Form(None),
):
    df = _parse_input(file, riesgos)

    errors = validate_input(df)
    if errors:
        raise HTTPException(422, {"errores": errors, "columnas_requeridas": REQUIRED_COLS})

    if anio is not None and (anio < 2000 or anio > 2025) and (ipc is None or trm is None):
        raise HTTPException(400, f"Año {anio} fuera de rango (2000-2025). Debes proporcionar ipc y trm manualmente.")

    try:
        result = quantitative_analysis.compute(
            df, anio=anio, ipc=ipc, trm=trm,
            n_iteraciones=n_iteraciones, incluir_ruido=incluir_ruido,
            valor_inicial=valor_inicial,
        )
    except Exception as e:
        raise HTTPException(500, f"Error en simulación Monte Carlo: {e}")

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
            anio=r["anio"],
            ipc=r["ipc"],
            trm=r["trm"],
            prediccion_ridge=r["prediccion_ridge"],
            probabilidad_alto_riesgo=r["probabilidad_alto_riesgo"],
            alerta=r["alerta"],
            sobrecosto_real=r["sobrecosto_real"],
            notas=r["notas"],
            factores_aumentan=_build_factor_info(fa),
            factores_disminuyen=_build_factor_info(fd),
        ))
    return {"data": out, "total": result["total"], "page": result["page"], "page_size": result["page_size"], "paginas": result["paginas"]}


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
