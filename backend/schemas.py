from pydantic import BaseModel, Field


class FactorInfo(BaseModel):
    feature: str = Field(..., description="Nombre interno de la variable")
    label: str = Field(..., description="Nombre legible para el usuario")
    coef: float = Field(..., description="Coeficiente del modelo")


class PrediccionSalida(BaseModel):
    id_contrato: str = Field(..., description="Identificador del contrato")
    sobrecosto_estimado: float = Field(..., description="Sobrecosto estimado en porcentaje")
    intervalo_confianza: str = Field(default="±15.6 pp (RMSE del modelo Ridge)")
    probabilidad_alto_riesgo: float = Field(..., description="Probabilidad de sobrecosto >25% (0-1)")
    alerta: str = Field(..., description="ALTO RIESGO o RIESGO MODERADO")
    riesgos_procesados: int = Field(default=0, description="Cantidad de riesgos procesados")
    factores_aumentan: list[FactorInfo] = Field(..., description="Variables que más aumentan el sobrecosto")
    factores_disminuyen: list[FactorInfo] = Field(..., description="Variables que más disminuyen el sobrecosto")
    modelo: str = Field(default="Ridge + LogisticRegression")
    r2_cv: float = Field(default=0.103)
    auc_cv: float = Field(default=0.639)
    accuracy: float = Field(default=0.706)
    history_id: int | None = Field(default=None, description="ID en el historial local")


class PrediccionHistorial(BaseModel):
    id: int
    created_at: str
    id_contrato: str
    n_riesgos: int
    anio: int | None
    ipc: float | None
    trm: float | None
    prediccion_ridge: float
    probabilidad_alto_riesgo: float
    alerta: str
    sobrecosto_real: float | None
    notas: str | None
    factores_aumentan: list[FactorInfo]
    factores_disminuyen: list[FactorInfo]

