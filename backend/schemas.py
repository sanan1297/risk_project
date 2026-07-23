from pydantic import BaseModel, Field


class FactorInfo(BaseModel):
    feature: str = Field(..., description="Nombre interno de la variable")
    label: str = Field(..., description="Nombre legible para el usuario")
    coef: float = Field(..., description="Importancia de la variable (feature importance)")


class PrediccionSalida(BaseModel):
    id_contrato: str = Field(..., description="Identificador del contrato")
    sobrecosto_estimado: float = Field(..., description="Sobrecosto estimado en porcentaje")
    intervalo_confianza: str = Field(default="±11.4 pp (RMSE del modelo RandomForest)")
    probabilidad_alto_riesgo: float = Field(..., description="Probabilidad de sobrecosto >25% (0-1)")
    alerta: str = Field(..., description="ALTO RIESGO o RIESGO MODERADO")
    riesgos_procesados: int = Field(default=0, description="Cantidad de riesgos procesados")
    factores_aumentan: list[FactorInfo] = Field(..., description="Variables que más aumentan el sobrecosto")
    factores_disminuyen: list[FactorInfo] = Field(..., description="Variables que más disminuyen el sobrecosto")
    modelo: str = Field(default="RandomForest + RandomForestClassifier")
    r2_cv: float = Field(default=0.235)
    auc_cv: float = Field(default=0.591)
    accuracy: float = Field(default=0.706)
    history_id: int | None = Field(default=None, description="ID en el historial local")


class RiesgoContribucion(BaseModel):
    riesgo: str = Field(..., description="Descripción del riesgo")
    tipo: str = Field(default="")
    categoria: str = Field(default="")
    probabilidad: int = Field(..., description="Valor de probabilidad (1-5)")
    impacto: int = Field(..., description="Valor de impacto (1-5)")
    peso_contribucion: float = Field(..., description="Peso proporcional del riesgo (0-1)")
    contribucion_porcentaje: float = Field(..., description="Contribución al sobrecosto en pp")


class ItemTornado(BaseModel):
    riesgo: str = Field(..., description="Descripción del riesgo")
    tipo: str = Field(default="")
    categoria: str = Field(default="")
    probabilidad_original: int = Field(..., description="Valor de probabilidad original (1-5)")
    impacto_original: int = Field(..., description="Valor de impacto original (1-5)")
    prediccion_alta: float = Field(..., description="Predicción si el riesgo sube 1")
    prediccion_baja: float = Field(..., description="Predicción si el riesgo baja 1")
    swing: float = Field(..., description="Diferencia absoluta entre alta y baja")
    direccion: str = Field(..., description="aumenta o disminuye")


class BinHistograma(BaseModel):
    bin_inicio: float = Field(..., description="Inicio del bin")
    bin_fin: float = Field(..., description="Fin del bin")
    frecuencia: int = Field(..., description="Número de simulaciones en este bin")


class MonteCarloSalida(BaseModel):
    prediccion_central: float = Field(..., description="Predicción base del modelo RandomForest (%)")
    percentiles: dict[str, float] = Field(..., description="Percentiles P5-P95 (%)")
    stats: dict[str, float] = Field(..., description="Estadísticas de la simulación (%)")
    histograma: list[BinHistograma] = Field(..., description="Histograma de 20 bins (%)")
    tornado: list[ItemTornado] = Field(..., description="Análisis de tornado por riesgo (%)")
    riesgos: list[RiesgoContribucion] = Field(..., description="Desglose de contribución por riesgo (%)")
    n_simulaciones: int = Field(default=1000, description="Número de iteraciones MC")
    rmse: float = Field(default=16.0, description="RMSE del modelo usado como ruido")
    ruido_incluido: bool = Field(default=True, description="Si se incluyó ruido gaussiano RMSE")
    valor_inicial: float | None = Field(default=None, description="Valor inicial del contrato (COP)")
    percentiles_cop: dict[str, float] | None = Field(default=None, description="Percentiles P5-P95 en COP")
    stats_cop: dict[str, float] | None = Field(default=None, description="Estadísticas en COP")
    histograma_cop: list[dict] | None = Field(default=None, description="Histograma en COP")
    tornado_cop: list[dict] | None = Field(default=None, description="Tornado en COP")
    riesgos_cop: list[dict] | None = Field(default=None, description="Desglose en COP")


class PrediccionHistorial(BaseModel):
    id: int
    created_at: str
    id_contrato: str
    n_riesgos: int
    anio: int | None = None
    ipc: float | None = None
    trm: float | None = None
    anio_inicio: int | None = None
    anio_fin: int | None = None
    duracion: int | None = None
    ipc_acumulado: float | None = None
    trm_promedio: float | None = None
    prediccion_ridge: float
    probabilidad_alto_riesgo: float
    alerta: str
    sobrecosto_real: float | None = None
    notas: str | None = None
    factores_aumentan: list[FactorInfo]
    factores_disminuyen: list[FactorInfo]

