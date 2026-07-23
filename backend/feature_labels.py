FEATURE_LABELS = {
    "tfidf_desarrollo": "Desarrollo",
    "interaccion_prob_x_impacto": "Prob. × Impacto",
    "tfidf_insumos": "Insumos",
    "prob_std": "Variación prob.",
    "tfidf_expedicion": "Expedición",
    "tfidf_materiales": "Materiales",
    "imp_promedio": "Impacto promedio",
    "tfidf_cualquier": "Cualquier",
    "tfidf_ejecucion": "Ejecución",
    "tfidf_ejecución": "Ejecución",
    "tfidf_contrato": "Contrato",
    "prop_tipo_operacional": "Riesgos operacionales",
    "prob_promedio": "Probabilidad promedio",
    "prop_cate_bajo": "Categoría bajo",
    "valor_inicial": "Valor del contrato",
    "tfidf_riesgo": "Riesgo",
    "tfidf_tecnicas": "Técnicas",
    "tfidf_municipio": "Municipio",
    "tfidf_obras": "Obras",
    "tfidf_informacion": "Información",
    "prop_fuen_externo": "Fuente externa",
    "tfidf_cuando": "Cuando",
    "tfidf_disenos": "Diseños",
    "tfidf_ejecucion contrato": "Ejec. contrato",
    "tfidf_calidad": "Calidad",
    "tfidf_manejo": "Manejo",
    "prop_cate_alto": "Categoría alto",
    "tfidf_pago": "Pago",
    "prop_tipo_economico": "Riesgos económicos",
    "prop_asig_entidad": "Asignado a entidad",
    "tfidf_falta": "Falta",
    "anio_inicio": "Año inicio",
    "anio_fin": "Año fin",
    "duracion": "Duración (años)",
    "ipc_acumulado": "IPC acumulado",
    "trm_promedio": "TRM promedio",
    "anio": "Año del contrato",
    "ipc": "Inflación (IPC)",
    "trm": "Tasa de cambio (TRM)",
    "pct_riesgos_con_mitigacion": "% Riesgos con mitigación",
    "avg_longitud_mitigacion": "Longitud mitigación prom.",
    "n_distinct_codes_mitigacion": "Tipos de mitigación",
}

_PROP_PREFIX_LABEL = {
    "tipo": "Tipo",
    "clas": "Clase",
    "asig": "Asignación",
    "fuen": "Fuente",
    "etap": "Etapa",
    "cate": "Categoría",
}


def label_feature(name: str) -> str:
    cached = FEATURE_LABELS.get(name)
    if cached is not None:
        return cached

    if name.startswith("tfidf_"):
        word = name[6:].replace("_", " ")
        return word.title()

    if name.startswith("prop_"):
        rest = name[5:]
        parts = rest.split("_", 1)
        if len(parts) == 2:
            prefix, value = parts
            label = _PROP_PREFIX_LABEL.get(prefix, prefix.capitalize())
            return f"{label}: {value.replace('_', ' ')}"
        return rest.replace("_", " ")

    return name
