"""
Pipeline de normalizacion de matriz de riesgos SECOP.
Uso: python estudio_data/normalizar.py

Transformaciones:
  1. Lowercase + quitar tildes + espacios multiples
  2. Estandarizar campos categoricos a taxonomia SECOP
  3. Uniformizar filas malformadas (padding/truncado)
  4. Preservar matriz.csv intacto como fuente original
"""

import csv
import re
import unicodedata
from pathlib import Path
from collections import Counter

ORIGEN = Path('docs/matriz.csv')
DESTINO = Path('docs/matriz_clean.csv')
NCOLS = 20


# ---------------------------------------------------------------------------
#  Utilidades de texto
# ---------------------------------------------------------------------------

def quitar_tildes(s: str) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def limpiar(s: str) -> str:
    s = s.strip().lower()
    s = quitar_tildes(s)
    s = re.sub(r'\s+', ' ', s)
    return s


# ---------------------------------------------------------------------------
#  TIPO — agrupar 244 variantes en ~16 canonicas
# ---------------------------------------------------------------------------

# Orden de prioridad: primero se evaluan multi-riesgo, despues simples
TIPO_PATRONES = [
    # Multi-riesgo explicito (contiene varios tipos distintos)
    (re.compile(r'(economico|financiero).*(operacional|tecnico|regulatorio|social|ambiental)'), 'economico'),
    (re.compile(r'(operacional|tecnico).*(economico|financiero)'), 'operacional'),
    (re.compile(r'(social|politico).*(operacional|naturaleza|ambiental|regulatorio)'), 'social'),
    (re.compile(r'(regulatorio).*(operacional|social)'), 'regulatorio'),
    (re.compile(r'(naturaleza|ambiental).*(social|regulatorio)'), 'naturaleza'),
    # Simples
    (re.compile(r'^operacional(?:es)?$|^operacion$|^operativ[oa]$|^operaciona$|^operacionalis$|^operative$|^riesgo\s*operacional'), 'operacional'),
    (re.compile(r'^economi[coas]+$|^economic$|^riesgo\s*economico'), 'economico'),
    (re.compile(r'^financier[oa]$|^financieras$|^riesgo\s*financiero'), 'financiero'),
    (re.compile(r'^regulatori[oa]$|^regulatorios$|^legales?$|^juridic[oa]s?$|^normativos?$|^documentales?$|^riesgo\s*regulatorio'), 'regulatorio'),
    (re.compile(r'^social(?:es)?$|^politic[oa]s?$|^corrupcion$|^riesgo\s*social'), 'social'),
    (re.compile(r'^naturale(?:za|s)$|^de la naturaleza$|^fuerza mayor$|^climatico$|^riesgo\s*de\s*naturaleza'), 'naturaleza'),
    (re.compile(r'^ambiental(?:es)?$|^riesgo\s*ambiental'), 'ambiental'),
    (re.compile(r'^tecnico$|^tecnologic[oa]s?$|^diseno$|^riesgo\s*tecnologic'), 'tecnico'),
    (re.compile(r'^contratacion$|^contractual$|^precontractual$|^seleccion$|^perfeccionamiento$|^riesgo\s*de\s*contratacion'), 'contratacion'),
    (re.compile(r'^ejecucion$|^cumplimiento$'), 'ejecucion'),
    (re.compile(r'^seguridad$|^salud$'), 'seguridad'),
    (re.compile(r'^administrati(?:vo|vos)'), 'administrativo'),
    (re.compile(r'^imagen$'), 'imagen'),
    (re.compile(r'^predial$|^adquisicion'), 'predial'),
    (re.compile(r'^tributari[oa]$'), 'tributario'),
    (re.compile(r'^cambiari[oa]$'), 'cambiario'),
    (re.compile(r'^constru'), 'construccion'),
    (re.compile(r'^laboral$'), 'laboral'),
    (re.compile(r'^no especificado$|^general$|^o$|^e$|^todas'), 'no especificado'),
]

# Palabras que identifican cada tipo dentro de frases compuestas
TIPO_KEYWORDS = {
    'operacional': re.compile(r'\boperacion(?:al|ales|a|)?\b|\boperativ[oa]\b'),
    'economico': re.compile(r'\beconomi[coas]\b'),
    'financiero': re.compile(r'\bfinancier[oa]\b'),
    'regulatorio': re.compile(r'\bregulatori[oa]\b|\blegal(?:es)?\b|\bjuri[d]ic[oa]s?\b|\bnormativos?\b|\bdocumentales?\b'),
    'social': re.compile(r'\bsocial(?:es)?\b|\bpolitic[oa]s?\b|\bcorrupcion\b'),
    'naturaleza': re.compile(r'\bnaturale(?:za|s)\b|\blluvia\b|\binvernal\b|\bclimatico\b|\bfuerza mayor\b|\bfortuito\b'),
    'ambiental': re.compile(r'\bambiental(?:es)?\b'),
    'tecnico': re.compile(r'\btecnic[oa]?\b|\btecnologic[oa]s?\b|\bdiseno\b'),
    'contratacion': re.compile(r'\bcontratacion\b|\bcontractual\b|\bseleccion\b|\bprecontractual\b|\bperfeccionamiento\b'),
    'seguridad': re.compile(r'\bseguridad\b|\bsalud\b'),
    'administrativo': re.compile(r'\badminis\b'),
    'imagen': re.compile(r'\bimagen\b'),
    'predial': re.compile(r'\bpredial\b|\badquisi\b'),
    'tributario': re.compile(r'\btributari[oa]\b'),
    'cambiario': re.compile(r'\bcambiari[oa]\b'),
    'construccion': re.compile(r'\bconstru\w*\b'),
}


TIPO_MAP_EXACTO = {
    'financieros': 'financiero',
    'economicos': 'economico', 'economicas': 'economico', 'economica': 'economico',
    'riesgos economicos': 'economico', 'riesgos financieros': 'financiero',
    'riesgos regulatorios': 'regulatorio', 'riesgo': 'no especificado',
    'riesgos': 'no especificado', 'del contratista': 'no especificado',
    'natural': 'naturaleza', 'recurso': 'no especificado',
    'tramite y gestion de otras entidades': 'administrativo',
    'que no se firme el contrato': 'contratacion',
    'planeacion': 'contratacion', 'cambiario': 'financiero',
    'redes': 'tecnico', 'riesgo administrativo': 'administrativo',
    'socialopolitico': 'social',
    'judiciales': 'regulatorio', 'judicial1': 'regulatorio',
    'economi': 'economico', 'economic': 'economico',
    'riesgo baio': 'no especificado',
    'operativos': 'operacional', 'operacionalis': 'operacional',
    'operaciona': 'operacional', 'operative': 'operacional',
    'riesgo de naturaleza': 'naturaleza', 'riesgos naturaleza': 'naturaleza',
    'riesgo tecnologico': 'tecnico',
    'riesgo de construccion': 'construccion',
    'declaratoria de desierta del proceso por ausencia de oferentes o incumplimiento de requisitos': 'contratacion',
    'colusion dentro del proceso': 'social',
    'que no se presenten las garantias requeridas en los documentos del proceso o que su presentacion sea tardia': 'contratacion',
}


def estandarizar_tipo(s: str) -> str:
    if not s:
        return 'no especificado'
    if s in TIPO_MAP_EXACTO:
        return TIPO_MAP_EXACTO[s]
    # Primero probar patrones exactos
    for pat, estandar in TIPO_PATRONES:
        if pat.match(s):
            return estandar
    # Si no matcheo, detectar palabras clave
    tipos_encontrados = []
    for tipo, pat in TIPO_KEYWORDS.items():
        if pat.search(s):
            tipos_encontrados.append(tipo)
    if len(tipos_encontrados) == 1:
        return tipos_encontrados[0]
    elif len(tipos_encontrados) > 1:
        return tipos_encontrados[0]
    return 'no especificado'


# ---------------------------------------------------------------------------
#  ETAPA — agrupar 86 variantes en 7 canonicas
# ---------------------------------------------------------------------------

ETAPA_PATRONES = [
    (re.compile(r'^planeacion$|^precontractual$|^pre-contractual$|^etapa\s*precontractual$|^precontratacion$|^precont\.'), 'planeacion'),
    (re.compile(r'^seleccion$|^adjudicacion$'), 'seleccion'),
    (re.compile(r'^contratacion$|^contractual$|^contratual$'), 'contratacion'),
    (re.compile(r'^ejecucion$|^ejecucuion$|^ejecucon$|^ejecucion\s*contractual$'), 'ejecucion'),
    (re.compile(r'^liquidacion$|^poscontractual$|^post$|^post\s*/?\s*contratacion$|^post\s*/?\s*contractural$|^cierre$'), 'liquidacion'),
    (re.compile(r'^(?:no\s+)?especificad[oa]$|^todas\s*las\s*etapas$|^[cepso]$|^general$|^transversal$|^intern[oa]$|^extern[oa]$'), 'no especificado'),
    (re.compile(r'^financier[oa]s?$|^fisico$'), 'no especificado'),
    (re.compile(r'^contratista$'), 'no especificado'),
    (re.compile(r'^(adquisicion|corrupcion|operativo|naturaleza|ambientales|economico|tecnico|legal)'), 'no especificado'),
]

# Palabras de etapa para dividir compuestos
ETAPA_KEYWORDS = {
    'planeacion': re.compile(r'\bplaneacion\b|\bprecontractual\b'),
    'seleccion': re.compile(r'\bseleccion\b|\badjudicacion\b'),
    'contratacion': re.compile(r'\bcontratacion\b|\bcontractual\b|\bcontratual\b|\bperfeccionamiento\b'),
    'ejecucion': re.compile(r'\bejecucion\b|\bejecucuion\b|\bejecucon\b|\boperacion\b'),
    'liquidacion': re.compile(r'\bliquidacion\b|\bposcontractual\b|\bpost\b|\bcierre\b'),
}

DIVISORES = re.compile(r'\s*[-/]\s*|\s+y\s+|,')


def estandarizar_etapa(s: str) -> str:
    if not s:
        return 'no especificado'
    # Texto demasiado largo = description erronea en columna etapa
    if len(s) > 60:
        return 'no especificado'
    # Verificar patron exacto primero
    for pat, estandar in ETAPA_PATRONES:
        if pat.match(s):
            return estandar
    # Si tiene divisores, separar y estandarizar cada parte
    if '/' in s or '-' in s or ' y ' in s or ',' in s:
        partes = [p.strip() for p in DIVISORES.split(s) if p.strip()]
        estandarizadas = []
        for p in partes:
            encontro = False
            for pat, estandar in ETAPA_PATRONES:
                if pat.match(p):
                    estandarizadas.append(estandar)
                    encontro = True
                    break
            if not encontro:
                estandarizadas.append(p)
        if estandarizadas:
            return ' / '.join(dict.fromkeys(estandarizadas))
    # Deteccion por palabra clave
    etapas = []
    for etapa, pat in ETAPA_KEYWORDS.items():
        if pat.search(s):
            etapas.append(etapa)
    if etapas:
        return ' / '.join(dict.fromkeys(etapas))
    return 'no especificado'


# ---------------------------------------------------------------------------
#  CLASE — 70 variantes -> 10 canonicas SECOP
# ---------------------------------------------------------------------------

CLASE_PATRONES = [
    (re.compile(r'^general$|^g$|^genera$'), 'general'),
    (re.compile(r'^especific[oa]$|^espec[ii]fica$|^expecifico$|^espeofico$|^e$'), 'especifico'),
    (re.compile(r'^a\.?\s*(juridico)\s*-?\s*(documental)\s*-?\s*(regulatorio)$'), 'a. juridico - documental - regulatorio'),
    (re.compile(r'^a\.?\s*juridico\s*documental\s*-?\s*regulatorio$'), 'a. juridico - documental - regulatorio'),
    (re.compile(r'^b\.?\s*financieros?\s*(y/o)?\s*(de\s*)?mercado$|^financiero\s*o\s*de\s*mercado$|^financieros?$'), 'b. financieros y/o de mercado'),
    (re.compile(r'^c\.?\s*estudios\s*(y/o)?\s*disenos?$'), 'c. estudios y/o disenos'),
    (re.compile(r'^d\.?\s*sociales\s*(y\s*ambientales|y/o\s*ambientales)$'), 'd. sociales y ambientales'),
    (re.compile(r'^sociales\s*(y/o)?\s*ambientales$'), 'd. sociales y ambientales'),
    (re.compile(r'^e\.?\s*tecnicos?\s*[-]\s*operativos?\s*[-]\s*constructivos?$'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^e\.\s*tecnicos\s*-\s*operativos\s*-\s*constructivos$'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^tecnicos?\s*(y/o)?\s*operativos?\s*(y/o)?\s*(de\s*)?ejecucion$'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^tecnicos?,\s*operativos?\s*(o\s*tecnologicos)?$'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^f\.?\s*supervision,?\s*seguimiento\s*(y\s*control)?(\s*\(.*?\))?$'), 'f. supervision, seguimiento y control'),
    (re.compile(r'^contractual\s*[-]\s*ejecucion$'), 'contractual - ejecucion'),
    (re.compile(r'^riesgos?\s*operacionales?$|^operacional(?:es)?$'), 'riesgos operacionales'),
    (re.compile(r'^riesgos?\s*financieros?\s*(y\s*legales)?$|^economicos\s*(y/o)?\s*financieros?$|^financieros?\s*(y/o)?\s*de\s*mercado$'), 'riesgos financieros y legales'),
    (re.compile(r'^derivados?\s*de\s*circunstancias\s*economicas?\s*(o\s*legales)?$'), 'derivados de circunstancias economicas o legales'),
    (re.compile(r'^precontractual\s*[-]\s*perfeccionamiento$'), 'precontractual - perfeccionamiento'),
    (re.compile(r'^riesgos?\s*sociales?\s*(o\s*politicos)?$|^sociales?\s*o\s*politicos$'), 'riesgos sociales o politicos'),
    (re.compile(r'^riesgos?\s*tecnologicos?$|^tecnologicos?$'), 'riesgos tecnologicos'),
    (re.compile(r'^de\s*la\s*naturaleza$|^naturaleza$|^causas?\s*de\s*la\s*naturaleza|^riesgos?\s*de\s*la\s*naturaleza'), 'de la naturaleza'),
    (re.compile(r'^administrativos?$'), 'administrativos'),
    (re.compile(r'^poscontractual$'), 'poscontractual'),
    (re.compile(r'^juridicos?\s*(y/o)?\s*legales?\s*(y/o)?\s*regulatorios?$'), 'a. juridico - documental - regulatorio'),
    (re.compile(r'^ambientales?$|^riesgos?\s*ambientales?$'), 'd. sociales y ambientales'),
    (re.compile(r'^economicos?$|^riesgos?\s*economicos?$'), 'b. financieros y/o de mercado'),
    (re.compile(r'^regulatorios?$|^riesgos?\s*regulatorios?$'), 'a. juridico - documental - regulatorio'),
    (re.compile(r'^contratista$|^contratante$'), 'general'),
    (re.compile(r'^por\s*riesgos?\s*en\s*la\s*planeacion'), 'general'),
    (re.compile(r'^otros$|^particular$|^soci$'), 'general'),
    (re.compile(r'^riesgo\s*de\s*diseno$|^riesgo\s*en\s*el\s*programa|^riesgo\s*en\s*el\s*proceso|^riesgo\s*en\s*el\s*equipo|^riesgo\s*en\s*la\s*falta'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^riesgo\s*en\s*relaciones\s*laborales$'), 'general'),
    (re.compile(r'^epafo$'), 'general'),
    (re.compile(r'^constru$'), 'e. tecnicos - operativos - constructivos'),
    (re.compile(r'^riesgos?\s*corrupcion$'), 'general'),
    (re.compile(r'^laboral$'), 'general'),
    (re.compile(r'^,$'), 'no especificado'),
    (re.compile(r'^dffgfgfg$'), 'no especificado'),
    (re.compile(r'^contratista$'), 'general'),
    (re.compile(r'^contratista\s*(y\s*contratante)?$|^contratante\s*/\s*contratista$'), 'general'),
    (re.compile(r'^sociales\s*(y/o)?\s*politicos$'), 'riesgos sociales o politicos'),
    (re.compile(r'^riesgos?\s*tecnologicos?\s*(e\s*infraestructura\s*publica)?$'), 'riesgos tecnologicos'),
    (re.compile(r'^riesgo\s*por\s*danos\s*causados\s*por\s*terceros'), 'general'),
    (re.compile(r'^social$'), 'riesgos sociales o politicos'),
    (re.compile(r'^ambientales\s*(y\s*prediales)?$'), 'd. sociales y ambientales'),
]


def estandarizar_clase(s: str) -> str:
    if not s:
        return 'no especificado'
    for pat, estandar in CLASE_PATRONES:
        if pat.match(s):
            return estandar
    return s


# ---------------------------------------------------------------------------
#  ASIGNACION — 98 variantes -> ~12 canonicas
# ---------------------------------------------------------------------------

ASIGNACION_MAP_EXACTO = {
    'no especificado': 'no especificado',
    'sin datos': 'no especificado',
    'n/a': 'no especificado',
    'indeterminado': 'no especificado',
    'x': 'no especificado',
    'compartido': 'compartido',
    'compartida': 'compartido',
    'compartido (definir porcentaje c/u)': 'compartido',
    'ambos': 'compartido',
    'contratista': 'contratista',
    'contratesta': 'contratista',
    'proponente': 'contratista',
    'proponente o contratista': 'contratista',
    'proponente y/o contratista': 'contratista',
    'adjudicatario': 'contratista',
    'oferente': 'contratista',
    'oferentes': 'contratista',
    'entidad': 'entidad',
    'entidad contratante': 'entidad',
    'entidad estatal': 'entidad',
    'institucion': 'entidad',
    'empresa': 'entidad',
    'desur': 'desur',
    'interventoria': 'interventoria',
    'interventor': 'interventoria',
    'supervisor': 'interventoria',
    'privado': 'privado',
    'publico': 'publico',
    'audiencia de aclaraciones': 'no especificado',
    'edu': 'edu',
}

ASIGNACION_PARRAFO_RE = re.compile(r'^.{50,}$')  # frases largas = error


def estandarizar_asignacion(s: str) -> str:
    if not s:
        return 'no especificado'
    if s in ASIGNACION_MAP_EXACTO:
        return ASIGNACION_MAP_EXACTO[s]

    if ASIGNACION_PARRAFO_RE.match(s):
        return 'no especificado'

    # Entidades especificas -> entidad
    entidades_municipios = ['alcaldia', 'gobernacion', 'secretaria', 'amva', 'edeso',
                            'cvc', 'cvp', 'ibal', 'cornare', 'itm', 'icbf', 'fondo adaptacion',
                            'transcaribe', 'acuatodos', 'adeli', 'coord', 'invisbu',
                            'lider de proceso', 'corporacion', 'universidad',
                            'oficina asesora juridica', 'comite evaluador', 'comite de contratacion']
    for e in entidades_municipios:
        if e in s:
            return 'entidad'

    es_contratista = any(t in s for t in ['contratista', 'contratesta', 'proponente',
                                           'oferente', 'adjudicatario'])
    es_entidad = any(t in s for t in ['entidad', 'municipio', 'departamento',
                                       'contratante', 'estatal', 'institucion'])
    es_interventor = 'interventor' in s or 'interventoria' in s
    es_aseguradora = 'aseguradora' in s

    if es_contratista and not es_entidad and not es_interventor:
        return 'contratista'
    if es_entidad and not es_contratista:
        return 'entidad'
    if es_interventor:
        return 'interventoria'
    if es_aseguradora:
        return 'aseguradora'

    # Compartido
    if es_contratista and es_entidad:
        return 'contratista / entidad'

    # Porcentajes -> compartido
    if re.search(r'\d+%', s):
        return 'compartido'
    # Frases descriptivas sueltas -> no especificado
    if ' ' in s and len(s) > 25:
        return 'no especificado'
    return 'no especificado'


# ---------------------------------------------------------------------------
#  FUENTE_RIESGO — 16 variantes -> 3 canonicas
# ---------------------------------------------------------------------------

FUENTE_PATRONES = [
    (re.compile(r'^intern[oa]$|^intermo$|^internmo$|^interna\s*externa$'), 'interno'),
    (re.compile(r'^extern[oa]$|^extremo$|^externo$|^exerno$|^exernc$|^exterio$|^extemo$'), 'externo'),
    (re.compile(r'^(no\s+)?especificad[oa]$|^general$|^especifico$|^e$|^i$|^durante$'), 'no especificado'),
    (re.compile(r'(interno|interna|internmo|intermo|int).*(externo|externa|extremo|exerno|ext)'), 'mixto'),
    (re.compile(r'(externo|externa|extremo|exerno|ext).*(interno|interna|internmo|intermo|int)'), 'mixto'),
    (re.compile(r'^int/ext$|^intern[oa]\s*[-/]\s*extern[oa]$'), 'mixto'),
]


def estandarizar_fuente(s: str) -> str:
    if not s:
        return 'no especificado'
    for pat, estandar in FUENTE_PATRONES:
        if pat.match(s):
            return estandar
    return 'no especificado'


# ---------------------------------------------------------------------------
#  PROBABILIDAD / IMPACTO — texto -> numero 1-5
# ---------------------------------------------------------------------------

PROBABILIDAD_MAP = {
    'raro': '1', 'poco probable': '1',
    'improbable': '2',
    'posible': '3', 'medio': '3', 'media': '3', 'mb': '3',
    'probable': '4', 'alta': '4', 'alto': '4', 'ma': '4', 'a': '4',
    'b': '2', 'baja': '2', 'bajo': '2',
}

IMPACTO_MAP = {
    'insignificante': '1',
    'menor': '2', 'minor': '2',
    'moderado': '3', 'medio': '3', 'mb': '3',
    'mayor': '4', 'alto': '4', 'ma': '4', 'a': '4',
    'catastrofico': '5',
    'bajo': '2', 'baja': '2', 'b': '2',
}

IMPACTO_TEXTO_LARGO = [
    (re.compile(r'baja$|no representan mas del cinco'), '2'),
    (re.compile(r'moderado|perturba los costos'), '3'),
    (re.compile(r'grave|sustancialmente|obstruye|mas del treinta'), '4'),
]

NUMERICO_RE = re.compile(r'^-?\d+(\.\d+)?$')


def _mapear_escala(v: float) -> str:
    """Mapea valores numericos fuera de rango 1-5 a la escala estandar."""
    if 1 <= v <= 5:
        return str(int(v))
    if v == 0:
        return '1'
    # Escala 0-1 (probabilidad)
    if 0 < v <= 1:
        m = round(v * 5)
        return str(max(1, min(5, m)))
    # Escala 0-10
    if v <= 10:
        m = round(v / 2)
        return str(max(1, min(5, m)))
    # Escala porcentual /10 (10, 20, 30...)
    m = round(v / 10)
    return str(max(1, min(5, m)))


def estandarizar_probabilidad(s: str) -> str:
    if not s:
        return 'no especificado'
    if s in PROBABILIDAD_MAP:
        return PROBABILIDAD_MAP[s]
    if NUMERICO_RE.match(s):
        v = float(s)
        if 1 <= v <= 5:
            return str(int(v))
        return _mapear_escala(v)
    return 'no especificado'


def estandarizar_impacto(s: str) -> str:
    if not s:
        return 'no especificado'
    s_clean = limpiar(s)
    if s_clean in IMPACTO_MAP:
        return IMPACTO_MAP[s_clean]
    for pat, val in IMPACTO_TEXTO_LARGO:
        if pat.search(s_clean):
            return val
    if NUMERICO_RE.match(s_clean):
        v = float(s_clean)
        if 1 <= v <= 5:
            return str(int(v))
        return _mapear_escala(v)
    return 'no especificado'


# ---------------------------------------------------------------------------
#  CATEGORIA / VALORACION
# ---------------------------------------------------------------------------

CATEGORIA_MAP = {
    'bajo': 'bajo', 'riesgo bajo': 'bajo', 'riesgo menor': 'bajo',
    'baja': 'bajo', 'minor': 'bajo', 'riesgo improbable': 'bajo',
    'medio': 'medio', 'riesgo medio': 'medio', 'riesgo moderado': 'medio',
    'media': 'medio', 'medio alto': 'medio', 'mediano': 'medio',
    'alto': 'alto', 'riesgo alto': 'alto', 'muy alto': 'alto',
    'alta': 'alto', 'riesgo mayor': 'alto', 'mayor': 'alto',
    'extremo': 'extremo', 'riesgo extremo': 'extremo',
    'no especificado': 'no especificado',
    'improbable': 'bajo', 'probable': 'alto',
    'riesgo probable': 'alto',
    'riesgo catastrofico': 'extremo',
}

VALORACION_TEXTO = {
    'bajo': '2', 'baja': '2', 'minor': '2',
    'medio': '7', 'media': '7', 'mediano': '7', 'moderado': '7',
    'aceptable': '7', 'tolerable': '7',
    'alto': '12', 'alta': '12', 'mayor': '12',
    'extremo': '20',
    'no especificado': '',
}

CATEGORIA_TEXTO_LARGO = [
    (re.compile(r'riesg[o]\s*ext\s*remo$|riesgo\s*ext\s*remo$'), 'extremo'),
    (re.compile(r'riesg[o]\s*baj[o]'), 'bajo'),
]


def estandarizar_categoria(s: str) -> str:
    if not s:
        return 'no especificado'
    s_clean = limpiar(s)
    if s_clean in CATEGORIA_MAP:
        return CATEGORIA_MAP[s_clean]
    for pat, val in CATEGORIA_TEXTO_LARGO:
        if pat.match(s_clean):
            return val
    if NUMERICO_RE.match(s_clean):
        v = float(s_clean)
        if v == 0:
            return 'no especificado'
        if 1 <= v <= 4:
            return 'bajo'
        if v <= 9:
            return 'medio'
        if v <= 14:
            return 'alto'
        return 'extremo'
    return 'no especificado'


def estandarizar_valoracion(s: str) -> str:
    if not s:
        return ''
    s_clean = limpiar(s)
    if s_clean in VALORACION_TEXTO:
        return VALORACION_TEXTO[s_clean]
    if s_clean in ('contratista', 'entidad', 'entidad, contratista'):
        return ''
    if NUMERICO_RE.match(s_clean):
        v = float(s_clean)
        return str(int(v)) if v == int(v) else s_clean
    return ''


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------

COLUMNAS_TEXTUALES = {
    'objeto', 'fuente', 'clase', 'fuente_riesgo', 'etapa', 'tipo',
    'descripcion_riesgo', 'consecuencia', 'probabilidad', 'impacto',
    'valoracion', 'categoria', 'asignacion', 'plan_mitigacion'
}

ESTANDARIZADORES = {
    'clase': estandarizar_clase,
    'asignacion': estandarizar_asignacion,
    'probabilidad': estandarizar_probabilidad,
    'impacto': estandarizar_impacto,
    'fuente_riesgo': estandarizar_fuente,
    'etapa': estandarizar_etapa,
    'tipo': estandarizar_tipo,
    'categoria': estandarizar_categoria,
    'valoracion': estandarizar_valoracion,
}

cambios_contados = {k: Counter() for k in COLUMNAS_TEXTUALES}


def normalizar_valor(col: str, s: str) -> str:
    original = s
    s = limpiar(s)
    if col in ESTANDARIZADORES:
        s = ESTANDARIZADORES[col](s)
    if original != s:
        cambios_contados[col][f'{original} -> {s}'] += 1
    return s


def arreglar_fila(row: list, ncols: int) -> list:
    if len(row) == ncols:
        return row
    if len(row) < ncols:
        return row + [''] * (ncols - len(row))
    return row[:ncols - 1] + [','.join(row[ncols - 1:])]


def main():
    with open(ORIGEN, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        filas = list(reader)

    idx_textuales = {i for i, h in enumerate(header) if h in COLUMNAS_TEXTUALES}
    n_original = len(filas)
    n_bad = 0

    with open(DESTINO, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in filas:
            if not row:
                continue
            if len(row) != NCOLS:
                n_bad += 1
            row = arreglar_fila(row, NCOLS)
            for i in idx_textuales:
                row[i] = normalizar_valor(header[i], row[i])
            writer.writerow(row)

    print(f'Hecho: {n_original} filas -> {DESTINO}')
    print(f'Filas malformadas reparadas: {n_bad}')
    total_cambios = sum(v for c in cambios_contados.values() for v in c.values())
    print(f'Valores normalizados: {total_cambios}')

    for col, c in sorted(cambios_contados.items()):
        if c and col in ESTANDARIZADORES:
            print(f'\n--- {col} ({len(c)} transformaciones) ---')
            for k, v in sorted(c.most_common(10)):
                if v > 1 or len(c) <= 15:
                    print(f'  [{v:3d}] {k}')
            if len(c) > 15:
                resto = sum(v for k_, v in c.most_common()[10:])
                if resto:
                    print(f'  ... y {len(c) - 10} mas ({resto} filas)')


if __name__ == '__main__':
    main()
