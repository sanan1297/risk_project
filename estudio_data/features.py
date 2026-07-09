"""
Pipeline de feature engineering: transforma matriz_clean.csv (riesgos)
en contratos_features.csv (contratos) listo para modelado ML.

Uso: python estudio_data/features.py

Salida: docs/contratos_features.csv (~340 filas, ~200+ columnas)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ORIGEN = Path('docs/matriz_clean.csv')
DESTINO = Path('docs/contratos_features.csv')

STOP_WORDS = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
    'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como',
    'más', 'pero', 'sus', 'le', 'ya', 'este', 'entre', 'porque', 'este',
    'esta', 'todo', 'tiene', 'sin', 'eso', 'muy', 'son', 'era', 'han',
    'ello', 'cada', 'dicho', 'dicha', 'tanto', 'donde', 'cual', 'quien',
    'sido', 'sea', 'ante', 'tras', 'durante', 'mediante', 'contra',
    'hasta', 'desde', 'para', 'según', 'sobre', 'entre', 'ser', 'haber',
    'estar', 'tener', 'hacer', 'poder', 'saber', 'nuestro', 'vuestra',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'aquel', 'aquella', 'aquellos', 'aquellas',
    'cualquier',
]


def _prepare_text(s):
    if pd.isna(s) or not isinstance(s, str):
        return ''
    return s.strip()

CAT_COLS = ['tipo', 'clase', 'asignacion', 'fuente_riesgo', 'etapa', 'categoria']
NUM_COLS = ['probabilidad', 'impacto', 'valoracion']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir numéricos
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Asegurar categóricas como string
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().replace('nan', 'no especificado')

    grupos = df.groupby('id_contrato', sort=False)
    rows = []
    texto_por_contrato = {}

    for cid, g in grupos:
        # A. Variables básicas (first)
        row = {
            'id_contrato': cid,
            'valor_inicial': g['valor_inicial'].iloc[0],
            'sobrecosto': g['sobrecosto'].iloc[0],
            'fuente': g['fuente'].iloc[0],
        }

        # B. Volumen
        row['n_riesgos'] = len(g)

        # C. Texto para TF-IDF
        texto_por_contrato[cid] = {
            'descripciones': ' '.join(g['descripcion_riesgo'].dropna().astype(str)),
            'objeto': str(g['objeto'].iloc[0]) if pd.notna(g['objeto'].iloc[0]) else '',
        }

        # D. Centralidad y dispersión por columna numérica
        for col, name in [('probabilidad', 'prob'), ('impacto', 'imp'), ('valoracion', 'val')]:
            vals = g[col].dropna()
            if len(vals) > 0:
                row[f'{name}_promedio'] = vals.mean()
                row[f'{name}_std'] = vals.std(ddof=0)
                row[f'{name}_min'] = vals.min()
                row[f'{name}_max'] = vals.max()
                row[f'{name}_rango'] = vals.max() - vals.min()
                row[f'{name}_p25'] = vals.quantile(0.25)
                row[f'{name}_p75'] = vals.quantile(0.75)
            else:
                for s in ['_promedio', '_std', '_min', '_max', '_rango', '_p25', '_p75']:
                    row[f'{name}{s}'] = np.nan

        # E. Carga total
        row['suma_valoracion'] = g['valoracion'].sum()
        row['suma_probabilidad'] = g['probabilidad'].sum()
        row['suma_impacto'] = g['impacto'].sum()

        # F. Conteo por categoría de riesgo
        for cat in ['bajo', 'medio', 'alto', 'extremo', 'no especificado']:
            row[f'n_categoria_{cat}'] = (g['categoria'] == cat).sum()
        row['n_categoria_alto_extremo'] = (g['categoria'].isin(['alto', 'extremo'])).sum()

        # G. Interacciones
        row['interaccion_prob_x_n'] = row.get('prob_promedio', 0) * row['n_riesgos']
        row['interaccion_val_x_n'] = row.get('val_promedio', 0) * row['n_riesgos']
        row['interaccion_prob_x_impacto'] = row.get('prob_promedio', 0) * row.get('imp_promedio', 0)

        # H. Proporción con plan de mitigación
        pm = g['plan_mitigacion'].astype(str).str.strip().replace('nan', '')
        row['prop_plan_mitigacion'] = pm.ne('').mean()

        rows.append(row)

    out = pd.DataFrame(rows)

    # J. Vector de composición (proporciones categóricas)
    for col in CAT_COLS:
        dummies = pd.get_dummies(df[col], prefix=col[:4])
        col_dummies = pd.concat([df['id_contrato'], dummies], axis=1)
        col_agg = col_dummies.groupby('id_contrato', sort=False).mean()
        col_agg.columns = [f'prop_{c}' for c in col_agg.columns]
        out = out.merge(col_agg, on='id_contrato', how='left')

    # K. Vector de texto TF-IDF
    df_texto = pd.DataFrame([
        {'id_contrato': cid, 'texto_completo': v['descripciones'] + ' ' + v['objeto']}
        for cid, v in texto_por_contrato.items()
    ])

    vectorizer = TfidfVectorizer(max_features=100, stop_words=STOP_WORDS, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_texto['texto_completo'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{w}' for w in vectorizer.get_feature_names_out()],
        index=df_texto['id_contrato'],
    )
    tfidf_df = tfidf_df.reset_index()
    out = out.merge(tfidf_df, on='id_contrato', how='left')

    # Rellenar nulos
    prop_cols = [c for c in out.columns if c.startswith('prop_')]
    out[prop_cols] = out[prop_cols].fillna(0.0)

    num_feat = [c for c in out.columns if c not in ['id_contrato', 'fuente', 'valor_inicial', 'sobrecosto']]
    out[num_feat] = out[num_feat].fillna(0)

    return out


def main():
    print('Leyendo matriz_clean.csv...')
    df = pd.read_csv(ORIGEN, encoding='utf-8-sig')
    print(f'  Filas: {len(df):,} | Contratos: {df["id_contrato"].nunique()}')

    print('Ingeniería de características...')
    df_feat = engineer_features(df)

    print(f'  Contratos: {len(df_feat)} | Columnas: {df_feat.shape[1]}')

    # Reporte breve
    prop_count = sum(1 for c in df_feat.columns if c.startswith('prop_'))
    tfidf_count = sum(1 for c in df_feat.columns if c.startswith('tfidf_'))
    print(f'  Proporciones: {prop_count} | TF-IDF: {tfidf_count} | Numéricas: {df_feat.shape[1] - prop_count - tfidf_count - 3}')

    df_feat.to_csv(DESTINO, index=False, encoding='utf-8-sig')
    print(f'Guardado: {DESTINO}')
    print(f'  Head(2):')
    print(df_feat.head(2).to_string())
    print('Listo.')


if __name__ == '__main__':
    main()
