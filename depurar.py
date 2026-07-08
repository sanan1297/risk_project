"""
DEPURAR DATASET UNIFICADO
Lee los cache de SECOP I y II, normaliza correctamente,
filtra solo proyectos terminados con URL, elimina columnas inutiles.
"""
import pandas as pd, numpy as np, os, sys, ast
sys.stdout.reconfigure(encoding="utf-8")

CARPETA = "contratos"

print("=" * 60)
print("DEPURACION DE DATASET UNIFICADO")
print("=" * 60)

# ---- CARGAR CACHES ----
print("\nCargando SECOP I...")
s1 = pd.read_csv(os.path.join(CARPETA, "secop1_cache.csv"), encoding="utf-8-sig", low_memory=False)
print(f"  {len(s1)} registros")

print("Cargando SECOP II...")
s2 = pd.read_csv(os.path.join(CARPETA, "secop2_cache.csv"), encoding="utf-8-sig", low_memory=False)
print(f"  {len(s2)} registros")


# ---- NORMALIZAR COLUMNAS COMUNES ----
def extraer_url(v):
    if isinstance(v, dict):
        return v.get("url", "")
    if isinstance(v, str) and v.strip() not in ("", "nan", "None"):
        try:
            d = ast.literal_eval(v)
            if isinstance(d, dict):
                return d.get("url", "")
        except:
            pass
        return v
    return ""


def limpiar_texto(v):
    if isinstance(v, str) and v.strip() in ("No Definido", "No definido", "nan", "None", "", "Sin Descripcion"):
        return ""
    if isinstance(v, str):
        return v.strip()
    return ""


def normalizar_secop1(df):
    out = pd.DataFrame({"fuente": ["SECOP I"] * len(df)})
    out["entidad"] = df["nombre_entidad"].apply(limpiar_texto)
    out["departamento"] = df["departamento_entidad"].apply(limpiar_texto)
    out["municipio"] = df["municipio_entidad"].apply(limpiar_texto)
    out["valor_inicial"] = pd.to_numeric(df["cuantia_contrato"], errors="coerce")
    out["valor_final"] = pd.to_numeric(df["valor_contrato_con_adiciones"], errors="coerce")
    out["adiciones_valor"] = pd.to_numeric(df["valor_total_de_adiciones"], errors="coerce")
    out["adiciones_dias"] = pd.to_numeric(df["tiempo_adiciones_en_dias"], errors="coerce")
    out["plazo_dias"] = pd.to_numeric(df["plazo_de_ejec_del_contrato"], errors="coerce")
    out["estado"] = df["estado_del_proceso"].apply(limpiar_texto)
    out["objeto"] = df["detalle_del_objeto_a_contratar"].apply(limpiar_texto)
    out["categoria_objeto"] = df["objeto_a_contratar"].apply(limpiar_texto)
    out["url"] = df["ruta_proceso_en_secop_i"].apply(extraer_url)
    out["contratista"] = df["nom_razon_social_contratista"].apply(limpiar_texto)
    out["fecha_firma"] = df["fecha_de_firma_del_contrato"].apply(limpiar_texto)
    out["fecha_inicio"] = df["fecha_ini_ejec_contrato"].apply(limpiar_texto)
    out["fecha_fin"] = df["fecha_fin_ejec_contrato"].apply(limpiar_texto)
    out["postconflicto"] = pd.to_numeric(df["es_postconflicto"], errors="coerce").fillna(0).astype(int)
    out["destino_gasto"] = "Inversión"
    out["numero_proceso"] = df["numero_de_proceso"].apply(limpiar_texto)
    out["numero_contrato"] = df["numero_de_contrato"].apply(limpiar_texto)
    # Paz columns
    pilar = df["pilar_acuerdo_paz"].apply(limpiar_texto)
    out["pilar_paz"] = pilar.where(pilar != "", np.nan)
    punto = df["punto_acuerdo_paz"].apply(limpiar_texto)
    out["punto_paz"] = punto.where(punto != "", np.nan)
    out["codigo_bpin"] = df["codigo_bpin"].apply(limpiar_texto)
    return out


def normalizar_secop2(df):
    out = pd.DataFrame({"fuente": ["SECOP II"] * len(df)})
    out["entidad"] = df["nombre_entidad"].apply(limpiar_texto)
    out["departamento"] = df["departamento"].apply(limpiar_texto)
    out["municipio"] = df["ciudad"].apply(limpiar_texto)
    out["valor_inicial"] = pd.to_numeric(df["valor_del_contrato"], errors="coerce")
    out["valor_final"] = pd.to_numeric(df["valor_pagado"], errors="coerce")
    out["adiciones_valor"] = 0
    out["adiciones_dias"] = pd.to_numeric(df["dias_adicionados"], errors="coerce")
    out["plazo_dias"] = np.nan
    out["estado"] = df["estado_contrato"].apply(limpiar_texto)
    out["objeto"] = df["objeto_del_contrato"].apply(limpiar_texto)
    out["categoria_objeto"] = ""
    out["url"] = df["urlproceso"].apply(extraer_url)
    out["contratista"] = df["proveedor_adjudicado"].apply(limpiar_texto)
    out["fecha_firma"] = ""
    out["fecha_inicio"] = df["fecha_de_inicio_del_contrato"].apply(limpiar_texto)
    out["fecha_fin"] = df["fecha_de_fin_del_contrato"].apply(limpiar_texto)
    out["postconflicto"] = 0
    out["destino_gasto"] = df["destino_gasto"].apply(limpiar_texto)
    out["numero_proceso"] = ""
    out["numero_contrato"] = ""
    out["pilar_paz"] = np.nan
    out["punto_paz"] = np.nan
    out["codigo_bpin"] = ""
    return out


# ---- NORMALIZAR ----
s1n = normalizar_secop1(s1)
s2n = normalizar_secop2(s2)

# Unificar
df = pd.concat([s1n, s2n], ignore_index=True)
print(f"\nTotal combinado: {len(df)}")

# Normalize destino_gasto
df["destino_gasto"] = df["destino_gasto"].str.replace("Inversion", "Inversión")

# ---- FILTRO 1: DESTINO = INVERSION ----
antes = len(df)
df = df[df["destino_gasto"] == "Inversión"].copy()
print(f"Filtro Inversión: {antes} -> {len(df)}")

# ---- FILTRO 2: VALOR >= $500M ----
antes = len(df)
df = df[df["valor_inicial"] >= 500e6].copy()
print(f"Filtro >= $500M: {antes} -> {len(df)}")

# ---- FILTRO 3: TERMINADOS ----
# SECOP I: Liquidado. SECOP II: terminado, Cerrado
estados_validos = ["liquidado", "terminado", "cerrado"]
antes = len(df)
df["estado_ok"] = df["estado"].str.lower().isin(estados_validos)
df = df[df["estado_ok"]].copy()
print(f"Filtro terminados: {antes} -> {len(df)}")

# ---- FILTRO 4: URL VALIDA ----
antes = len(df)
tiene_url = df["url"].notna() & (df["url"].astype(str).str.strip() != "") & (df["url"].astype(str) != "nan")
df = df[tiene_url].copy()
print(f"Filtro URL valida: {antes} -> {len(df)}")

# ---- FILTRO 5: ENTIDAD CON NOMBRE ----
antes = len(df)
df = df[df["entidad"].astype(str).str.strip() != ""].copy()
print(f"Filtro entidad con nombre: {antes} -> {len(df)}")

# ---- CALCULAR SOBRECOSTO ----
mask_sc = (df["valor_inicial"] > 0) & (df["valor_final"] > 0)
df["sobrecosto_pct"] = np.nan
df.loc[mask_sc, "sobrecosto_pct"] = (
    (df.loc[mask_sc, "valor_final"] - df.loc[mask_sc, "valor_inicial"])
    / df.loc[mask_sc, "valor_inicial"]
) * 100

# Retraso
mask_rt = df["plazo_dias"] > 0
df["retraso_pct"] = np.nan
df.loc[mask_rt, "retraso_pct"] = (df.loc[mask_rt, "adiciones_dias"] / df.loc[mask_rt, "plazo_dias"]) * 100

# ---- FILTRO 6: SOLO SOBRECOSTO >= 0 ----
# Objetivo: hallar proyectos con sobrecosto. Se descartan ahorros (negativos)
antes = len(df)
df = df[(df["sobrecosto_pct"] >= 0) | (df["sobrecosto_pct"].isna())].copy()
print(f"Filtro sobrecosto >= 0: {antes} -> {len(df)}")

# ---- SELECCIONAR COLUMNAS FINALES ----
cols_finales = [
    "fuente", "entidad", "departamento", "municipio",
    "valor_inicial", "valor_final", "adiciones_dias", "plazo_dias",
    "sobrecosto_pct", "retraso_pct",
    "estado", "objeto", "url",
    "contratista", "fecha_inicio", "fecha_fin",
    "postconflicto", "destino_gasto",
]

df = df.sort_values("valor_inicial", ascending=False).reset_index(drop=True)

# ---- EXPORTAR ----
archivo = os.path.join(CARPETA, "proyectos_depurados.csv")
df[cols_finales].to_csv(archivo, index=False, encoding="utf-8-sig")
print(f"\nGuardado: {archivo}")
print(f"  {len(df)} registros, {len(cols_finales)} columnas")

# Tambien sobrecosto no cero
con_sc = df[df["sobrecosto_pct"].notna() & (df["sobrecosto_pct"] != 0)].copy()
archivo2 = os.path.join(CARPETA, "proyectos_depurados_con_sobrecosto.csv")
con_sc.to_csv(archivo2, index=False, encoding="utf-8-sig")
print(f"Guardado: {archivo2} ({len(con_sc)} registros)")

# ---- RESUMEN ----
print(f"\n{'=' * 60}")
print("RESUMEN FINAL")
print(f"{'=' * 60}")
print(f"Total: {len(df)}")
print(f"  SECOP I: {(df['fuente']=='SECOP I').sum()}")
print(f"  SECOP II: {(df['fuente']=='SECOP II').sum()}")
print(f"  Con sobrecosto != 0: {(df['sobrecosto_pct'] != 0).sum()}")
print(f"  Con sobrecosto = 0: {(df['sobrecosto_pct'] == 0).sum()}")
print(f"  Sin dato sobrecosto: {df['sobrecosto_pct'].isna().sum()}")
s = df["sobrecosto_pct"].dropna()
if len(s) > 0:
    print(f"  Media sobrecosto: {s.mean():.1f}%")
    print(f"  Mediana: {s.median():.1f}%")
    print(f"  > 0%: {(s>0).sum()} | < 0%: {(s<0).sum()}")

print(f"\nTop 10 por valor:")
for _, r in df.head(10).iterrows():
    obj = str(r.get("objeto", ""))[:100]
    sc = r.get("sobrecosto_pct", np.nan)
    sc_s = f" ({sc:+.1f}%)" if pd.notna(sc) else ""
    print(f"  ${r['valor_inicial']:>15,.0f}{sc_s} | {r['entidad'][:40]}")
