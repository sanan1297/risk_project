"""
UNIFICAR SECOP I + SECOP II
Proyectos de desarrollo: Obra + Inversion + Liquidado/Terminado + >= $500M
Resultado: contratos/proyectos_unificados.csv
"""
import pandas as pd, numpy as np, os, sys, ast, requests, time
sys.stdout.reconfigure(encoding='utf-8')

CARPETA = "contratos"
os.makedirs(CARPETA, exist_ok=True)

MAX_POR_PAGINA = 50000

def extraer_url(v):
    if isinstance(v, dict): return v.get("url", "")
    if isinstance(v, str) and v.strip() not in ("", "nan", "None", "No Definido"):
        try:
            d = ast.literal_eval(v)
            if isinstance(d, dict): return d.get("url", "")
        except: pass
        return v
    return ""

def descargar_o_cargar(nombre, archivo_cache, cfg):
    path = os.path.join(CARPETA, archivo_cache)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  Cargando cache: {archivo_cache}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        print(f"  {len(df)} registros, {len(df.columns)} columnas")
        return df, True

    print(f"  Descargando desde API...")
    todos = []
    for offset in range(0, cfg["max"], MAX_POR_PAGINA):
        pagina = offset // MAX_POR_PAGINA + 1
        order = "fecha_de_firma_del_contrato DESC" if nombre == "SECOP I" else "fecha_de_inicio_del_contrato DESC"
        params = {"$limit": MAX_POR_PAGINA, "$offset": offset, "$where": cfg["where"], "$order": order}
        try:
            r = requests.get(cfg["url"], params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                print(f"    Pag {pagina} - Error: {data.get('message','')}"); break
            if not data: print(f"    Pag {pagina} - Fin."); break
            todos.extend(data)
            print(f"    Pag {pagina}: +{len(data)} = {len(todos)}")
            time.sleep(0.3)
        except Exception as e:
            print(f"    Pag {pagina} - Error: {e}"); time.sleep(2)
    if todos:
        df = pd.DataFrame(todos)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  Guardado cache: {archivo_cache}")
        return df, True
    return None, False

# ---- MAIN ----
print("=" * 60)
print("PROYECTOS DE DESARROLLO - DATASET UNIFICADO")
print("Obra + Inversion + Terminado/Liquidado + >= $500M")
print("=" * 60)

FUENTES = [
    ("SECOP I", "secop1_cache.csv", {
        "url": "https://www.datos.gov.co/resource/f789-7hwg.json",
        "max": 200000,
        "where": "tipo_de_contrato = 'Obra' AND destino_gasto = 'Inversion' AND estado_del_proceso = 'Liquidado'",
    }),
    ("SECOP II", "secop2_cache.csv", {
        "url": "https://www.datos.gov.co/resource/jbjy-vk9h.json",
        "max": 300000,
        "where": "tipo_de_contrato = 'Obra' AND (estado_contrato = 'terminado' OR estado_contrato = 'Cerrado')",
    }),
]

dataframes = []
for nombre, cache, cfg in FUENTES:
    print(f"\n{'---'} {nombre} {'---'}")
    df, ok = descargar_o_cargar(nombre, cache, cfg)
    if ok:
        df["fuente"] = nombre
        # Normalize destino_gasto
        if "destino_gasto" in df.columns:
            df["destino_gasto"] = df["destino_gasto"].astype(str).str.strip()
            if nombre == "SECOP I":
                df["destino_gasto"] = df["destino_gasto"].replace({"Inversion": "Inversión"})
        else:
            df["destino_gasto"] = ""
        dataframes.append(df)

if not dataframes:
    print("ERROR: No hay datos disponibles.")
    exit()

# ---- UNIFICAR ----
print(f"\n{'='} UNIFICANDO DATASETS {'='}")
df_u = pd.concat(dataframes, ignore_index=True, sort=False)
print(f"Total combinado: {len(df_u)}")

# Normalize key columns using positions (safe)
df_u["url"] = ""
df_u["nombre_entidad"] = ""
df_u["departamento"] = ""
df_u["municipio"] = ""
df_u["contratista"] = ""
df_u["objeto_texto"] = ""
df_u["estado"] = ""
df_u["fecha_inicio"] = ""
df_u["fecha_fin"] = ""
df_u["fecha_firma"] = ""
df_u["postconflicto"] = 0
df_u["pilar_paz"] = ""
df_u["punto_paz"] = ""
df_u["codigo_bpin"] = ""
df_u["numero_proceso"] = ""
df_u["numero_contrato"] = ""

for i in range(len(df_u)):
    r = df_u.iloc[i]
    src = r["fuente"]

    # URL
    if src == "SECOP I":
        url_raw = r.get("ruta_proceso_en_secop_i", "")
        df_u.iloc[i, df_u.columns.get_loc("url")] = extraer_url(url_raw)
        df_u.iloc[i, df_u.columns.get_loc("nombre_entidad")] = r.get("nombre_entidad", "")
        df_u.iloc[i, df_u.columns.get_loc("departamento")] = r.get("departamento_entidad", "")
        df_u.iloc[i, df_u.columns.get_loc("municipio")] = r.get("municipio_entidad", "")
        df_u.iloc[i, df_u.columns.get_loc("contratista")] = r.get("nom_razon_social_contratista", "")
        df_u.iloc[i, df_u.columns.get_loc("objeto_texto")] = r.get("detalle_del_objeto_a_contratar", "") or r.get("objeto_a_contratar", "")
        df_u.iloc[i, df_u.columns.get_loc("estado")] = r.get("estado_del_proceso", "")
        df_u.iloc[i, df_u.columns.get_loc("fecha_firma")] = r.get("fecha_de_firma_del_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("fecha_inicio")] = r.get("fecha_ini_ejec_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("fecha_fin")] = r.get("fecha_fin_ejec_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("postconflicto")] = r.get("es_postconflicto", 0)
        df_u.iloc[i, df_u.columns.get_loc("pilar_paz")] = r.get("pilar_acuerdo_paz", "")
        df_u.iloc[i, df_u.columns.get_loc("punto_paz")] = r.get("punto_acuerdo_paz", "")
        df_u.iloc[i, df_u.columns.get_loc("codigo_bpin")] = r.get("codigo_bpin", "")
        df_u.iloc[i, df_u.columns.get_loc("numero_proceso")] = r.get("numero_de_proceso", "")
        df_u.iloc[i, df_u.columns.get_loc("numero_contrato")] = r.get("numero_de_contrato", "")
    else:
        url_raw = r.get("urlproceso", "")
        df_u.iloc[i, df_u.columns.get_loc("url")] = extraer_url(url_raw)
        df_u.iloc[i, df_u.columns.get_loc("nombre_entidad")] = r.get("nombre_de_la_entidad", "") or r.get("nombre_entidad", "")
        df_u.iloc[i, df_u.columns.get_loc("departamento")] = r.get("departamento", "")
        df_u.iloc[i, df_u.columns.get_loc("municipio")] = r.get("ciudad", "")
        df_u.iloc[i, df_u.columns.get_loc("contratista")] = r.get("proveedor_adjudicado", "")
        df_u.iloc[i, df_u.columns.get_loc("objeto_texto")] = r.get("objeto_del_contrato", "") or r.get("descripcion_del_proceso", "")
        df_u.iloc[i, df_u.columns.get_loc("estado")] = r.get("estado_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("fecha_inicio")] = r.get("fecha_de_inicio_del_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("fecha_fin")] = r.get("fecha_de_fin_del_contrato", "")
        df_u.iloc[i, df_u.columns.get_loc("postconflicto")] = 0

# Numeric columns
df_u["cuantia_contrato"] = pd.to_numeric(
    df_u.apply(lambda r: r.get("cuantia_contrato") if r["fuente"] == "SECOP I" else r.get("valor_del_contrato") or 0, axis=1),
    errors="coerce"
)

df_u["valor_final"] = pd.to_numeric(
    df_u.apply(lambda r: r.get("valor_contrato_con_adiciones") if r["fuente"] == "SECOP I" else r.get("valor_pagado") or 0, axis=1),
    errors="coerce"
)

df_u["adiciones_valor"] = pd.to_numeric(df_u.apply(lambda r: r.get("valor_total_de_adiciones") or 0, axis=1), errors="coerce")
df_u["adiciones_dias"] = pd.to_numeric(
    df_u.apply(lambda r: r.get("tiempo_adiciones_en_dias") if r["fuente"] == "SECOP I" else r.get("dias_adicionados") or 0, axis=1),
    errors="coerce"
)
df_u["plazo_dias"] = pd.to_numeric(df_u.apply(lambda r: r.get("plazo_de_ejec_del_contrato") or 0, axis=1), errors="coerce")

# ---- SOBRECOSTO ----
mask = (df_u["cuantia_contrato"] > 0) & (df_u["valor_final"] > 0)
df_u["sobrecosto_pct"] = np.nan
df_u.loc[mask, "sobrecosto_pct"] = ((df_u.loc[mask, "valor_final"] - df_u.loc[mask, "cuantia_contrato"]) / df_u.loc[mask, "cuantia_contrato"]) * 100

# Time overrun
mask_t = df_u["plazo_dias"] > 0
df_u["retraso_pct"] = np.nan
df_u.loc[mask_t, "retraso_pct"] = (df_u.loc[mask_t, "adiciones_dias"] / df_u.loc[mask_t, "plazo_dias"]) * 100

# ---- FILTRO >= $500M ----
antes = len(df_u)
df_u = df_u[df_u["cuantia_contrato"] >= 500e6].copy()
print(f"Filtro >= $500M: {antes} -> {len(df_u)}")

# ---- FILTRO INVERSION ----
antes = len(df_u)
df_u = df_u[df_u["destino_gasto"] == "Inversión"].copy()
print(f"Filtro Inversión: {antes} -> {len(df_u)}")

# Sort by value
df_u = df_u.sort_values("cuantia_contrato", ascending=False).reset_index(drop=True)

# ---- EXPORTAR ----
cols_salida = [
    "fuente", "nombre_entidad", "departamento", "municipio",
    "cuantia_contrato", "valor_final", "adiciones_valor", "adiciones_dias", "plazo_dias",
    "sobrecosto_pct", "retraso_pct",
    "destino_gasto", "estado", "objeto_texto",
    "url", "contratista",
    "fecha_firma", "fecha_inicio", "fecha_fin",
    "postconflicto", "pilar_paz", "punto_paz",
    "codigo_bpin", "numero_proceso", "numero_contrato",
]

archivo = os.path.join(CARPETA, "proyectos_unificados.csv")
df_u[cols_salida].to_csv(archivo, index=False, encoding="utf-8-sig")
print(f"\nGuardado: {archivo}")
print(f"  {len(df_u)} registros, {len(cols_salida)} columnas")

# Also filtered
con_sc = df_u[df_u["sobrecosto_pct"].notna() & (df_u["sobrecosto_pct"] != 0)].copy()
archivo2 = os.path.join(CARPETA, "proyectos_con_sobrecosto.csv")
con_sc.to_csv(archivo2, index=False, encoding="utf-8-sig")
print(f"Guardado: {archivo2} ({len(con_sc)} registros)")

# ---- SUMMARY ----
print(f"\n{'=' * 60}")
print("RESUMEN FINAL")
print(f"{'=' * 60}")
print(f"Total: {len(df_u)}")
print(f"  SECOP I: {(df_u['fuente']=='SECOP I').sum()}")
print(f"  SECOP II: {(df_u['fuente']=='SECOP II').sum()}")
print(f"  Con URL: {df_u['url'].notna().sum()}")
print(f"  Sobrecosto != 0: {(df_u['sobrecosto_pct'] != 0).sum()}")
print(f"  Sobrecosto = 0%: {(df_u['sobrecosto_pct'] == 0).sum()}")
print(f"  Sin data: {df_u['sobrecosto_pct'].isna().sum()}")

s = df_u["sobrecosto_pct"].dropna()
if len(s) > 0:
    print(f"\n  Media: {s.mean():.1f}%")
    print(f"  Mediana: {s.median():.1f}%")
    print(f"  >0: {(s > 0).sum()}  <0: {(s < 0).sum()}")
