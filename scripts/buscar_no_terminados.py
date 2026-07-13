import pandas as pd
import numpy as np

CARPETA = "contratos"

s1 = pd.read_csv(f"{CARPETA}/secop1_cache.csv", encoding="utf-8-sig", low_memory=False)
s2 = pd.read_csv(f"{CARPETA}/secop2_cache.csv", encoding="utf-8-sig", low_memory=False)

col_fecha_inicio = "fecha_ini_ejec_contrato"
col_fecha_fin = "fecha_fin_ejec_contrato"
s1[col_fecha_inicio] = pd.to_datetime(s1[col_fecha_inicio], errors="coerce")
s1[col_fecha_fin] = pd.to_datetime(s1[col_fecha_fin], errors="coerce")
s1["dg_norm"] = s1["destino_gasto"].str.strip().str.lower()

col_fecha_inicio2 = "fecha_de_inicio_del_contrato"
col_fecha_fin2 = "fecha_de_fin_del_contrato"
s2[col_fecha_inicio2] = pd.to_datetime(s2[col_fecha_inicio2], errors="coerce")
s2[col_fecha_fin2] = pd.to_datetime(s2[col_fecha_fin2], errors="coerce")
s2["dg_norm"] = s2["destino_gasto"].str.strip().str.lower()

# =====================
# SECOP I: ¿Hay ALGÚN contrato NO Liquidado?
# =====================
print("=== SECOP I: estados únicos (cualquier valor) ===")
print(s1["estado_del_proceso"].value_counts().head(20).to_string())

excluir = ["liquidado"]
m_no_liq = ~s1["estado_del_proceso"].str.strip().str.lower().isin(excluir)
m_obra = s1["tipo_de_contrato"] == "Obra"
m_inv = s1["dg_norm"] == "inversion"

print(f"\nObra + Inversión + NO Liquidado: {((m_no_liq & m_obra & m_inv).sum())}")
print(f"Obra + NO Liquidado (sin Inversión): {((m_no_liq & m_obra).sum())}")

if (m_no_liq & m_obra).any():
    df = s1[m_no_liq & m_obra].sort_values("cuantia_contrato", ascending=False)
    print("\nTop 20 Obra NO Liquidados (cualquier valor):")
    for _, r in df.head(20).iterrows():
        ini = r[col_fecha_inicio].date() if pd.notna(r[col_fecha_inicio]) else "?"
        fin = r[col_fecha_fin].date() if pd.notna(r[col_fecha_fin]) else "?"
        print(f"  ${r['cuantia_contrato']:>14,.0f} | {ini} → {fin} | {r['estado_del_proceso']:25s} | {r['destino_gasto']:25s} | {str(r['objeto_a_contratar'])[:60]}")

# =====================
# SECOP II: ¿Hay ALGÚN contrato NO terminado/Cerrado?
# =====================
print("\n\n=== SECOP II: estados únicos (cualquier valor) ===")
print(s2["estado_contrato"].value_counts().head(20).to_string())

excluir2 = ["terminado", "cerrado"]
m_no_term = ~s2["estado_contrato"].str.strip().str.lower().isin(excluir2)

print(f"\n>= $0 + NO terminado/Cerrado: {m_no_term.sum()}")
print(f">= $500M + NO terminado/Cerrado: {(m_no_term & (s2['valor_del_contrato'] >= 500e6)).sum()}")
print(f">= $100M + NO terminado/Cerrado: {(m_no_term & (s2['valor_del_contrato'] >= 100e6)).sum()}")

# =====================
# ¿Qué contratos NO terminados existen en SECOP II?
# =====================
if m_no_term.any():
    df2 = s2[m_no_term].sort_values("valor_del_contrato", ascending=False)
    print(f"\nTop 50 SECOP II NO terminados:")
    for i, (_, r) in enumerate(df2.iterrows()):
        if i >= 50:
            break
        ini = r[col_fecha_inicio2].date() if pd.notna(r[col_fecha_inicio2]) else "?"
        fin = r[col_fecha_fin2].date() if pd.notna(r[col_fecha_fin2]) else "?"
        print(f"  ${r['valor_del_contrato']:>14,.0f} | {ini} → {fin} | {r['estado_contrato']:25s} | {r['destino_gasto']:25s} | {str(r['objeto_del_contrato'])[:70]}")
else:
    print("\nNo hay ningún contrato NO terminado en SECOP II cache.")
    print("Todos los contratos en cache ya están finalizados.")

print("\n\n=== Conclusión ===")
print("Los caches disponibles solo tienen contratos finalizados (Liquidado/terminado/Cerrado).")
print("No hay contratos 'en ejecución' en la data descargada.")
