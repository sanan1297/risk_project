import pandas as pd
import numpy as np

CARPETA = "contratos"

# --- SECOP I ---
s1 = pd.read_csv(f"{CARPETA}/secop1_cache.csv", encoding="utf-8-sig", low_memory=False)
col_fecha = "fecha_ini_ejec_contrato"
s1[col_fecha] = pd.to_datetime(s1[col_fecha], errors="coerce")

# The raw data has "Inversion" (no accent), normalize
s1["destino_gasto_norm"] = s1["destino_gasto"].str.strip().str.lower()

m_obra = s1["tipo_de_contrato"] == "Obra"
m_2025 = s1[col_fecha].dt.year >= 2025
m_valor = s1["cuantia_contrato"] >= 500e6
m_url = s1["ruta_proceso_en_secop_i"].notna() & (s1["ruta_proceso_en_secop_i"] != "")
m_inv = s1["destino_gasto_norm"] == "inversion"

mask = m_obra & m_2025 & m_valor & m_url & m_inv
print(f"SECOP I Obra + Inversion + >=$500M + 2025+ + URL: {mask.sum()}")
if mask.any():
    df = s1[mask].sort_values("cuantia_contrato", ascending=False)
    for _, r in df.iterrows():
        print(f"  ${r['cuantia_contrato']:>14,.0f} | {r[col_fecha].date()} | {str(r['objeto_a_contratar'])[:80]}")
        print(f"    Estado: {r['estado_del_proceso']} | URL: {str(r['ruta_proceso_en_secop_i'])[:100]}")

# --- SECOP II ---
s2 = pd.read_csv(f"{CARPETA}/secop2_cache.csv", encoding="utf-8-sig", low_memory=False)
col_fecha2 = "fecha_de_inicio_del_contrato"
s2[col_fecha2] = pd.to_datetime(s2[col_fecha2], errors="coerce")

# Normalize destino_gasto
s2["destino_gasto_norm"] = s2["destino_gasto"].str.strip().str.lower()

m2_2025 = s2[col_fecha2].dt.year >= 2025
m2_valor = s2["valor_del_contrato"] >= 500e6
m2_url = s2["urlproceso"].notna() & (s2["urlproceso"] != "")
m2_inv = s2["destino_gasto_norm"] == "inversión"
# States that are NOT finished
m2_no_terminado = ~s2["estado_contrato"].str.strip().str.lower().isin(["terminado", "cerrado"])

print(f"\n--- SECOP II: Contratos 2025+ NO terminados ---")
mask = m2_2025 & m2_valor & m2_url & m2_inv & m2_no_terminado
print(f"Inversión + >=$500M + 2025+ + URL + NO terminado: {mask.sum()}")
if mask.any():
    df2 = s2[mask].sort_values("valor_del_contrato", ascending=False)
    for _, r in df2.iterrows():
        print(f"\n  ${r['valor_del_contrato']:>14,.0f} | {r[col_fecha2].date()}")
        print(f"  Estado: {r['estado_contrato']}")
        print(f"  Objeto: {str(r['objeto_del_contrato'])[:120]}")
        print(f"  URL: {str(r.get('urlproceso', ''))[:100]}")

# Also show available states for 2025+ contracts
print(f"\nEstados disponibles en SECOP II 2025+ (>= $500M):")
mask = m2_2025 & m2_valor
print(s2[mask]["estado_contrato"].value_counts().to_string())

# Show top 10 biggest 2025+ contracts with all estados
print(f"\nTop 10 contratos SECOP II 2025+ >= $500M:")
mask = m2_2025 & m2_valor & m2_url
df2 = s2[mask].sort_values("valor_del_contrato", ascending=False).head(10)
for _, r in df2.iterrows():
    print(f"  ${r['valor_del_contrato']:>14,.0f} | Estado: {r['estado_contrato']:30s} | {r[col_fecha2].date()} | {str(r['objeto_del_contrato'])[:80]}")
