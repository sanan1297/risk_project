"""
EXCEL LITE - Version reducida para extraccion manual de datos
Columnas esenciales: entidad, url (primero), valor_inicial, valor_final, sobrecosto_pct
SECOP I: solo registros con sobrecosto_pct > 0
SECOP II: todos los registros
"""
import pandas as pd, os, sys
sys.stdout.reconfigure(encoding="utf-8")

CARPETA = "contratos"
print("=" * 60)
print("EXCEL LITE - Version reducida")
print("=" * 60)

cols_lite = ["entidad", "url", "departamento", "municipio",
             "valor_inicial", "valor_final", "sobrecosto_pct",
             "objeto", "contratista", "estado"]

for fuente, archivo_in, archivo_out, solo_con_sc in [
    ("SECOP I", "proyectos_secop1.csv", "proyectos_secop1_lite.csv", True),
]:
    df = pd.read_csv(os.path.join(CARPETA, archivo_in), encoding="utf-8-sig", low_memory=False)

    antes = len(df)
    if solo_con_sc:
        df = df[df["sobrecosto_pct"] > 0].copy()

    disponibles = [c for c in cols_lite if c in df.columns]
    df_out = df[disponibles].copy()
    df_out.to_csv(os.path.join(CARPETA, archivo_out), index=False, encoding="utf-8-sig")
    print(f"\n{fuente}:")
    print(f"  Leidos: {antes}")
    if solo_con_sc:
        print(f"  Con sobrecosto_pct: {len(df)}")
    print(f"  Columnas: {disponibles}")
    print(f"  Guardado: contratos/{archivo_out}")
