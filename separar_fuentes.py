"""
SEPARAR DATASET DEPURADO POR FUENTE
Lee proyectos_depurados.csv y crea dos CSVs:
  - proyectos_secop1.csv (solo SECOP I, sin duplicados por URL)
  - proyectos_secop2.csv (solo SECOP II)
"""
import pandas as pd, os, sys
sys.stdout.reconfigure(encoding="utf-8")

CARPETA = "contratos"
archivo = os.path.join(CARPETA, "proyectos_depurados.csv")

print("=" * 60)
print("SEPARAR POR FUENTE: SECOP I vs SECOP II")
print("=" * 60)

df = pd.read_csv(archivo, encoding="utf-8-sig", low_memory=False)
print(f"Total registros: {len(df)}")

s1 = df[df["fuente"] == "SECOP I"].copy()
s2 = df[df["fuente"] == "SECOP II"].copy()

# Eliminar duplicados por URL en SECOP I
antes = len(s1)
s1 = s1.drop_duplicates(subset="url", keep="first")
print(f"\nSECOP I: {antes} -> {len(s1)} (eliminados {antes - len(s1)} duplicados por URL)")

print(f"SECOP II: {len(s2)} registros")

s1.to_csv(os.path.join(CARPETA, "proyectos_secop1.csv"), index=False, encoding="utf-8-sig")
s2.to_csv(os.path.join(CARPETA, "proyectos_secop2.csv"), index=False, encoding="utf-8-sig")

print(f"\nGuardado: contratos/proyectos_secop1.csv")
print(f"Guardado: contratos/proyectos_secop2.csv")
