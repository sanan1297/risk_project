"""
Calcula ipc_acumulado (compuesto) y trm_promedio por rango de fechas
para los 351 contratos y genera contrato_features_con_rango.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs"

IPC_TRM = {
    2000: {"ipc": 8.75, "trm": 2052}, 2001: {"ipc": 7.65, "trm": 2200},
    2002: {"ipc": 6.99, "trm": 2504}, 2003: {"ipc": 6.49, "trm": 2878},
    2004: {"ipc": 5.50, "trm": 2628}, 2005: {"ipc": 4.85, "trm": 2322},
    2006: {"ipc": 4.48, "trm": 2358}, 2007: {"ipc": 5.69, "trm": 2014},
    2008: {"ipc": 7.67, "trm": 1973}, 2009: {"ipc": 2.00, "trm": 2047},
    2010: {"ipc": 3.17, "trm": 1898}, 2011: {"ipc": 3.73, "trm": 1848},
    2012: {"ipc": 2.44, "trm": 1798}, 2013: {"ipc": 1.94, "trm": 1887},
    2014: {"ipc": 3.66, "trm": 2020}, 2015: {"ipc": 6.77, "trm": 2742},
    2016: {"ipc": 5.75, "trm": 3055}, 2017: {"ipc": 4.09, "trm": 2951.32},
    2018: {"ipc": 3.18, "trm": 2956.55}, 2019: {"ipc": 3.80, "trm": 3281.09},
    2020: {"ipc": 1.61, "trm": 3693.36}, 2021: {"ipc": 5.62, "trm": 3743.09},
    2022: {"ipc": 13.12, "trm": 4255.44}, 2023: {"ipc": 9.28, "trm": 4325.05},
    2024: {"ipc": 5.20, "trm": 4071.28}, 2025: {"ipc": 5.10, "trm": 4052.86},
    2026: {"ipc": 6.40, "trm": 4200}, 2027: {"ipc": 4.80, "trm": 4100},
}

MAX_DURACION = 5


def compute_range_features(anio_inicio: int, anio_fin: int) -> dict:
    if anio_fin < anio_inicio:
        anio_fin = anio_inicio
    duracion = anio_fin - anio_inicio
    if duracion > MAX_DURACION:
        anio_fin = anio_inicio + MAX_DURACION
        duracion = MAX_DURACION
    ipc_acumulado = 1.0
    trm_values = []
    for y in range(anio_inicio, anio_fin + 1):
        d = IPC_TRM.get(y, {"ipc": 3.0, "trm": 4000})
        ipc_acumulado *= (1 + d["ipc"] / 100)
        trm_values.append(d["trm"])
    ipc_acumulado = (ipc_acumulado - 1) * 100
    trm_promedio = np.mean(trm_values)
    return {
        "anio_inicio": anio_inicio,
        "anio_fin": anio_fin,
        "duracion": duracion,
        "ipc_acumulado": round(ipc_acumulado, 2),
        "trm_promedio": round(trm_promedio, 2),
    }


def main():
    print("=" * 55)
    print("  COMPUTE IPC RANGE — Features por rango de fechas")
    print("=" * 55)

    print("\n1. Cargando matriz_clean.csv...")
    matriz = pd.read_csv(DATA_DIR / "matriz_clean.csv", encoding="utf-8-sig")
    contratos_ids = sorted(matriz["id_contrato"].unique())
    print(f"   Contratos únicos: {len(contratos_ids)}")

    print("\n2. Cargando contratos_anios.csv...")
    anios = pd.read_csv(DATA_DIR / "contratos_anios.csv")
    anios_dict = {}
    for _, r in anios.iterrows():
        cid = r["id_contrato"]
        if cid not in anios_dict:
            anios_dict[cid] = {
                "anio_raw": r["anio"],
                "fecha_ini": r["fecha_ini_ejec_contrato"],
                "fecha_fin": r["fecha_fin_ejec_contrato"],
            }

    print("\n3. Cargando contratos_features.csv (para anio fallback)...")
    feats = pd.read_csv(DATA_DIR / "contratos_features.csv")
    anio_fallback = dict(zip(feats["id_contrato"], feats["anio"]))

    end_dates_usuario = {
        "C-110": "2022-05-07",
        "C-111": "2025-07-01",
        "C-112": "2024-12-27",
        "C-113": "2023-08-05",
        "C-114": "2022-12-30",
    }

    rows = []
    stats = {"con_rango": 0, "sin_rango": 0, "con_fin_usuario": 0}
    for cid in contratos_ids:
        info = anios_dict.get(cid, {})
        fecha_ini = info.get("fecha_ini", "") if info else ""
        fecha_fin = info.get("fecha_fin", "") if info else ""

        if fecha_ini and pd.notna(fecha_ini) and str(fecha_ini).strip():
            anio_ini = int(str(fecha_ini).split("-")[0])
        else:
            anio_ini = int(anio_fallback.get(cid, 2022))

        if cid in end_dates_usuario:
            fecha_fin = end_dates_usuario[cid]
            anio_fin = int(fecha_fin.split("-")[0])
            stats["con_fin_usuario"] += 1
        elif fecha_fin and pd.notna(fecha_fin) and str(fecha_fin).strip():
            anio_fin = int(str(fecha_fin).split("-")[0])
        else:
            anio_fin = anio_ini

        if anio_fin > anio_ini:
            stats["con_rango"] += 1
        else:
            stats["sin_rango"] += 1

        feat = compute_range_features(anio_ini, anio_fin)
        feat["id_contrato"] = cid
        rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"\n   Con rango (>1 año): {stats['con_rango']}")
    print(f"   Sin rango (1 año): {stats['sin_rango']}")
    print(f"   Con fecha fin usuario: {stats['con_fin_usuario']}")

    out_path = DATA_DIR / "contratos_macro.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n4. Guardado: {out_path}")
    print(f"   Head:")
    print(df.head(10).to_string())

    print("\n5. Verificación C-110 a C-114 + C-053:")
    check = df[df["id_contrato"].isin(["C-053", "C-110", "C-111", "C-112", "C-113", "C-114"])]
    print(check.to_string())

    print("\nListo.")


if __name__ == "__main__":
    main()
