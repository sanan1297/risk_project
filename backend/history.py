import os
import sqlite3
from pathlib import Path
from datetime import datetime
import json
from collections import Counter

DB_PATH = Path(os.environ.get("HISTORY_DB_PATH", str(Path(__file__).resolve().parent / "history.db")))


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            id_contrato TEXT NOT NULL,
            n_riesgos INTEGER NOT NULL,
            anio INTEGER,
            ipc REAL,
            trm REAL,
            anio_inicio INTEGER,
            anio_fin INTEGER,
            duracion INTEGER,
            ipc_acumulado REAL,
            trm_promedio REAL,
            prediccion_ridge REAL NOT NULL,
            probabilidad_alto_riesgo REAL NOT NULL,
            alerta TEXT NOT NULL,
            sobrecosto_real REAL,
            notas TEXT,
            factores_aumentan TEXT,
            factores_disminuyen TEXT,
            mc_iteraciones INTEGER
        )
    """)
    for col in ["mc_iteraciones", "resultado_json", "anio_inicio", "anio_fin", "duracion", "ipc_acumulado", "trm_promedio"]:
        try:
            conn.execute(f"ALTER TABLE predicciones ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def guardar(
    id_contrato: str,
    n_riesgos: int,
    anio_inicio: int | None = None,
    anio_fin: int | None = None,
    ipc_override: float | None = None,
    trm_override: float | None = None,
    ipc_acumulado: float | None = None,
    trm_promedio: float | None = None,
    anio: int | None = None,
    ipc: float | None = None,
    trm: float | None = None,
    prediccion_ridge: float = 0,
    probabilidad_alto_riesgo: float = 0,
    alerta: str = "",
    factores_aumentan: list[dict] | None = None,
    factores_disminuyen: list[dict] | None = None,
    sobrecosto_real: float | None = None,
    notas: str | None = None,
    mc_iteraciones: int | None = None,
) -> int:
    conn = _get_conn()
    duracion = (anio_fin - anio_inicio) if (anio_inicio and anio_fin) else None
    cur = conn.execute(
        """INSERT INTO predicciones
        (created_at, id_contrato, n_riesgos, anio, ipc, trm,
         anio_inicio, anio_fin, duracion, ipc_acumulado, trm_promedio,
         prediccion_ridge, probabilidad_alto_riesgo, alerta,
         sobrecosto_real, notas, factores_aumentan, factores_disminuyen,
         mc_iteraciones)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            id_contrato,
            n_riesgos,
            anio,
            ipc,
            trm,
            anio_inicio,
            anio_fin,
            duracion,
            ipc_acumulado,
            trm_promedio,
            round(prediccion_ridge, 2),
            round(probabilidad_alto_riesgo, 4),
            alerta,
            sobrecosto_real,
            notas,
            json.dumps(factores_aumentan, ensure_ascii=False) if factores_aumentan else None,
            json.dumps(factores_disminuyen, ensure_ascii=False) if factores_disminuyen else None,
            mc_iteraciones,
        ),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def actualizar_mc(pred_id: int, mc_iteraciones: int) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE predicciones SET mc_iteraciones = ? WHERE id = ?",
        (mc_iteraciones, pred_id),
    )
    conn.commit()
    conn.close()


def guardar_resultado_completo(pred_id: int, resultado: dict) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE predicciones SET resultado_json = ? WHERE id = ?",
        (json.dumps(resultado, ensure_ascii=False), pred_id),
    )
    conn.commit()
    conn.close()


def obtener_resultado_completo(pred_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT resultado_json FROM predicciones WHERE id = ?", (pred_id,)
    ).fetchone()
    conn.close()
    if row and row["resultado_json"]:
        return json.loads(row["resultado_json"])
    return None


def listar_paginado(page: int = 1, page_size: int = 20) -> dict:
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) as cnt FROM predicciones").fetchone()["cnt"]
    offset = (page - 1) * page_size
    rows = conn.execute(
        "SELECT * FROM predicciones ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, offset),
    ).fetchall()
    conn.close()
    data = []
    for r in rows:
        d = dict(r)
        d.pop("resultado_json", None)
        data.append(d)
    return {
        "data": data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "paginas": max(1, (total + page_size - 1) // page_size),
    }


def actualizar_sobrecosto_real(pred_id: int, valor: float, notas: str | None = None) -> None:
    conn = _get_conn()
    if notas:
        conn.execute(
            "UPDATE predicciones SET sobrecosto_real = ?, notas = ? WHERE id = ?",
            (valor, notas, pred_id),
        )
    else:
        conn.execute(
            "UPDATE predicciones SET sobrecosto_real = ? WHERE id = ?",
            (valor, pred_id),
        )
    conn.commit()
    conn.close()


def stats() -> dict:
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as total_predicciones,
            COUNT(DISTINCT id_contrato) as contratos_unicos,
            COALESCE(SUM(n_riesgos), 0) as riesgos_totales,
            AVG(CASE WHEN alerta LIKE '%ALTO%' THEN 1.0 ELSE 0 END) as porcentaje_alto_riesgo,
            AVG(prediccion_ridge) as sobrecosto_promedio
        FROM predicciones
    """).fetchone()

    alertas = conn.execute(
        "SELECT alerta, COUNT(*) as cnt FROM predicciones GROUP BY alerta"
    ).fetchall()

    temporal = conn.execute("""
        SELECT DATE(created_at) as fecha,
               COUNT(*) as cantidad,
               ROUND(AVG(prediccion_ridge), 1) as promedio
        FROM predicciones
        GROUP BY DATE(created_at)
        ORDER BY fecha
    """).fetchall()

    reales = conn.execute("""
        SELECT prediccion_ridge as predicho, sobrecosto_real as real
        FROM predicciones
        WHERE sobrecosto_real IS NOT NULL
    """).fetchall()

    riesgos_dist = conn.execute("""
        SELECT
            CASE
                WHEN n_riesgos <= 5 THEN '0-5'
                WHEN n_riesgos <= 10 THEN '6-10'
                WHEN n_riesgos <= 20 THEN '11-20'
                ELSE '21+'
            END as rango,
            COUNT(*) as cantidad
        FROM predicciones
        GROUP BY rango
    """).fetchall()

    conn.close()

    # Parse top factors from JSON stored in DB
    conn2 = _get_conn()
    fa_rows = conn2.execute(
        "SELECT factores_aumentan FROM predicciones WHERE factores_aumentan IS NOT NULL"
    ).fetchall()
    fd_rows = conn2.execute(
        "SELECT factores_disminuyen FROM predicciones WHERE factores_disminuyen IS NOT NULL"
    ).fetchall()
    mc_row = conn2.execute(
        "SELECT COUNT(*) as con_mc, COALESCE(SUM(mc_iteraciones), 0) as total_iter FROM predicciones WHERE mc_iteraciones IS NOT NULL"
    ).fetchone()

    counter = Counter()
    for r in fa_rows:
        try:
            for f in json.loads(r["factores_aumentan"]):
                counter[f["label"]] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for r in fd_rows:
        try:
            for f in json.loads(r["factores_disminuyen"]):
                counter[f["label"]] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    top_factores = [{"label": k, "apariciones": v} for k, v in counter.most_common(10)]

    conn2.close()

    return {
        "total_predicciones": row["total_predicciones"],
        "contratos_unicos": row["contratos_unicos"],
        "riesgos_totales": row["riesgos_totales"],
        "porcentaje_alto_riesgo": round(row["porcentaje_alto_riesgo"] or 0, 4),
        "sobrecosto_promedio": round(row["sobrecosto_promedio"] or 0, 1),
        "serie_temporal": [dict(r) for r in temporal],
        "distribucion_alertas": {r["alerta"]: r["cnt"] for r in alertas},
        "top_factores": top_factores,
        "histograma_riesgos": [dict(r) for r in riesgos_dist],
        "predicciones_vs_reales": [dict(r) for r in reales],
        "mc_predicciones": mc_row["con_mc"],
        "mc_iteraciones_totales": mc_row["total_iter"],
    }


def obtener_por_id(pred_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM predicciones WHERE id = ?", (pred_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    resultado_raw = d.pop("resultado_json", None)
    d["resultado_json"] = json.loads(resultado_raw) if resultado_raw else None
    for col in ("factores_aumentan", "factores_disminuyen"):
        if d.get(col):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                d[col] = []
    return d


def eliminar(pred_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM predicciones WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()


def eliminar_todos() -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM predicciones")
    conn.commit()
    conn.close()


init()
