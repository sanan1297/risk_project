import sqlite3
from pathlib import Path
from datetime import datetime
import json
from collections import Counter

DB_PATH = Path(__file__).resolve().parent / "history.db"


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
            prediccion_ridge REAL NOT NULL,
            probabilidad_alto_riesgo REAL NOT NULL,
            alerta TEXT NOT NULL,
            sobrecosto_real REAL,
            notas TEXT,
            factores_aumentan TEXT,
            factores_disminuyen TEXT
        )
    """)
    conn.commit()
    conn.close()


def guardar(
    id_contrato: str,
    n_riesgos: int,
    anio: int | None,
    ipc: float | None,
    trm: float | None,
    prediccion_ridge: float,
    probabilidad_alto_riesgo: float,
    alerta: str,
    factores_aumentan: list[dict],
    factores_disminuyen: list[dict],
    sobrecosto_real: float | None = None,
    notas: str | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO predicciones
        (created_at, id_contrato, n_riesgos, anio, ipc, trm,
         prediccion_ridge, probabilidad_alto_riesgo, alerta,
         sobrecosto_real, notas, factores_aumentan, factores_disminuyen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            id_contrato,
            n_riesgos,
            anio,
            ipc,
            trm,
            round(prediccion_ridge, 2),
            round(probabilidad_alto_riesgo, 4),
            alerta,
            sobrecosto_real,
            notas,
            json.dumps(factores_aumentan, ensure_ascii=False),
            json.dumps(factores_disminuyen, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def listar_paginado(page: int = 1, page_size: int = 20) -> dict:
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) as cnt FROM predicciones").fetchone()["cnt"]
    offset = (page - 1) * page_size
    rows = conn.execute(
        "SELECT * FROM predicciones ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, offset),
    ).fetchall()
    conn.close()
    return {
        "data": [dict(r) for r in rows],
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
    conn2.close()

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
    }


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
