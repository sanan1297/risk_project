import streamlit as st
import pandas as pd
import requests
import io
import plotly.graph_objects as go
import plotly.io as pio

# Configurar tema global para que el texto sea negro (para modo claro)
pio.templates.default = "plotly_white"

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Risk Control Dashboard",
    page_icon=":material/analytics:",
    layout="wide",
)

DARK = st.toggle(":material/dark_mode: Dark Mode", key="dark_mode", value=False)
st.space("small")

BG_COLOR = "#0F172A" if DARK else "#F4F7FD"
CARD_BG = "#1E293B" if DARK else "#FFFFFF"
TEXT_COLOR = "#F1F5F9" if DARK else "#1E293B"
MUTED = "#64748B"
BORDER_COLOR = "#334155" if DARK else "#E2E8F0"

AZUL = "#4facfe"
MORADO = "#7B5CE4"
VERDE = "#1ABC9C"
NARANJA = "#F39C12"

view = st.query_params.get("view", "dashboard")

st.markdown('<style>'
    '[role="dialog"]{background:#F4F7FD!important}'
    '[role="dialog"] [data-testid="stVerticalBlock"]{background:transparent!important}'
    '</style>', unsafe_allow_html=True)

st.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  * {{ font-family: 'Inter', sans-serif; }}
  #root > div:first-child, .main, .block-container {{ background: {BG_COLOR}; }}
  .main, .block-container {{ padding: 1rem 1.5rem !important; max-width: 100% !important; }}

  .header-container {{
    display: flex; justify-content: space-between; align-items: center;
    background-color: #90D5FF; padding: 16px 40px; width: 100%;
    box-sizing: border-box; font-family: sans-serif; border-radius: 12px;
    margin-bottom: 1.25rem;
  }}
  .nav-menu {{ display: flex; gap: 40px; align-items: center; }}
  .nav-menu a {{
    text-decoration: none; color: #1F2937; font-weight: 500;
    font-size: 16px; display: flex; align-items: center; gap: 4px;
  }}
  .nav-menu a:hover {{ opacity: 0.7; }}
  .header-btn {{
    background-color: white; border: 1px solid #1F2937;
    padding: 8px 16px; border-radius: 6px; color: #1F2937;
    text-decoration: none; font-weight: 500; font-size: 14px;
  }}
  .header-btn:hover {{ background-color: #f0f0f0; }}

  .kpi-row {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
    margin-bottom: 1.25rem;
  }}
  .kpi-card {{
    border-radius: 20px; padding: 1.25rem 1.5rem;
    color: white; position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }}
  .kpi-title {{ font-size: 0.8rem; font-weight: 500; opacity: 0.85; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 2rem; font-weight: 800; line-height: 1.2; }}
  .kpi-badge {{ display: inline-block; margin-top: 0.4rem; padding: 0.2rem 0.7rem; border-radius: 100px; font-size: 0.75rem; font-weight: 600; }}

  .chart-card {{
    background: {CARD_BG}; border-radius: 20px; padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    border: 1px solid {BORDER_COLOR};
    margin-bottom: 1.25rem;
  }}

  .chart-title {{ font-size: 1rem; font-weight: 600; color: {TEXT_COLOR}; margin-bottom: 0.25rem; }}
  .chart-subtitle {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: 0.75rem; }}

  .pill-group {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; }}
  .pill-stat {{
    padding: 0.35rem 1rem; border-radius: 100px; font-size: 0.8rem; font-weight: 500;
    border: 1px solid {BORDER_COLOR};
    background: {CARD_BG}; color: {TEXT_COLOR};
  }}
  .pill-stat.active {{ background: {AZUL}; color: white; border-color: {AZUL}; }}
  .pill-stat .amount {{ font-weight: 700; }}

  .pred-card {{
    background: {CARD_BG}; border-radius: 20px; padding: 1.5rem;
    border: 1px solid {BORDER_COLOR}; margin-bottom: 1.25rem;
  }}
  .pred-card h3 {{ color: {TEXT_COLOR}; font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; }}
  .risk-bar {{ transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }}

  .stFileUploader section {{ padding: 0.75rem; border: 1px dashed #CBD5E1; border-radius: 12px; background: #F8FAFC; }}
  .stTextArea textarea {{ background: #F8FAFC !important; color: #1E293B !important; border: 1px solid #CBD5E1 !important; border-radius: 12px !important; }}
  div[data-testid="stVerticalBlockBorder"] {{ border: 1px solid {BORDER_COLOR} !important; border-radius: 16px !important; background: {CARD_BG} !important; }}
  div[data-baseweb="select"] {{ background: #F8FAFC !important; border: 1px solid #CBD5E1 !important; border-radius: 10px !important; }}
  div[data-baseweb="select"] * {{ color: #1E293B !important; }}
  .stSelectbox label, .stSelectbox div[role="listbox"] {{ color: #1E293B !important; }}
  .stRadio label {{ color: #1E293B !important; font-weight: 500 !important; }}

  .st-key-procesar button {{
    padding: 0.25rem 1rem !important; height: auto !important; min-height: 0 !important;
    font-size: 0.85rem !important; border-radius: 100px !important;
    background: linear-gradient(135deg, {AZUL}, {VERDE}) !important; border: none !important;
  }}
  .st-key-clear_all button {{ padding: 0.15rem 0.75rem !important; height: auto !important; min-height: 0 !important; font-size: 0.8rem !important; border-radius: 100px !important; }}
  .st-key-del_ button {{ padding: 0.1rem 0.5rem !important; height: auto !important; min-height: 0 !important; font-size: 0.75rem !important; border-radius: 100px !important; }}

  .history-metric {{
    background: {CARD_BG}; border-radius: 8px; padding: 0.4rem 0.6rem;
    border: 1px solid {BORDER_COLOR}; text-align: center;
  }}
  .history-metric .label {{ font-size: 0.7rem; color: {MUTED}; }}
  .history-metric .value {{ font-size: 1rem; font-weight: 600; color: {TEXT_COLOR}; }}
  [class*="st-key-edit_"] button, [class*="st-key-del_"] button {{ margin-top: -12px !important; }}

  /* RESULTADOS: forzar texto oscuro en tarjetas de predicción */
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMarkdown h3,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMarkdown p,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMarkdown strong,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stCaption,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMetric,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMetric label,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMetric div[data-testid="stMetricValue"],
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stBadge,
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stBadge span {{
    color: #111827 !important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stMetric {{
    background: #FFFFFF !important; border: 1px solid #D1D5DB !important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] code {{
    background: #F3F4F6 !important; color: #111827 !important; padding: 2px 6px !important; border-radius: 4px !important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stBadge {{
    background: #F3F4F6 !important; color: #111827 !important; border: 1px solid #D1D5DB !important;
  }}

  /* === FORZAR MÉTRICAS Y TEXTOS EN RESULTADOS === */
  div[data-testid="stMetric"] {{
    background: #FFFFFF !important;
    border: 2px solid #374151 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
  }}
  div[data-testid="stMetric"] label {{
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
  }}
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 1.5rem !important;
  }}
  div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
    color: #059669 !important;
    font-weight: 600 !important;
  }}
  .stBadge {{
    background: #F3F4F6 !important;
    color: #111827 !important;
    border: 1px solid #374151 !important;
    font-weight: 600 !important;
  }}
  .stCaption {{
    color: #374151 !important;
    font-weight: 500 !important;
  }}

  /* === FORZAR TEXTO NEGRO EN CONTENEDORES DE RESULTADOS === */
  .stContainer > div[data-testid="stVerticalBlockBorder"] {{
    background: #FFFFFF !important; border: 1px solid #374151 !important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] * {{
    color: #111827;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] h3,
  .stContainer > div[data-testid="stVerticalBlockBorder"] h4,
  .stContainer > div[data-testid="stVerticalBlockBorder"] p,
  .stContainer > div[data-testid="stVerticalBlockBorder"] strong,
  .stContainer > div[data-testid="stVerticalBlockBorder"] code {{
    color: #111827 ! important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] code {{
    background: #F3F4F6 ! important; padding: 2px 6px ! important; border-radius: 4px ! important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stBadge {{
    background: #F3F4F6 ! important; color: #111827 ! important; border: 1px solid #374151 ! important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] .stCaption {{
    color: #374151 ! important;
  }}
  .stContainer > div[data-testid="stVerticalBlockBorder"] [data-testid="stMarkdownContainer"] * {{
    color: #111827 ! important;
  }}

  /* === TABS === */
  div[data-testid="stTabs"] {{
    background: transparent !important;
  }}
  div[data-testid="stTabs"] button[data-baseweb="tab"],
  div[data-testid="stTabs"] button[role="tab"] {{
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    background: transparent !important;
  }}
  div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
  div[data-testid="stTabs"] button[role="tab"]:hover {{
    color: #000000 !important;
    background: rgba(79,172,254,0.1) !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"],
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: #000000 !important;
    border-bottom: 2px solid #4facfe !important;
  }}
</style>
""")

# ─── HEADER ────────────────────────────────────────────
st.markdown("""
<style>
    .header-container { display: flex; justify-content: space-between; align-items: center; background-color: #90D5FF; padding: 16px 40px; width: 100%; box-sizing: border-box; font-family: sans-serif; border-radius: 12px; margin-bottom: 1.25rem; }
    .nav-menu { display: flex; gap: 40px; align-items: center; margin-right: 20px; }
    .nav-menu a { text-decoration: none; color: #1F2937; font-weight: 500; font-size: 16px; display: flex; align-items: center; gap: 4px; }
    .nav-menu a:hover { opacity: 0.7; }
</style>
<div class="header-container">
    <span style="font-weight:700;font-size:20px;color:#1F2937;">Risk Control</span>
    <div class="nav-menu">
        <a href="?view=dashboard" target="_self">Home</a>
        <a href="?view=predict" target="_self">Predicci&oacute;n</a>
        <a href="?view=history" target="_self">Historial</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────
RANGO_ANOS = list(range(2000, 2026))
IPC_TRM = {
    2000: {"ipc": 8.75, "trm": 2052.0}, 2001: {"ipc": 7.65, "trm": 2200.0},
    2002: {"ipc": 6.99, "trm": 2504.0}, 2003: {"ipc": 6.49, "trm": 2878.0},
    2004: {"ipc": 5.50, "trm": 2628.0}, 2005: {"ipc": 4.85, "trm": 2322.0},
    2006: {"ipc": 4.48, "trm": 2358.0}, 2007: {"ipc": 5.69, "trm": 2014.0},
    2008: {"ipc": 7.67, "trm": 1973.0}, 2009: {"ipc": 2.00, "trm": 2047.0},
    2010: {"ipc": 3.17, "trm": 1898.0}, 2011: {"ipc": 3.73, "trm": 1848.0},
    2012: {"ipc": 2.44, "trm": 1798.0}, 2013: {"ipc": 1.94, "trm": 1887.0},
    2014: {"ipc": 3.66, "trm": 2020.0}, 2015: {"ipc": 6.77, "trm": 2742.0},
    2016: {"ipc": 5.75, "trm": 3055.0}, 2017: {"ipc": 4.09, "trm": 2951.32},
    2018: {"ipc": 3.18, "trm": 2956.55}, 2019: {"ipc": 3.80, "trm": 3281.09},
    2020: {"ipc": 1.61, "trm": 3693.36}, 2021: {"ipc": 5.62, "trm": 3743.09},
    2022: {"ipc": 13.12, "trm": 4255.44}, 2023: {"ipc": 9.28, "trm": 4325.05},
    2024: {"ipc": 5.20, "trm": 4071.28}, 2025: {"ipc": 5.10, "trm": 4052.86},
}

with st.sidebar:
    st.markdown("### :material/tune: Parámetros IPC / TRM")
    editable = view == "predict"
    anio_sel = st.selectbox(
        "Año del contrato",
        [None] + RANGO_ANOS,
        format_func=lambda x: "Seleccionar..." if x is None else str(x),
        disabled=not editable,
    )
    def_val = IPC_TRM.get(anio_sel, {}) if anio_sel else {}
    ipc_val = st.number_input("IPC (%)", value=def_val.get("ipc"), step=0.01, format="%.2f", disabled=not editable)
    trm_val = st.number_input("TRM (COP/USD)", value=def_val.get("trm"), step=1.0, format="%.0f", disabled=not editable)
    if anio_sel and anio_sel in IPC_TRM:
        st.caption(f":material/auto_awesome: Datos precargados")
    elif anio_sel and anio_sel not in IPC_TRM:
        st.warning("Ingresá IPC y TRM manualmente", icon=":material/edit:")

    st.space("large")
    with st.container(border=True):
        st.markdown("**:material/bar_chart: Rendimiento del Modelo**")
        st.caption("R²: 0.149")
        st.caption("AUC CV: 0.662 ± 0.026")
        st.caption("RMSE: 16.0 pp")

    st.space("large")
    st.caption("Risk Control Dashboard v0.5")


# ─── MODEL METRICS LABELS ──────────────────────────────
MODEL_METRICS_LABELS = {
    "r2_cv": "R²",
    "auc_cv": "AUC (CV)",
    "rmse": "RMSE",
}


# ─── APIS ──────────────────────────────────────────────
def _call_api(data_bytes, text_data, filename):
    files, form = None, {}
    if data_bytes is not None:
        files = {"file": (filename or "datos.csv", data_bytes, "text/csv")}
    elif text_data is not None:
        form["riesgos"] = text_data
    if anio_sel is not None:
        form["anio"] = str(anio_sel)
    if ipc_val is not None:
        form["ipc"] = str(ipc_val)
    if trm_val is not None:
        form["trm"] = str(trm_val)
    try:
        return requests.post(f"{API_URL}/predict", files=files, data=form, timeout=30)
    except requests.ConnectionError:
        st.error(":material/cloud_off: Backend no disponible. Ejecutá `uv run uvicorn backend.main:app --reload`")
        return None


def _validate_required_params():
    """Verifica que `anio_sel`, `ipc_val` y `trm_val` estén presentes en la barra lateral.
    Muestra un error en la UI y devuelve False si falta alguno.
    """
    missing = []
    if anio_sel is None:
        missing.append("Año")
    if ipc_val is None:
        missing.append("IPC")
    if trm_val is None:
        missing.append("TRM")
    if missing:
        st.error(f"Parámetros obligatorios faltantes: {', '.join(missing)}")
        return False
    return True


def _risk_style(pct):
    if pct < 40:
        return "MODERADO", "green", "#1ABC9C"
    elif pct < 65:
        return "ALERTA", "orange", "#F39C12"
    return "CRÍTICO", "red", "#EF4444"


def kpi_card(title, value, badge, gradient, delta_up=True):
    badge_color = "#1ABC9C" if delta_up else "#EF4444"
    arrow = "↑" if delta_up else "↓"
    return f"""
    <div class="kpi-card" style="background: linear-gradient(135deg, {gradient[0]}, {gradient[1]});">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-badge" style="background: white; color: {badge_color};">{arrow} {badge}</div>
    </div>
    """


# ─── DASHBOARD ─────────────────────────────────────────
def _cargar_stats_usage():
    try:
        r = requests.get(f"{API_URL}/stats/usage", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        pass
    return None

def _cargar_stats_training():
    try:
        r = requests.get(f"{API_URL}/stats/training", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        pass
    return None

def _render_tab_uso():
    with st.spinner(":material/bar_chart: Cargando estadísticas de uso..."):
        stats = _cargar_stats_usage()
    if stats is None:
        st.warning("No se pudieron cargar las estadísticas de uso. Verificá que el backend esté corriendo.")
        return
    if stats["total_predicciones"] == 0:
        st.info("Aún no hay predicciones. Procesá al menos un contrato para ver estadísticas de uso.")
        return

    tp = stats["total_predicciones"]
    cu = stats["contratos_unicos"]
    rt = stats["riesgos_totales"]
    par = stats["porcentaje_alto_riesgo"]
    sp = stats["sobrecosto_promedio"]
    mc_pred = stats.get("mc_predicciones", 0)
    mc_iter = stats.get("mc_iteraciones_totales", 0)

    st.html(f"""
    <div class="kpi-row">
      {kpi_card("Predicciones", str(tp), f"contratos: {cu}", ("#4facfe", "#00f2fe"), True)}
      {kpi_card("Riesgos Procesados", str(rt), f"{rt//max(tp,1)} por pred.", ("#7B5CE4", "#b06ff2"), True)}
      {kpi_card("Alto Riesgo", f"{par*100:.0f}%", f"de {tp} predicciones", ("#EF4444", "#F97316"), par > 0.3)}
      {kpi_card("Sobrecosto Prom.", f"{sp:.1f}%", f"estimado promedio", ("#1ABC9C", "#2ECC71"), sp < 25)}
    </div>
    <div class="kpi-row" style="grid-template-columns: repeat(2, 1fr);">
      {kpi_card("Análisis Cuantitativo", str(mc_pred), f"con Monte Carlo", ("#7B5CE4", "#b06ff2"), True)}
      {kpi_card("Simulaciones MC", f"{mc_iter:,}".replace(",", "."), f"{mc_iter//max(mc_pred,1):,} por pred.".replace(",", "."), ("#4facfe", "#00f2fe"), True)}
    </div>
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.html(f'<div class="chart-card"><div class="chart-title">Evolución de Predicciones</div><div class="chart-subtitle">Predicciones por día y sobrecosto promedio</div>')
        df_t = pd.DataFrame(stats["serie_temporal"])
        fig = go.Figure()
        if not df_t.empty:
            fig.add_trace(go.Bar(name="Cantidad", x=df_t["fecha"], y=df_t["cantidad"],
                marker_color=AZUL, text=df_t["cantidad"], textposition="outside", textfont=dict(color="#000000", size=11)))
            fig.add_trace(go.Scatter(name="Promedio %", x=df_t["fecha"], y=df_t["promedio"],
                mode="lines+markers", line=dict(color=NARANJA, width=2.5), marker=dict(size=6, color=NARANJA), yaxis="y2"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#000000", size=11),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#000000", size=11)),
            xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
            yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
            yaxis2=dict(overlaying="y", side="right", gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
            hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_uso_serie")
        st.html("</div>")

    with c2:
        st.html(f'<div class="chart-card"><div class="chart-title">Distribución de Alertas</div><div class="chart-subtitle">ALTO RIESGO vs RIESGO MODERADO</div>')
        da = stats["distribucion_alertas"]
        labels = list(da.keys())
        values = list(da.values())
        colors = ["#EF4444" if "ALTO" in l else "#1ABC9C" for l in labels]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=values, textposition="outside", textfont=dict(color="#000000", size=14, weight=700),
            width=[0.4, 0.4],
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#000000", size=13),
            xaxis=dict(gridcolor="#E2E8F0", linecolor="#E2E8F0", tickfont=dict(color="#000000", size=13)),
            yaxis=dict(gridcolor="#E2E8F0", linecolor="#E2E8F0", tickfont=dict(color="#000000", size=12)),
            hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_uso_alertas")
        st.html("</div>")

    c3, c4 = st.columns(2)
    with c3:
        st.html(f'<div class="chart-card"><div class="chart-title">Top Factores</div><div class="chart-subtitle">Factores que más aparecen en explicaciones</div>')
        df_f = pd.DataFrame(stats["top_factores"])
        if not df_f.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_f["apariciones"], y=df_f["label"],
                orientation="h", marker_color=VERDE,
                text=df_f["apariciones"], textposition="outside", textfont=dict(color="#000000", size=11)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000", size=12)),
                hovermode="y unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_uso_factores")
        st.html("</div>")

    with c4:
        st.html(f'<div class="chart-card"><div class="chart-title">Riesgos por Predicción</div><div class="chart-subtitle">Distribución de cantidad de riesgos procesados</div>')
        df_h = pd.DataFrame(stats["histograma_riesgos"])
        if not df_h.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_h["rango"], y=df_h["cantidad"],
                marker_color=MORADO, text=df_h["cantidad"], textposition="outside", textfont=dict(color="#000000", size=11)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_uso_hist")
        st.html("</div>")

    pred_vs_real = stats["predicciones_vs_reales"]
    if pred_vs_real:
        st.html(f'<div class="chart-card"><div class="chart-title">Estimado vs Real</div><div class="chart-subtitle">Predicciones con sobrecosto real registrado</div>')
        df_vr = pd.DataFrame(pred_vs_real)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_vr["predicho"], y=df_vr["real"],
            mode="markers", marker=dict(color=AZUL, size=10, line=dict(color="#FFFFFF", width=1)),
            name="Contratos"))
        max_val = max(df_vr["predicho"].max(), df_vr["real"].max()) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(color="#CBD5E1", width=1.5, dash="dash"),
            name="y=x", showlegend=False))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#000000", size=11),
            xaxis=dict(title=dict(text="Estimado (%)", font=dict(color="#000000")), gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
            yaxis=dict(title=dict(text="Real (%)", font=dict(color="#000000")), gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
            hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_uso_vsreal")
        st.html("</div>")

def _render_tab_entrenamiento():
    with st.spinner(":material/model_training: Cargando datos de entrenamiento..."):
        stats = _cargar_stats_training()
    if stats is None:
        st.warning("No se pudieron cargar las estadísticas de entrenamiento.")
        return

    tc = stats["total_contratos"]
    sp = stats["sobrecosto_promedio"]
    sm = stats["sobrecosto_mediana"]
    par = stats["porcentaje_alto_riesgo"]
    trm = stats["total_riesgos_matriz"]
    trm_f = stats["total_riesgos_filtro"]
    cm = stats["contratos_en_matriz"]
    pool = stats.get("contratos_pool_secop1", 0)
    secop2 = stats.get("contratos_secop2_incluidos", 0)
    dn = stats.get("dist_n_riesgos", {})

    st.html(f"""
    <div class="kpi-row" style="grid-template-columns: repeat(3, 1fr);">
      {kpi_card("Contratos Entrenados", f"{tc:,}", f"{cm} raw · {pool} pool SECOP I", ("#4facfe", "#00f2fe"), True)}
      {kpi_card("Sobrecosto Promedio", f"{sp:.1f}%", f"mediana: {sm:.1f}%", ("#EF4444", "#F97316"), sp < 25)}
      {kpi_card("Alto Riesgo (>25%)", f"{par*100:.0f}%", f"de {tc} contratos", ("#F39C12", "#FDB813"), par < 0.5)}
    </div>
    <div class="kpi-row" style="grid-template-columns: repeat(3, 1fr);">
      {kpi_card("Matriz de Riesgos", f"{trm_f:,}", f"{tc} contratos (+{secop2} SECOP II)", ("#7B5CE4", "#b06ff2"), True)}
      {kpi_card("Riesgos x Contrato", f"{dn.get('media', 0):.0f}", f"min: {dn.get('min', 0)} · máx: {dn.get('max', 0)}", ("#1ABC9C", "#2ECC71"), True)}
      {kpi_card("Features del Modelo", "33", "top: tfidf_diseños +1.24", ("#E91E63", "#FF6F61"), True)}
    </div>
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.html(f'<div class="chart-card"><div class="chart-title">Distribución del Sobrecosto Real</div><div class="chart-subtitle">Contratos de entrenamiento</div>')
        df_d = pd.DataFrame(stats["distribucion_sobrecosto"])
        if not df_d.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_d["rango"], y=df_d["cantidad"],
                marker_color=AZUL, text=df_d["cantidad"], textposition="outside", textfont=dict(color="#000000", size=11)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_ent_dist")
        st.html("</div>")

    with c2:
        st.html(f'<div class="chart-card"><div class="chart-title">Coeficientes del Modelo</div><div class="chart-subtitle">Top factores que suben / bajan el sobrecosto</div>')
        df_c = pd.DataFrame(stats["top_coeficientes"])
        if not df_c.empty:
            df_c["color"] = df_c["tipo"].map({"aumenta": "#EF4444", "disminuye": "#1ABC9C"})
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_c["coef"], y=df_c["feature"],
                orientation="h", marker_color=df_c["color"],
                text=df_c["coef"].round(3), textposition="outside", textfont=dict(color="#000000", size=10)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000", size=12)),
                hovermode="y unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_ent_coef")
        st.html("</div>")

    c3, c4 = st.columns(2)
    with c3:
        st.html(f'<div class="chart-card"><div class="chart-title">Categorías de Riesgo</div><div class="chart-subtitle">Distribución en la matriz de entrenamiento</div>')
        df_cat = pd.DataFrame(stats["categorias_riesgo"])
        if not df_cat.empty:
            colors_cat = {"bajo": "#1ABC9C", "medio": "#F39C12", "alto": "#EF4444", "extremo": "#8B0000", "no especificado": "#95A5A6"}
            df_cat["color"] = df_cat["categoria"].map(colors_cat).fillna(MORADO)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_cat["cantidad"], y=df_cat["categoria"],
                orientation="h", marker_color=df_cat["color"],
                text=df_cat["cantidad"], textposition="outside", textfont=dict(color="#000000", size=11)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000", size=12)),
                hovermode="y unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_ent_cat")
        st.html("</div>")

    with c4:
        st.html(f'<div class="chart-card"><div class="chart-title">IPC / TRM por Año</div><div class="chart-subtitle">Variables macroeconómicas del modelo</div>')
        df_it = pd.DataFrame(stats["ipc_trm"])
        if not df_it.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="IPC", x=df_it["anio"], y=df_it["ipc"],
                marker_color=AZUL, text=df_it["ipc"].round(1), textposition="outside", textfont=dict(color="#000000", size=10)))
            fig.add_trace(go.Scatter(name="TRM", x=df_it["anio"], y=df_it["trm"],
                mode="lines+markers", line=dict(color=NARANJA, width=2.5), marker=dict(size=6, color=NARANJA), yaxis="y2"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#000000", size=11)),
                xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis2=dict(overlaying="y", side="right", gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="graf_ent_ipctrm")
        st.html("</div>")

    c5, c6 = st.columns(2)
    with c5:
        top_sc = stats.get("top_sobrecosto", [])
        if top_sc:
            st.html(f'<div class="chart-card"><div class="chart-title">Contratos con Mayor Sobrecosto</div><div class="chart-subtitle">Top 5 de la matriz de entrenamiento</div>')
            rows_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #e0e0e0;">'
                f'<span style="font-weight:600;">{r["id_contrato"]}</span>'
                f'<span style="color:#EF4444;font-weight:700;">{r["sobrecosto"]:.1f}%</span>'
                f'<span style="color:#64748B;">{int(r["n_riesgos"])} riesgos</span>'
                f'</div>'
                for r in top_sc
            )
            st.html(f'<div style="margin-top:4px;">{rows_html}</div></div>')

    with c6:
        top_riesgos = stats.get("top_n_riesgos", [])
        if top_riesgos:
            st.html(f'<div class="chart-card"><div class="chart-title">Contratos con Más Riesgos</div><div class="chart-subtitle">Top 5 por cantidad de riesgos</div>')
            rows_html = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #e0e0e0;">'
                f'<span style="font-weight:600;">{r["id_contrato"]}</span>'
                f'<span style="color:#7B5CE4;font-weight:700;">{int(r["n_riesgos"])} riesgos</span>'
                f'<span style="color:{NARANJA};">{r["sobrecosto"]:.1f}%</span>'
                f'</div>'
                for r in top_riesgos
            )
            st.html(f'<div style="margin-top:4px;">{rows_html}</div></div>')

    st.html('<div style="margin-top:1rem;">')
    modelo = stats.get("modelo", {})
    if modelo:
        DESCRIPCIONES_METRICAS = {
            "r2_cv": "Variación del sobrecosto explicada",
            "auc_cv": "Capacidad de separar alto riesgo",
            "rmse": "Error típico de la predicción",
        }
        st.html(f'<div class="chart-card"><div class="chart-title">Rendimiento del Modelo</div><div class="chart-subtitle">Métricas del modelo actual (33 features)</div>')
        met_cols = st.columns([2.2, 1.6, 1.6, 1.6])
        for i, (k, v) in enumerate(modelo.items()):
            if k == "modelo":
                with met_cols[0]:
                    st.markdown(
                        f'<div style="background:white;border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);height:100%;display:flex;flex-direction:column;justify-content:center;">'
                        f'<div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">Modelo</div>'
                        f'<div style="font-size:1.5rem;font-weight:800;color:#1E293B;white-space:nowrap;">{v}</div>'
                        f'<div style="font-size:0.7rem;color:transparent;padding:7px 0;">‎</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                with met_cols[-(i) if i > 0 else i]:
                    label = MODEL_METRICS_LABELS.get(k, k)
                    desc = DESCRIPCIONES_METRICAS.get(k, "")
                    st.markdown(
                        f'<div style="background:white;border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
                        f'<div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>'
                        f'<div style="font-size:1.8rem;font-weight:800;color:#1E293B;">{v:.3f}</div>'
                        f'<div style="font-size:0.7rem;color:#94A3B8;margin-top:2px;">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
        st.html("</div>")
    st.html('</div>')

def _render_dashboard():
    pio.templates.default = "plotly_white"
    st.html("""
    <style>
      div[data-testid="stTabs"] button[data-baseweb="tab"],
      div[data-testid="stTabs"] button[role="tab"],
      div[data-testid="stTabs"] button p {
        color: #000000 !important;
        font-weight: 600;
        font-size: 0.95rem;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #000000 !important;
        border-bottom-color: #4facfe !important;
      }
    </style>
    """)
    st.markdown("""
<style>
    /* Forzamos el color negro en las pestañas normales y en la pestaña activa */
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] div[role="tab"] {
        color: #000000 !important;
    }
    /* También aseguramos que la pestaña seleccionada (activa) sea negra */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Uso del Modelo", "🧠 Entrenamiento"])
    with tab1:
        _render_tab_uso()
    with tab2:
        _render_tab_entrenamiento()
    


# ─── PREDECIR ──────────────────────────────────────────
def _mostrar_resultados(resp, df_original):
    if resp.status_code == 200:
        results = resp.json()
    elif resp.status_code == 422:
        detail = resp.json().get("detail", {})
        st.error(":material/error: Error de validación")
        for e in detail.get("errores", []):
            st.warning(f"- {e}")
        st.info(f"Columnas requeridas: {', '.join(detail.get('columnas_requeridas', []))}")
        return
    else:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        st.error(f":material/error: Error {resp.status_code}")
        st.code(str(msg))
        return

    n_ctos = len(results)
    st.markdown(f'<div class="chart-title" style="margin-bottom:0.75rem; color: #000000;">Análisis Cualitativo - Alertas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-title" style="margin-bottom:1rem; color: #000000;"></div>', unsafe_allow_html=True)

    # --- KPIs (fuera de las tarjetas) ---
    kpi_val = st.columns(3)
    with kpi_val[0]:
        st.metric("Contratos analizados", n_ctos, border=True)
    with kpi_val[1]:
        st.metric("Riesgos totales", f"{len(df_original):,}", border=True)
    with kpi_val[2]:
        n_alto = sum(1 for r in results if r["alerta"] == "ALTO RIESGO")
        st.metric("Alertas activas", f"{n_alto}/{n_ctos}", border=True)

    # --- Inyección de CSS específico para los contenedores de resultados ---
    st.markdown("""
    <style>
        /* Forzamos el fondo oscuro y texto blanco en los contenedores que tengan key="res_..." */
        div[data-testid="stContainer"][key*="res_"] {
            background-color: #0F172A !important;
            border: 1px solid #1E293B !important;
            border-radius: 16px !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
        }
        /* Todo el texto dentro de esos contenedores debe ser blanco */
        div[data-testid="stContainer"][key*="res_"] * {
            color: #F8FAFC !important;
        }
        /* Excepción para los badges y métricas internas */
        div[data-testid="stContainer"][key*="res_"] .stMetric {
            background: transparent !important;
            border: none !important;
        }
        div[data-testid="stContainer"][key*="res_"] .stMetric label,
        div[data-testid="stContainer"][key*="res_"] .stMetric div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
        }
        div[data-testid="stContainer"][key*="res_"] .stBadge {
            background-color: rgba(251, 146, 60, 0.2) !important;
            border-color: #FB923C !important;
            color: #F8FAFC !important;
        }
        div[data-testid="stContainer"][key*="res_"] .stCaption {
            color: #CBD5E1 !important;
        }
        /* Inputs dentro del contenedor oscuro: fondo blanco y texto negro */
        div[data-testid="stContainer"][key*="res_"] div[data-baseweb="input"] {
            background-color: #FFFFFF !important;
        }
        div[data-testid="stContainer"][key*="res_"] div[data-baseweb="input"] > div {
            background-color: #FFFFFF !important;
        }
        div[data-testid="stContainer"][key*="res_"] div[data-baseweb="input"] input {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        div[data-testid="stContainer"][key*="res_"] div[data-baseweb="input"] input::placeholder {
            color: #64748B !important;
            -webkit-text-fill-color: #64748B !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Loop de resultados ---
    for idx, r in enumerate(results):
        pred = r["sobrecosto_estimado"]
        prob = r["probabilidad_alto_riesgo"]
        pct = prob * 100
        n_riesgos = r["riesgos_procesados"]
        hid = r.get("history_id")
        label, badge_color, bar_color = _risk_style(pct)

        # ✅ Usamos st.container con una clave que empiece por "res_" para que el CSS lo detecte
        with st.container(border=True, key=f"res_{hid or idx}_{pred}"):
            
            top_cols = st.columns([1, 1.2, 2.8], gap="large")
            with top_cols[0]:
                color = "green" if pred < 10 else ("blue" if pred < 20 else ("orange" if pred < 50 else "red"))
                cid = r.get("id_contrato", "")
                st.markdown(
                    f'<div style="color:#000;font-size:1rem;font-weight:600;margin-bottom:-8px;">'
                    f'Contrato: {cid}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'''
<div style="font-size: 2.8rem; font-weight: 900; color: {color}; padding-right: 10px;">
    {pred:.1f}%
</div>
''', unsafe_allow_html=True)
                st.markdown(f"#### :{color}[**Sobrecosto estimado por el modelo**]")
            
            with top_cols[1]:
                st.markdown(f'<span style="color: #000000; font-weight: 700; padding-right: 200px;">Prob. riesgo alto</span>', unsafe_allow_html=True)
                meta_cols = st.columns([2, 4])
                with meta_cols[0]:
                    st.markdown(f"#### :{color}[{pct:.0f}%]")
                with meta_cols[1]:
                    st.markdown(
                        f"""<div style="height:100%;display:flex;align-items:center; padding-top: 16px;">
                            <div style="background:#334155;border-radius:8px;height:10px;width:100%">
                                <div class="risk-bar" style="background:{bar_color};width:{pct}%;height:10px;border-radius:8px"></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with st.container():
                    st.markdown("""
                            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                        """, unsafe_allow_html=True)

                st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-top: -30px;">
                <span style="display: inline-flex; align-items: center; 
                            background-color: {badge_color}22; 
                            color: {badge_color}; 
                            font-size: 1.3rem; font-weight: 700;">
                    <span class="material-icons" style="font-size: 1.6rem;">warning</span>
                    {label}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            with top_cols[2]:
                st.markdown(
                    '<div style="display:flex;gap:20px;">'
                    '<div style="flex:1;">'
                    '<span style="color:#000000;font-weight:700;">'
                    '<span class="material-icons" style="font-size:1.2rem;vertical-align:middle;">trending_up</span> Suben el costo'
                    '</span>'
                    '<div style="width:100%">'
                    + ''.join(
                        f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #e0e0e0;">
                            <span style="color:#000000;font-size:0.85rem;line-height:1.3;padding-right:8px;">{f["label"]}</span>
                            <span style="background-color:#000000;color:#FFFFFF;padding:1px 10px;border-radius:4px;font-family:monospace;font-weight:bold;font-size:0.85rem;white-space:nowrap;flex-shrink:0;">+{f["coef"]:.2f}</span>
                        </div>'''
                        for f in r["factores_aumentan"][:4]
                    )
                    + '</div></div>'
                    '<div style="flex:1;">'
                    '<span style="color:#000000;font-weight:700;">'
                    '<span class="material-icons" style="font-size:1.2rem;vertical-align:middle;">trending_down</span> Bajan el costo'
                    '</span>'
                    '<div style="width:100%">'
                    + ''.join(
                        f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #e0e0e0;">
                            <span style="color:#000000;font-size:0.85rem;line-height:1.3;padding-right:8px;">{f["label"]}</span>
                            <span style="background-color:#000000;color:#FFFFFF;padding:1px 10px;border-radius:4px;font-family:monospace;font-weight:bold;font-size:0.85rem;white-space:nowrap;flex-shrink:0;">{f["coef"]:.2f}</span>
                        </div>'''
                        for f in r["factores_disminuyen"][:4]
                    )
                    + '</div></div></div>',
                    unsafe_allow_html=True,
                )
            
                                                 
def _render_predict():
    st.markdown("""
    <style>
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #000000 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="pred-card"><div style="font-size: 24px; font-weight: 700; color: #000000; margin-bottom: 15px;">Predecir Sobrecosto</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Método de entrada", 
        ["Subir CSV", "Pegar texto"],
        horizontal=True,
    )
    
    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)

    raw = None
    texto_csv = None
    df_orig = None
    file_name = None
    ready = False

    if input_mode == "Subir CSV":
        uploaded = st.file_uploader("Seleccioná un archivo CSV", type="csv", label_visibility="collapsed")
        if uploaded:
            raw = uploaded.getvalue()
            df_orig = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            file_name = uploaded.name
            n_r = len(df_orig); n_c = df_orig['id_contrato'].nunique()
            st.info(f"{n_r} {'riesgo' if n_r == 1 else 'riesgos'} · {n_c} {'contrato' if n_c == 1 else 'contratos'}")
            ready = True
    else:
        st.markdown("""
            <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; font-family: monospace; color: #000000;">
                Ingrese los datos en formato CSV. Una fila por riesgo: <br>
                id_contrato,descripcion_riesgo,probabilidad,impacto,tipo,categoria
            </div>
            <br>
        """, unsafe_allow_html=True)
        texto_csv = st.text_area(
            "Contenido CSV", height=160,
            placeholder="id_contrato,descripcion_riesgo,probabilidad,impacto,tipo,categoria",
            label_visibility="collapsed",
        )
        raw = None
        file_name = None
        if texto_csv.strip():
            try:
                df_orig = pd.read_csv(io.StringIO(texto_csv))
                n_r = len(df_orig); n_c = df_orig['id_contrato'].nunique()
                st.info(f"{n_r} {'riesgo' if n_r == 1 else 'riesgos'} · {n_c} {'contrato' if n_c == 1 else 'contratos'}")
                ready = True
            except Exception as e:
                st.error(f":material/error: No se pudo interpretar el texto como CSV: {e}")
                ready = False

    st.markdown('<hr style="margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <style>
        .st-key-vi_val label, .st-key-mc_iter label { color: #000000 !important; font-weight: 600 !important; }
        .st-key-vi_val input, .st-key-mc_iter input { background-color: #FFFFFF !important; color: #000000 !important; }
        .st-key-vi_val button, .st-key-mc_iter button { background-color: #FFFFFF !important; color: #000000 !important; border: 1px solid #CBD5E1 !important; }
        .st-key-vi_val button:hover, .st-key-mc_iter button:hover { background-color: #F1F5F9 !important; }

        div[data-testid="stCheckbox"] { padding-top: 24px !important; }
        div[data-testid="stCheckbox"] label p { color: #000000 !important; font-weight: 500 !important; }
        div[data-testid="stCheckbox"] label span { color: #000000 !important; font-weight: 500 !important; }
    </style>
    """, unsafe_allow_html=True)

    mc_row = st.columns([3, 2, 2, 1.5, 4])
    with mc_row[0]:
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    with mc_row[1]:
        vi_val = st.number_input(
            "Valor inicial (COP)",
            value=0.0, min_value=0.0, step=1_000_000.0, format="%.0f", key="vi_val",
        )
        if vi_val > 0:
            st.caption(f"$ {vi_val:,.0f}".replace(",", "."))
    with mc_row[2]:
        mc_iter = st.number_input("Iteraciones", value=1000, min_value=100, max_value=10000, step=500, key="mc_iter")
    with mc_row[3]:
        incluir_mc = st.checkbox("Incluir Monte Carlo", value=True, key="incluir_mc")
    with mc_row[4]:
        mc_ruido = st.checkbox("Ruido RMSE", value=True, key="mc_ruido")

    st.markdown('<p style="color:#000000;font-size:0.8rem;margin:0 0 0.5rem 0;">Aviso: Primero carga el análisis cualitativo (alertas), posteriormente el cuantitativo (Impacto Monetario por Riesgo).</p>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2.5, 1, 2.5])
    with btn_col:
        procesar_clicked = st.button("Procesar", key="procesar", type="primary", use_container_width=True)

    if procesar_clicked and ready:
        if not _validate_required_params():
            st.warning("Completá los parámetros obligatorios en la barra lateral antes de procesar.")
        else:
            with st.spinner(":material/sync: Procesando riesgos..."):
                resp = _call_api(data_bytes=raw, text_data=texto_csv, filename=file_name)
            if resp is not None:
                _mostrar_resultados(resp, df_orig)

            if incluir_mc:
                mc_history_id = None
                if resp is not None and resp.status_code == 200:
                    preds = resp.json()
                    if isinstance(preds, list) and len(preds) > 0:
                        mc_history_id = preds[0].get("history_id")
                with st.spinner(f":material/sync: Ejecutando {mc_iter} simulaciones Monte Carlo..."):
                    mc_resp = _call_mc_api(
                        data_bytes=raw, text_data=texto_csv, filename=file_name,
                        n_iteraciones=mc_iter, incluir_ruido=mc_ruido,
                        valor_inicial=vi_val if vi_val > 0 else None,
                        history_id=mc_history_id,
                    )
                if mc_resp is not None and mc_resp.status_code == 200:
                    st.session_state.mc_data = mc_resp.json()
                    _mostrar_resultados_mc(st.session_state.mc_data)
                elif mc_resp is not None:
                    st.error(f":material/error: Error en MC: {mc_resp.status_code}")

    elif "mc_data" in st.session_state:
        _mostrar_resultados_mc(st.session_state.mc_data)
                    
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.html('<div style="height: 20px;"></div>')
# ─── HISTORIAL ─────────────────────────────────────────
def _render_history():
    st.markdown(f'<div class="pred-card"><div style="font-size: 24px; font-weight: 700; color: #000000; margin-bottom: 15px;">Historial de Predicciones</div>', unsafe_allow_html=True)
    st.caption("Cada predicción se guarda automáticamente como evidencia para la tesis.")
    st.space("small")

    if "hist_page" not in st.session_state:
        st.session_state.hist_page = 1
    page_size = 20

    try:
        with st.spinner(":material/history: Cargando historial..."):
            resp = requests.get(f"{API_URL}/history", params={"page": st.session_state.hist_page, "page_size": page_size}, timeout=10)
        if resp.status_code == 200:
            body = resp.json()
            rows = body["data"]
            total = body["total"]
            paginas = body["paginas"]
            pagina = body["page"]
            if total == 0:
                st.info(":material/info: Aún no hay predicciones. Debe procesar al menos un contrato.")
            else:
                top_col = st.columns([1, 4, 1])
                with top_col[0]:
                    if st.button(":material/delete_sweep: Limpiar todo", key="clear_all", type="secondary"):
                        try:
                            rr = requests.delete(f"{API_URL}/history", timeout=10)
                            if rr.status_code == 200:
                                st.session_state.hist_page = 1
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                with top_col[2]:
                    st.caption(f"{total} registro(s) en total")

                for h in rows:
                    real_str = f"{h['sobrecosto_real']:.1f}%" if h['sobrecosto_real'] is not None else "—"
                    rid = h['prediccion_ridge']
                    prob_val = h['probabilidad_alto_riesgo'] * 100
                    hid = h['id']
                    alerta = h['alerta']
                    is_alto = "ALTO" in alerta
                    badge_color = "#EF4444" if is_alto else "#1ABC9C"
                    with st.container(border=True):
                        hcols = st.columns([2.2, 1, 1, 1, 0.5])
                        with hcols[0]:
                            st.markdown(
                                f'<span style="color: #000000; font-weight: bold; font-size: 1rem; margin-left:8px;">'
                                f':material/description: {h["id_contrato"]}</span>'
                                f'<span style="display:inline-block; background:{badge_color}18; color:{badge_color}; '
                                f'padding:0 8px; border-radius:4px; font-size:0.75rem; font-weight:700; margin-left:8px;">'
                                f'{alerta}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<span style="color: #000000; font-size: 0.8rem;">'
                                f'🕐 {h["created_at"][:16]} · '
                                f'📋 {h["n_riesgos"]} riesgos · '
                                f'📅 año {h["anio"] or "?"}</span>',
                                unsafe_allow_html=True,
                            )
                            fa = h.get("factores_aumentan", [])[:2]
                            fd = h.get("factores_disminuyen", [])[:2]
                            parts = []
                            for f in fa:
                                parts.append(f'<span style="color:#DC2626;font-size:0.75rem;">▲ {f["label"]}</span>')
                            for f in fd:
                                parts.append(f'<span style="color:#16A34A;font-size:0.75rem;">▼ {f["label"]}</span>')
                            if parts:
                                st.markdown(
                                    f'<div style="margin-top:2px; display:flex; gap:10px; flex-wrap:wrap;">'
                                    + ''.join(parts) + '</div>',
                                    unsafe_allow_html=True,
                                )
                        with hcols[1]:
                            st.markdown(f'<div class="history-metric"><div class="label" style="color:{MUTED};">Ridge</div><div class="value" style="color:{TEXT_COLOR};">{rid:.1f}%</div></div>', unsafe_allow_html=True)
                        with hcols[2]:
                            st.markdown(f'<div class="history-metric"><div class="label" style="color:{MUTED};">Prob.</div><div class="value" style="color:{TEXT_COLOR};">{prob_val:.0f}%</div></div>', unsafe_allow_html=True)
                        with hcols[3]:
                            st.markdown(f'<div class="history-metric"><div class="label" style="color:{MUTED};">Real</div><div class="value" style="color:{TEXT_COLOR};">{real_str}</div></div>', unsafe_allow_html=True)
                        with hcols[4]:
                            editando = st.session_state.get(f"editando_{hid}", False)
                            with st.container():
                                if st.button(":material/edit:", key=f"edit_{hid}", help="Editar", use_container_width=True):
                                    st.session_state[f"editando_{hid}"] = not editando
                                    st.rerun()
                                if not editando:
                                    if st.button(":material/delete:", key=f"del_{hid}", help="Eliminar", use_container_width=True):
                                        try:
                                            rr = requests.delete(f"{API_URL}/history/{hid}", timeout=10)
                                            if rr.status_code == 200:
                                                st.rerun()
                                        except Exception:
                                            pass
                        vcols = st.columns([1, 2, 1])
                        with vcols[1]:
                            ver_key = f"ver_full_{hid}"
                            if st.button(":material/analytics: Ver completo", key=ver_key, use_container_width=True):
                                try:
                                    rr = requests.get(f"{API_URL}/history/{hid}", timeout=10)
                                    if rr.status_code == 200:
                                        _dialogo_full_analysis(rr.json())
                                except Exception:
                                    pass
                        if editando:
                            evcols = st.columns([1, 1.5, 2, 1])
                            with evcols[0]:
                                st.caption("Real %:")
                            with evcols[1]:
                                ev_real = st.number_input("Real", key=f"evr_{hid}", min_value=0.0, max_value=500.0, step=0.1, format="%.1f", label_visibility="collapsed", placeholder="Real %")
                            with evcols[2]:
                                ev_nota = st.text_input("Nota", key=f"evn_{hid}", label_visibility="collapsed", placeholder="Nota")
                            with evcols[3]:
                                if st.button("Guardar", key=f"evs_{hid}", type="primary", use_container_width=True):
                                    rr = requests.put(f"{API_URL}/history/{hid}", data={"sobrecosto_real": str(ev_real), "notas": ev_nota or ""}, timeout=10)
                                    if rr.status_code == 200:
                                        st.session_state[f"editando_{hid}"] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {rr.text}")
                        if h.get("notas"):
                            st.caption(f":material/note: {h['notas']}")

                if paginas > 1:
                    pcols = st.columns([1, 2, 1])
                    with pcols[0]:
                        if st.button("◀ Anterior", key="hist_prev", disabled=(pagina <= 1), use_container_width=True):
                            st.session_state.hist_page = max(1, pagina - 1)
                            st.rerun()
                    with pcols[1]:
                        inicio = (pagina - 1) * page_size + 1
                        fin = min(pagina * page_size, total)
                        st.markdown(
                            f'<div style="text-align:center;font-weight:600;font-size:0.95rem;padding-top:4px;">'
                            f'Página {pagina} de {paginas}</div>'
                            f'<div style="text-align:center;font-size:0.8rem;color:#475569;">'
                            f'Mostrando registros {inicio}–{fin} de {total}</div>',
                            unsafe_allow_html=True,
                        )
                    with pcols[2]:
                        if st.button("Siguiente ▶", key="hist_next", disabled=(pagina >= paginas), use_container_width=True):
                            st.session_state.hist_page = min(paginas, pagina + 1)
                            st.rerun()
        else:
            st.error(f":material/error: Error al cargar historial: {resp.text}")
    except requests.ConnectionError:
        st.error(":material/cloud_off: Backend no disponible.")
    st.html("</div>")


# ─── MONTE CARLO (helpers) ──────────────────────────────
def _fmt_cop(val):
    if val is None:
        return None
    if abs(val) >= 1_000_000_000:
        return f"${val/1_000_000_000:,.2f}B"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:,.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.0f}"


def _call_mc_api(data_bytes, text_data, filename, n_iteraciones=1000, incluir_ruido=True, valor_inicial=None, history_id=None):
    files, form = None, {}
    if data_bytes is not None:
        files = {"file": (filename or "datos.csv", data_bytes, "text/csv")}
    elif text_data is not None:
        form["riesgos"] = text_data
    if anio_sel is not None:
        form["anio"] = str(anio_sel)
    if ipc_val is not None:
        form["ipc"] = str(ipc_val)
    if trm_val is not None:
        form["trm"] = str(trm_val)
    form["n_iteraciones"] = str(n_iteraciones)
    form["incluir_ruido"] = str(incluir_ruido).lower()
    if valor_inicial is not None:
        form["valor_inicial"] = str(valor_inicial)
    if history_id is not None:
        form["history_id"] = str(history_id)
    try:
        return requests.post(f"{API_URL}/predict/montecarlo", files=files, data=form, timeout=60)
    except requests.ConnectionError:
        st.error(":material/cloud_off: Backend no disponible.")
        return None


@st.dialog("Análisis completo", width="large")
def _dialogo_full_analysis(full: dict):
    pred = full.get("prediccion_ridge", 0)
    prob = full.get("probabilidad_alto_riesgo", 0) * 100
    alerta = full.get("alerta", "")
    is_alto = "ALTO" in alerta
    badge_color = "#EF4444" if is_alto else "#1ABC9C"
    color = "red" if pred > 30 else ("orange" if pred > 15 else "green")
    st.markdown(f'<div style="font-size:1.2rem;font-weight:700;margin-bottom:0.5rem;">{full.get("id_contrato","")}</div>', unsafe_allow_html=True)
    rc = st.columns([1, 1, 2])
    with rc[0]:
        st.markdown(f'<div style="font-size:2.5rem;font-weight:900;color:{color};">{pred:.1f}%</div><span style="font-weight:600;">Sobrecosto estimado</span>', unsafe_allow_html=True)
    with rc[1]:
        st.markdown(f'<div style="font-size:2rem;font-weight:800;color:{badge_color};">{prob:.0f}%</div><span style="font-weight:600;">Prob. alto riesgo</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="display:inline-block;background:{badge_color}22;color:{badge_color};padding:2px 12px;border-radius:100px;font-weight:700;">{alerta}</span>', unsafe_allow_html=True)
    with rc[2]:
        fa = full.get("factores_aumentan", [])[:4]
        fd = full.get("factores_disminuyen", [])[:4]
        parts_html = '<div style="display:flex;gap:20px;">'
        if fa:
            parts_html += '<div><div style="font-weight:600;">▲ Suben</div>' + ''.join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:2px 0;border-bottom:1px solid #e0e0e0;"><span style="font-size:0.85rem;">{f["label"]}</span><span style="color:#DC2626;font-weight:700;">+{f["coef"]:.2f}</span></div>' for f in fa) + '</div>'
        if fd:
            parts_html += '<div><div style="font-weight:600;">▼ Bajan</div>' + ''.join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:2px 0;border-bottom:1px solid #e0e0e0;"><span style="font-size:0.85rem;">{f["label"]}</span><span style="color:#16A34A;font-weight:700;">{f["coef"]:.2f}</span></div>' for f in fd) + '</div>'
        parts_html += '</div>'
        st.markdown(parts_html, unsafe_allow_html=True)
    if full.get("resultado_json"):
        _mostrar_resultados_mc(full["resultado_json"])



def _mostrar_resultados_mc(data: dict):
    pred_central = data["prediccion_central"]
    percentiles = data["percentiles"]
    stats = data["stats"]
    histograma = data["histograma"]
    tornado = data["tornado"]
    riesgos = data["riesgos"]
    n_sim = data["n_simulaciones"]
    ruido_incl = data["ruido_incluido"]
    tiene_cop = "valor_inicial" in data and data["valor_inicial"] is not None
    vi = data.get("valor_inicial")

    st.markdown("""
    <style>
        div[data-testid="stRadio"] div[role="radiogroup"] { display: flex; gap: 0; justify-content: center; }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            width: auto; text-align: center; padding: 8px 20px; margin: 0;
            border: 1px solid #CBD5E1; border-radius: 0; background: #F8FAFC;
            cursor: pointer; color: #000000 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:first-child { border-radius: 8px 0 0 8px; }
        div[data-testid="stRadio"] div[role="radiogroup"] label:last-child { border-radius: 0 8px 8px 0; }
        div[data-testid="stRadio"] div[role="radiogroup"] label:not(:last-child) { border-right: none; }
        div[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] {
            background: #FFFFFF; border-color: #4facfe; font-weight: 700; color: #000000 !important;
            box-shadow: inset 0 -2px 0 #4facfe;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p { color: #000000 !important; font-weight: 500; }
        div[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p { color: #000000 !important; font-weight: 700; }
        .chart-card .chart-title { color: #000000 !important; }
        .chart-card .chart-subtitle { color: #374151 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:1.5rem 0 1rem 0;border:0;border-top:2px solid #E2E8F0;">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-title" style="margin-bottom:1.2rem; color: #000000;">Análisis Cuantitativo — Impacto Monetario por Riesgo</div>', unsafe_allow_html=True)

    if tiene_cop and vi:
        st.markdown(f'<p style="color:#000000;font-size:0.85rem;margin:0;">Valor inicial del contrato: {_fmt_cop(vi)} COP  |  {n_sim} simulaciones  |  {"Ruido RMSE incluido" if ruido_incl else "Solo perturbación de inputs"}</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-title" style="margin-bottom:1.2rem; color: #000000;"></div>', unsafe_allow_html=True)
        
    # ─── Tabla de impacto por riesgo (tornado en COP + %) ───
    if tiene_cop:
        tdata = data["tornado"]
        tdata_cop = data["tornado_cop"]
    else:
        tdata = tornado
        tdata_cop = None
    if tdata:
        df_t = pd.DataFrame(tdata)
        rows_html = ""
        for idx, r in df_t.iterrows():
            riesgo = r["riesgo"]
            prob = int(r["probabilidad_original"]) if "probabilidad_original" in r and pd.notna(r["probabilidad_original"]) else "-"
            imp = int(r["impacto_original"]) if "impacto_original" in r and pd.notna(r["impacto_original"]) else "-"
            direccion = r["direccion"]
            swing_val = r["swing"]
            if tiene_cop:
                rc = tdata_cop[idx]
                alta = f"{_fmt_cop(rc['prediccion_alta'])} ({r['prediccion_alta']:.1f}%)"
                baja = f"{_fmt_cop(rc['prediccion_baja'])} ({r['prediccion_baja']:.1f}%)"
                swing = f"{_fmt_cop(rc['swing'])} ({r['swing']:.2f}pp)"
                bar_swing = rc["swing"]
            else:
                alta = f"{r['prediccion_alta']:.1f}%"
                baja = f"{r['prediccion_baja']:.1f}%"
                swing = f"{r['swing']:.2f}pp"
                bar_swing = swing_val
            max_swing = max(abs(r2["swing"]) for r2 in (tdata_cop if tiene_cop else tdata))
            bar_pct = min(abs(bar_swing) / (max_swing or 1) * 100, 100)
            bar_color = "#EF4444" if direccion == "aumenta" else "#1ABC9C"

            rows_html += f"""
            <tr>
                <td style="padding:10px 8px;font-weight:600;color:#000;">{riesgo}</td>
                <td style="padding:10px 8px;text-align:center;color:#000;">{prob}</td>
                <td style="padding:10px 8px;text-align:center;color:#000;">{imp}</td>
                <td style="padding:10px 8px;text-align:center;color:#EF4444;font-weight:700;">{alta}</td>
                <td style="padding:10px 8px;text-align:center;color:#1ABC9C;font-weight:700;">{baja}</td>
                <td style="padding:10px 8px;text-align:center;color:#000;font-weight:700;">{swing}</td>
                <td style="padding:10px 8px;">
                    <div style="background:#E2E8F0;border-radius:4px;height:10px;width:100%;">
                        <div style="background:{bar_color};width:{bar_pct:.0f}%;height:10px;border-radius:4px;"></div>
                    </div>
                </td>
            </tr>"""

        st.html(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:4px;margin-bottom:1.25rem;overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
            <thead>
                <tr style="border-bottom:2px solid #E2E8F0;">
                    <th style="padding:8px;text-align:left;color:#64748B;font-weight:600;">Tipo de Riesgo</th>
                    <th style="padding:8px;text-align:center;color:#64748B;font-weight:600;">Prob</th>
                    <th style="padding:8px;text-align:center;color:#64748B;font-weight:600;">Imp</th>
                    <th style="padding:8px;text-align:center;color:#EF4444;font-weight:600;">Si suben todos (+1)</th>
                    <th style="padding:8px;text-align:center;color:#1ABC9C;font-weight:600;">Si bajan todos (-1)</th>
                    <th style="padding:8px;text-align:center;color:#64748B;font-weight:600;">Impacto neto</th>
                    <th style="padding:8px;text-align:center;color:#64748B;font-weight:600;"></th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        </div>
        """)

    # ─── KPIs ───
    fmt_pc = _fmt_cop(data["stats_cop"]["prediccion_central"]) if tiene_cop else f"{pred_central:.1f}%"
    fmt_media = _fmt_cop(data["stats_cop"]["media"]) if tiene_cop else f"{stats['media']:.1f}%"
    fmt_std = _fmt_cop(data["stats_cop"]["std"]) if tiene_cop else f"{stats['std']:.1f}pp"
    fmt_p10 = _fmt_cop(data["percentiles_cop"]["P10"]) if tiene_cop else f"{percentiles['P10']:.1f}%"
    fmt_p90 = _fmt_cop(data["percentiles_cop"]["P90"]) if tiene_cop else f"{percentiles['P90']:.1f}%"

    kpi = st.columns(5)
    with kpi[0]:
        st.metric("Sobrecosto estimado", fmt_pc, border=True)
        
    with kpi[1]:
        st.metric("Media MC", fmt_media, border=True)
    with kpi[2]:
        st.metric("Desv. estándar", fmt_std, border=True)
    with kpi[3]:
        st.metric("P10 (optimista)", fmt_p10, border=True)
    with kpi[4]:
        st.metric("P90 (pesimista)", fmt_p90, border=True)

    # ─── Tabs secundarios ───
    st.markdown(f'<div class="chart-title" style="margin-bottom:1.2rem; color: #000000;"></div>', unsafe_allow_html=True)
    sel = st.radio("", ["📊 Distribución", "🔍 Contribución por Riesgo"], horizontal=True, label_visibility="collapsed", key="mc_tab_sel")

    if sel == "📊 Distribución":
        h = data["histograma_cop"] if tiene_cop else histograma
        st.html(f'<div class="chart-card"><div class="chart-title">Distribución del Sobrecosto Total</div>'
                f'<div class="chart-subtitle">{n_sim} simulaciones · {"COP" if tiene_cop else "%"}</div>')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{_fmt_cop(b['bin_inicio']) if tiene_cop else f'{b['bin_inicio']:.0f}'}" for b in h],
            y=[b["frecuencia"] for b in h],
            marker_color=AZUL,
            text=[b["frecuencia"] for b in h],
            textposition="outside",
            textfont=dict(color="#000000", size=9),
        ))
        max_freq = max(b["frecuencia"] for b in h)
        if tiene_cop:
            fig.add_vline(x=data["stats_cop"]["prediccion_central"], line_dash="dash", line_color=NARANJA,
                          annotation_text=f"Central: {_fmt_cop(data['stats_cop']['prediccion_central'])}", annotation_position="top")
            fig.add_vline(x=data["percentiles_cop"]["P10"], line_dash="dot", line_color=VERDE,
                          annotation_text=f"P10: {_fmt_cop(data['percentiles_cop']['P10'])}", annotation_position="bottom")
            fig.add_vline(x=data["percentiles_cop"]["P90"], line_dash="dot", line_color=VERDE,
                          annotation_text=f"P90: {_fmt_cop(data['percentiles_cop']['P90'])}", annotation_position="bottom")
        else:
            fig.add_vline(x=pred_central, line_dash="dash", line_color=NARANJA,
                          annotation_text=f"Central: {pred_central:.1f}%", annotation_position="top")
            fig.add_vline(x=percentiles["P10"], line_dash="dot", line_color=VERDE,
                          annotation_text=f"P10: {percentiles['P10']:.1f}%", annotation_position="bottom")
            fig.add_vline(x=percentiles["P90"], line_dash="dot", line_color=VERDE,
                          annotation_text=f"P90: {percentiles['P90']:.1f}%", annotation_position="bottom")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#000000", size=11),
            xaxis=dict(
                gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR,
                tickfont=dict(color="#000000", size=9),
                tickangle=45,
            ),
            yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000"),
                       range=[0, max_freq * 1.25]),
            hovermode="x unified",
            bargap=0.1,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="mc_hist")
        st.html("</div>")

        pdata = data["percentiles_cop"] if tiene_cop else percentiles
        st.html(f'<div class="chart-card"><div class="chart-title">Percentiles</div></div>')
        pcols = st.columns(7)
        for i, (k, v) in enumerate(sorted(pdata.items())):
            with pcols[i]:
                fmt_v = _fmt_cop(v) if tiene_cop else f"{v:.1f}%"
                st.metric(k, fmt_v, border=True)
        st.html("</div>")

    elif sel == "🔍 Contribución por Riesgo":
        rdata = data["riesgos_cop"] if tiene_cop else riesgos
        if not rdata:
            st.info("No hay datos de desglose disponibles.")
        else:
            st.html(f'<div class="chart-card"><div class="chart-title">Contribución por Riesgo al Sobrecosto Total</div>'
                    f'<div class="chart-subtitle">Distribución del sobrecosto estimado según peso relativo (prob × imp) de cada riesgo</div>')
            df_r = pd.DataFrame(rdata)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_r["riesgo"].tolist(),
                x=df_r["contribucion_porcentaje"].tolist(),
                orientation="h",
                marker_color=VERDE,
                text=[_fmt_cop(c) if tiene_cop else f"{c:.2f}pp" for c in df_r["contribucion_porcentaje"]],
                textposition="outside",
                textfont=dict(color="#000000", size=10),
            ))
            fig.update_layout(height=max(250, len(rdata) * 30), margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#000000", size=11),
                xaxis=dict(title=dict(text="COP" if tiene_cop else "pp", font=dict(color="#000000")), gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000")),
                yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color="#000000", size=11)),
                hovermode="y unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="mc_desglose")
            st.html("</div>")

            with st.expander("Ver tabla detallada"):
                df_display = df_r[["riesgo", "tipo", "categoria", "probabilidad", "impacto", "peso_contribucion", "contribucion_porcentaje"]].copy()
                if tiene_cop:
                    df_display["contribucion_porcentaje"] = df_display["contribucion_porcentaje"].apply(_fmt_cop)
                df_display.columns = ["Riesgo", "Tipo", "Categoría", "Prob", "Imp", "Peso", "Contribución"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)


# ─── RENDER ────────────────────────────────────────────
{"dashboard": _render_dashboard, "predict": _render_predict, "history": _render_history}.get(
    view, _render_dashboard
)()
