"""
App Streamlit - Conciliación Cerro Negro Norte (CNN) v1
Basada en la arquitectura de app_conciliacion_mer_v9 / app_conciliacion_plc_v1
Particularidades CNN:
  - Ley de corte: ue_fe >= 1 AND fem >= 22 AND fe_dtt >= 65.5
  - Marginal:     ue_fe >= 1 AND (fem < 22 OR fe_dtt < 65.5)
  - Ocurrencias:  mac, bre, gui/dis → gyd, est
  - Fases:        análisis completo o por fase (1, 2, 3)
  - Bloques:      10×10×12.5 m
  - Navegación:   vertical de bancos (banco base ± 2 niveles)
  - Scatter:      3 paneles — FeM planeado / Ley de corte / FeDTT
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter, MultipleLocator
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import pearsonr, linregress
from pandas.api.types import CategoricalDtype
import calendar
import io

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Conciliación CNN",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCHEMA_CNN = [
    "block_id",
    "centroid_x", "centroid_y", "centroid_z",
    "dim_x", "dim_y", "dim_z",
    "proportional_volume", "densidad",
    "fe", "fem", "fe_dtt", "dtt", "p", "s",
    "sio2", "al2o3", "v", "mag",
    "ocurrencia", "ue_fe", "categoria",
    "axb", "bwi_cab", "bwi_conc",
    "extraccion", "fase",
]

COLS_DROP_CNN = [
    'volume', 'litologia', 'alteracion', 'intensidad',
    'bound', 'dominio', 'mine', 'fedtt',  # fedtt = alias, usamos fe_dtt
]

COLUMNAS_REQUERIDAS = SCHEMA_CNN

VARS_CALIDAD    = ['fe', 'fem', 'fe_dtt', 'p', 's']
VARS_DISPERSION = ['fe', 'fem', 'fe_dtt', 'p', 's', 'mag']

MESES = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
         7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

# Ocurrencias CNN: gui y dis se unifican en gyd
OCURRENCIAS_CNN = ['mac', 'bre', 'gyd', 'est']

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #f0c040; letter-spacing: -0.5px; }
    .metric-card { background: #1a1d27; border: 1px solid #2a2d3a; border-left: 3px solid #f0c040;
                   padding: 12px 16px; border-radius: 4px; margin-bottom: 8px; }
    .metric-card.red  { border-left-color: #cc4444; }
    .metric-card.green{ border-left-color: #44aa66; }
    .metric-card.blue { border-left-color: #6699ff; }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;
                    font-family: 'IBM Plex Mono', monospace; }
    .metric-value { font-size: 22px; font-weight: 600; color: #f0c040; font-family: 'IBM Plex Mono', monospace; }
    .metric-value.red  { color: #cc4444; }
    .metric-value.green{ color: #44aa66; }
    .metric-value.blue { color: #6699ff; }
    .metric-sub { font-size: 12px; color: #777; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }
    .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #888; }
    .stTabs [aria-selected="true"] { color: #f0c040 !important; border-bottom: 2px solid #f0c040 !important; }
    .cutoff-box { background: #1a1d27; border: 1px solid #2a2d3a; border-left: 3px solid #6699ff;
                  padding: 10px 14px; border-radius: 4px; font-size: 12px; color: #aaa;
                  font-family: 'IBM Plex Mono', monospace; margin: 6px 0; }
    .warn-box { background: #2a1f00; border-left: 3px solid #f0c040; padding: 10px 14px;
                font-size: 12px; color: #ccc; border-radius: 0 4px 4px 0; margin: 8px 0; }
    .fase-box { background: #1a2a1a; border: 1px solid #2a3d2a; border-left: 3px solid #44aa66;
                padding: 10px 14px; border-radius: 4px; font-size: 12px; color: #aaa;
                font-family: 'IBM Plex Mono', monospace; margin: 6px 0; }
    .col-tag { display: inline-block; background: #1e2235; border: 1px solid #2a2d3a; color: #f0c040;
               font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 6px;
               border-radius: 2px; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def leer_csv_cnn(uploaded_file, columnas_extra=None):
    """Lee CSV formato Vulcan. Aplica schema maestro + elimina BOM."""
    try:
        # Leer bytes y eliminar BOM
        if hasattr(uploaded_file, 'read'):
            raw = uploaded_file.read()
        else:
            raw = uploaded_file
        if raw[:3] == b'\xef\xbb\xbf':
            raw = raw[3:]
        buf = io.BytesIO(raw)

        header_df = pd.read_csv(buf, nrows=1, header=None, encoding='utf-8')
        buf.seek(0)
        col_names = [str(c).strip().lower() for c in header_df.iloc[0].tolist()]

        df = pd.read_csv(buf, skiprows=4, header=None,
                         names=col_names, low_memory=False, encoding='utf-8')

        # Normalizar fe_dtt
        if 'fedtt' in df.columns and 'fe_dtt' not in df.columns:
            df = df.rename(columns={'fedtt': 'fe_dtt'})

        # Aplicar schema: solo columnas necesarias, mismo orden
        cols_schema = SCHEMA_CNN + (columnas_extra or [])
        df = df.reindex(columns=[c for c in cols_schema if c in df.columns or c in SCHEMA_CNN])

        # Densidad -99 → 2.7
        if 'densidad' in df.columns:
            df.loc[df['densidad'] == -99, 'densidad'] = 2.7

        # Ocurrencia: gui/dis → gyd
        if 'ocurrencia' in df.columns:
            df['ocurrencia'] = df['ocurrencia'].replace({'gui': 'gyd', 'dis': 'gyd'})

        cols_faltantes = [c for c in SCHEMA_CNN if c not in df.columns]
        return df, cols_faltantes, list(df.columns)
    except Exception as e:
        return None, [], str(e)


def ponderado(df, col_ley, col_ton):
    df_v = df[(df[col_ton] > 0) & (df[col_ley].notna())]
    total = df_v[col_ton].sum()
    if total == 0:
        return np.nan
    return (df_v[col_ley] * df_v[col_ton]).sum() / total


def clasificar_ore_cnn(row, criterios, sufijo=''):
    """
    Clasificación CNN con criterios dinámicos:
      mineral:  ue_fe >= 1 AND todos los criterios se cumplen
      marginal: ue_fe >= 1 AND algún criterio no se cumple
      esteril:  ue_fe < 1
    """
    ue = row.get(f'ue_fe{sufijo}', 0)
    if pd.isna(ue): ue = 0
    if ue < 1:
        return 'esteril'
    # Evaluar criterios
    for c in criterios:
        val = row.get(f"{c['var']}{sufijo}", np.nan)
        if pd.isna(val):
            return 'marginal'
        op = c['op']
        if   op == '>=' and not (val >= c['val']): return 'marginal'
        elif op == '<=' and not (val <= c['val']): return 'marginal'
        elif op == '>'  and not (val >  c['val']): return 'marginal'
        elif op == '<'  and not (val <  c['val']): return 'marginal'
    return 'mineral'


def conciliacion_fn(a, b):
    return f"{a}_{b}"


def make_fig(figsize=(14, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    return fig, ax


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#cccccc')
    ax.tick_params(colors='#444444')
    ax.xaxis.label.set_color('#444444')
    ax.yaxis.label.set_color('#444444')


def save_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf.getvalue()


def plotly_to_html(fig):
    return fig.to_html(include_plotlyjs='cdn', full_html=True).encode('utf-8')


def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')


def annotate_bars(ax, bars, val_series, fmt='{h:.1f}\n{v:.1f}%'):
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if pd.isna(h) or h == 0:
            continue
        try:
            v = val_series.iloc[i]
            ax.annotate(fmt.format(h=h, v=v),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=7, ha='center', va='bottom', color='#333')
        except Exception:
            pass


def _metric_html(label, value, cls=''):
    return (f"<div class='metric-card {cls}'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div>"
            f"</div>")


def titulo_cnn(base, modo_fase, fase_sel, cutoff_str):
    """Genera título según modo de análisis."""
    if modo_fase == "Por fase":
        return f"{base}\n{cutoff_str}\nFase: {fase_sel}"
    return f"{base}\n{cutoff_str}"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ CNN · Conciliación")
    st.markdown("---")

    st.markdown("### 📂 Archivos CSV")
    file_lp  = st.file_uploader("Modelo LP (`output_lp.csv`)", type="csv", key="lp")
    file_cp  = st.file_uploader("Modelo CP (`output_cp.csv`)", type="csv", key="cp")
    st.markdown("**Modelos Mensuales (MP)**")
    files_mp = st.file_uploader("Archivos MP (output_mp_1 … output_mp_12)",
                                type="csv", accept_multiple_files=True, key="mp")
    st.markdown("---")

    # ── MODO DE ANÁLISIS (por encima del período) ──
    st.markdown("### 🔬 Modo de análisis")
    modo_fase = st.radio("Tipo de análisis:",
                         ["Análisis completo", "Por fase"],
                         horizontal=False, key="modo_fase")

    fase_sel = None
    if modo_fase == "Por fase":
        fase_sel = st.selectbox("Fase CNN:", [1, 2, 3], index=0, key="fase_sel")
        st.markdown(f"<div class='fase-box'>✅ Filtrando por Fase {fase_sel}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='fase-box'>🌐 Análisis global (todas las fases)</div>",
                    unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📅 Período")
    ca_p, cb_p = st.columns(2)
    with ca_p:
        anio = st.number_input("Año", value=2026, step=1, min_value=2000, max_value=2100)
    with cb_p:
        mes = st.selectbox("Mes", options=list(MESES.keys()),
                           format_func=lambda x: f"{MESES[x]}", index=0)
    st.markdown("---")

    st.markdown("### ✂️ Ley de Corte")
    st.markdown("<div class='cutoff-box'>CNN base: ue_fe ≥ 1 AND fem ≥ 22 AND fe_dtt ≥ 65.5<br>"
                "Marginal: ue_fe ≥ 1 AND por debajo del corte</div>",
                unsafe_allow_html=True)

    VARS_CORTE_CNN = ["fem", "fe_dtt", "fe", "mag", "al2o3", "p", "s", "sio2", "ue_fe"]
    OPS = ['>=', '<=', '>', '<']
    n_criterios = st.number_input("Número de criterios", min_value=1, max_value=4, value=2, step=1)
    DEFAULTS_CNN = [
        {'var': 'fem',    'op': '>=', 'val': 22.0},
        {'var': 'fe_dtt', 'op': '>=', 'val': 65.5},
        {'var': 'fe',     'op': '>=', 'val': 0.0},
        {'var': 'mag',    'op': '>=', 'val': 0.0},
    ]
    criterios_cnn = []
    for i in range(int(n_criterios)):
        d = DEFAULTS_CNN[i] if i < len(DEFAULTS_CNN) else {'var': 'fem', 'op': '>=', 'val': 0.0}
        st.markdown(f"**Criterio {i+1}**")
        ca_c, cb_c, cc_c = st.columns([2, 1, 1.5])
        with ca_c:
            var = st.selectbox("Var", VARS_CORTE_CNN,
                               index=VARS_CORTE_CNN.index(d['var'])
                               if d['var'] in VARS_CORTE_CNN else 0,
                               key=f"cnn_var_{i}", label_visibility="collapsed")
        with cb_c:
            op = st.selectbox("Op", OPS, index=OPS.index(d['op']),
                              key=f"cnn_op_{i}", label_visibility="collapsed")
        with cc_c:
            val = st.number_input("Val", value=d['val'], step=0.5,
                                  key=f"cnn_val_{i}", label_visibility="collapsed")
        criterios_cnn.append({'var': var, 'op': op, 'val': val})

    cutoff_str = " AND ".join([f"{c['var']} {c['op']} {c['val']}" for c in criterios_cnn])
    st.markdown(f"<div class='cutoff-box'>🎯 {cutoff_str}<br>"
                f"Marginal: ue_fe ≥ 1 y por debajo del corte</div>",
                unsafe_allow_html=True)
    # Mantener referencias a fem/dtt para uso en scatter (tomar del primer criterio que coincida)
    cutoff_fem_lc = next((c['val'] for c in criterios_cnn if c['var'] == 'fem'), 22.0)
    cutoff_dtt_lc = next((c['val'] for c in criterios_cnn if c['var'] == 'fe_dtt'), 65.5)
    st.markdown("---")

    st.markdown("### 📐 Variable de calidad")
    var_cal = st.selectbox("Eje secundario (gráficos)", VARS_CALIDAD,
                           format_func=lambda v: v.upper())
    st.markdown("---")

    st.markdown("### 🎯 Objetivo anual")
    target_ton = st.number_input("Tonelaje objetivo (kt)", value=0, step=100, min_value=0)
    st.markdown("---")

    st.markdown("### 🔍 Columnas requeridas")
    st.markdown("".join([f"<span class='col-tag'>{c}</span>" for c in COLUMNAS_REQUERIDAS]),
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos_raw(lp_bytes, cp_bytes, mp_files_bytes, modo_fase_key, fase_key,
                     _cache_key=None):
    """Solo lee y filtra por fase. NO clasifica ore."""
    df_lp, wl, _ = leer_csv_cnn(lp_bytes)
    df_cp, wc, _ = leer_csv_cnn(cp_bytes)
    df_mp = pd.DataFrame()
    for name, content in mp_files_bytes:
        dt, _, _ = leer_csv_cnn(content)
        if dt is not None:
            try:
                num = int(''.join(filter(str.isdigit, name.replace('.csv','').split('_')[-1])))
                dt  = dt[dt['extraccion'] == num]
            except Exception: pass
            dt['ARCHIVO'] = name
            df_mp = pd.concat([df_mp, dt], ignore_index=True)
    if modo_fase_key == "Por fase" and fase_key is not None:
        if 'fase' in df_lp.columns: df_lp = df_lp[df_lp['fase'] == fase_key]
        if 'fase' in df_cp.columns: df_cp = df_cp[df_cp['fase'] == fase_key]
        if not df_mp.empty and 'fase' in df_mp.columns:
            df_mp = df_mp[df_mp['fase'] == fase_key]
    return df_lp, df_cp, df_mp, wl, wc


st.markdown("# Conciliación Mensual · Cerro Negro Norte")

if not file_lp or not file_cp:
    st.markdown("<div class='warn-box'>⚠️ Carga <strong>output_lp.csv</strong> y "
                "<strong>output_cp.csv</strong> en el panel lateral.</div>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Columna requerida": COLUMNAS_REQUERIDAS}), hide_index=True)
    st.stop()

lp_bytes       = file_lp.read()
cp_bytes       = file_cp.read()
mp_files_bytes = [(f.name, f.read()) for f in files_mp] if files_mp else []

# Hash de cache: detecta archivos modificados con mismo nombre
cache_key = (
    (file_lp.name,  len(lp_bytes)),
    (file_cp.name,  len(cp_bytes)),
    tuple((n, len(c)) for n,c in mp_files_bytes),
)

with st.spinner("Cargando datos..."):
    df_lp, df_cp, df_mp, warn_lp, warn_cp = cargar_datos_raw(
        lp_bytes, cp_bytes, mp_files_bytes,
        modo_fase, fase_sel, _cache_key=cache_key
    )

if warn_lp: st.warning(f"⚠️ LP — columnas faltantes: {', '.join(warn_lp)}")
if warn_cp: st.warning(f"⚠️ CP — columnas faltantes: {', '.join(warn_cp)}")

# ── Clasificación en tiempo real (no recarga archivos al cambiar criterios) ──
def _ore(row, sufijo, crit):
    ue = row.get(f'ue_fe{sufijo}', 0)
    if pd.isna(ue): ue = 0
    if ue < 1: return 'esteril'
    for c in crit:
        col = f"{c['var']}{sufijo}"
        val = row.get(col, np.nan)
        if pd.isna(val): return 'marginal'
        op = c['op']
        if   op == '>=' and not (val >= c['val']): return 'marginal'
        elif op == '<=' and not (val <= c['val']): return 'marginal'
        elif op == '>'  and not (val >  c['val']): return 'marginal'
        elif op == '<'  and not (val <  c['val']): return 'marginal'
    return 'mineral'

def _ore_lp_row(row): return _ore(row, '',    criterios_cnn)
def _ore_cp_row(row): return _ore(row, '_cp', criterios_cnn)

df = pd.merge(df_lp, df_cp, on='block_id', how='outer', suffixes=('', '_cp'))
df['ore_lp']       = df.apply(_ore_lp_row, axis=1)
df['ore_cp']       = df.apply(_ore_cp_row, axis=1)
df['conciliacion'] = df['ore_lp'] + '_' + df['ore_cp']
df['tonelaje_lp']  = df['densidad']    * df['proportional_volume']
df['tonelaje_cp']  = df['densidad_cp'] * df['proportional_volume']

if not df_mp.empty:
    df_mp = df_mp.copy()
    df_mp['ore_mp']      = df_mp.apply(_ore_lp_row, axis=1)
    df_mp['tonelaje_mp'] = df_mp['densidad'] * df_mp['proportional_volume']
    if 'ocurrencia' in df_mp.columns:
        df_mp['ocurrencia'] = df_mp['ocurrencia'].replace({'gui':'gyd','dis':'gyd'})

# ── dfmp_final: merge df_mp + df_cp por block_id (Cell 31/32) — para cascadas ──
dfmp_final = pd.DataFrame()
if not df_mp.empty:
    df_cp_ren = df_cp.rename(columns={
        col: f"{col}_cp" for col in df_cp.columns
        if col not in {'block_id', 'extraccion'}
    })
    if 'fe_dtt_cp' not in df_cp_ren.columns and 'fedtt_cp' in df_cp_ren.columns:
        df_cp_ren = df_cp_ren.rename(columns={'fedtt_cp': 'fe_dtt_cp'})
    df_cp_ren['tonelaje_cp'] = df_cp_ren['densidad_cp'] * df_cp_ren['proportional_volume_cp']
    df_cp_ren['ore_cp']      = df_cp_ren.apply(_ore_cp_row, axis=1)
    dfmp_final = df_mp.merge(
        df_cp_ren.drop(columns=['extraccion'], errors='ignore'),
        on='block_id', how='left'
    )
    dfmp_final['conciliacion'] = (
        dfmp_final['ore_mp'] + '_' +
        dfmp_final['ore_cp'].fillna('esteril')
    )


# ── Tabla mensual ──
def build_tabla(df, df_mp):
    def agg_model(df_in, ore_col, ton_col, suf):
        return (
            df_in[(df_in['extraccion'] > 0) & (df_in[ore_col] == "mineral")]
            .groupby('extraccion', group_keys=False)
            .apply(lambda x: pd.Series({
                f'fe{suf}':       ponderado(x, f'fe{suf}',     ton_col),
                f'fem{suf}':      ponderado(x, f'fem{suf}',    ton_col),
                f'fe_dtt{suf}':   ponderado(x, f'fe_dtt{suf}', ton_col),
                f'p{suf}':        ponderado(x, f'p{suf}',      ton_col),
                f's{suf}':        ponderado(x, f's{suf}',      ton_col),
                f'tonelaje{suf}': x[ton_col].sum() / 1_000,
            }), include_groups=False)
            .reset_index()
        )

    t  = agg_model(df, 'ore_lp', 'tonelaje_lp', '')
    t2 = agg_model(df, 'ore_cp', 'tonelaje_cp', '_cp')
    tabla = pd.merge(t, t2, on='extraccion', how='outer')

    if not df_mp.empty:
        t3 = (
            df_mp[df_mp['ore_mp'] == "mineral"]
            .groupby('extraccion', group_keys=False)
            .apply(lambda x: (lambda f: pd.Series({
                'fe_mp':       ponderado(f, 'fe',     'tonelaje_mp'),
                'fem_mp':      ponderado(f, 'fem',    'tonelaje_mp'),
                'fe_dtt_mp':   ponderado(f, 'fe_dtt', 'tonelaje_mp'),
                'p_mp':        ponderado(f, 'p',      'tonelaje_mp'),
                's_mp':        ponderado(f, 's',      'tonelaje_mp'),
                'tonelaje_mp': f['tonelaje_mp'].sum() / 1_000,
            }))(x[x['ARCHIVO'] == f'output_mp_{int(x.name)}.csv']),
                include_groups=False)
            .dropna(how='all')
            .reset_index()
        )
        tabla = pd.merge(tabla, t3, on='extraccion', how='outer')

    df_m = pd.DataFrame({'extraccion': range(1, 13),
                         'mes': [MESES[i] for i in range(1, 13)]})
    tabla = pd.merge(df_m, tabla, on='extraccion', how='left')

    for suf in ['', '_mp', '_cp']:
        fe_c = f'fe{suf}'; ton_c = f'tonelaje{suf}'; fin_c = f'fino{suf}'
        if fe_c in tabla.columns and ton_c in tabla.columns:
            tabla[fin_c] = tabla[fe_c] * tabla[ton_c] / 100

    total = {'mes': 'Total', 'extraccion': 0}
    for suf in ['', '_mp', '_cp']:
        for col in ['fe', 'fem', 'fe_dtt', 'p', 's']:
            cn, tn = f'{col}{suf}', f'tonelaje{suf}'
            if cn in tabla.columns and tn in tabla.columns:
                total[cn] = ponderado(tabla, cn, tn)
        for col in ['tonelaje', 'fino']:
            cn = f'{col}{suf}'
            if cn in tabla.columns:
                total[cn] = tabla[cn].sum()

    return pd.concat([tabla, pd.DataFrame([total])], ignore_index=True).round(2)


tabla_mensual = build_tabla(df, df_mp)

# Garantizar 12 meses en tabla_plot
_df_12 = pd.DataFrame({'extraccion': range(1, 13), 'mes': [MESES[i] for i in range(1, 13)]})
_tabla_sin_total = tabla_mensual[tabla_mensual['mes'] != 'Total']
tabla_plot = pd.merge(_df_12, _tabla_sin_total.drop(columns=['mes'], errors='ignore'),
                      on='extraccion', how='left').reset_index(drop=True)

mes_nombre = calendar.month_name[mes]
mes_abr    = MESES[mes]
_mes_rows  = tabla_mensual[tabla_mensual['extraccion'] == mes]
mes_row    = _mes_rows if not _mes_rows.empty else tabla_mensual[tabla_mensual['mes'] == 'Total']

fase_lbl = f" · Fase {fase_sel}" if modo_fase == "Por fase" else ""


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Balance", "📈 Gráficos", "🗺️ Visualización",
    "🌊 Cascadas", "🔬 Dispersión", "📐 Cuadrantes FeM", "🧮 Matrices"
])


# ─────────────────────────────────────────────
# TAB 1 — BALANCE
# ─────────────────────────────────────────────
with tab1:
    st.markdown(f"### Balance Mensual — Cerro Negro Norte{fase_lbl}")
    st.markdown(f"*Mineral: {cutoff_str} · Marginal: ue_fe ≥ 1 y por debajo del corte*")

    total_row = tabla_mensual[tabla_mensual['mes'] == 'Total']

    def sv(df_src, col, fmt="{:,.1f}"):
        try:
            v = df_src[col].values[0]
            return "—" if pd.isna(v) else fmt.format(v)
        except Exception:
            return "—"

    st.markdown(f"#### Mes seleccionado — {mes_abr}")
    c1, c2, c3 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_dtt in [
        (c1, "LP", "tonelaje",    "fe",       "fem",    "fe_dtt"),
        (c2, "MP", "tonelaje_mp", "fe_mp",    "fem_mp", "fe_dtt_mp"),
        (c3, "CP", "tonelaje_cp", "fe_cp",    "fem_cp", "fe_dtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} (kt)</div>
                <div class='metric-value'>{sv(mes_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(mes_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(mes_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    FeDTT: {sv(mes_row, k_dtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Acumulado anual")
    c4, c5, c6 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_dtt in [
        (c4, "LP", "tonelaje",    "fe",    "fem",    "fe_dtt"),
        (c5, "MP", "tonelaje_mp", "fe_mp", "fem_mp", "fe_dtt_mp"),
        (c6, "CP", "tonelaje_cp", "fe_cp", "fem_cp", "fe_dtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} acum. (kt)</div>
                <div class='metric-value'>{sv(total_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(total_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(total_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    FeDTT: {sv(total_row, k_dtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    cols_d = [c for c in [
        'mes', 'fe', 'fem', 'fe_dtt', 'p', 's', 'tonelaje', 'fino',
        'fe_mp', 'fem_mp', 'fe_dtt_mp', 'p_mp', 's_mp', 'tonelaje_mp', 'fino_mp',
        'fe_cp', 'fem_cp', 'fe_dtt_cp', 'p_cp', 's_cp', 'tonelaje_cp', 'fino_cp'
    ] if c in tabla_mensual.columns]

    def _highlight_rows(row):
        if row['mes'] == 'Total':
            return ['background-color:#1e2235;font-weight:bold'] * len(row)
        elif row['mes'] == mes_abr:
            return ['font-weight:700;border-left:3px solid #E8650A'] * len(row)
        return [''] * len(row)

    st.dataframe(
        tabla_mensual[cols_d].style
            .format({c: '{:.2f}' for c in cols_d if c != 'mes'})
            .apply(_highlight_rows, axis=1),
        use_container_width=True, hide_index=True
    )
    st.download_button("⬇️ Tabla CSV", csv_bytes(tabla_mensual[cols_d]),
                       "balance_mensual_cnn.csv", "text/csv", key="dl_tabla")


# ─────────────────────────────────────────────
# TAB 2 — GRÁFICOS
# ─────────────────────────────────────────────
with tab2:
    plt.rcParams['font.family'] = 'DejaVu Sans'

    st.markdown("#### ⚙️ Opciones de visualización")
    ca, cb, cc, cd = st.columns(4)
    with ca: ton_ymin = st.number_input("Ton. mín (kt)", value=0,    step=50,  key="ton_ymin")
    with cb: ton_ymax = st.number_input("Ton. máx (kt)", value=1500, step=50,  key="ton_ymax")
    with cc: ley_ymin = st.number_input("Ley mín (%)",  value=15.0, step=1.0, key="ley_ymin")
    with cd: ley_ymax = st.number_input("Ley máx (%)",  value=45.0, step=1.0, key="ley_ymax")

    show_annot = st.checkbox("Mostrar etiquetas en barras", value=True, key="show_annot")

    with st.expander("🎨 Paleta de colores — Barras y Líneas"):
        st.markdown("*Defaults = colores originales del notebook*")
        _pc1, _pc2, _pc3 = st.columns(3)
        with _pc1:
            col_lp      = st.color_picker("LP (barra)",  "#D3D3D3", key="c_lp_bar")
            col_lp_line = st.color_picker("LP (línea)",  "#008000", key="c_lp_line")
        with _pc2:
            col_mp      = st.color_picker("MP (barra)",  "#696969", key="c_mp_bar")
            col_mp_line = st.color_picker("MP (línea)",  "#9370DB", key="c_mp_line")
        with _pc3:
            col_cp      = st.color_picker("CP (barra)",  "#000000", key="c_cp_bar")
            col_cp_line = st.color_picker("CP (línea)",  "#4682B4", key="c_cp_line")

    # ── Gráfico Mensual ──
    titulo_base_mens = f'Extracción Mensual {anio} — Cerro Negro Norte{fase_lbl}'
    st.markdown(f"### Extracción Mensual — LP o MP vs CP  ·  {var_cal.upper()}")
    modelo_sel = st.radio("Comparar CP contra:", ["LP", "MP"], horizontal=True, key="r_mensual")

    col_ton_m  = 'tonelaje'     if modelo_sel == "LP" else 'tonelaje_mp'
    col_cal_m  = var_cal        if modelo_sel == "LP" else f'{var_cal}_mp'
    col_cal_cp = f'{var_cal}_cp'
    bar_color  = col_lp         if modelo_sel == "LP" else col_mp
    line_color = col_lp_line    if modelo_sel == "LP" else col_mp_line

    n_meses = 12
    pos = np.arange(n_meses)
    meses_labels = [MESES[i] for i in range(1, 13)]

    def _safe_array(df, col):
        if col not in df.columns:
            return np.full(12, np.nan)
        arr = df[col].values.copy().astype(float)
        if len(arr) < 12:
            arr = np.concatenate([arr, np.full(12 - len(arr), np.nan)])
        return arr[:12]

    ton_mod = _safe_array(tabla_plot, col_ton_m)
    cal_mod = _safe_array(tabla_plot, col_cal_m)
    ton_cp  = _safe_array(tabla_plot, 'tonelaje_cp')
    cal_cp  = _safe_array(tabla_plot, col_cal_cp)

    fig, ax1 = make_fig((14, 5))
    bw = 0.4
    b1 = ax1.bar(pos - bw/2, ton_mod, bw, color=bar_color, alpha=0.85, label=f'Ton {modelo_sel}')
    b2 = ax1.bar(pos + bw/2, ton_cp,  bw, color=col_cp,   alpha=0.85, label='Ton CP')
    if show_annot:
        annotate_bars(ax1, b1, pd.Series(cal_mod))
        annotate_bars(ax1, b2, pd.Series(cal_cp))

    ax1.set_ylabel('Tonelaje (kt)', color='#444')
    ax1.set_xticks(pos); ax1.set_xticklabels(meses_labels, color='#444')
    ax1.set_xlim(-0.5, 11.5)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.tick_params(colors='#444'); ax1.set_ylim(ton_ymin, ton_ymax)
    style_ax(ax1)

    ax2 = ax1.twinx(); ax2.set_facecolor('white')
    ax2.plot(pos, cal_mod, marker='.', color=line_color,  lw=1.8, label=f'{var_cal.upper()} {modelo_sel}')
    ax2.plot(pos, cal_cp,  marker='.', color=col_cp_line, lw=1.8, label=f'{var_cal.upper()} CP')
    ax2.set_ylabel(f'Ley {var_cal.upper()} (%)', color='#444')
    ax2.set_ylim(ley_ymin, ley_ymax); ax2.tick_params(colors='#444')
    for sp in ['top','bottom']: ax2.spines[sp].set_visible(False)
    for sp in ['left','right']:  ax2.spines[sp].set_color('#ccc')

    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5,-0.12),
               frameon=False, ncol=4, labelcolor='#333', fontsize=9)
    ax1.set_title(f'{titulo_base_mens}\n{cutoff_str}', color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_mensual = save_png(fig)
    st.pyplot(fig); plt.close()
    st.download_button("⬇️ PNG Mensual", png_mensual, "extraccion_mensual_cnn.png",
                       "image/png", key="dl_png_mensual")

    # ── Gráfico Trimestral ──
    st.markdown(f"### Extracción Trimestral — LP / MP / CP  ·  {var_cal.upper()}")
    tp2 = tabla_mensual[tabla_mensual['mes'] != 'Total'].copy()
    tp2['trimestre'] = np.select(
        [tp2['mes'].isin(['Ene','Feb','Mar']), tp2['mes'].isin(['Abr','May','Jun']),
         tp2['mes'].isin(['Jul','Ago','Sep']), tp2['mes'].isin(['Oct','Nov','Dic'])],
        ['1T','2T','3T','4T'], default=pd.NA
    )
    trim_rows = []
    for t in ['1T','2T','3T','4T']:
        g = tp2[tp2['trimestre'] == t]
        row = {'trimestre': t}
        for suf, cal_c, ton_c in [('lp', var_cal, 'tonelaje'),
                                   ('mp', f'{var_cal}_mp', 'tonelaje_mp'),
                                   ('cp', f'{var_cal}_cp', 'tonelaje_cp')]:
            row[f'ton_{suf}'] = g[ton_c].sum(skipna=True) if ton_c in g.columns else np.nan
            row[f'cal_{suf}'] = (ponderado(g, cal_c, ton_c)
                                 if (cal_c in g.columns and ton_c in g.columns) else np.nan)
        trim_rows.append(row)
    tabla_trim = pd.DataFrame(trim_rows)

    fig2, ax1 = make_fig((12, 5))
    pos2 = np.arange(len(tabla_trim)); bw2 = 0.28
    for tc, cc_col, label, color, idx in [('ton_lp','cal_lp','LP',col_lp,0),
                                            ('ton_mp','cal_mp','MP',col_mp,1),
                                            ('ton_cp','cal_cp','CP',col_cp,2)]:
        if tc in tabla_trim.columns:
            bars = ax1.bar(pos2+(idx-1)*bw2, tabla_trim[tc], bw2,
                           color=color, alpha=0.85, label=f'Ton {label}')
            if show_annot:
                annotate_bars(ax1, bars, tabla_trim[cc_col])

    ax1.set_ylabel('Tonelaje (kt)', color='#444')
    ax1.set_xticks(pos2); ax1.set_xticklabels(tabla_trim['trimestre'], color='#444')
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.tick_params(colors='#444'); ax1.set_ylim(ton_ymin, ton_ymax)
    style_ax(ax1)

    ax2b = ax1.twinx(); ax2b.set_facecolor('white')
    for cc_col, label, bar_c, line_c in [
        ('cal_lp', f'{var_cal.upper()} LP', col_lp, col_lp_line),
        ('cal_mp', f'{var_cal.upper()} MP', col_mp, col_mp_line),
        ('cal_cp', f'{var_cal.upper()} CP', col_cp, col_cp_line),
    ]:
        if cc_col in tabla_trim.columns:
            ax2b.plot(pos2, tabla_trim[cc_col], marker='.', color=line_c, lw=1.8, label=label)
    ax2b.set_ylabel(f'Ley {var_cal.upper()} (%)', color='#444')
    ax2b.set_ylim(ley_ymin, ley_ymax); ax2b.tick_params(colors='#444')
    for sp in ['top','bottom']: ax2b.spines[sp].set_visible(False)
    for sp in ['left','right']:  ax2b.spines[sp].set_color('#ccc')

    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5,-0.1),
               frameon=False, ncol=6, labelcolor='#333', fontsize=9)
    ax1.set_title(f'Extracción Trimestral {anio} — Cerro Negro Norte{fase_lbl}\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_trim = save_png(fig2)
    st.pyplot(fig2); plt.close()
    st.download_button("⬇️ PNG Trimestral", png_trim, "extraccion_trimestral_cnn.png",
                       "image/png", key="dl_png_trim")

    # ── Gráfico Acumulado ──
    st.markdown("### Extracción Acumulada")
    fig3, ax3 = make_fig((14, 5))
    for col, label, color, offset in [('tonelaje','LP',col_lp,-18),
                                       ('tonelaje_mp','MP',col_mp,-10),
                                       ('tonelaje_cp','CP',col_cp,8)]:
        if col in tabla_plot.columns:
            cum = tabla_plot[col].cumsum()
            ax3.plot(tabla_plot['mes'], cum, marker='o', label=label, color=color, lw=1.8)
            for i, (m, v) in enumerate(zip(tabla_plot['mes'], cum)):
                if not pd.isna(v):
                    ax3.annotate(f'{v:,.1f}', xy=(i, v), xytext=(0, offset),
                                 textcoords='offset points', ha='center', fontsize=7, color=color)
    if target_ton > 0:
        monthly_target = target_ton / 12
        cum_target = np.cumsum([monthly_target]*12)
        ax3.plot(tabla_plot['mes'], cum_target, '--', color='#cc4444',
                 lw=1.5, label=f'Objetivo ({target_ton:,.0f} kt)')
        ax3.axhline(y=target_ton, color='#cc4444', lw=0.8, linestyle=':', alpha=0.4)
    ax3.set_ylabel('Tonelaje acumulado (kt)', color='#444')
    ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax3.tick_params(colors='#444'); style_ax(ax3)
    ax3.legend(frameon=False, labelcolor='#333')
    ax3.set_title(f'Gráfico Acumulado {anio} — Cerro Negro Norte{fase_lbl}\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_acum = save_png(fig3)
    st.pyplot(fig3); plt.close()
    st.download_button("⬇️ PNG Acumulado", png_acum, "extraccion_acumulada_cnn.png",
                       "image/png", key="dl_png_acum")


# ─────────────────────────────────────────────
# TAB 3 — VISUALIZACIÓN DE BLOQUES (10×10×12.5)
# Navegación vertical: banco base ± hasta 2 bancos
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Visualización de Bloques por Banco")
    st.markdown("*Bloques 10×10×12.5 m — CNN*")

    if df_mp.empty:
        st.info("Carga los archivos MP mensuales para ver la visualización de bloques.")
    else:
        tipo_mapa = st.selectbox("Visualizar por:", ["Clasificación (Ore)", "Ocurrencia", "Ley Fe"])

        cv1, cv2 = st.columns(2)
        with cv1:
            panel_modo = st.radio("Paneles:", ["MP y CP (doble)", "Solo MP", "Solo CP"],
                                  horizontal=True, key="panel_modo")
        with cv2:
            mostrar_grilla = st.checkbox("Mostrar grilla", value=True)

        cg1, cg2 = st.columns(2)
        with cg1:
            grid_x = st.number_input("Espaciado grilla X (m)", value=100, step=50, min_value=10)
        with cg2:
            grid_y = st.number_input("Espaciado grilla Y (m)", value=100, step=50, min_value=10)

        # ── Navegación de bancos ──
        BENCH_HEIGHT = 12.5
        HALF_BENCH   = BENCH_HEIGHT / 2  # 6.25

        # Obtener todas las cotas disponibles para el mes (ordenadas descendente)
        z_disponibles = sorted(
            df_mp.loc[df_mp['extraccion'] == mes, 'centroid_z'].dropna().unique(),
            reverse=False  # ascendente: banco más bajo primero
        )

        if len(z_disponibles) == 0:
            st.warning(f"No hay bloques con extraccion == {mes} en los archivos MP.")
        else:
            n_bancos = min(3, len(z_disponibles))  # máx 3 bancos (base + 2 superiores)

            st.markdown("#### 🏔️ Navegación de bancos")
            nb1, nb2, nb3 = st.columns([1, 1, 3])
            with nb1:
                banco_idx = st.number_input(
                    "Banco (0=más bajo, +1, +2)",
                    min_value=0, max_value=n_bancos - 1, value=0, step=1,
                    key="banco_idx",
                    help="0 = banco más bajo del mes; incrementar sube un banco (12.5 m)"
                )
            with nb2:
                cota_sel = z_disponibles[banco_idx]
                banco_real_label = cota_sel - HALF_BENCH
                st.markdown(f"<div class='cutoff-box'>Banco seleccionado<br>"
                            f"<b>Z = {banco_real_label:.1f} m</b><br>"
                            f"(centroide Z = {cota_sel:.1f})</div>",
                            unsafe_allow_html=True)

            try:
                tam_bloque = 10
                df_mp_cota = df_mp[
                    (np.abs(df_mp['centroid_z'] - cota_sel) < HALF_BENCH + 0.01) &
                    (df_mp['extraccion'] <= mes)
                ].copy()
                df_cp_cota = df.merge(
                    df_mp_cota[['centroid_x','centroid_y','centroid_z']],
                    on=['centroid_x','centroid_y','centroid_z'], how='inner'
                )

                if panel_modo == "Solo MP":
                    paneles = [(df_mp_cota, 'MP')]
                elif panel_modo == "Solo CP":
                    paneles = [(df_cp_cota, 'CP')]
                else:
                    paneles = [(df_mp_cota, 'MP'), (df_cp_cota, 'CP')]

                n_paneles = len(paneles)
                fig, axs = plt.subplots(1, n_paneles, figsize=(7*n_paneles, 6),
                                        facecolor='white', squeeze=False)
                axs = axs[0]
                for ax in axs: ax.set_facecolor('white')

                def _style_ax_blq(ax_i, titulo):
                    ax_i.set_title(titulo, color='#222', fontsize=10, fontweight='bold')
                    ax_i.set_xlabel('Este (m)', color='#444')
                    ax_i.set_ylabel('Norte (m)', color='#444')
                    ax_i.tick_params(colors='#444')
                    if mostrar_grilla:
                        ax_i.xaxis.set_major_locator(MultipleLocator(grid_x))
                        ax_i.yaxis.set_major_locator(MultipleLocator(grid_y))
                        ax_i.grid(True, color='#cccccc', linewidth=0.5)
                    for sp in ['top','right']: ax_i.spines[sp].set_visible(False)
                    for sp in ['bottom','left']: ax_i.spines[sp].set_color('#ccc')
                    ax_i.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
                    ax_i.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))

                def plot_blocks(ax, df_i, col_cat, color_map, is_fn=False):
                    if df_i.empty or col_cat not in df_i.columns: return
                    x    = df_i['centroid_x'].values
                    y    = df_i['centroid_y'].values
                    cats = df_i[col_cat].values
                    exts = df_i['extraccion'].values if 'extraccion' in df_i.columns else np.zeros(len(df_i))
                    patches, facecolors = [], []
                    for xi, yi, cat, ext in zip(x, y, cats, exts):
                        if is_fn:
                            rgb = color_map(cat)
                        else:
                            key = str(cat).lower() if isinstance(cat, str) else str(cat)
                            rgb = color_map.get(key, (0.5, 0.5, 0.5))
                        alpha = 1.0 if ext == mes else 0.25
                        facecolors.append((*rgb[:3], alpha))
                        patches.append(Rectangle((xi - tam_bloque/2, yi - tam_bloque/2),
                                                   tam_bloque, tam_bloque))
                    pc = PatchCollection(patches, facecolor=facecolors, edgecolor='none')
                    ax.add_collection(pc)
                    ax.autoscale(); ax.set_aspect('equal')

                titulo_banco = f"Banco = {banco_real_label:.1f} m"

                if tipo_mapa == "Clasificación (Ore)":
                    # CNN: mineral=violeta, marginal=azul-morado, esteril=gris claro
                    colores = {
                        'mineral': (255/255, 136/255, 255/255),
                        'marginal': (153/255, 136/255, 255/255),
                        'esteril': (238/255, 238/255, 238/255)
                    }
                    col_ore_map = {'MP':'ore_mp', 'CP':'ore_cp'}
                    for ax_i, (df_i, lbl) in zip(axs, paneles):
                        plot_blocks(ax_i, df_i, col_ore_map[lbl], colores)
                        _style_ax_blq(ax_i, f'{lbl} — {titulo_banco}')
                    ley = [Patch(facecolor=c, label=k) for k,c in colores.items()]
                    for ax_i in axs:
                        ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

                elif tipo_mapa == "Ocurrencia":
                    # CNN: mac=rojo, bre=azul, gyd(gui+dis)=azul claro, est=gris
                    co = {
                        'mac': (1.0, 0.0, 0.0),
                        'bre': (0.0, 0.0, 1.0),
                        'gyd': (102/255, 153/255, 255/255),
                        'est': (0.93, 0.93, 0.93)
                    }
                    col_occ_map = {'MP':'ocurrencia', 'CP':'ocurrencia_cp'}
                    for ax_i, (df_i, lbl) in zip(axs, paneles):
                        plot_blocks(ax_i, df_i, col_occ_map[lbl], co)
                        _style_ax_blq(ax_i, f'{lbl} — {titulo_banco}')
                    ley = [Patch(facecolor=c, label=k.upper()) for k,c in co.items()]
                    for ax_i in axs:
                        ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

                else:  # Ley Fe — rangos originales CNN
                    def gfe(fe):
                        if pd.isna(fe): return (0.5,0.5,0.5)
                        for lim, c in [
                            (10, (238/255, 238/255, 238/255)),
                            (15, (0/255,   218/255, 255/255)),
                            (22, (200/255, 255/255, 0/255)),
                            (30, (255/255, 236/255, 0/255)),
                            (40, (255/255, 188/255, 0/255)),
                            (57, (255/255, 0/255,   0/255)),
                        ]:
                            if fe < lim: return c
                        return (255/255, 0/255, 255/255)

                    col_fe_map = {'MP':'fe', 'CP':'fe_cp'}
                    for ax_i, (df_i, lbl) in zip(axs, paneles):
                        plot_blocks(ax_i, df_i, col_fe_map[lbl], gfe, is_fn=True)
                        _style_ax_blq(ax_i, f'{lbl} — {titulo_banco}')
                    ley = [Patch(facecolor=c, label=l) for c, l in [
                        ((238/255,238/255,238/255),'<10'),
                        ((0/255,218/255,255/255),  '10–15'),
                        ((200/255,255/255,0/255),  '15–22'),
                        ((255/255,236/255,0/255),  '22–30'),
                        ((255/255,188/255,0/255),  '30–40'),
                        ((255/255,0/255,0/255),    '40–57'),
                        ((255/255,0/255,255/255),  '≥57'),
                    ]]
                    for ax_i in axs:
                        ax_i.legend(handles=ley, frameon=False, labelcolor='#333', fontsize=8)

                plt.suptitle(f'Cerro Negro Norte{fase_lbl} — {mes_abr} {anio}',
                             color='#222', fontsize=11, y=1.01)
                plt.tight_layout()
                png_bloques = save_png(fig)
                st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Bloques", png_bloques,
                                   "mapa_bloques_cnn.png", "image/png", key="dl_png_bloques")

            except Exception as e:
                st.error(f"Error al generar mapa de bloques: {e}")


# ─────────────────────────────────────────────
# TAB 4 — CASCADAS
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Gráficos de Cascada")

    if df_mp.empty or dfmp_final.empty:
        st.info("Carga los archivos MP para ver las cascadas.")
    else:
        # conc_mp (Cell 33): agrupa dfmp_final por extraccion+conciliacion
        # tonelaje = tonelaje_mp (MP), tonelaje_cp = tonelaje_cp (CP)
        conc_mp = (
            dfmp_final[dfmp_final['extraccion'] != -99]
            .groupby(['extraccion', 'conciliacion'], group_keys=False)
            .apply(lambda x: pd.Series({
                'tonelaje':    x['tonelaje_mp'].sum() if 'tonelaje_mp' in x.columns else 0,
                'tonelaje_cp': x['tonelaje_cp'].sum() if 'tonelaje_cp' in x.columns else 0,
            }), include_groups=False)
            .reset_index()
        )

        def cascada(df_sel, lbl_proy, lbl_real, title, yrange=None):
            def ton(mask, col='tonelaje'):
                sub = df_sel[mask]
                return sub[col].sum() / 1000 if col in sub.columns else 0

            proy  = ton(df_sel['conciliacion'].isin(['mineral_mineral','mineral_marginal','mineral_esteril']))
            mm    = ton(df_sel['conciliacion'] == 'mineral_marginal')
            me    = ton(df_sel['conciliacion'] == 'mineral_esteril')
            mmin  = ton(df_sel['conciliacion'] == 'mineral_mineral')
            aj    = ton(df_sel['conciliacion'] == 'mineral_mineral','tonelaje_cp') - mmin
            marg  = ton(df_sel['conciliacion'] == 'marginal_mineral','tonelaje_cp')
            est   = ton(df_sel['conciliacion'] == 'esteril_mineral', 'tonelaje_cp')
            real  = ton(df_sel['conciliacion'].isin(['mineral_mineral','marginal_mineral','esteril_mineral']),'tonelaje_cp')

            pasos = [lbl_proy, "Mineral Marginal", "Mineral Esteril", "Mineral Mineral",
                     "Ajuste dens.", "Marg→Min", "Est→Min", lbl_real]
            vals  = [proy, -mm, -me, mmin, aj, marg, est, real]
            tipos = ['Proyectado','Pérdida','Pérdida','Subtotal','Ajuste','Ganancia','Ganancia','Real']

            df_res = pd.DataFrame({
                'Paso':          pasos,
                'Tonelaje (kt)': [round(v, 1) for v in vals],
                'Tipo':          tipos,
            })
            proy_ref = proy if proy != 0 else 1
            df_res['% vs Proy.'] = [
                f"{v/proy_ref*100:+.1f}%" if t not in ('Proyectado','Subtotal','Real') else '—'
                for v, t in zip(vals, tipos)
            ]

            fw = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute","relative","relative","total",
                         "relative","relative","relative","total"],
                x=pasos,
                text=[f"{v:,.1f}" for v in vals],
                textposition="outside",
                y=vals,
                increasing={"marker": {"color": "#43A047"}},
                decreasing={"marker": {"color": "#EF5350"}},
                totals={"marker":     {"color": "#42A5F5"}},
                connector={"line": {"color": "#999", "width": 1}},
            ))
            fw.update_layout(
                title=dict(text=f"<b>{title}</b>",
                           font=dict(size=24, color='#000000'), x=0.0, xanchor='left'),
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#000000', size=14),
                height=420, margin=dict(t=80, b=60, l=60, r=30)
            )
            fw.update_yaxes(title_text="Tonelaje (kt)", range=yrange, tickformat=",d",
                            showgrid=False, zerolinecolor='#ccc',
                            tickfont=dict(color='#000000', size=12),
                            title_font=dict(color='#000000', size=12))
            fw.update_xaxes(tickfont=dict(color='#000000', size=12), showgrid=False)
            return fw, df_res

        # ── Cascada Mensual ──
        st.markdown("#### Cascada Mensual")
        modelo_casc_mes = st.radio("Modelo vs CP:", ["MP","LP"], horizontal=True, key="r_casc_mes")
        _cm1, _cm2, _ = st.columns([1,1,3])
        with _cm1: casc_mes_ymin = st.number_input("Ton. mín (kt)", value=0,   step=50, key="casc_mes_ymin")
        with _cm2: casc_mes_ymax = st.number_input("Ton. máx (kt)", value=600, step=50, key="casc_mes_ymax")

        # Cell 36: df_mes_seleccionado = conc_mp[conc_mp['extraccion'] == mes]
        df_mes_sel = conc_mp[conc_mp['extraccion'] == mes]
        if modelo_casc_mes == "MP":
            df_casc_mes  = df_mes_sel
            lbl_proy_mes = f"Proyectado MP {mes_abr}"
        else:
            df_lp_mes = df[df['extraccion'] == mes].copy()
            if not df_lp_mes.empty:
                df_casc_mes = (
                    df_lp_mes.groupby('conciliacion')
                    .apply(lambda x: pd.Series({
                        'tonelaje':    x['tonelaje_lp'].sum(),
                        'tonelaje_cp': x['tonelaje_cp'].sum()
                    }), include_groups=False).reset_index()
                )
            else:
                df_casc_mes = pd.DataFrame(columns=['conciliacion','tonelaje','tonelaje_cp'])
            lbl_proy_mes = f"Budget LP {mes_abr}"

        fig_c1, df_res_c1 = cascada(
            df_casc_mes, lbl_proy_mes, f"Real CP {mes_abr}",
            f"Conciliación Mensual CNN{fase_lbl} — {mes_abr} ({modelo_casc_mes} vs CP)",
            yrange=[casc_mes_ymin, casc_mes_ymax]
        )
        st.plotly_chart(fig_c1, use_container_width=True)

        def _color_tipo(row):
            if row['Tipo']=='Pérdida':  return ['color:#cc4444']*len(row)
            if row['Tipo']=='Ganancia': return ['color:#44aa66']*len(row)
            if row['Tipo'] in ('Proyectado','Real'): return ['font-weight:bold']*len(row)
            return ['']*len(row)

        with st.expander("📋 Ver tabla de cascada mensual"):
            try: st.dataframe(df_res_c1.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except: st.dataframe(df_res_c1, hide_index=True, use_container_width=True)

        d1, d2, _ = st.columns([1,1,3])
        with d1: st.download_button("⬇️ HTML Cascada Mensual", plotly_to_html(fig_c1),
                                    f"cascada_mensual_cnn_{mes_abr}.html", "text/html", key="dl_casc_mes")
        with d2: st.download_button("⬇️ CSV Cascada Mensual", csv_bytes(df_res_c1),
                                    f"cascada_mensual_cnn_{mes_abr}.csv", "text/csv", key="dl_casc_mes_csv")

        st.markdown("---")

        # ── Cascada Acumulada ──
        st.markdown("#### Cascada Acumulada")
        modelo_casc_acum = st.radio("Modelo vs CP:", ["MP","LP"], horizontal=True, key="r_casc_acum")
        _ca1, _ca2, _ = st.columns([1,1,3])
        with _ca1: casc_acum_ymin = st.number_input("Ton. mín (kt)", value=0,    step=50,  key="casc_acum_ymin")
        with _ca2: casc_acum_ymax = st.number_input("Ton. máx (kt)", value=2000, step=100, key="casc_acum_ymax")

        if modelo_casc_acum == "MP":
            # Cell 38: usa conc_mp (dfmp_final) filtrado hasta el mes seleccionado
            df_casc_acum  = conc_mp[
                (conc_mp['extraccion'] > 0) & (conc_mp['extraccion'] <= mes)
            ]
            lbl_proy_acum = f"Proyectado MP (hasta {mes_abr})"
        else:
            # Cell 40: usa df (LP+CP) con tonelaje_lp y tonelaje_cp
            df_lp_g = df[
                (df['extraccion'] > 0) & (df['extraccion'] <= mes)
            ].copy()
            df_casc_acum = (
                df_lp_g.groupby('conciliacion')
                .apply(lambda x: pd.Series({
                    'tonelaje':    x['tonelaje_lp'].sum(),
                    'tonelaje_cp': x['tonelaje_cp'].sum()
                }), include_groups=False).reset_index()
            )
            lbl_proy_acum = f"Budget LP (hasta {mes_abr})"

        fig_c2, df_res_c2 = cascada(
            df_casc_acum, lbl_proy_acum, "Real CP acum.",
            f"Conciliación Acumulada CNN{fase_lbl} — hasta {mes_abr} ({modelo_casc_acum} vs CP)",
            yrange=[casc_acum_ymin, casc_acum_ymax]
        )
        st.plotly_chart(fig_c2, use_container_width=True)

        with st.expander("📋 Ver tabla de cascada acumulada"):
            try: st.dataframe(df_res_c2.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except: st.dataframe(df_res_c2, hide_index=True, use_container_width=True)

        d3, d4, _ = st.columns([1,1,3])
        with d3: st.download_button("⬇️ HTML Cascada Acumulada", plotly_to_html(fig_c2),
                                    f"cascada_acumulada_cnn_{mes_abr}.html", "text/html", key="dl_casc_acum")
        with d4: st.download_button("⬇️ CSV Cascada Acumulada", csv_bytes(df_res_c2),
                                    f"cascada_acumulada_cnn_{mes_abr}.csv", "text/csv", key="dl_casc_acum_csv")


# ─────────────────────────────────────────────
# TAB 5 — DISPERSIÓN (3 paneles exclusivo CNN)
# Panel 1: Mineral Planeado (FeM+FeDTT en MP)
# Panel 2: Conciliación por Ley de Corte (ore_mp vs ore_cp)
# Panel 3: Impacto FeDTT independiente del ore
# ─────────────────────────────────────────────
with tab5:
    st.markdown("### Análisis de Dispersión — Fe y FeDTT")
    st.markdown("*Bloques filtrados por UE_FE. Selector LP o MP vs CP.*")

    if df_mp.empty:
        st.info("Carga los archivos MP para el análisis de dispersión.")
    else:
        _d1, _d2, _d3 = st.columns(3)
        with _d1:
            modelo_disp = st.radio("Modelo vs CP:", ["MP", "LP"],
                                   horizontal=True, key="modelo_disp")
        with _d2:
            periodo_disp = st.radio("Período:", ["Mes", "Acumulado"],
                                    horizontal=True, key="periodo_disp")
        with _d3:
            ue_fe_disp = st.selectbox("UE_FE =", [1, 2, 3, 4], index=0, key="ue_fe_disp")

        try:
            import seaborn as _sns
            arch_d    = f'output_mp_{mes}.csv'
            periodo_lbl = f"Ene–{mes_abr}" if periodo_disp == "Acumulado" else mes_abr
            color_puntos = _sns.color_palette()[0]

            if modelo_disp == "MP":
                if periodo_disp == "Acumulado":
                    df_mp_d = df_mp[df_mp['extraccion'] <= mes].drop_duplicates('block_id').copy()
                    df_cp_d = df_cp[df_cp['extraccion'] <= mes].drop_duplicates('block_id').copy()
                else:
                    df_mp_d = df_mp[df_mp['ARCHIVO'] == arch_d].drop_duplicates('block_id').copy()
                    df_cp_d = df_cp[df_cp['extraccion'] == mes].drop_duplicates('block_id').copy()
                merged = pd.merge(
                    df_cp_d[['block_id','extraccion','fe','fe_dtt','ue_fe']],
                    df_mp_d[['block_id','extraccion','fe','fe_dtt','ue_fe']],
                    on=['block_id','extraccion'], suffixes=('_cp','_mp')
                )
                merged = merged[
                    (merged['ue_fe_cp'] == ue_fe_disp) &
                    (merged['ue_fe_mp'] == ue_fe_disp)
                ].copy()
                lbl_modelo = "MP"
            else:
                if periodo_disp == "Acumulado":
                    df_d = df[df['extraccion'] <= mes].drop_duplicates('block_id').copy()
                else:
                    df_d = df[df['extraccion'] == mes].drop_duplicates('block_id').copy()
                merged = df_d[['block_id','extraccion',
                               'fe','fe_dtt','ue_fe',
                               'fe_cp','fe_dtt_cp','ue_fe_cp']].copy()
                merged = merged.rename(columns={
                    'fe':     'fe_mp',
                    'fe_dtt': 'fe_dtt_mp',
                    'ue_fe':  'ue_fe_mp',
                })
                merged = merged[
                    (merged['ue_fe_cp'] == ue_fe_disp) &
                    (merged['ue_fe_mp'] == ue_fe_disp)
                ].copy()
                lbl_modelo = "LP"

            merged = merged.dropna(subset=['fe_mp','fe_cp']).copy()

            if merged.empty:
                st.warning(f"Sin datos para UE_FE = {ue_fe_disp}.")
            else:
                # Dos paneles: Fe y FeDTT
                comparaciones = [
                    ('fe_cp',     'fe_mp',     'Fe (%)',    (0,  70), (0,  70)),
                    ('fe_dtt_cp', 'fe_dtt_mp', 'FeDTT (%)', (40, 75), (40, 75)),
                ]

                fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
                for ax in axes: ax.set_facecolor('white')

                for idx, (col_cp, col_mp, titulo, xlim, ylim) in enumerate(comparaciones):
                    ax = axes[idx]
                    df_plot = merged[[col_cp, col_mp]].dropna()

                    if df_plot.empty:
                        ax.text(0.5, 0.5, f'{titulo}\nSin datos',
                                transform=ax.transAxes, ha='center', color='#888')
                        continue

                    # Scatterplot (sin línea de tendencia visible separada - solo identidad)
                    _sns.scatterplot(x=col_cp, y=col_mp, data=df_plot,
                                     alpha=1, edgecolors='none', s=20,
                                     ax=ax, color=color_puntos)

                    # Línea identidad 1:1
                    ax.plot(xlim, ylim, color='gray', linestyle='-', lw=1)

                    # Línea de ajuste (sesgo) — igual que notebook
                    sl, ic, *_ = linregress(df_plot[col_cp], df_plot[col_mp])
                    xf = np.linspace(xlim[0], xlim[1], 100)
                    ax.plot(xf, sl*xf+ic, color=color_puntos, linestyle='--', linewidth=1)

                    ax.set_xlim(xlim); ax.set_ylim(ylim)
                    ax.set_xlabel(f'{titulo} CP', color='#444')
                    ax.set_ylabel(f'{titulo} {lbl_modelo}', color='#444')
                    ax.set_title(f'{titulo} - UE_FE = {ue_fe_disp}: {periodo_lbl}',
                                 color='#222', fontsize=11)
                    style_ax(ax)

                    # Pearson en verde (formato notebook original)
                    pr, _ = pearsonr(df_plot[col_cp], df_plot[col_mp])
                    ax.text(0.05, 0.92,
                            f'UE_FE = {ue_fe_disp} ({periodo_lbl})\nCoef. Pearson = {pr:.2f}',
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', color='green')

                    # Histogramas marginales (igual que notebook)
                    ax_top = ax.inset_axes([0, 1.02, 1, 0.2], sharex=ax)
                    ax_top.hist(df_plot[col_cp], bins=20,
                                color=color_puntos, alpha=0.9, linewidth=0)
                    ax_top.axis('off')

                    ax_right = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
                    ax_right.hist(df_plot[col_mp], bins=20,
                                  orientation='horizontal',
                                  color=color_puntos, alpha=0.9, linewidth=0)
                    ax_right.axis('off')

                plt.suptitle(f'Cerro Negro Norte{fase_lbl} — {lbl_modelo} vs CP',
                             color='#222', fontsize=11, y=1.04)
                plt.tight_layout()
                png_disp = save_png(fig, dpi=150)
                st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Dispersión", png_disp,
                                   f"dispersion_cnn_{lbl_modelo}_{mes_abr}.png",
                                   "image/png", key="dl_png_disp")

        except Exception as e:
            st.error(f"Error en dispersión: {e}")


with tab6:
    st.markdown("### Conciliación por Cuadrantes FeM")
    st.markdown("*Scatter y matriz Ore/Waste usan exactamente los mismos bloques.*")

    if df_mp.empty:
        st.info("Carga los archivos MP para el análisis de cuadrantes.")
    else:
        _cq1, _cq2, _cq3 = st.columns(3)
        with _cq1:
            cutoff_fem_q = st.number_input("Corte FeM (%)", value=cutoff_fem_lc,
                                           step=0.5, key="cutoff_fem_q")
        with _cq2:
            periodo_q = st.radio("Período:", ["Mes", "Acumulado"], horizontal=True, key="periodo_q")
        with _cq3:
            HEATMAP_PAL_Q = {"Blues":"Blues","YlOrBr":"YlOrBr","Greens":"Greens",
                             "Purples":"Purples","Oranges":"Oranges","RdYlGn":"RdYlGn",
                             "viridis":"viridis","coolwarm":"coolwarm"}
            hmap_q_lbl = st.selectbox("🎨 Paleta matriz", list(HEATMAP_PAL_Q.keys()),
                                      index=0, key="hmap_q")
        hmap_q_cmap = HEATMAP_PAL_Q[hmap_q_lbl]

        try:
            arch_q = f'output_mp_{mes}.csv'
            if periodo_q == "Acumulado":
                dmp_q = df_mp[df_mp['extraccion'] <= mes].drop_duplicates('block_id').copy()
                dcp_q = df[df['extraccion'] <= mes].drop_duplicates('block_id').copy()
            else:
                dmp_q = df_mp[df_mp['ARCHIVO'] == arch_q].drop_duplicates('block_id').copy()
                dcp_q = df[df['extraccion'] == mes].drop_duplicates('block_id').copy()

            dmp_q['ore_bin'] = dmp_q['ore_mp'].replace({'marginal':'esteril'})
            dcp_q['ore_bin'] = dcp_q['ore_cp'].replace({'marginal':'esteril'})

            union_q = pd.merge(
                dmp_q[['block_id','extraccion','fem','ore_bin',
                        'proportional_volume','dim_x','dim_y','dim_z']],
                dcp_q[['block_id','extraccion','fem','ore_bin']],
                on=['block_id','extraccion'], suffixes=('_mp','_cp')
            )
            union_q = union_q[
                union_q['proportional_volume'] >= 0.75 *
                union_q['dim_x'] * union_q['dim_y'] * union_q['dim_z']
            ].copy()

            df_fem = union_q[['fem_mp','fem_cp']].dropna()

            if df_fem.empty:
                st.warning("Sin datos suficientes.")
            else:
                pr_q, _ = pearsonr(df_fem['fem_cp'], df_fem['fem_mp'])

                coincidencia = (df_fem['fem_cp'] >= cutoff_fem_q) & (df_fem['fem_mp'] >= cutoff_fem_q)
                perdida      = (df_fem['fem_cp'] <  cutoff_fem_q) & (df_fem['fem_mp'] >= cutoff_fem_q)
                esteril_q    = (df_fem['fem_cp'] <  cutoff_fem_q) & (df_fem['fem_mp'] <  cutoff_fem_q)
                ganancia     = (df_fem['fem_cp'] >= cutoff_fem_q) & (df_fem['fem_mp'] <  cutoff_fem_q)

                conteos = {"I": int(coincidencia.sum()), "II": int(perdida.sum()),
                           "III": int(esteril_q.sum()), "IV": int(ganancia.sum())}

                r1 = st.columns(4)
                for col_w, (lbl, val), cls in zip(r1,
                    [("Coincidencia Mineral (I)", conteos["I"]),
                     ("Pérdida Mineral (II)",     conteos["II"]),
                     ("Coincidencia Estéril (III)",conteos["III"]),
                     ("Ganancia Mineral (IV)",    conteos["IV"])],
                    ['green','red','','blue']):
                    with col_w:
                        st.markdown(_metric_html(lbl, f'{val:,}', cls), unsafe_allow_html=True)

                # Clasificación fem para que scatter y matriz sean consistentes
                df_fem = df_fem.copy()
                df_fem['ore_mp_fem'] = np.where(df_fem['fem_mp'] >= cutoff_fem_q, 'mineral','esteril')
                df_fem['ore_cp_fem'] = np.where(df_fem['fem_cp'] >= cutoff_fem_q, 'mineral','esteril')

                # Scatter
                sc_col, _ = st.columns([1,1])
                with sc_col:
                    MAX_FEM_Q = 70
                    fig, ax = make_fig((7, 6))
                    ax.plot([0, MAX_FEM_Q], [0, MAX_FEM_Q], color='black', lw=1)
                    ax.axvline(cutoff_fem_q, ls='--', color='black', lw=1)
                    ax.axhline(cutoff_fem_q, ls='--', color='black', lw=1)
                    ax.scatter(df_fem.loc[esteril_q,   'fem_cp'], df_fem.loc[esteril_q,   'fem_mp'],
                               s=18, color='#d9d9d9', label='Estéril', alpha=0.8)
                    ax.scatter(df_fem.loc[coincidencia,'fem_cp'], df_fem.loc[coincidencia,'fem_mp'],
                               s=18, color='#2ca02c', label='Coincidencia', alpha=0.8)
                    ax.scatter(df_fem.loc[perdida,     'fem_cp'], df_fem.loc[perdida,     'fem_mp'],
                               s=18, color='#ff7f0e', label='Pérdida', alpha=0.8)
                    ax.scatter(df_fem.loc[ganancia,    'fem_cp'], df_fem.loc[ganancia,    'fem_mp'],
                               s=18, color='#1f77b4', label='Ganancia', alpha=0.8)
                    ax.set_xlim(0, MAX_FEM_Q); ax.set_ylim(0, MAX_FEM_Q)
                    ax.set_xlabel('FeM CP (%)', color='#444')
                    ax.set_ylabel('FeM MP (%)', color='#444')
                    style_ax(ax)
                    c = cutoff_fem_q; hi = (c + MAX_FEM_Q)/2; lo = c/2
                    for txt, xp, yp in [('I',hi,hi),('II',lo,hi),('III',lo,lo),('IV',hi,lo)]:
                        ax.text(xp, yp, txt, fontsize=13, weight='bold', color='#555',
                                ha='center', va='center')
                    ax.text(0.02, 0.98,
                            f'{mes_abr} · FeM corte = {cutoff_fem_q}%\n'
                            f'Coef. Pearson: {pr_q:.3f}\n'
                            f'Coincidencia (I): {conteos["I"]:,}\n'
                            f'Pérdida (II): {conteos["II"]:,}\n'
                            f'Estéril (III): {conteos["III"]:,}\n'
                            f'Ganancia (IV): {conteos["IV"]:,}',
                            transform=ax.transAxes, fontsize=9, color='#333', va='top',
                            bbox=dict(facecolor='#f5f5f5', edgecolor='#ddd', boxstyle='round,pad=0.4'))
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.08),
                              ncol=4, frameon=False, fontsize=9, labelcolor='#333')
                    ax.set_title(f'Cuadrantes FeM — {mes_abr}', color='#222', fontsize=11)
                    plt.tight_layout()
                    png_cuad = save_png(fig)
                    st.pyplot(fig); plt.close()
                    st.download_button("⬇️ PNG Cuadrantes", png_cuad,
                                       f"cuadrantes_fem_cnn_{mes_abr}.png",
                                       "image/png", key="dl_png_cuad")

                # Matriz Ore/Waste (mismo criterio fem >= cutoff)
                mx_col, _ = st.columns([1,1])
                with mx_col:
                    orden_ow = ['mineral','esteril']
                    df_fem['ore_mp_fem'] = pd.Categorical(df_fem['ore_mp_fem'], categories=orden_ow, ordered=True)
                    df_fem['ore_cp_fem'] = pd.Categorical(df_fem['ore_cp_fem'], categories=orden_ow, ordered=True)
                    ct_ow = (pd.crosstab(df_fem['ore_mp_fem'], df_fem['ore_cp_fem'])
                               .reindex(index=orden_ow[::-1], columns=orden_ow).fillna(0))
                    ct_ow_pct = ct_ow.div(ct_ow.sum(axis=1), axis=0).mul(100).round(1).fillna(0)
                    annot_ow  = (ct_ow.astype(int).apply(lambda c: c.map('{:,}'.format)).astype(str)
                                 + '\n(' + ct_ow_pct.astype(str) + '%)')
                    periodo_lbl_q = f"Ene–{mes_abr}" if periodo_q == "Acumulado" else mes_nombre
                    fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor='white')
                    ax2.set_facecolor('white')
                    sns.heatmap(ct_ow_pct, cmap=hmap_q_cmap, annot=annot_ow, fmt='',
                                square=True, linewidths=0.5, cbar=True,
                                annot_kws={'size':10}, ax=ax2)
                    ax2.set_title(f'Ore/Waste — {periodo_lbl_q}', color='#222', pad=10)
                    ax2.tick_params(colors='#333')
                    ax2.set_xlabel('CP', color='#444'); ax2.set_ylabel('MP', color='#444')
                    plt.tight_layout()
                    png_ow = save_png(fig2)
                    st.pyplot(fig2); plt.close()
                    st.download_button("⬇️ PNG Ore/Waste", png_ow,
                                       f"matriz_ow_cnn_{mes_abr}.png",
                                       "image/png", key="dl_png_ow")

                if 'mineral' in ct_ow.index and 'mineral' in ct_ow.columns:
                    TP = int(ct_ow.loc['mineral','mineral'])
                    TN = int(ct_ow.loc['esteril','esteril']) if 'esteril' in ct_ow.index else 0
                    FP = int(ct_ow.loc['esteril','mineral']) if 'esteril' in ct_ow.index else 0
                    FN = int(ct_ow.loc['mineral','esteril']) if 'esteril' in ct_ow.columns else 0
                    perd_v = FN/(TP+FN)*100 if (TP+FN)>0 else 0
                    r2 = st.columns(3)
                    with r2[0]: st.markdown(_metric_html('Mineral Proyectado (MP)', f'{TP+FN:,}'), unsafe_allow_html=True)
                    with r2[1]: st.markdown(_metric_html('Mineral Real (CP)', f'{TP+FP:,}'), unsafe_allow_html=True)
                    with r2[2]: st.markdown(_metric_html('Pérdida (Min→Est)', f'{FN:,} ({perd_v:.1f}%)', 'red'), unsafe_allow_html=True)

                with st.expander("📋 Ver tabla de conteos Ore/Waste"):
                    ct_raw_ow = ct_ow.astype(int).copy()
                    ct_raw_ow.index.name = 'MP \\ CP'; ct_raw_ow.columns.name = None
                    st.dataframe(ct_raw_ow, use_container_width=True)

                dl_ow1, dl_ow2, _ = st.columns([1,1,3])
                with dl_ow1: st.download_button("⬇️ CSV Ore/Waste (%)", csv_bytes(ct_ow_pct.reset_index()),
                                                f"ow_pct_cnn_{mes_abr}.csv", "text/csv", key="dl_ow_csv")
                with dl_ow2: st.download_button("⬇️ CSV Conteos", csv_bytes(ct_ow.astype(int).reset_index()),
                                                f"ow_raw_cnn_{mes_abr}.csv", "text/csv", key="dl_ow_raw")

        except Exception as e:
            st.error(f"Error en cuadrantes FeM: {e}")


# ─────────────────────────────────────────────
# TAB 7 — MATRIZ DE OCURRENCIA
# CNN: mac, bre, gyd (gui+dis unificados), est
# ─────────────────────────────────────────────
with tab7:
    st.markdown("### Matriz de Ocurrencia — Modelo Geológico")
    st.markdown("*CNN: mac, bre, gyd (gui+dis unificados), est*")

    if df_mp.empty:
        st.info("Carga los archivos MP para la matriz de ocurrencia.")
    else:
        _mo1, _mo2 = st.columns(2)
        with _mo1:
            HEATMAP_PAL_O = {"Blues":"Blues","YlOrBr":"YlOrBr","Greens":"Greens",
                             "Purples":"Purples","Oranges":"Oranges","RdYlGn":"RdYlGn",
                             "viridis":"viridis","coolwarm":"coolwarm"}
            hmap_o_lbl = st.selectbox("🎨 Paleta de colores", list(HEATMAP_PAL_O.keys()),
                                      index=0, key="hmap_o")
        with _mo2:
            periodo_occ = st.radio("Período:",
                                   ["Mes seleccionado", f"Acumulado (Ene–{mes_abr})"],
                                   horizontal=True, key="r_mat_occ")
        hmap_o_cmap = HEATMAP_PAL_O[hmap_o_lbl]
        es_acum_occ = periodo_occ.startswith("Acum")

        arch_o    = f'output_mp_{mes}.csv'
        df_mp_occ = df_mp[df_mp['ARCHIVO'] == arch_o].drop_duplicates('block_id')
        df_cp_occ = df_cp[df_cp['extraccion'] == mes].drop_duplicates('block_id')

        try:
            # CNN: mac, bre, gyd, est
            orden = ['mac', 'bre', 'gyd', 'est']
            cat_t = CategoricalDtype(categories=orden, ordered=True)

            if es_acum_occ:
                dmp_occ = df_mp[df_mp['extraccion'] <= mes].drop_duplicates('block_id').copy()
                dcp_occ = df_cp[df_cp['extraccion'] <= mes].drop_duplicates('block_id').copy()
            else:
                dmp_occ = df_mp_occ.copy()
                dcp_occ = df_cp_occ.copy()

            # Normalización: gui/dis → gyd (ya hecha en leer_csv, pero por si acaso)
            for d in [dmp_occ, dcp_occ]:
                if 'ocurrencia' in d.columns:
                    d['ocurrencia'] = d['ocurrencia'].replace({'gui':'gyd','dis':'gyd'})

            # En df_cp la ocurrencia puede estar como ocurrencia o ocurrencia_cp
            occ_col_cp = 'ocurrencia_cp' if 'ocurrencia_cp' in dcp_occ.columns else 'ocurrencia'
            if occ_col_cp in dcp_occ.columns:
                dcp_occ[occ_col_cp] = dcp_occ[occ_col_cp].replace({'gui':'gyd','dis':'gyd'})

            dmp = dmp_occ.assign(ocurrencia=lambda d: d['ocurrencia'].astype(cat_t))
            dcp_merge = dcp_occ.copy()
            if occ_col_cp != 'ocurrencia':
                dcp_merge['ocurrencia'] = dcp_merge[occ_col_cp]
            dcp_f = dcp_merge.assign(ocurrencia=lambda d: d['ocurrencia'].astype(cat_t))

            union_occ = (
                dmp[['block_id','extraccion','ocurrencia',
                      'proportional_volume','dim_x','dim_y','dim_z']]
                .merge(dcp_f[['block_id','extraccion','ocurrencia']],
                       on=['block_id','extraccion'], suffixes=('_mp','_cp'))
                .query("proportional_volume >= 0.75 * dim_x * dim_y * dim_z")
            )
            ct_occ = (pd.crosstab(union_occ['ocurrencia_mp'], union_occ['ocurrencia_cp'])
                        .reindex(index=orden[::-1], columns=orden).fillna(0))
            ct_occ_pct = ct_occ.div(ct_occ.sum(axis=1), axis=0).mul(100).round(1).fillna(0)
            annot_occ  = (ct_occ.astype(int).apply(lambda c: c.map('{:,}'.format)).astype(str)
                          + '\n(' + ct_occ_pct.astype(str) + '%)')

            periodo_lbl_o = f"Ene–{mes_abr}" if es_acum_occ else mes_nombre
            fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
            ax.set_facecolor('white')
            sns.heatmap(ct_occ_pct, cmap=hmap_o_cmap, annot=annot_occ, fmt='',
                        square=True, linewidths=0.5, cbar=True,
                        annot_kws={'size':12}, ax=ax)
            ax.set_title(f'Cumplimiento Ore/Waste — CNN: {periodo_lbl_o}', color='#222',
                         fontsize=12, pad=10)
            ax.tick_params(colors='#333')
            ax.set_xlabel('Ocurrencia CP', color='#444')
            ax.set_ylabel('Ocurrencia MP', color='#444')
            plt.tight_layout()

            col_o, _ = st.columns([1,1])
            with col_o:
                png_occ = save_png(fig)
                st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Ocurrencia", png_occ,
                                   f"matriz_occ_cnn_{mes_abr}.png",
                                   "image/png", key="dl_png_occ")

            with st.expander("📋 Ver tabla de conteos"):
                ct_raw_occ = ct_occ.astype(int).copy()
                ct_raw_occ.index.name = 'MP \\ CP'; ct_raw_occ.columns.name = None
                st.dataframe(ct_raw_occ, use_container_width=True)

            dl_o1, dl_o2, _ = st.columns([1,1,3])
            with dl_o1: st.download_button("⬇️ CSV Ocurrencia (%)", csv_bytes(ct_occ_pct.reset_index()),
                                           f"occ_pct_cnn_{mes_abr}.csv", "text/csv", key="dl_occ_csv")
            with dl_o2: st.download_button("⬇️ CSV Conteos", csv_bytes(ct_raw_occ.reset_index()),
                                           f"occ_raw_cnn_{mes_abr}.csv", "text/csv", key="dl_occ_raw")

        except Exception as e:
            st.error(f"Error en matriz de ocurrencia: {e}")
