"""
App Streamlit - Conciliación Minas El Romeral (MER) v9
Mejoras sobre v8:
  FIX 1. Gráfico Mensual — muestra TODOS los meses (1-12) aunque no tengan datos,
          replicando el comportamiento original del notebook Jupyter rev5.
  NEW 2. Paleta de colores editable para barras Y líneas del gráfico mensual/trimestral.
          Defaults idénticos a los colores originales del notebook (lightgray/black/green/steelblue).
  NEW 3. Colores de la heatmap/matrices usan la paleta original del notebook ('Blues').
  NEW 4. Menú de selección de paleta de colores para la heatmap/matrices (Blues, YlOrBr,
          Greens, Purples, Oranges, RdYlGn, viridis), con 'Blues' como default original.
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
    page_title="Conciliación MER",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLUMNAS_REQUERIDAS = [
    "block_id", "centroid_x", "centroid_y", "centroid_z",
    "dim_x", "dim_y", "dim_z", "volume", "proportional_volume",
    "fe", "fem", "fedtt", "dtt", "al2o3", "al2o3dtt",
    "p", "pdtt", "s", "sdtt", "sio2", "sio2dtt",
    "v", "vdtt", "densidad", "mag", "ocurrencia",
    "litologia", "alteracion", "intensidad", "bound",
    "dominio", "ue_fe", "mine", "categoria", "extraccion"
]

VARS_CORTE_CANDIDATAS = [
    "fe", "fem", "fedtt", "dtt", "mag", "al2o3", "al2o3dtt",
    "p", "pdtt", "s", "sdtt", "sio2", "sio2dtt", "v", "vdtt", "ue_fe"
]

VARS_CALIDAD    = ['fe', 'fem', 'fedtt', 'p', 's']
VARS_DISPERSION = ['fe', 'fem', 'fedtt', 'p', 's', 'mag']

MESES = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
         7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

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
    .col-tag { display: inline-block; background: #1e2235; border: 1px solid #2a2d3a; color: #f0c040;
               font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 2px 6px;
               border-radius: 2px; margin: 2px; }
    div[data-testid="stDownloadButton"] > button { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def leer_csv_mer(uploaded_file, columnas_extra=None):
    try:
        try:
            header_df = pd.read_csv(uploaded_file, nrows=1, header=None)
            uploaded_file.seek(0)
            col_names = header_df.iloc[0].tolist()
            df = pd.read_csv(uploaded_file, skiprows=4, header=None, names=col_names, low_memory=False)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        cols_a_usar = list(COLUMNAS_REQUERIDAS) + (columnas_extra or [])
        cols_presentes = [c for c in cols_a_usar if c in df.columns]
        cols_faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
        return df[cols_presentes], cols_faltantes, list(df.columns)
    except Exception as e:
        return None, [], str(e)


def ponderado(df, col_ley, col_ton):
    df_v = df[(df[col_ton] > 0) & (df[col_ley].notna())]
    total = df_v[col_ton].sum()
    if total == 0:
        return np.nan
    return (df_v[col_ley] * df_v[col_ton]).sum() / total


def clasificar_ore(row, criterios, sufijo=''):
    ue_val = row.get(f'ue_fe{sufijo}', 0)
    if pd.isna(ue_val):
        ue_val = 0
    cumple = True
    for c in criterios:
        val = row.get(f"{c['var']}{sufijo}", np.nan)
        if pd.isna(val):
            cumple = False; break
        op = c['op']
        if   op == '>=' and not (val >= c['val']): cumple = False; break
        elif op == '<=' and not (val <= c['val']): cumple = False; break
        elif op == '>'  and not (val >  c['val']): cumple = False; break
        elif op == '<'  and not (val <  c['val']): cumple = False; break
    if   ue_val >= 1 and cumple: return 'mineral'
    elif ue_val >= 1:             return 'marginal'
    else:                         return 'esteril'


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


def plot_blocks_vec(ax, df_i, col_cat, color_src, mes, tam_bloque, is_fn=False):
    if df_i.empty or col_cat not in df_i.columns:
        return
    x    = df_i['centroid_x'].values
    y    = df_i['centroid_y'].values
    cats = df_i[col_cat].values
    exts = df_i['extraccion'].values if 'extraccion' in df_i.columns else np.zeros(len(df_i))

    patches = [Rectangle((xi - tam_bloque/2, yi - tam_bloque/2), tam_bloque, tam_bloque)
               for xi, yi in zip(x, y)]
    facecolors = []
    for cat, ext in zip(cats, exts):
        if is_fn:
            rgb = color_src(cat)
        else:
            key = str(cat).lower() if isinstance(cat, str) else str(cat)
            rgb = color_src.get(key, (0.5, 0.5, 0.5))
        alpha = 1.0 if ext == mes else 0.25
        facecolors.append((*rgb[:3], alpha))
    pc = PatchCollection(patches, facecolor=facecolors, edgecolor='none')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')


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


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ MER · Conciliación")
    st.markdown("---")

    st.markdown("### 📂 Archivos CSV")
    file_lp  = st.file_uploader("Modelo LP (`output_lp.csv`)", type="csv", key="lp")
    file_cp  = st.file_uploader("Modelo CP (`output_cp.csv`)", type="csv", key="cp")
    st.markdown("**Modelos Mensuales (MP)**")
    files_mp = st.file_uploader("Archivos MP (output_mp_1 ... output_mp_12)",
                                type="csv", accept_multiple_files=True, key="mp")
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
    st.markdown("<div class='cutoff-box'>Criterios para mineral/marginal/estéril.</div>",
                unsafe_allow_html=True)
    n_criterios = st.number_input("Número de criterios", min_value=1, max_value=3, value=2, step=1)
    DEFAULTS = [{'var':'fem','op':'>=','val':20.0},
                {'var':'mag','op':'>=','val':65.0},
                {'var':'fe', 'op':'>=','val':0.0}]
    OPS = ['>=', '<=', '>', '<']
    criterios = []
    for i in range(int(n_criterios)):
        d = DEFAULTS[i] if i < len(DEFAULTS) else {'var':'fe','op':'>=','val':0.0}
        st.markdown(f"**Criterio {i+1}**")
        ca, cb, cc = st.columns([2, 1, 1.5])
        with ca:
            var = st.selectbox("Var", VARS_CORTE_CANDIDATAS,
                               index=VARS_CORTE_CANDIDATAS.index(d['var'])
                               if d['var'] in VARS_CORTE_CANDIDATAS else 0,
                               key=f"var_{i}", label_visibility="collapsed")
        with cb:
            op  = st.selectbox("Op", OPS, index=OPS.index(d['op']),
                               key=f"op_{i}", label_visibility="collapsed")
        with cc:
            val = st.number_input("Val", value=d['val'], step=0.5,
                                  key=f"val_{i}", label_visibility="collapsed")
        criterios.append({'var': var, 'op': op, 'val': val})

    resumen_corte = " AND ".join([f"{c['var']} {c['op']} {c['val']}" for c in criterios])
    st.markdown(f"<div class='cutoff-box'>🎯 ue_fe ≥ 1 AND {resumen_corte}</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📐 Variable de calidad")
    var_cal         = st.selectbox("Eje secundario (gráficos)", VARS_CALIDAD,
                                   format_func=lambda v: v.upper())
    var_disp_global = st.selectbox("Dispersión (Tab 5)", VARS_DISPERSION,
                                   format_func=lambda v: v.upper())
    st.markdown("---")

    st.markdown("---")

    st.markdown("### 🎯 Objetivo anual")
    target_ton = st.number_input("Tonelaje objetivo (kt)", value=0, step=100, min_value=0,
                                  help="Si > 0, se traza línea de objetivo en el gráfico acumulado")
    st.markdown("---")

    st.markdown("### ⚙️ Columnas adicionales")
    cols_extra_input = st.text_input("Nombres separados por coma",
                                     placeholder="ej: litologia, dominio")
    columnas_extra = ([c.strip().lower() for c in cols_extra_input.split(",") if c.strip()]
                      if cols_extra_input else [])
    st.markdown("---")
    st.markdown("### 🔍 Columnas requeridas")
    st.markdown("".join([f"<span class='col-tag'>{c}</span>" for c in COLUMNAS_REQUERIDAS]),
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos(lp_bytes, cp_bytes, mp_files_bytes, columnas_extra_tuple):
    ce = list(columnas_extra_tuple)
    df_lp, wl, _ = leer_csv_mer(io.BytesIO(lp_bytes), ce)
    df_cp, wc, _ = leer_csv_mer(io.BytesIO(cp_bytes), ce)
    df_mp = pd.DataFrame()
    for name, content in mp_files_bytes:
        dt, _, _ = leer_csv_mer(io.BytesIO(content), ce)
        if dt is not None:
            try:
                num = int(''.join(filter(str.isdigit, name.replace('.csv', '').split('_')[-1])))
                dt = dt[dt['extraccion'] == num]
            except Exception:
                pass
            dt['ARCHIVO'] = name
            df_mp = pd.concat([df_mp, dt], ignore_index=True)
    return df_lp, df_cp, df_mp, wl, wc


st.markdown("# Conciliación Mensual · Minas El Romeral")

if not file_lp or not file_cp:
    st.markdown("<div class='warn-box'>⚠️ Carga <strong>output_lp.csv</strong> y "
                "<strong>output_cp.csv</strong> en el panel lateral.</div>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Columna requerida": COLUMNAS_REQUERIDAS}), hide_index=True)
    st.stop()

lp_bytes       = file_lp.read()
cp_bytes       = file_cp.read()
mp_files_bytes = [(f.name, f.read()) for f in files_mp] if files_mp else []

with st.spinner("Cargando datos..."):
    df_lp, df_cp, df_mp, warn_lp, warn_cp = cargar_datos(
        lp_bytes, cp_bytes, mp_files_bytes, tuple(columnas_extra)
    )

if warn_lp: st.warning(f"⚠️ LP — columnas faltantes: {', '.join(warn_lp)}")
if warn_cp: st.warning(f"⚠️ CP — columnas faltantes: {', '.join(warn_cp)}")

# ── Merge + clasificación ──
df = pd.merge(df_lp, df_cp, on='block_id', how='outer', suffixes=('', '_cp'))
df['ore_lp']       = df.apply(lambda r: clasificar_ore(r, criterios, ''),    axis=1)
df['ore_cp']       = df.apply(lambda r: clasificar_ore(r, criterios, '_cp'), axis=1)
df['conciliacion'] = df.apply(lambda r: conciliacion_fn(r['ore_lp'], r['ore_cp']), axis=1)
df['tonelaje_lp']  = df['densidad']    * df['proportional_volume']
df['tonelaje_cp']  = df['densidad_cp'] * df['proportional_volume']

if not df_mp.empty:
    df_mp['ore_mp']      = df_mp.apply(lambda r: clasificar_ore(r, criterios, ''), axis=1)
    df_mp['tonelaje_mp'] = df_mp['densidad'] * df_mp['proportional_volume']


# ── Tabla mensual ──
def build_tabla(df, df_mp):
    def agg_model(df_in, ore_col, ton_col, suf):
        return (
            df_in[(df_in['extraccion'] > 0) & (df_in[ore_col] == "mineral")]
            .groupby('extraccion', group_keys=False)
            .apply(lambda x: pd.Series({
                f'fe{suf}':       ponderado(x, f'fe{suf}',    ton_col),
                f'fem{suf}':      ponderado(x, f'fem{suf}',   ton_col),
                f'fedtt{suf}':    ponderado(x, f'fedtt{suf}', ton_col),
                f'p{suf}':        ponderado(x, f'p{suf}',     ton_col),
                f's{suf}':        ponderado(x, f's{suf}',     ton_col),
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
                'fe_mp':       ponderado(f, 'fe',    'tonelaje_mp'),
                'fem_mp':      ponderado(f, 'fem',   'tonelaje_mp'),
                'fedtt_mp':    ponderado(f, 'fedtt', 'tonelaje_mp'),
                'p_mp':        ponderado(f, 'p',     'tonelaje_mp'),
                's_mp':        ponderado(f, 's',     'tonelaje_mp'),
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
        fe_c, ton_c, fin_c = f'fe{suf}', f'tonelaje{suf}', f'fino{suf}'
        if fe_c in tabla.columns and ton_c in tabla.columns:
            tabla[fin_c] = tabla[fe_c] * tabla[ton_c] / 100

    total = {'mes': 'Total', 'extraccion': 0}
    for suf in ['', '_mp', '_cp']:
        for col in ['fe', 'fem', 'fedtt', 'p', 's']:
            cn, tn = f'{col}{suf}', f'tonelaje{suf}'
            if cn in tabla.columns and tn in tabla.columns:
                total[cn] = ponderado(tabla, cn, tn)
        for col in ['tonelaje', 'fino']:
            cn = f'{col}{suf}'
            if cn in tabla.columns:
                total[cn] = tabla[cn].sum()

    return pd.concat([tabla, pd.DataFrame([total])], ignore_index=True).round(2)


tabla_mensual = build_tabla(df, df_mp)

# Garantizar que tabla_plot tenga SIEMPRE los 12 meses en orden, con NaN para los vacíos
_df_12 = pd.DataFrame({'extraccion': range(1, 13), 'mes': [MESES[i] for i in range(1, 13)]})
_tabla_sin_total = tabla_mensual[tabla_mensual['mes'] != 'Total']
tabla_plot = pd.merge(_df_12, _tabla_sin_total.drop(columns=['mes'], errors='ignore'),
                      on='extraccion', how='left').reset_index(drop=True)
cutoff_str    = " AND ".join([f"{c['var']} {c['op']} {c['val']}" for c in criterios])
mes_nombre    = calendar.month_name[mes]
mes_abr       = MESES[mes]

_mes_rows = tabla_mensual[tabla_mensual['extraccion'] == mes]
mes_row   = _mes_rows if not _mes_rows.empty else tabla_mensual[tabla_mensual['mes'] == 'Total']


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Balance", "📈 Gráficos", "🗺️ Visualización",
    "🌊 Cascadas", "🔬 Dispersión", "🧮 Matrices"
])

# ─────────────────────────────────────────────
# TAB 1 — BALANCE
# ─────────────────────────────────────────────
with tab1:
    st.markdown("### Balance Mensual de Mineral de Alimentación")
    st.markdown(f"*Mineral: ue_fe ≥ 1 AND {cutoff_str}*")

    total_row = tabla_mensual[tabla_mensual['mes'] == 'Total']

    def sv(df_src, col, fmt="{:,.1f}"):
        try:
            v = df_src[col].values[0]
            return "—" if pd.isna(v) else fmt.format(v)
        except Exception:
            return "—"

    st.markdown(f"#### Mes seleccionado — {mes_abr}")
    c1, c2, c3 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_fedtt in [
        (c1, "LP", "tonelaje",    "fe",    "fem",    "fedtt"),
        (c2, "MP", "tonelaje_mp", "fe_mp", "fem_mp", "fedtt_mp"),
        (c3, "CP", "tonelaje_cp", "fe_cp", "fem_cp", "fedtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} (kt)</div>
                <div class='metric-value'>{sv(mes_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(mes_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(mes_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    FeDTT: {sv(mes_row, k_fedtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Acumulado anual")
    c4, c5, c6 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_fedtt in [
        (c4, "LP", "tonelaje",    "fe",    "fem",    "fedtt"),
        (c5, "MP", "tonelaje_mp", "fe_mp", "fem_mp", "fedtt_mp"),
        (c6, "CP", "tonelaje_cp", "fe_cp", "fem_cp", "fedtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} acum. (kt)</div>
                <div class='metric-value'>{sv(total_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(total_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(total_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    FeDTT: {sv(total_row, k_fedtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    cols_d = [c for c in [
        'mes', 'fe', 'fem', 'fedtt', 'p', 's', 'tonelaje', 'fino',
        'fe_mp', 'fem_mp', 'fedtt_mp', 'p_mp', 's_mp', 'tonelaje_mp', 'fino_mp',
        'fe_cp', 'fem_cp', 'fedtt_cp', 'p_cp', 's_cp', 'tonelaje_cp', 'fino_cp'
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

    dl1, dl2, _ = st.columns([1, 1, 3])
    with dl1:
        st.download_button("⬇️ Tabla CSV", csv_bytes(tabla_mensual[cols_d]),
                           "balance_mensual.csv", "text/csv", key="dl_tabla")

    st.markdown("### Desviaciones CP vs ...")
    modelo_desv = st.radio("Comparar CP contra:", ["MP", "LP"], horizontal=True, key="r_desv")
    suf_desv    = "_mp" if modelo_desv == "MP" else ""
    lbl_desv    = modelo_desv

    vd = [v for v in ['fe', 'fem', 'fedtt', 'p', 's', 'tonelaje', 'fino']
          if f'{v}_cp' in tabla_mensual.columns and f'{v}{suf_desv}' in tabla_mensual.columns]
    desv = pd.DataFrame({'mes': tabla_mensual['mes']})
    for v in vd:
        ref = tabla_mensual[f'{v}{suf_desv}']
        desv[v] = ((tabla_mensual[f'{v}_cp'] - ref) / ref * 100).replace([np.inf, -np.inf], np.nan)

    st.markdown(f"*Desviación (%) = (CP − {lbl_desv}) / {lbl_desv} × 100*")

    def _color_dev(val):
        if pd.isna(val) or val == 0: return ''
        return 'color:#cc4444;font-weight:600' if val < 0 else 'color:#44aa66;font-weight:600'

    try:
        styled_desv = desv.style.map(_color_dev, subset=vd)
    except AttributeError:
        styled_desv = desv.style.applymap(_color_dev, subset=vd)

    st.dataframe(styled_desv.format({v: '{:+.2f}%' for v in vd}, na_rep='—'),
                 use_container_width=True, hide_index=True)

    with dl2:
        st.download_button("⬇️ Desviaciones CSV", csv_bytes(desv),
                           "desviaciones.csv", "text/csv", key="dl_desv")


# ─────────────────────────────────────────────
# TAB 2 — GRÁFICOS
# ─────────────────────────────────────────────
with tab2:
    plt.rcParams['font.family'] = 'DejaVu Sans'

    st.markdown("#### ⚙️ Opciones de visualización")
    ca, cb, cc, cd = st.columns(4)
    with ca:
        ton_ymin = st.number_input("Ton. mín (kt)", value=0,    step=50,  key="ton_ymin")
    with cb:
        ton_ymax = st.number_input("Ton. máx (kt)", value=700,  step=50,  key="ton_ymax")
    with cc:
        ley_ymin = st.number_input("Ley mín (%)",  value=10.0, step=1.0, key="ley_ymin")
    with cd:
        ley_ymax = st.number_input("Ley máx (%)",  value=70.0, step=1.0, key="ley_ymax")

    show_annot = st.checkbox("Mostrar etiquetas en barras", value=True, key="show_annot")

    with st.expander("🎨 Paleta de colores — Barras y Líneas"):
        st.markdown("*Defaults = colores originales del notebook Jupyter*")
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

    st.markdown(f"### Extracción Mensual — LP o MP vs CP  ·  {var_cal.upper()}")
    modelo_sel = st.radio("Comparar CP contra:", ["LP", "MP"], horizontal=True, key="r_mensual")

    col_ton_m  = 'tonelaje'  if modelo_sel == "LP" else 'tonelaje_mp'
    col_cal_m  = var_cal     if modelo_sel == "LP" else f'{var_cal}_mp'
    col_cal_cp = f'{var_cal}_cp'
    # Colores originales del notebook: LP=lightgray/green, MP=dimgray/purple, CP=black/steelblue
    bar_color  = col_lp      if modelo_sel == "LP" else col_mp
    line_color = col_lp_line if modelo_sel == "LP" else col_mp_line
    label_mod  = modelo_sel

    # ── FIX: usar SIEMPRE los 12 meses (tabla_plot ya tiene los 12 con NaN para vacíos) ──
    # tabla_plot = tabla_mensual sin fila Total, con los 12 meses del año
    n_meses = 12
    pos = np.arange(n_meses)
    meses_labels = [MESES[i] for i in range(1, 13)]

    def _safe_array(df, col):
        """Extrae columna como array de 12 elementos, NaN donde no hay dato."""
        if col not in df.columns:
            return np.full(12, np.nan)
        arr = df[col].values.copy().astype(float)
        # Asegurar longitud 12
        if len(arr) < 12:
            arr = np.concatenate([arr, np.full(12 - len(arr), np.nan)])
        return arr[:12]

    ton_mod = _safe_array(tabla_plot, col_ton_m)
    cal_mod = _safe_array(tabla_plot, col_cal_m)
    ton_cp  = _safe_array(tabla_plot, 'tonelaje_cp')
    cal_cp  = _safe_array(tabla_plot, col_cal_cp)

    fig, ax1 = make_fig((14, 5))
    bw = 0.4
    b1 = ax1.bar(pos - bw/2, ton_mod, bw, color=bar_color,  alpha=0.85, label=f'Ton {label_mod}')
    b2 = ax1.bar(pos + bw/2, ton_cp,  bw, color=col_cp,     alpha=0.85, label='Ton CP')
    if show_annot:
        annotate_bars(ax1, b1, pd.Series(cal_mod))
        annotate_bars(ax1, b2, pd.Series(cal_cp))

    ax1.set_ylabel('Tonelaje (kt)', color='#444')
    ax1.set_xticks(pos)
    ax1.set_xticklabels(meses_labels, color='#444')
    ax1.set_xlim(-0.5, 11.5)  # forzar siempre los 12 meses en el eje X
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.tick_params(colors='#444')
    ax1.set_ylim(ton_ymin, ton_ymax)
    style_ax(ax1)

    ax2 = ax1.twinx(); ax2.set_facecolor('white')
    ax2.plot(pos, cal_mod, marker='.', color=line_color,   lw=1.8, label=f'{var_cal.upper()} {label_mod}')
    ax2.plot(pos, cal_cp,  marker='.', color=col_cp_line,  lw=1.8, label=f'{var_cal.upper()} CP')
    ax2.set_ylabel(f'Ley {var_cal.upper()} (%)', color='#444')
    ax2.set_ylim(ley_ymin, ley_ymax)
    ax2.tick_params(colors='#444')
    for sp in ['top','bottom']: ax2.spines[sp].set_visible(False)
    for sp in ['left','right']: ax2.spines[sp].set_color('#ccc')

    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5,-0.12),
               frameon=False, ncol=4, labelcolor='#333', fontsize=9)
    ax1.set_title(f'Extracción Mensual {anio} — Minas El Romeral\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_mensual = save_png(fig)
    st.pyplot(fig); plt.close()
    st.download_button("⬇️ PNG Mensual", png_mensual,
                       "extraccion_mensual.png", "image/png", key="dl_png_mensual")

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
    ax1.tick_params(colors='#444')
    ax1.set_ylim(ton_ymin, ton_ymax)
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
    ax2b.set_ylim(ley_ymin, ley_ymax)
    ax2b.tick_params(colors='#444')
    for sp in ['top','bottom']: ax2b.spines[sp].set_visible(False)
    for sp in ['left','right']: ax2b.spines[sp].set_color('#ccc')

    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5,-0.1),
               frameon=False, ncol=6, labelcolor='#333', fontsize=9)
    ax1.set_title(f'Extracción Trimestral {anio} — Minas El Romeral\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_trim = save_png(fig2)
    st.pyplot(fig2); plt.close()
    st.download_button("⬇️ PNG Trimestral", png_trim,
                       "extraccion_trimestral.png", "image/png", key="dl_png_trim")

    st.markdown("### Extracción Acumulada")
    fig3, ax3 = make_fig((14, 5))
    for col, label, color, offset in [('tonelaje','LP',col_lp,-18),
                                       ('tonelaje_mp','MP',col_mp,-10),
                                       ('tonelaje_cp','CP',col_cp,8)]:
        if col in tabla_plot.columns:
            cum = tabla_plot[col].cumsum()
            ax3.plot(tabla_plot['mes'], cum, marker='o', label=label, color=color, lw=1.8)
            for i, (m, v) in enumerate(zip(tabla_plot['mes'], cum)):
                ax3.annotate(f'{v:,.1f}', xy=(i, v), xytext=(0, offset),
                             textcoords='offset points', ha='center', fontsize=7, color=color)

    if target_ton > 0:
        monthly_target = target_ton / 12
        cum_target = np.cumsum([monthly_target] * n_meses)
        ax3.plot(tabla_plot['mes'], cum_target, '--', color='#cc4444',
                 lw=1.5, label=f'Objetivo ({target_ton:,.0f} kt)')
        ax3.axhline(y=target_ton, color='#cc4444', lw=0.8, linestyle=':', alpha=0.4)

    ax3.set_ylabel('Tonelaje acumulado (kt)', color='#444')
    ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax3.tick_params(colors='#444')
    style_ax(ax3)
    ax3.legend(frameon=False, labelcolor='#333')
    ax3.set_title(f'Gráfico Acumulado {anio} — Minas El Romeral\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_acum = save_png(fig3)
    st.pyplot(fig3); plt.close()
    st.download_button("⬇️ PNG Acumulado", png_acum,
                       "extraccion_acumulada.png", "image/png", key="dl_png_acum")


# ─────────────────────────────────────────────
# TAB 3 — VISUALIZACIÓN DE BLOQUES
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Visualización de Bloques por Banco")
    if df_mp.empty:
        st.info("Carga los archivos MP mensuales para ver la visualización de bloques.")
    else:
        tipo_mapa  = st.selectbox("Visualizar por:", ["Clasificación (Ore)", "Ocurrencia", "Ley Fe"])
        tam_bloque = st.slider("Tamaño de bloque (m)", 5, 25, 10)

        cv1, cv2 = st.columns(2)
        with cv1:
            panel_modo = st.radio("Paneles:", ["MP y CP (doble)", "Solo MP", "Solo CP"],
                                  horizontal=True, key="panel_modo")
        with cv2:
            mostrar_grilla = st.checkbox("Mostrar grilla", value=True)

        cg2, cg3 = st.columns(2)
        with cg2:
            grid_x = st.number_input("Espaciado grilla X (m)", value=20, step=5, min_value=5)
        with cg3:
            grid_y = st.number_input("Espaciado grilla Y (m)", value=20, step=5, min_value=5)

        try:
            cota       = df_mp.loc[df_mp['extraccion'] == mes, 'centroid_z'].min() + 12.5
            banco_real = cota - 6.25
            df_mp_cota = df_mp[(df_mp['centroid_z'] == cota) & (df_mp['extraccion'] <= mes)].copy()
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
            fig, axs = plt.subplots(1, n_paneles,
                                    figsize=(7*n_paneles, 6), facecolor='white',
                                    squeeze=False)
            axs = axs[0]
            for ax in axs:
                ax.set_facecolor('white')

            def _apply_grid_and_style(ax_i, titulo):
                ax_i.set_title(titulo, color='#222', fontsize=10)
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

            if tipo_mapa == "Clasificación (Ore)":
                colores = {'mineral':(0.85,0.4,0.85), 'marginal':(0.5,0.4,0.85),
                           'esteril':(0.75,0.75,0.75)}
                col_ore_map = {'MP':'ore_mp', 'CP':'ore_cp'}
                for ax_i, (df_i, lbl) in zip(axs, paneles):
                    plot_blocks_vec(ax_i, df_i, col_ore_map[lbl], colores, mes, tam_bloque)
                    _apply_grid_and_style(ax_i, f'{lbl} (Banco={banco_real:.1f})')
                ley = [Patch(facecolor=c, label=k) for k, c in colores.items()]
                for ax_i in axs: ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

            elif tipo_mapa == "Ocurrencia":
                co = {'mac':(0.9,0.1,0.1),'bre':(0.1,0.3,0.9),'gyd':(0.4,0.6,1.0),
                      'dis':(1.0,0.9,0.1),'est':(0.75,0.75,0.75)}
                col_occ_map = {'MP':'ocurrencia', 'CP':'ocurrencia_cp'}
                for ax_i, (df_i, lbl) in zip(axs, paneles):
                    plot_blocks_vec(ax_i, df_i, col_occ_map[lbl], co, mes, tam_bloque)
                    _apply_grid_and_style(ax_i, f'{lbl} (Banco={banco_real:.1f})')
                ley = [Patch(facecolor=c, label=k.upper()) for k, c in co.items()]
                for ax_i in axs: ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

            else:
                def gfe(fe):
                    if pd.isna(fe): return (0.5,0.5,0.5)
                    for lim,c in [(10,(0.93,0.93,0.93)),(15,(0.0,0.85,1.0)),
                                  (22,(0.78,1.0,0.0)),(30,(1.0,0.93,0.0)),
                                  (40,(1.0,0.74,0.0)),(57,(1.0,0.0,0.0))]:
                        if fe < lim: return c
                    return (1.0,0.0,1.0)
                col_fe_map = {'MP':'fe', 'CP':'fe_cp'}
                for ax_i, (df_i, lbl) in zip(axs, paneles):
                    plot_blocks_vec(ax_i, df_i, col_fe_map[lbl], gfe, mes, tam_bloque, is_fn=True)
                    _apply_grid_and_style(ax_i, f'{lbl} (Banco={banco_real:.1f})')
                ley = [Patch(facecolor=c,label=l) for c,l in [
                    ((0.93,0.93,0.93),'<10'),((0.0,0.85,1.0),'10–15'),
                    ((0.78,1.0,0.0),'15–22'),((1.0,0.93,0.0),'22–30'),
                    ((1.0,0.74,0.0),'30–40'),((1.0,0.0,0.0),'40–57'),((1.0,0.0,1.0),'≥57')]]
                for ax_i in axs: ax_i.legend(handles=ley, frameon=False, labelcolor='#333', fontsize=8)

            plt.tight_layout()
            png_bloques = save_png(fig)
            st.pyplot(fig); plt.close()
            st.download_button("⬇️ PNG Bloques", png_bloques,
                               "mapa_bloques.png", "image/png", key="dl_png_bloques")

        except Exception as e:
            st.error(f"Error al generar mapa de bloques: {e}")


# ─────────────────────────────────────────────
# TAB 4 — CASCADAS
# NEW 1: cascada() retorna (fig, df_resumen) con tabla de kt por paso
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Gráficos de Cascada")

    if df_mp.empty:
        st.info("Carga los archivos MP para ver las cascadas.")
    else:
        dfmp_c = df_mp.copy()
        dfcp_c = df_cp.copy().rename(columns={
            col: f"{col}_cp" for col in df_cp.columns if col not in {'block_id','extraccion'}
        })
        dfcp_c['tonelaje_cp'] = dfcp_c['densidad_cp'] * dfcp_c['proportional_volume_cp']
        dfcp_c['ore_cp'] = dfcp_c.apply(lambda r: clasificar_ore(r, criterios, '_cp'), axis=1)

        dfmp_final = dfmp_c.merge(
            dfcp_c.drop(columns=['extraccion'], errors='ignore'),
            on='block_id', how='left'
        )
        dfmp_final['conciliacion'] = dfmp_final.apply(
            lambda r: conciliacion_fn(r.get('ore_mp','esteril'), r.get('ore_cp','esteril')), axis=1
        )

        conc_mp = (
            dfmp_final[dfmp_final['extraccion'] != -99]
            .groupby(['extraccion','conciliacion'], group_keys=False)
            .apply(lambda x: (lambda f: pd.Series({
                'tonelaje':    f['tonelaje_mp'].sum() if not f.empty else 0,
                'tonelaje_cp': f['tonelaje_cp'].sum() if ('tonelaje_cp' in f.columns and not f.empty) else 0,
            }))(x[x['ARCHIVO'] == f'output_mp_{int(x.name[0])}.csv']),
                include_groups=False)
            .reset_index()
        )

        # NEW 1: cascada retorna (figura, tabla_resumen)
        def cascada(df_sel, lbl_proy, lbl_real, title):
            def ton(mask, col='tonelaje'):
                sub = df_sel[mask]
                return sub[col].sum() / 1000 if col in sub.columns else 0

            proy = ton(df_sel['conciliacion'].isin(['mineral_mineral','mineral_marginal','mineral_esteril']))
            mm   = ton(df_sel['conciliacion'] == 'mineral_marginal')
            me   = ton(df_sel['conciliacion'] == 'mineral_esteril')
            mmin = ton(df_sel['conciliacion'] == 'mineral_mineral')
            aj   = ton(df_sel['conciliacion'] == 'mineral_mineral','tonelaje_cp') - mmin
            marg = ton(df_sel['conciliacion'] == 'marginal_mineral','tonelaje_cp')
            est  = ton(df_sel['conciliacion'] == 'esteril_mineral', 'tonelaje_cp')
            real = ton(df_sel['conciliacion'].isin(['mineral_mineral','marginal_mineral','esteril_mineral']),'tonelaje_cp')

            pasos = [lbl_proy, "Min→Marg", "Min→Est", "Min→Min",
                     "Ajuste dens.", "Marg→Min", "Est→Min", lbl_real]
            vals  = [proy, -mm, -me, mmin, aj, marg, est, real]
            tipos = ['Proyectado', 'Pérdida', 'Pérdida', 'Subtotal',
                     'Ajuste', 'Ganancia', 'Ganancia', 'Real']

            df_res = pd.DataFrame({
                'Paso':          pasos,
                'Tonelaje (kt)': [round(v, 1) for v in vals],
                'Tipo':          tipos,
            })
            # Columna % sobre proyectado (para pasos intermedios)
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
                connector={"line":{"color":"rgba(63,63,63,0.5)"}},
            ))
            fw.update_layout(
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(size=18, color='#222'),
                    x=0.0, xanchor='left'
                ),
                paper_bgcolor='white',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333', size=13),
                height=420,
                margin=dict(t=70, b=60, l=60, r=30)
            )
            fw.update_yaxes(
                title_text="Tonelaje (kt)",
                tickformat=",d",
                gridcolor='#eee',
                zerolinecolor='#ccc'
            )
            return fw, df_res

        df_lp_c = df[df['extraccion'] > 0].copy()
        df_lp_c['conciliacion'] = df_lp_c.apply(
            lambda r: conciliacion_fn(r['ore_lp'], r['ore_cp']), axis=1
        )
        df_lp_g = (
            df_lp_c.groupby('conciliacion')
            .apply(lambda x: pd.Series({
                'tonelaje':    x['tonelaje_lp'].sum(),
                'tonelaje_cp': x['tonelaje_cp'].sum()
            }), include_groups=False)
            .reset_index()
        )

        df_mes_sel = conc_mp[conc_mp['extraccion'] == mes]

        # ── Cascada Mensual ──
        st.markdown("#### Cascada Mensual")
        modelo_casc_mes = st.radio("Modelo vs CP:", ["MP", "LP"], horizontal=True, key="r_casc_mes")

        if modelo_casc_mes == "MP":
            df_casc_mes  = df_mes_sel
            lbl_proy_mes = f"Proy. MP {mes_abr}"
        else:
            df_lp_mes = df[df['extraccion'] == mes].copy()
            if not df_lp_mes.empty:
                df_lp_mes['conciliacion'] = df_lp_mes.apply(
                    lambda r: conciliacion_fn(r['ore_lp'], r['ore_cp']), axis=1)
                df_casc_mes = (
                    df_lp_mes.groupby('conciliacion')
                    .apply(lambda x: pd.Series({
                        'tonelaje':    x['tonelaje_lp'].sum(),
                        'tonelaje_cp': x['tonelaje_cp'].sum()
                    }), include_groups=False)
                    .reset_index()
                )
            else:
                df_casc_mes = pd.DataFrame(columns=['conciliacion','tonelaje','tonelaje_cp'])
            lbl_proy_mes = f"Budget LP {mes_abr}"

        fig_c1, df_res_c1 = cascada(df_casc_mes, lbl_proy_mes, f"Real CP {mes_abr}",
                                     f"Conciliación Mensual MER — {mes_abr} ({modelo_casc_mes} vs CP)")
        st.plotly_chart(fig_c1, use_container_width=True)

        # NEW 1: tabla resumen cascada mensual
        with st.expander("📋 Ver tabla de cascada mensual"):
            def _color_tipo(row):
                if row['Tipo'] == 'Pérdida':    return ['color:#cc4444']*len(row)
                if row['Tipo'] == 'Ganancia':   return ['color:#44aa66']*len(row)
                if row['Tipo'] in ('Proyectado','Real'): return ['font-weight:bold']*len(row)
                return ['']*len(row)
            try:
                st.dataframe(df_res_c1.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except Exception:
                st.dataframe(df_res_c1, hide_index=True, use_container_width=True)

        d_casc_m1, d_casc_m2, _ = st.columns([1, 1, 3])
        with d_casc_m1:
            st.download_button("⬇️ HTML Cascada Mensual", plotly_to_html(fig_c1),
                               f"cascada_mensual_{mes_abr}.html", "text/html", key="dl_casc_mes")
        with d_casc_m2:
            st.download_button("⬇️ CSV Cascada Mensual", csv_bytes(df_res_c1),
                               f"cascada_mensual_{mes_abr}.csv", "text/csv", key="dl_casc_mes_csv")

        st.markdown("---")

        # ── Cascada Acumulada ──
        st.markdown("#### Cascada Acumulada")
        modelo_casc_acum = st.radio("Modelo vs CP:", ["MP", "LP"], horizontal=True, key="r_casc_acum")

        if modelo_casc_acum == "MP":
            df_casc_acum  = conc_mp[conc_mp['extraccion'] > 0]
            lbl_proy_acum = "Proy. MP acum."
        else:
            df_casc_acum  = df_lp_g
            lbl_proy_acum = "Budget LP acum."

        fig_c2, df_res_c2 = cascada(df_casc_acum, lbl_proy_acum, "Real CP acum.",
                                     f"Conciliación Acumulada MER — hasta {mes_abr} ({modelo_casc_acum} vs CP)")
        st.plotly_chart(fig_c2, use_container_width=True)

        # NEW 1: tabla resumen cascada acumulada
        with st.expander("📋 Ver tabla de cascada acumulada"):
            try:
                st.dataframe(df_res_c2.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except Exception:
                st.dataframe(df_res_c2, hide_index=True, use_container_width=True)

        d_casc_a1, d_casc_a2, _ = st.columns([1, 1, 3])
        with d_casc_a1:
            st.download_button("⬇️ HTML Cascada Acumulada", plotly_to_html(fig_c2),
                               f"cascada_acumulada_{mes_abr}.html", "text/html", key="dl_casc_acum")
        with d_casc_a2:
            st.download_button("⬇️ CSV Cascada Acumulada", csv_bytes(df_res_c2),
                               f"cascada_acumulada_{mes_abr}.csv", "text/csv", key="dl_casc_acum_csv")


# ─────────────────────────────────────────────
# TAB 5 — DISPERSIÓN
# ─────────────────────────────────────────────
with tab5:
    st.markdown(f"### Análisis de Dispersión — {var_disp_global.upper()}")
    if df_mp.empty:
        st.info("Carga los archivos MP para el análisis de dispersión.")
    else:
        ue_sel = st.selectbox("UE_FE", [1, 2, 3], index=0)

        arch    = f'output_mp_{mes}.csv'
        df_mp_f = df_mp[df_mp['ARCHIVO'] == arch].drop_duplicates('block_id')
        df_cp_f = df_cp[df_cp['extraccion'] == mes].drop_duplicates('block_id')
        merged  = pd.merge(df_cp_f, df_mp_f, on='block_id', suffixes=('_cp','_mp'))

        if 'ue_fe_cp' in merged.columns and 'ue_fe_mp' in merged.columns:
            df_filt = merged[(merged['ue_fe_cp'] == ue_sel) & (merged['ue_fe_mp'] == ue_sel)].copy()
        else:
            df_filt = merged.copy()

        col_x = f'{var_disp_global}_cp'
        col_y = f'{var_disp_global}_mp'

        if df_filt.empty or col_x not in df_filt.columns or col_y not in df_filt.columns:
            st.warning(f"Sin datos o columnas '{col_x}'/'{col_y}' no disponibles.")
        else:
            try:
                clean = df_filt[[col_x, col_y]].dropna()
                xv, yv = clean[col_x], clean[col_y]
                if len(xv) < 3:
                    st.warning("Datos insuficientes para calcular regresión.")
                else:
                    pr, _       = pearsonr(xv, yv)
                    sl, ic, *_ = linregress(xv, yv)
                    max_val     = max(xv.max(), yv.max()) * 1.08

                    fig, ax = make_fig((6, 6))
                    ax.scatter(xv, yv, s=15, alpha=0.6, color='#4472C4', edgecolors='none')
                    ax.plot([0, max_val], [0, max_val], color='#aaa', lw=1, linestyle=':')
                    xf = np.linspace(0, max_val, 100)
                    ax.plot(xf, sl*xf+ic, color='#cc4444', linestyle='--', lw=1.8)
                    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
                    ax.set_xlabel(f'{var_disp_global.upper()} CP (%)', color='#444')
                    ax.set_ylabel(f'{var_disp_global.upper()} MP (%)', color='#444')
                    style_ax(ax)
                    ax.text(0.05, 0.93,
                            f'UE_FE = {ue_sel} · {mes_abr}\n'
                            f'Pearson r = {pr:.3f}\nn = {len(xv):,}',
                            transform=ax.transAxes, fontsize=10, color='#333', va='top',
                            bbox=dict(facecolor='#f5f5f5', edgecolor='#ddd', boxstyle='round,pad=0.4'))
                    ax_t = ax.inset_axes([0, 1.02, 1, 0.18], sharex=ax)
                    ax_r = ax.inset_axes([1.02, 0, 0.18, 1], sharey=ax)
                    ax_t.hist(xv, bins=20, color='#4472C4', alpha=0.75, linewidth=0)
                    ax_t.set_facecolor('white'); ax_t.axis('off')
                    ax_r.hist(yv, bins=20, orientation='horizontal', color='#4472C4', alpha=0.75, linewidth=0)
                    ax_r.set_facecolor('white'); ax_r.axis('off')
                    ax.set_title(f'{var_disp_global.upper()} MP vs CP — {mes_abr}',
                                 color='#222', pad=28, fontsize=11)
                    plt.tight_layout()
                    cd, _ = st.columns([1, 1])
                    with cd:
                        png_disp = save_png(fig)
                        st.pyplot(fig); plt.close()
                        st.download_button("⬇️ PNG Dispersión", png_disp,
                                           f"dispersion_{var_disp_global}_{mes_abr}.png",
                                           "image/png", key="dl_png_disp")
            except Exception as e:
                st.error(f"Error en dispersión: {e}")


# ─────────────────────────────────────────────
# TAB 6 — MATRICES
# NEW 2: precisión, recall, F1, dilución, pérdida (binario)
# NEW 3: opción acumulado para matriz binaria
# ─────────────────────────────────────────────
with tab6:
    st.markdown("### Matrices de Cumplimiento")
    if df_mp.empty:
        st.info("Carga los archivos MP para las matrices.")
    else:
        tipo_m = st.radio("Tipo:", [
            "Ocurrencia MP vs CP",
            "Ore/Waste (binario)"
        ], horizontal=True)

        # Selector de paleta de colores para el heatmap
        HEATMAP_PALETTES = {
            "Blues":    "Blues",
            "YlOrBr":   "YlOrBr",
            "Greens":   "Greens",
            "Purples":  "Purples",
            "Oranges":  "Oranges",
            "RdYlGn":   "RdYlGn",
            "viridis":  "viridis",
            "coolwarm": "coolwarm",
        }
        _hpal_col, _ = st.columns([1, 2])
        with _hpal_col:
            heatmap_palette_label = st.selectbox(
                "🎨 Paleta de colores",
                options=list(HEATMAP_PALETTES.keys()),
                index=0,
                key="heatmap_palette"
            )
        heatmap_cmap = HEATMAP_PALETTES[heatmap_palette_label]

        arch       = f'output_mp_{mes}.csv'
        df_mp_mes2 = df_mp[df_mp['ARCHIVO'] == arch].drop_duplicates('block_id')
        df_cp_mes2 = df_cp[df_cp['extraccion'] == mes].drop_duplicates('block_id')
        min_proy   = None
        min_real   = None

        # NEW 3: selector de período para ambas matrices
        periodo_mat = st.radio(
            "Período:",
            ["Mes seleccionado", f"Acumulado (Ene–{mes_abr})"],
            horizontal=True, key="r_mat_periodo"
        )
        es_acumulado = periodo_mat.startswith("Acum")

        try:
            if tipo_m == "Ocurrencia MP vs CP":
                orden = ['bre','dis','gyd','est']
                cat_t = CategoricalDtype(categories=orden, ordered=True)

                if es_acumulado:
                    dmp_occ = df_mp[df_mp['extraccion'] <= mes].drop_duplicates('block_id')
                    dcp_occ = df_cp[df_cp['extraccion'] <= mes].drop_duplicates('block_id')
                else:
                    dmp_occ = df_mp_mes2
                    dcp_occ = df_cp_mes2

                dmp = dmp_occ.assign(ocurrencia=lambda d: d['ocurrencia'].astype(cat_t))
                dcp = dcp_occ.assign(ocurrencia=lambda d: d['ocurrencia'].astype(cat_t))
                union = (
                    dmp[['block_id','extraccion','ocurrencia',
                          'proportional_volume','dim_x','dim_y','dim_z']]
                    .merge(dcp[['block_id','extraccion','ocurrencia']],
                           on=['block_id','extraccion'], suffixes=('_mp','_cp'))
                    .query("proportional_volume >= 0.75 * dim_x * dim_y * dim_z")
                )
                ct = (pd.crosstab(union['ocurrencia_mp'], union['ocurrencia_cp'])
                        .reindex(index=orden[::-1], columns=orden).fillna(0))

            else:  # Binario
                orden = ['mineral','esteril']
                ore_t = CategoricalDtype(categories=orden, ordered=True)

                if es_acumulado:
                    dmp = df_mp[df_mp['extraccion'] <= mes].drop_duplicates('block_id').copy()
                    dcp = df[df['extraccion'] <= mes].drop_duplicates('block_id').copy()
                else:
                    dmp = df_mp_mes2.copy()
                    dcp = df[df['extraccion'] == mes].drop_duplicates('block_id').copy()

                dmp['ore_mp'] = dmp['ore_mp'].replace({'marginal':'esteril'}).astype(ore_t)
                dcp['ore_cp'] = dcp['ore_cp'].replace({'marginal':'esteril'}).astype(ore_t)
                union = (
                    dmp[['block_id','extraccion','ore_mp',
                          'proportional_volume','dim_x','dim_y','dim_z']]
                    .merge(dcp[['block_id','extraccion','ore_cp']],
                           on=['block_id','extraccion'])
                )
                union = union[union['proportional_volume'] >= 0.75*union['dim_x']*union['dim_y']*union['dim_z']]
                ct = (pd.crosstab(union['ore_mp'], union['ore_cp'])
                        .reindex(index=orden[::-1], columns=orden).fillna(0))
                min_proy = int(ct.loc['mineral'].sum()) if 'mineral' in ct.index else 0
                min_real = int(ct['mineral'].sum())     if 'mineral' in ct.columns else 0

            # ── Métricas base ──
            total_b  = int(ct.values.sum())
            diag_b   = int(np.diag(ct.values).sum())
            accuracy = diag_b / total_b * 100 if total_b > 0 else 0

            if tipo_m == "Ore/Waste (binario)" and 'mineral' in ct.index and 'mineral' in ct.columns:
                TP = int(ct.loc['mineral', 'mineral'])
                TN = int(ct.loc['esteril', 'esteril']) if ('esteril' in ct.index and 'esteril' in ct.columns) else 0
                FP = int(ct.loc['esteril', 'mineral']) if 'esteril' in ct.index else 0
                FN = int(ct.loc['mineral', 'esteril']) if 'esteril' in ct.columns else 0

                perdida_v  = FN / (TP + FN) * 100 if (TP + FN) > 0 else 0
                ganancia_v = FP / (TN + FP) * 100 if (TN + FP) > 0 else 0

                r1 = st.columns(3)
                with r1[0]:
                    st.markdown(_metric_html('Coincidencia Mineral', f'{TP:,}', 'green'), unsafe_allow_html=True)
                with r1[1]:
                    st.markdown(_metric_html('Coincidencia Estéril', f'{TN:,}', 'green'), unsafe_allow_html=True)
                with r1[2]:
                    st.markdown(_metric_html('Pérdida (Min→Est)', f'{FN:,} ({perdida_v:.1f}%)', 'red'), unsafe_allow_html=True)

                r2 = st.columns(3)
                with r2[0]:
                    st.markdown(_metric_html('Ganancia (Est→Min)', f'{FP:,} ({ganancia_v:.1f}%)', 'blue'), unsafe_allow_html=True)
                with r2[1]:
                    st.markdown(_metric_html('Mineral Proyectado (MP)', f'{min_proy:,}' if min_proy is not None else '—'), unsafe_allow_html=True)
                with r2[2]:
                    st.markdown(_metric_html('Mineral Real (CP)', f'{min_real:,}' if min_real is not None else '—'), unsafe_allow_html=True)

            else:
                # Ocurrencia: sin métricas adicionales
                pass

            # ── Heatmap ──
            ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1).fillna(0)
            annot  = (ct.astype(int).apply(lambda c: c.map('{:,}'.format)).astype(str)
                      + '\n(' + ct_pct.astype(str) + '%)')

            periodo_lbl = f"Ene–{mes_abr}" if es_acumulado else mes_nombre
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
            ax.set_facecolor('white')
            sns.heatmap(ct_pct, cmap=heatmap_cmap, annot=annot, fmt='',
                        square=True, linewidths=0.5, cbar=True,
                        annot_kws={'size':10}, ax=ax)
            ax.set_title(f'Cumplimiento — {periodo_lbl}', color='#222', pad=10)
            ax.tick_params(colors='#333')
            ax.set_xlabel('CP', color='#444'); ax.set_ylabel('MP', color='#444')
            plt.tight_layout()

            col_m, _ = st.columns([1, 1])
            with col_m:
                png_mat = save_png(fig)
                st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Matriz", png_mat,
                                   f"matriz_{mes_abr}.png", "image/png", key="dl_png_matriz")

            # Tabla de conteos raw expandible
            with st.expander("📋 Ver tabla de conteos"):
                ct_raw = ct.astype(int).copy()
                ct_raw.index.name   = 'MP \\ CP'
                ct_raw.columns.name = None
                st.dataframe(ct_raw, use_container_width=True)

            dl_m1, dl_m2, _ = st.columns([1, 1, 3])
            with dl_m1:
                st.download_button("⬇️ CSV Matriz (%)", csv_bytes(ct_pct.reset_index()),
                                   f"matriz_pct_{mes_abr}.csv", "text/csv", key="dl_mat_csv")
            with dl_m2:
                st.download_button("⬇️ CSV Conteos", csv_bytes(ct_raw.reset_index()),
                                   f"matriz_raw_{mes_abr}.csv", "text/csv", key="dl_mat_raw_csv")

        except Exception as e:
            st.error(f"Error en matriz: {e}")
