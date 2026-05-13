"""
App Streamlit - Conciliación Mina Los Colorados (MLC) v1
Particularidades MLC vs CNN/MER/PLC:
  - 12 LP + 12 MP + 12 CP (un archivo por mes para cada modelo)
  - Clave de merge: id = block_id + "_" + periodo (no block_id solo)
  - Columna temporal: 'periodo' (no 'extraccion')
  - Fases: f6a, f6b, fde (strings, no numéricas)
  - Ley de corte: ue_fe >= 1 AND fem >= 28 (sin fe_dtt)
  - Análisis por ocurrencia: mac, bre, gyd (filtro adicional en tabla mensual)
  - Finos calculados con dtt (no fe)
  - Dispersión: Fe MP vs Fe CP, un gráfico por UE_FE seleccionado (sin FeDTT)
  - Bloques 12.5x12.5 (tam_bloque=12.5), columnas con sufijos _mp y _cp en df
  - Merge completo con add_suffix (todas las columnas llevan sufijo)
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
    page_title="Conciliación MLC",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MESES = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
         7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

VARS_CALIDAD    = ['fe', 'fem', 'fedtt', 'dtt', 'p', 's']
OCURRENCIAS_MLC = ['mac', 'bre', 'gyd', 'est']
FASES_MLC       = ['f6a', 'f6b', 'fde']

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

# Columnas necesarias (usecols) — incluye las solicitadas
COLS_BASE = [
    'block_id', 'fase', 'ocurrencia',
    'centroid_x', 'centroid_y', 'centroid_z',
    'dim_x', 'dim_y', 'dim_z',
    'proportional_volume', 'densidad',
    'ue_fe', 'fe', 'fem', 'fedtt', 'dtt', 'p', 's',
    'sio2', 'al2o3', 'v', 'axb', 'bwi_cab', 'bwi_conc',
    'extraccion',
]
COLS_F32 = [
    'centroid_x', 'centroid_y', 'centroid_z',
    'dim_x', 'dim_y', 'dim_z', 'proportional_volume', 'densidad',
    'ue_fe', 'fe', 'fem', 'fedtt', 'dtt', 'p', 's',
    'sio2', 'al2o3', 'v', 'axb', 'bwi_cab', 'bwi_conc',
]
COLS_CAT = ['fase', 'ocurrencia']
COLS_SUFIJO = [
    'fe', 'fem', 'fedtt', 'dtt', 'p', 's',
    'sio2', 'al2o3', 'v', 'axb', 'bwi_cab', 'bwi_conc',
    'densidad', 'proportional_volume',
    'ue_fe', 'ocurrencia', 'fase',
    'centroid_x', 'centroid_y', 'centroid_z',
    'dim_x', 'dim_y', 'dim_z',
]


def leer_csv_mlc(uploaded_file):
    """Lectura CSV formato Vulcan. Optimizado: usecols + float32 + category."""
    try:
        header_df = pd.read_csv(uploaded_file, nrows=1, header=None, encoding='latin1')
        uploaded_file.seek(0)
        col_names = [str(c).strip().lower() for c in header_df.iloc[0].tolist()]

        cols_usar = [c for c in COLS_BASE if c in col_names]
        col_idx   = [col_names.index(c) for c in cols_usar]
        dtype_map = {c: 'float32' for c in COLS_F32 if c in cols_usar}

        try:
            df = pd.read_csv(
                uploaded_file, skiprows=4, header=None,
                names=col_names, usecols=col_idx,
                dtype=dtype_map, low_memory=False, encoding='latin1'
            )
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='latin1')
            df.columns = [str(c).strip().lower() for c in df.columns]
            df = df[[c for c in COLS_BASE if c in df.columns]]

        if 'densidad' in df.columns:
            df['densidad'] = df['densidad'].replace(-99.0, 2.7).astype('float32')
        for c in COLS_CAT:
            if c in df.columns:
                df[c] = df[c].astype('category')
        return df, []
    except Exception as e:
        return None, [str(e)]


def ponderado(df, col_ley, col_ton):
    df_v = df[(df[col_ton] > 0) & (df[col_ley].notna())]
    total = df_v[col_ton].sum()
    if total == 0: return np.nan
    return (df_v[col_ley] * df_v[col_ton]).sum() / total


@st.cache_data(show_spinner=False)
def cargar_datos_mlc(lp_files_bytes, mp_files_bytes, cp_files_bytes,
                     modo_fase_key, fase_key, cutoff_fem):
    def _leer_grupo(files_bytes):
        df_out = pd.DataFrame()
        for name, content in files_bytes:
            dt, _ = leer_csv_mlc(io.BytesIO(content))
            if dt is None: continue
            try:
                num = int(''.join(filter(str.isdigit, name.replace('.csv','').split('_')[-1])))
            except Exception:
                num = 1
            dt['periodo'] = np.int16(num)
            dt['archivo'] = name
            if 'block_id' in dt.columns:
                dt['id'] = dt['block_id'].astype(str) + "_" + dt['periodo'].astype(str)
            df_out = pd.concat([df_out, dt], ignore_index=True)
        return df_out

    df_lp = _leer_grupo(lp_files_bytes)
    df_mp = _leer_grupo(mp_files_bytes)
    df_cp = _leer_grupo(cp_files_bytes)

    if modo_fase_key == "Por fase" and fase_key:
        for d in [df_lp, df_mp, df_cp]:
            if 'fase' in d.columns:
                d.drop(d[d['fase'].astype(str) != fase_key].index, inplace=True)

    if df_lp.empty: return pd.DataFrame(), pd.DataFrame()

    # Merge liviano: solo columnas útiles con sufijo
    def _preparar_sufijo(df_src, suf):
        cols_disp = [c for c in COLS_SUFIJO if c in df_src.columns]
        rename_map = {c: f"{c}{suf}" for c in cols_disp}
        return df_src[['id'] + cols_disp].rename(columns=rename_map)

    df_mp_ren = _preparar_sufijo(df_mp, '_mp')
    df_cp_ren = _preparar_sufijo(df_cp, '_cp')

    df = df_lp.merge(df_mp_ren, on='id', how='outer')
    df = df.merge(df_cp_ren, on='id', how='outer')
    del df_mp_ren, df_cp_ren

    # Tonelajes float32
    if 'densidad' in df.columns and 'proportional_volume' in df.columns:
        df['tonelaje'] = (df['densidad'] * df['proportional_volume']).astype('float32')
    pv = 'proportional_volume_mp' if 'proportional_volume_mp' in df.columns else 'proportional_volume'
    if 'densidad_mp' in df.columns:
        df['tonelaje_mp'] = (df['densidad_mp'] * df[pv]).astype('float32')
    pv2 = 'proportional_volume_cp' if 'proportional_volume_cp' in df.columns else 'proportional_volume'
    if 'densidad_cp' in df.columns:
        df['tonelaje_cp'] = (df['densidad_cp'] * df[pv2]).astype('float32')

    # Clasificación ore vectorizada
    def _ore_vec(ue_col, fem_col):
        ue  = pd.to_numeric(df.get(ue_col,  pd.Series(0,       index=df.index)), errors='coerce').fillna(0)
        fem = pd.to_numeric(df.get(fem_col, pd.Series(np.nan,  index=df.index)), errors='coerce')
        ore = pd.Series('esteril', index=df.index, dtype=object)
        ore[ue >= 1] = 'marginal'
        ore[(ue >= 1) & (fem >= cutoff_fem)] = 'mineral'
        return ore.astype('category')

    df['ore_lp'] = _ore_vec('ue_fe',    'fem')
    df['ore_mp'] = _ore_vec('ue_fe_mp', 'fem_mp')
    df['ore_cp'] = _ore_vec('ue_fe_cp', 'fem_cp')

    df['conciliacion']    = (df['ore_mp'].astype(str) + '_' + df['ore_cp'].astype(str)).astype('category')
    df['conciliacion_lp'] = (df['ore_lp'].astype(str) + '_' + df['ore_cp'].astype(str)).astype('category')

    return df, df_mp


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ MLC · Conciliación")
    st.markdown("---")

    st.markdown("### 📂 Archivos CSV")
    st.markdown("*MLC: 12 LP + 12 MP + 12 CP (uno por mes)*")
    files_lp = st.file_uploader("Modelos LP (output_lp_1 … output_lp_12)",
                                type="csv", accept_multiple_files=True, key="lp")
    files_mp = st.file_uploader("Modelos MP (output_mp_1 … output_mp_12)",
                                type="csv", accept_multiple_files=True, key="mp")
    files_cp = st.file_uploader("Modelos CP (output_cp_1 … output_cp_12)",
                                type="csv", accept_multiple_files=True, key="cp")
    st.markdown("---")

    st.markdown("### 🔬 Modo de análisis")
    modo_fase = st.radio("Tipo de análisis:", ["Análisis completo", "Por fase"],
                         horizontal=False, key="modo_fase")
    fase_sel = None
    if modo_fase == "Por fase":
        fase_sel = st.selectbox("Fase MLC:", FASES_MLC, index=0, key="fase_sel",
                                format_func=str.upper)
        st.markdown(f"<div class='fase-box'>✅ Filtrando por Fase {fase_sel.upper()}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='fase-box'>🌐 Análisis global (todas las fases)</div>",
                    unsafe_allow_html=True)

    st.markdown("### 🔍 Filtro por Ocurrencia")
    modo_ocu = st.radio("Análisis por ocurrencia:", ["Completo", "Por ocurrencia"],
                        horizontal=False, key="modo_ocu")
    ocu_sel = None
    if modo_ocu == "Por ocurrencia":
        ocu_sel = st.selectbox("Ocurrencia:", ['mac', 'bre', 'gyd'],
                               format_func=str.upper, key="ocu_sel")
        st.markdown(f"<div class='fase-box'>✅ Filtrando por Ocurrencia {ocu_sel.upper()}</div>",
                    unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📅 Período")
    ca_p, cb_p = st.columns(2)
    with ca_p: anio = st.number_input("Año", value=2026, step=1, min_value=2000, max_value=2100)
    with cb_p: mes  = st.selectbox("Mes", options=list(MESES.keys()),
                                    format_func=lambda x: MESES[x], index=0)
    st.markdown("---")

    st.markdown("### ✂️ Ley de Corte")
    st.markdown("<div class='cutoff-box'>MLC: ue_fe ≥ 1 AND fem ≥ 28%<br>"
                "Marginal: ue_fe ≥ 1 AND fem &lt; 28%</div>", unsafe_allow_html=True)
    cutoff_fem_lc = st.number_input("Corte FeM (%)", value=28.0, step=0.5, key="lc_fem")
    cutoff_str = f"FeM ≥ {cutoff_fem_lc}%"
    st.markdown(f"<div class='cutoff-box'>🎯 {cutoff_str}</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📐 Variable de calidad")
    var_cal = st.selectbox("Eje secundario (gráficos)", VARS_CALIDAD,
                           format_func=lambda v: v.upper())
    st.markdown("---")
    st.markdown("### 🎯 Objetivo anual")
    target_ton = st.number_input("Tonelaje objetivo (kt)", value=0, step=100, min_value=0)
    st.markdown("---")

if not files_lp or not files_mp or not files_cp:
    st.markdown("<div class='warn-box'>⚠️ Carga los archivos LP, MP y CP (12 por cada modelo) "
                "en el panel lateral.</div>", unsafe_allow_html=True)
    st.stop()

lp_bytes = [(f.name, f.read()) for f in files_lp]
mp_bytes = [(f.name, f.read()) for f in files_mp]
cp_bytes = [(f.name, f.read()) for f in files_cp]

with st.spinner("Cargando y mergeando datos..."):
    df, df_mp = cargar_datos_mlc(
        lp_bytes, mp_bytes, cp_bytes, modo_fase, fase_sel, cutoff_fem_lc
    )

if df.empty:
    st.error("No se pudieron cargar los datos. Verifica los archivos."); st.stop()

# Período y etiquetas
mes_nombre = calendar.month_name[mes]
mes_abr    = MESES[mes]
fase_lbl   = f" · Fase {fase_sel.upper()}" if modo_fase == "Por fase" and fase_sel else ""
ocu_lbl    = f" · {ocu_sel.upper()}" if modo_ocu == "Por ocurrencia" and ocu_sel else ""


# ─────────────────────────────────────────────
# TABLA MENSUAL
# ─────────────────────────────────────────────
def build_tabla_mlc(df, cutoff_fem, ocu_sel=None):
    """Tabla mensual LP/MP/CP agrupada por periodo."""

    def filtro_ore(df, ore_col, ocu_col=None):
        mask = df[ore_col] == 'mineral'
        if ocu_sel and ocu_col and ocu_col in df.columns:
            mask = mask & (df[ocu_col] == ocu_sel)
        return mask & (df['periodo'] > 0)

    def agg_grupo(df_in, ore_col, ton_col, ocu_col, suf):
        mask = filtro_ore(df_in, ore_col, ocu_col)
        return (
            df_in[mask].groupby('periodo', group_keys=False)
            .apply(lambda x: pd.Series({
                f'fe{suf}':       ponderado(x, f'fe{suf}',    ton_col),
                f'fem{suf}':      ponderado(x, f'fem{suf}',   ton_col),
                f'fedtt{suf}':    ponderado(x, f'fedtt{suf}', ton_col),
                f'dtt{suf}':      ponderado(x, f'dtt{suf}',   ton_col),
                f'p{suf}':        ponderado(x, f'p{suf}',     ton_col),
                f's{suf}':        ponderado(x, f's{suf}',     ton_col),
                f'volumen{suf}':  x['proportional_volume'].sum() if 'proportional_volume' in x else np.nan,
                f'tonelaje{suf}': x[ton_col].sum() / 1_000,
            }), include_groups=False)
            .reset_index()
        )

    t_lp = agg_grupo(df, 'ore_lp', 'tonelaje',    'ocurrencia',    '')
    t_mp = agg_grupo(df, 'ore_mp', 'tonelaje_mp',  'ocurrencia_mp', '_mp')
    t_cp = agg_grupo(df, 'ore_cp', 'tonelaje_cp',  'ocurrencia_cp', '_cp')

    tabla = pd.merge(t_lp, t_mp, on='periodo', how='outer')
    tabla = pd.merge(tabla, t_cp, on='periodo', how='outer')

    # Asegurar 12 periodos
    df_p = pd.DataFrame({'periodo': range(1, 13), 'mes': [MESES[i] for i in range(1, 13)]})
    tabla = pd.merge(df_p, tabla, on='periodo', how='left')

    # Finos con DTT
    for suf in ['', '_mp', '_cp']:
        dtt_c = f'dtt{suf}'; ton_c = f'tonelaje{suf}'; fin_c = f'fino{suf}'
        if dtt_c in tabla.columns and ton_c in tabla.columns:
            tabla[fin_c] = tabla[dtt_c] * tabla[ton_c] / 100

    # Fila total
    total = {'mes': 'Total', 'periodo': 0}
    for suf in ['', '_mp', '_cp']:
        for col in ['fe', 'fem', 'fedtt', 'dtt', 'p', 's']:
            cn, tn = f'{col}{suf}', f'tonelaje{suf}'
            if cn in tabla.columns and tn in tabla.columns:
                total[cn] = ponderado(tabla, cn, tn)
        for col in ['tonelaje', 'volumen', 'fino']:
            cn = f'{col}{suf}'
            if cn in tabla.columns: total[cn] = tabla[cn].sum()

    return pd.concat([tabla, pd.DataFrame([total])], ignore_index=True).round(2)


tabla_mensual = build_tabla_mlc(df, cutoff_fem_lc, ocu_sel if modo_ocu == "Por ocurrencia" else None)

# tabla_plot: 12 filas garantizadas
_df_12 = pd.DataFrame({'periodo': range(1, 13), 'mes': [MESES[i] for i in range(1, 13)]})
_tabla_sin_total = tabla_mensual[tabla_mensual['mes'] != 'Total']
tabla_plot = pd.merge(_df_12, _tabla_sin_total.drop(columns=['mes'], errors='ignore'),
                      on='periodo', how='left').reset_index(drop=True)

_mes_rows = tabla_mensual[tabla_mensual['periodo'] == mes]
mes_row   = _mes_rows if not _mes_rows.empty else tabla_mensual[tabla_mensual['mes'] == 'Total']
total_row = tabla_mensual[tabla_mensual['mes'] == 'Total']


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
    st.markdown(f"### Balance Mensual — Mina Los Colorados{fase_lbl}{ocu_lbl}")
    st.markdown(f"*Mineral: {cutoff_str} · Marginal: ue_fe ≥ 1 y FeM < {cutoff_fem_lc}%*")

    def sv(df_src, col, fmt="{:,.1f}"):
        try:
            v = df_src[col].values[0]
            return "—" if pd.isna(v) else fmt.format(v)
        except: return "—"

    st.markdown(f"#### Mes seleccionado — {mes_abr}")
    c1, c2, c3 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_dtt in [
        (c1, "LP", "tonelaje",    "fe",    "fem",    "dtt"),
        (c2, "MP", "tonelaje_mp", "fe_mp", "fem_mp", "dtt_mp"),
        (c3, "CP", "tonelaje_cp", "fe_cp", "fem_cp", "dtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} (kt)</div>
                <div class='metric-value'>{sv(mes_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(mes_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(mes_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    DTT: {sv(mes_row, k_dtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Acumulado anual")
    c4, c5, c6 = st.columns(3)
    for widget, modelo, k_ton, k_fe, k_fem, k_dtt in [
        (c4, "LP", "tonelaje",    "fe",    "fem",    "dtt"),
        (c5, "MP", "tonelaje_mp", "fe_mp", "fem_mp", "dtt_mp"),
        (c6, "CP", "tonelaje_cp", "fe_cp", "fem_cp", "dtt_cp"),
    ]:
        with widget:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tonelaje {modelo} acum. (kt)</div>
                <div class='metric-value'>{sv(total_row, k_ton)}</div>
                <div class='metric-sub'>
                    Fe: {sv(total_row, k_fe, "{:.2f}")} % &nbsp;|&nbsp;
                    FeM: {sv(total_row, k_fem, "{:.2f}")} % &nbsp;|&nbsp;
                    DTT: {sv(total_row, k_dtt, "{:.2f}")} %
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    cols_d = [c for c in [
        'mes', 'fe', 'fem', 'fedtt', 'dtt', 'p', 's', 'tonelaje', 'fino',
        'fe_mp', 'fem_mp', 'fedtt_mp', 'dtt_mp', 'p_mp', 's_mp', 'tonelaje_mp', 'fino_mp',
        'fe_cp', 'fem_cp', 'fedtt_cp', 'dtt_cp', 'p_cp', 's_cp', 'tonelaje_cp', 'fino_cp'
    ] if c in tabla_mensual.columns]

    def _highlight_rows(row):
        if row['mes'] == 'Total': return ['background-color:#1e2235;font-weight:bold'] * len(row)
        elif row['mes'] == mes_abr: return ['font-weight:700;border-left:3px solid #E8650A'] * len(row)
        return [''] * len(row)

    st.dataframe(
        tabla_mensual[cols_d].style
            .format({c: '{:.2f}' for c in cols_d if c != 'mes'})
            .apply(_highlight_rows, axis=1),
        use_container_width=True, hide_index=True
    )
    st.download_button("⬇️ Tabla CSV", csv_bytes(tabla_mensual[cols_d]),
                       "balance_mlc.csv", "text/csv", key="dl_tabla")


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
            col_mp_line = st.color_picker("MP (línea)",  "#4682B4", key="c_mp_line")
        with _pc3:
            col_cp      = st.color_picker("CP (barra)",  "#000000", key="c_cp_bar")
            col_cp_line = st.color_picker("CP (línea)",  "#9370DB", key="c_cp_line")

    # ── Gráfico Mensual ──
    titulo_mlc = f"Mina Los Colorados{fase_lbl}{ocu_lbl}"
    st.markdown(f"### Extracción Mensual — LP o MP vs CP  ·  {var_cal.upper()}")
    modelo_sel = st.radio("Comparar CP contra:", ["LP", "MP"], horizontal=True, key="r_mensual")

    col_ton_m  = 'tonelaje'  if modelo_sel == "LP" else 'tonelaje_mp'
    col_cal_m  = var_cal     if modelo_sel == "LP" else f'{var_cal}_mp'
    col_cal_cp = f'{var_cal}_cp'
    bar_color  = col_lp      if modelo_sel == "LP" else col_mp
    line_color = col_lp_line if modelo_sel == "LP" else col_mp_line

    n_meses = 12; pos = np.arange(n_meses)
    meses_labels = [MESES[i] for i in range(1, 13)]

    def _safe_array(df, col):
        if col not in df.columns: return np.full(12, np.nan)
        arr = df[col].values.copy().astype(float)
        if len(arr) < 12: arr = np.concatenate([arr, np.full(12-len(arr), np.nan)])
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
    ax1.set_title(f'Extracción Mensual {anio} — {titulo_mlc}\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_mensual = save_png(fig)
    st.pyplot(fig); plt.close()
    st.download_button("⬇️ PNG Mensual", png_mensual, "extraccion_mensual_mlc.png",
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
            if show_annot: annotate_bars(ax1, bars, tabla_trim[cc_col])

    ax1.set_ylabel('Tonelaje (kt)', color='#444')
    ax1.set_xticks(pos2); ax1.set_xticklabels(tabla_trim['trimestre'], color='#444')
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.tick_params(colors='#444'); ax1.set_ylim(ton_ymin, ton_ymax)
    style_ax(ax1)

    ax2b = ax1.twinx(); ax2b.set_facecolor('white')
    for cc_col, label, line_c in [
        ('cal_lp', f'{var_cal.upper()} LP', col_lp_line),
        ('cal_mp', f'{var_cal.upper()} MP', col_mp_line),
        ('cal_cp', f'{var_cal.upper()} CP', col_cp_line),
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
    ax1.set_title(f'Extracción Trimestral {anio} — {titulo_mlc}\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_trim = save_png(fig2)
    st.pyplot(fig2); plt.close()
    st.download_button("⬇️ PNG Trimestral", png_trim, "extraccion_trimestral_mlc.png",
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
        cum_target = np.cumsum([target_ton/12]*12)
        ax3.plot(tabla_plot['mes'], cum_target, '--', color='#cc4444',
                 lw=1.5, label=f'Objetivo ({target_ton:,.0f} kt)')
    ax3.set_ylabel('Tonelaje acumulado (kt)', color='#444')
    ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax3.tick_params(colors='#444'); style_ax(ax3)
    ax3.legend(frameon=False, labelcolor='#333')
    ax3.set_title(f'Gráfico Acumulado {anio} — {titulo_mlc}\n{cutoff_str}',
                  color='#222', fontsize=11, pad=10)
    plt.tight_layout()
    png_acum = save_png(fig3)
    st.pyplot(fig3); plt.close()
    st.download_button("⬇️ PNG Acumulado", png_acum, "extraccion_acumulada_mlc.png",
                       "image/png", key="dl_png_acum")


# ─────────────────────────────────────────────
# TAB 3 — VISUALIZACIÓN DE BLOQUES
# MLC: bloques 12.5x12.5, columnas con sufijo _mp/_cp en df
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Visualización de Bloques por Banco")
    st.markdown("*Bloques 12.5×12.5 m — MLC*")

    tipo_mapa = st.selectbox("Visualizar por:", ["Clasificación (Ore)", "Ocurrencia"])
    cv1, cv2 = st.columns(2)
    with cv1:
        panel_modo = st.radio("Paneles:", ["MP y CP (doble)", "Solo MP", "Solo CP"],
                              horizontal=True, key="panel_modo")
    with cv2:
        mostrar_grilla = st.checkbox("Mostrar grilla", value=True)

    cg1, cg2 = st.columns(2)
    with cg1: grid_x = st.number_input("Espaciado grilla X (m)", value=100, step=50, min_value=10)
    with cg2: grid_y = st.number_input("Espaciado grilla Y (m)", value=100, step=50, min_value=10)

    # Navegación de bancos (MLC: tam_bloque=12.5, banco_real = cota - 7.5)
    TAM_BLOQUE_MLC = 12.5
    col_z_mp = 'centroid_z' if 'centroid_z' in df.columns else 'centroid_z_mp'

    try:
        z_disponibles = sorted(
            df.loc[df['periodo'] == mes, col_z_mp].dropna().unique(),
            reverse=False
        )
    except Exception:
        z_disponibles = []

    if not z_disponibles:
        st.warning(f"No hay bloques para periodo == {mes}.")
    else:
        n_bancos = min(3, len(z_disponibles))
        nb1, nb2, nb3 = st.columns([1, 1, 3])
        with nb1:
            banco_idx = st.number_input("Banco (0=más bajo, +1, +2)",
                                        min_value=0, max_value=n_bancos-1,
                                        value=0, step=1, key="banco_idx")
        with nb2:
            cota_sel = z_disponibles[banco_idx]
            banco_real_label = cota_sel - 7.5
            st.markdown(f"<div class='cutoff-box'>Banco seleccionado<br>"
                        f"<b>Z = {banco_real_label:.1f} m</b><br>"
                        f"(centroide Z = {cota_sel:.1f})</div>", unsafe_allow_html=True)

        try:
            # MLC: filtrar por cota en columnas con sufijo
            cx_mp = 'centroid_x_mp' if 'centroid_x_mp' in df.columns else 'centroid_x'
            cy_mp = 'centroid_y_mp' if 'centroid_y_mp' in df.columns else 'centroid_y'
            cx_cp = 'centroid_x_cp' if 'centroid_x_cp' in df.columns else 'centroid_x'
            cy_cp = 'centroid_y_cp' if 'centroid_y_cp' in df.columns else 'centroid_y'

            df_mp_cota = df[
                (np.abs(df[col_z_mp] - cota_sel) < TAM_BLOQUE_MLC/2 + 0.01) &
                (df['periodo'] <= mes)
            ].copy()
            # Resolver superposición XY: prioridad al periodo más alto
            df_mp_cota = (
                df_mp_cota
                .sort_values('periodo', ascending=False)
                .drop_duplicates(subset=[cx_mp, cy_mp], keep='first')
            )
            df_cp_cota = df_mp_cota.copy()  # mismos bloques que MP

            if panel_modo == "Solo MP":    paneles = [(df_mp_cota, 'MP', cx_mp, cy_mp)]
            elif panel_modo == "Solo CP":  paneles = [(df_cp_cota, 'CP', cx_cp, cy_cp)]
            else:                          paneles = [(df_mp_cota, 'MP', cx_mp, cy_mp),
                                                      (df_cp_cota, 'CP', cx_cp, cy_cp)]

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

            titulo_banco = f"Banco = {banco_real_label:.1f} m"

            if tipo_mapa == "Clasificación (Ore)":
                colores = {
                    'mineral': (255/255, 136/255, 255/255),
                    'marginal': (153/255, 136/255, 255/255),
                    'esteril': (238/255, 238/255, 238/255)
                }
                col_ore_map = {'MP': 'ore_mp', 'CP': 'ore_cp'}
                for ax_i, (df_i, lbl, cx, cy) in zip(axs, paneles):
                    for _, row in df_i.iterrows():
                        ore_val = row.get(col_ore_map[lbl], 'esteril')
                        color = colores.get(str(ore_val), (0.8, 0.8, 0.8))
                        alpha = 1.0 if row['periodo'] == mes else 0.25
                        ax_i.add_patch(Rectangle(
                            (row[cx] - TAM_BLOQUE_MLC/2, row[cy] - TAM_BLOQUE_MLC/2),
                            TAM_BLOQUE_MLC, TAM_BLOQUE_MLC,
                            facecolor=(*color[:3], alpha), edgecolor='none'))
                    ax_i.autoscale(); ax_i.set_aspect('equal')
                    _style_ax_blq(ax_i, f'{lbl} — {titulo_banco}')
                ley = [Patch(facecolor=c, label=k) for k, c in colores.items()]
                for ax_i in axs: ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

            else:  # Ocurrencia
                col_occ_map = {'MP': 'ocurrencia_mp', 'CP': 'ocurrencia_cp'}
                co = {
                    'mac': (1.0, 0.0, 0.0),
                    'bre': (0.0, 0.0, 1.0),
                    'gyd': (102/255, 153/255, 255/255),
                    'est': (0.85, 0.85, 0.85)
                }
                for ax_i, (df_i, lbl, cx, cy) in zip(axs, paneles):
                    occ_col = col_occ_map.get(lbl, 'ocurrencia_mp')
                    for _, row in df_i.iterrows():
                        occ_val = str(row.get(occ_col, 'est')).lower()
                        color = co.get(occ_val, (0.8, 0.8, 0.8))
                        alpha = 1.0 if row['periodo'] == mes else 0.25
                        ax_i.add_patch(Rectangle(
                            (row[cx] - TAM_BLOQUE_MLC/2, row[cy] - TAM_BLOQUE_MLC/2),
                            TAM_BLOQUE_MLC, TAM_BLOQUE_MLC,
                            facecolor=(*color[:3], alpha), edgecolor='none'))
                    ax_i.autoscale(); ax_i.set_aspect('equal')
                    _style_ax_blq(ax_i, f'{lbl} — {titulo_banco}')
                ley = [Patch(facecolor=c, label=k.upper()) for k, c in co.items()]
                for ax_i in axs: ax_i.legend(handles=ley, frameon=False, labelcolor='#333')

            plt.suptitle(f'Mina Los Colorados{fase_lbl} — {mes_abr} {anio}',
                         color='#222', fontsize=11, y=1.01)
            plt.tight_layout()
            png_bloques = save_png(fig)
            st.pyplot(fig); plt.close()
            st.download_button("⬇️ PNG Bloques", png_bloques,
                               "mapa_bloques_mlc.png", "image/png", key="dl_png_bloques")

        except Exception as e:
            st.error(f"Error al generar mapa de bloques: {e}")


# ─────────────────────────────────────────────
# TAB 4 — CASCADAS
# MLC: cascada mensual usa conc_mp (MP vs CP)
#       cascada acumulada usa conc_lp (LP vs CP)
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Gráficos de Cascada")

    def cascada(df_sel, lbl_proy, lbl_real, title, conc_col='conciliacion', yrange=None):
        def ton(mask, col='tonelaje'):
            sub = df_sel[mask]
            return sub[col].sum() / 1000 if col in sub.columns else 0

        proy = ton(df_sel[conc_col].isin(['mineral_mineral','mineral_marginal','mineral_esteril']))
        mm   = ton(df_sel[conc_col] == 'mineral_marginal')
        me   = ton(df_sel[conc_col] == 'mineral_esteril')
        mmin = ton(df_sel[conc_col] == 'mineral_mineral')
        aj   = ton(df_sel[conc_col] == 'mineral_mineral','tonelaje_cp') - mmin
        marg = ton(df_sel[conc_col] == 'marginal_mineral','tonelaje_cp')
        est  = ton(df_sel[conc_col] == 'esteril_mineral', 'tonelaje_cp')
        real = ton(df_sel[conc_col].isin(['mineral_mineral','marginal_mineral','esteril_mineral']),'tonelaje_cp')

        pasos = [lbl_proy, "Mineral Marginal", "Mineral Esteril", "Mineral Mineral",
                 "Ajuste por densidad", "Marginal Mineral", "Esteril Mineral", lbl_real]
        vals  = [proy, -mm, -me, mmin, aj, marg, est, real]
        tipos = ['Proyectado','Pérdida','Pérdida','Subtotal','Ajuste','Ganancia','Ganancia','Real']

        df_res = pd.DataFrame({
            'Paso': pasos, 'Tonelaje (kt)': [round(v,1) for v in vals], 'Tipo': tipos,
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
            x=pasos, text=[f"{v:,.1f}" for v in vals],
            textposition="outside", y=vals,
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

    # Preparar conc_mp y conc_lp
    try:
        # conc_mp: MP vs CP por periodo
        conc_mp = (
            df[df['periodo'].notna() & (df['periodo'] > 0)]
            .groupby(['periodo','conciliacion'], group_keys=False)
            .apply(lambda x: pd.Series({
                'tonelaje_mp': x['tonelaje_mp'].sum(),
                'tonelaje_cp': x['tonelaje_cp'].sum(),
            }), include_groups=False).reset_index()
        )

        # conc_lp: LP vs CP por periodo
        conc_lp = (
            df[df['periodo'].notna() & (df['periodo'] > 0)]
            .groupby(['periodo','conciliacion_lp'], group_keys=False)
            .apply(lambda x: pd.Series({
                'tonelaje': x['tonelaje'].sum(),
                'tonelaje_cp': x['tonelaje_cp'].sum(),
            }), include_groups=False).reset_index()
        )

        # ── Cascada Mensual (MP vs CP) ──
        st.markdown("#### Cascada Mensual")
        _cm1, _cm2, _ = st.columns([1,1,3])
        with _cm1: casc_mes_ymin = st.number_input("Ton. mín (kt)", value=0,   step=50, key="casc_mes_ymin")
        with _cm2: casc_mes_ymax = st.number_input("Ton. máx (kt)", value=600, step=50, key="casc_mes_ymax")

        df_mes_sel = conc_mp[conc_mp['periodo'] == mes].copy()
        df_mes_sel = df_mes_sel.rename(columns={'tonelaje_mp': 'tonelaje'})

        fig_c1, df_res_c1 = cascada(
            df_mes_sel, f"Proyectado MP {mes_abr}", f"Real CP {mes_abr}",
            f"Mineral Proyectado/Real ({mes_abr}): {cutoff_str}",
            conc_col='conciliacion', yrange=[casc_mes_ymin, casc_mes_ymax]
        )
        st.plotly_chart(fig_c1, use_container_width=True)

        def _color_tipo(row):
            if row['Tipo']=='Pérdida':  return ['color:#cc4444']*len(row)
            if row['Tipo']=='Ganancia': return ['color:#44aa66']*len(row)
            if row['Tipo'] in ('Proyectado','Real'): return ['font-weight:bold']*len(row)
            return ['']*len(row)

        with st.expander("📋 Ver tabla cascada mensual"):
            try: st.dataframe(df_res_c1.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except: st.dataframe(df_res_c1, hide_index=True, use_container_width=True)

        d1, d2, _ = st.columns([1,1,3])
        with d1: st.download_button("⬇️ HTML", plotly_to_html(fig_c1),
                                    f"cascada_mes_mlc_{mes_abr}.html", "text/html", key="dl_casc_mes")
        with d2: st.download_button("⬇️ CSV", csv_bytes(df_res_c1),
                                    f"cascada_mes_mlc_{mes_abr}.csv", "text/csv", key="dl_casc_mes_csv")

        st.markdown("---")

        # ── Cascada Acumulada (LP vs CP) ──
        st.markdown("#### Cascada Acumulada LP vs CP")
        _ca1, _ca2, _ = st.columns([1,1,3])
        with _ca1: casc_acum_ymin = st.number_input("Ton. mín (kt)", value=0,    step=100, key="casc_acum_ymin")
        with _ca2: casc_acum_ymax = st.number_input("Ton. máx (kt)", value=2000, step=100, key="casc_acum_ymax")

        df_acum_lp = conc_lp[conc_lp['periodo'] <= mes].copy()
        df_acum_lp = df_acum_lp.rename(columns={'conciliacion_lp': 'conciliacion',
                                                  'tonelaje': 'tonelaje'})

        fig_c2, df_res_c2 = cascada(
            df_acum_lp, "Proyectado LP acum.", "Real CP acum.",
            f"Mineral Proyectado Budget/Real (hasta {mes_abr})<br>{cutoff_str}",
            conc_col='conciliacion', yrange=[casc_acum_ymin, casc_acum_ymax]
        )
        st.plotly_chart(fig_c2, use_container_width=True)

        with st.expander("📋 Ver tabla cascada acumulada"):
            try: st.dataframe(df_res_c2.style.apply(_color_tipo, axis=1), hide_index=True, use_container_width=True)
            except: st.dataframe(df_res_c2, hide_index=True, use_container_width=True)

        d3, d4, _ = st.columns([1,1,3])
        with d3: st.download_button("⬇️ HTML", plotly_to_html(fig_c2),
                                    f"cascada_acum_mlc_{mes_abr}.html", "text/html", key="dl_casc_acum")
        with d4: st.download_button("⬇️ CSV", csv_bytes(df_res_c2),
                                    f"cascada_acum_mlc_{mes_abr}.csv", "text/csv", key="dl_casc_acum_csv")

    except Exception as e:
        st.error(f"Error en cascadas: {e}")


# ─────────────────────────────────────────────
# TAB 5 — DISPERSIÓN
# MLC: Fe MP vs Fe CP, un gráfico por UE_FE (selector)
# Notebook usa ue_fe 4 y 5 como default
# ─────────────────────────────────────────────
with tab5:
    st.markdown("### Análisis de Dispersión — Fe")
    st.markdown("*Fe MP vs Fe CP filtrado por UE_FE. Un gráfico por unidad de estimación.*")

    _d1, _d2 = st.columns(2)
    with _d1:
        periodo_disp = st.radio("Período:", ["Mes", "Acumulado"],
                                horizontal=True, key="periodo_disp")
    with _d2:
        # Obtener UE_FE disponibles en los datos
        ue_disponibles = sorted([int(x) for x in df['ue_fe_mp'].dropna().unique()
                                 if str(x).replace('.0','').isdigit() and float(x) >= 1])
        ue_disponibles = ue_disponibles if ue_disponibles else [1, 2, 3, 4, 5]
        ue_fe_sel = st.multiselect("UE_FE a mostrar:", ue_disponibles,
                                   default=[u for u in [4, 5] if u in ue_disponibles] or ue_disponibles[:2],
                                   key="ue_fe_sel")

    if not ue_fe_sel:
        st.warning("Selecciona al menos una UE_FE.")
    else:
        try:
            import seaborn as _sns
            color_puntos = _sns.color_palette()[0]
            periodo_lbl  = f"Ene–{mes_abr}" if periodo_disp == "Acumulado" else mes_abr

            if periodo_disp == "Acumulado":
                df_d = df[df['periodo'] <= mes].copy()
            else:
                df_d = df[df['periodo'] == mes].copy()

            n_ue = len(ue_fe_sel)
            fig, axes = plt.subplots(1, n_ue, figsize=(6*n_ue, 6), facecolor='white',
                                     squeeze=False)
            axes = axes[0]
            for ax in axes: ax.set_facecolor('white')

            for idx, ue in enumerate(sorted(ue_fe_sel)):
                ax = axes[idx]
                # Filtrar por ue_fe en MP y CP
                df_ue = df_d[
                    (df_d['ue_fe_mp'] == ue) & (df_d['ue_fe_cp'] == ue)
                ][['fe', 'fe_cp']].dropna()
                df_ue = df_ue.rename(columns={'fe': 'fe_mp'})

                if df_ue.empty:
                    ax.text(0.5, 0.5, f'UE_FE = {ue}\nSin datos',
                            transform=ax.transAxes, ha='center', color='#888')
                    ax.set_title(f'Fe (%) - UE_FE = {ue}: {periodo_lbl}', color='#222')
                    continue

                # Scatterplot formato notebook
                _sns.scatterplot(x='fe_cp', y='fe_mp', data=df_ue,
                                 alpha=1, edgecolors='none', s=20,
                                 ax=ax, color=color_puntos)

                # Línea identidad 1:1
                ax.plot([0, 70], [0, 70], color='gray', linestyle='-', lw=1)

                # Línea de ajuste
                sl, ic, *_ = linregress(df_ue['fe_cp'], df_ue['fe_mp'])
                xf = np.linspace(0, 70, 100)
                ax.plot(xf, sl*xf+ic, color=color_puntos, linestyle='--', linewidth=1)

                ax.set_xlim(0, 70); ax.set_ylim(0, 70)
                ax.set_xlabel('Fe_cp (%)', color='#444')
                ax.set_ylabel('Fe_mp (%)', color='#444')
                ax.set_title(f'Gráfico de dispersión de UE_FE = {ue}: {periodo_lbl}',
                             color='#222', fontsize=11)
                style_ax(ax)

                # Pearson en verde (formato notebook)
                pr, _ = pearsonr(df_ue['fe_cp'], df_ue['fe_mp'])
                ax.text(0.05, 0.95, f'Coeficiente de Pearson: {pr:.2f}',
                        transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', color='green')

                # Histogramas marginales
                ax_top = ax.inset_axes([0, 1.02, 1, 0.2], sharex=ax)
                ax_top.hist(df_ue['fe_cp'], bins=20, color=color_puntos, alpha=0.9, linewidth=0)
                ax_top.axis('off')
                ax_right = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
                ax_right.hist(df_ue['fe_mp'], bins=20, orientation='horizontal',
                              color=color_puntos, alpha=0.9, linewidth=0)
                ax_right.axis('off')

            plt.suptitle(f'Mina Los Colorados{fase_lbl} — MP vs CP',
                         color='#222', fontsize=11, y=1.04)
            plt.tight_layout()
            png_disp = save_png(fig, dpi=150)
            st.pyplot(fig); plt.close()
            st.download_button("⬇️ PNG Dispersión", png_disp,
                               f"dispersion_mlc_{mes_abr}.png", "image/png", key="dl_png_disp")

        except Exception as e:
            st.error(f"Error en dispersión: {e}")


# ─────────────────────────────────────────────
# TAB 6 — CUADRANTES FeM + MATRIZ ORE/WASTE
# ─────────────────────────────────────────────
with tab6:
    st.markdown("### Conciliación por Cuadrantes FeM")
    st.markdown("*Scatter y matriz Ore/Waste usan exactamente los mismos bloques.*")

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
        hmap_q_lbl = st.selectbox("🎨 Paleta matriz", list(HEATMAP_PAL_Q.keys()), index=0, key="hmap_q")
    hmap_q_cmap = HEATMAP_PAL_Q[hmap_q_lbl]

    try:
        if periodo_q == "Acumulado":
            df_q = df[(df['periodo'] > 0) & (df['periodo'] <= mes)].copy()
        else:
            df_q = df[df['periodo'] == mes].copy()

        # Filtro volumen
        if 'proportional_volume' in df_q.columns and 'dim_x' in df_q.columns:
            df_q = df_q[
                df_q['proportional_volume'] >= 0.75 *
                df_q['dim_x'] * df_q['dim_y'] * df_q['dim_z']
            ].copy()

        df_fem = df_q[['fem', 'fem_cp']].dropna().copy()
        df_fem = df_fem.rename(columns={'fem': 'fem_mp'})

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

            df_fem = df_fem.copy()
            df_fem['ore_mp_fem'] = np.where(df_fem['fem_mp'] >= cutoff_fem_q, 'mineral', 'esteril')
            df_fem['ore_cp_fem'] = np.where(df_fem['fem_cp'] >= cutoff_fem_q, 'mineral', 'esteril')

            sc_col, _ = st.columns([1, 1])
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
                ax.set_xlabel('FeM CP (%)', color='#444'); ax.set_ylabel('FeM MP (%)', color='#444')
                style_ax(ax)
                c = cutoff_fem_q; hi = (c + MAX_FEM_Q)/2; lo = c/2
                for txt, xp, yp in [('I',hi,hi),('II',lo,hi),('III',lo,lo),('IV',hi,lo)]:
                    ax.text(xp, yp, txt, fontsize=13, weight='bold', color='#555', ha='center', va='center')
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
                                   f"cuadrantes_fem_mlc_{mes_abr}.png",
                                   "image/png", key="dl_png_cuad")

            mx_col, _ = st.columns([1, 1])
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
                                   f"matriz_ow_mlc_{mes_abr}.png",
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

    except Exception as e:
        st.error(f"Error en cuadrantes FeM: {e}")


# ─────────────────────────────────────────────
# TAB 7 — MATRIZ DE OCURRENCIA
# MLC: mac, bre, gyd, est
# ─────────────────────────────────────────────
with tab7:
    st.markdown("### Matriz de Ocurrencia — Modelo Geológico")
    st.markdown("*MLC: mac, bre, gyd, est*")

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

    try:
        orden = ['mac','bre','gyd','est']
        cat_t = CategoricalDtype(categories=orden, ordered=True)

        if es_acum_occ:
            df_occ = df[(df['periodo'] > 0) & (df['periodo'] <= mes)].copy()
        else:
            df_occ = df[df['periodo'] == mes].copy()

        if 'proportional_volume' in df_occ.columns and 'dim_x' in df_occ.columns:
            df_occ = df_occ[
                df_occ['proportional_volume'] >= 0.75 *
                df_occ['dim_x'] * df_occ['dim_y'] * df_occ['dim_z']
            ].copy()

        # Normalizar ocurrencias
        for col in ['ocurrencia', 'ocurrencia_mp', 'ocurrencia_cp']:
            if col in df_occ.columns:
                df_occ[col] = df_occ[col].str.lower().replace({'nn': 'est'})

        occ_mp_col = 'ocurrencia_mp' if 'ocurrencia_mp' in df_occ.columns else 'ocurrencia'
        occ_cp_col = 'ocurrencia_cp'

        if occ_mp_col not in df_occ.columns or occ_cp_col not in df_occ.columns:
            st.warning("Columnas de ocurrencia no disponibles en los datos.")
        else:
            df_occ[occ_mp_col] = df_occ[occ_mp_col].astype(cat_t)
            df_occ[occ_cp_col] = df_occ[occ_cp_col].astype(cat_t)

            ct_occ = (pd.crosstab(df_occ[occ_mp_col], df_occ[occ_cp_col])
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
            ax.set_title(f'Ocurrencia MP vs CP — MLC: {periodo_lbl_o}',
                         color='#222', fontsize=12, pad=10)
            ax.tick_params(colors='#333')
            ax.set_xlabel('Ocurrencia CP', color='#444')
            ax.set_ylabel('Ocurrencia MP', color='#444')
            plt.tight_layout()

            col_o, _ = st.columns([1, 1])
            with col_o:
                png_occ = save_png(fig)
                st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Ocurrencia", png_occ,
                                   f"matriz_occ_mlc_{mes_abr}.png",
                                   "image/png", key="dl_png_occ")

            with st.expander("📋 Ver tabla de conteos"):
                ct_raw_occ = ct_occ.astype(int).copy()
                ct_raw_occ.index.name = 'MP \\ CP'; ct_raw_occ.columns.name = None
                st.dataframe(ct_raw_occ, use_container_width=True)

            dl_o1, dl_o2, _ = st.columns([1,1,3])
            with dl_o1: st.download_button("⬇️ CSV (%)", csv_bytes(ct_occ_pct.reset_index()),
                                           f"occ_pct_mlc_{mes_abr}.csv", "text/csv", key="dl_occ_csv")
            with dl_o2: st.download_button("⬇️ CSV Conteos", csv_bytes(ct_occ.astype(int).reset_index()),
                                           f"occ_raw_mlc_{mes_abr}.csv", "text/csv", key="dl_occ_raw")

    except Exception as e:
        st.error(f"Error en matriz de ocurrencia: {e}")
