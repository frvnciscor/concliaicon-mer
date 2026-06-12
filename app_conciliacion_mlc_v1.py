"""
App Streamlit - Conciliación Mina Los Colorados (MLC) v1
Lógica 100% fiel al notebook conciliacion_mensual_mlc_rev7.ipynb
"""
import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.ticker as mtick
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import pearsonr, linregress
from pandas.api.types import CategoricalDtype
import calendar, io

st.set_page_config(page_title="Conciliación MLC", page_icon="⛏️",
                   layout="wide", initial_sidebar_state="expanded")

MESES = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
         7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
VARS_CALIDAD = ['fe','fem','fedtt','dtt','p','s',
                'sio2','al2o3','v','axb','bwi_cab','bwi_conc']
FASES_MLC = ['f6a','f6b','fde']

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&family=Poppins:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.stApp{background-color:#0f1117;color:#e0e0e0;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;color:#f0c040;}
.metric-card{background:#1a1d27;border:1px solid #2a2d3a;border-left:3px solid #f0c040;
             padding:12px 16px;border-radius:4px;margin-bottom:8px;}
.metric-card.red{border-left-color:#cc4444;}.metric-card.green{border-left-color:#44aa66;}
.metric-card.blue{border-left-color:#6699ff;}
.metric-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;
              font-family:'IBM Plex Mono',monospace;}
.metric-value{font-size:24px;font-weight:700;color:#f0c040;font-family:'Poppins',sans-serif;}
.metric-value.red{color:#cc4444;}.metric-value.green{color:#44aa66;}.metric-value.blue{color:#6699ff;}
.metric-sub{font-size:12px;color:#777;font-family:'IBM Plex Mono',monospace;margin-top:2px;}
.metric-dev{font-size:12px;font-family:'Poppins',sans-serif;font-weight:600;margin-top:4px;}
.metric-dev.pos{color:#44aa66;}.metric-dev.neg{color:#cc4444;}.metric-dev.neu{color:#888;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#888;}
.stTabs [aria-selected="true"]{color:#f0c040!important;border-bottom:2px solid #f0c040!important;}
.cutoff-box{background:#1a1d27;border:1px solid #2a2d3a;border-left:3px solid #6699ff;
            padding:10px 14px;border-radius:4px;font-size:12px;color:#aaa;
            font-family:'IBM Plex Mono',monospace;margin:6px 0;}
.warn-box{background:#2a1f00;border-left:3px solid #f0c040;padding:10px 14px;
          font-size:12px;color:#ccc;border-radius:0 4px 4px 0;margin:8px 0;}
.fase-box{background:#1a2a1a;border:1px solid #2a3d2a;border-left:3px solid #44aa66;
          padding:10px 14px;border-radius:4px;font-size:12px;color:#aaa;
          font-family:'IBM Plex Mono',monospace;margin:6px 0;}
</style>""", unsafe_allow_html=True)

# ── Helpers ──
def _metric_html(label, value, cls=''):
    return (f"<div class='metric-card {cls}'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div></div>")

def make_fig(fs=(14,5)):
    fig,ax=plt.subplots(figsize=fs,facecolor='white'); ax.set_facecolor('white'); return fig,ax

def style_ax(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#ccc')
    ax.tick_params(colors='#444'); ax.xaxis.label.set_color('#444'); ax.yaxis.label.set_color('#444')

def save_png(fig,dpi=300):
    b=io.BytesIO(); fig.savefig(b,format='png',dpi=dpi,bbox_inches='tight',facecolor='white')
    b.seek(0); return b.getvalue()

def csv_bytes(df): return df.to_csv(index=False).encode('utf-8-sig')
def html_bytes(fig): return fig.to_html(include_plotlyjs='cdn',full_html=True).encode('utf-8')

def ann_bars(ax, bars, ley, fmt='{h:.1f}\n{v:.1f}%'):
    for i,b in enumerate(bars):
        h=b.get_height()
        if pd.isna(h) or h==0: continue
        try: ax.annotate(fmt.format(h=h,v=ley.iloc[i]),
                         xy=(b.get_x()+b.get_width()/2,h),
                         xytext=(0,3),textcoords='offset points',
                         fontsize=7,ha='center',va='bottom',color='#333')
        except: pass

def sv(src, col, fmt="{:,.1f}"):
    try: v=src[col].values[0]; return "—" if pd.isna(v) else fmt.format(v)
    except: return "—"

def pond(df, col_ley, col_ton):
    m=(df[col_ton]>0)&df[col_ley].notna()&df[col_ton].notna()
    t=df.loc[m,col_ton].sum()
    return np.nan if t==0 else (df.loc[m,col_ley]*df.loc[m,col_ton]).sum()/t

# ── Schema maestro — columnas fijas en orden definido ──
SCHEMA = [
    'block_id',
    'centroid_x','centroid_y','centroid_z',
    'dim_x','dim_y','dim_z',
    'volume','proportional_volume','densidad',
    'ue_fe','fe','fem','fedtt','dtt','p','s',
    'sio2','al2o3','v','axb','bwi_cab','bwi_conc',
    'fase','ocurrencia',
]

# Columnas que se pueden dropear después del merge — no las usa ningún cálculo
COLS_DROP_POST_MERGE = [
    'archivo','archivo_mp','archivo_cp',
    'block_id_mp','block_id_cp',
    'periodo_mp','periodo_cp',
    'volume_mp','volume_cp',
    'id',                           # ya no se necesita después del merge
]

# ── Lectura CSV (Cell 4/6/8) ──
def leer_csv(file_bytes):
    """
    Lee CSV formato Vulcan (4 filas cabecera).
    Aplica schema maestro: mismas columnas, mismo orden, independiente del archivo.
    Columnas faltantes → NaN. Columnas extra → descartadas.
    """
    raw = file_bytes
    if raw[:3] == b'\xef\xbb\xbf': raw = raw[3:]   # eliminar BOM
    buf = io.BytesIO(raw)
    hdr = pd.read_csv(buf, nrows=1, header=None, encoding='utf-8')
    buf.seek(0)
    names = [str(c).strip().lower() for c in hdr.iloc[0].tolist()]
    df = pd.read_csv(buf, skiprows=4, header=None, names=names,
                     low_memory=False, encoding='utf-8')
    # Aplicar schema: solo columnas necesarias, en orden fijo
    df = df.reindex(columns=SCHEMA)
    # Densidad -99 → 2.7
    if 'densidad' in df.columns:
        df.loc[df['densidad'] == -99, 'densidad'] = 2.7
    return df

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## ⛏️ MLC · Conciliación")
    st.markdown("---")
    st.markdown("### 📂 Archivos CSV")
    st.markdown("*12 LP + 12 MP + 12 CP*")
    files_lp=st.file_uploader("LP (output_lp_1…12)",type="csv",accept_multiple_files=True,key="lp")
    files_mp=st.file_uploader("MP (output_mp_1…12)",type="csv",accept_multiple_files=True,key="mp")
    files_cp=st.file_uploader("CP (output_cp_1…12)",type="csv",accept_multiple_files=True,key="cp")
    st.markdown("---")
    st.markdown("### 🔬 Modo de análisis")
    modo_fase=st.radio("Tipo:",["Análisis completo","Por fase"],horizontal=False,key="modo_fase")
    fase_sel=None
    if modo_fase=="Por fase":
        fase_sel=st.selectbox("Fase:",FASES_MLC,index=0,key="fase_sel",format_func=str.upper)
        st.markdown(f"<div class='fase-box'>✅ Fase {fase_sel.upper()}</div>",unsafe_allow_html=True)
    else:
        st.markdown("<div class='fase-box'>🌐 Todas las fases</div>",unsafe_allow_html=True)
    st.markdown("### 🔍 Filtro por Ocurrencia")
    modo_ocu=st.radio("Ocurrencia:",["Completo","Por ocurrencia"],horizontal=False,key="modo_ocu")
    ocu_sel=None
    if modo_ocu=="Por ocurrencia":
        ocu_sel=st.selectbox("Ocurrencia:",['mac','bre','gyd'],format_func=str.upper,key="ocu_sel")
        st.markdown(f"<div class='fase-box'>✅ {ocu_sel.upper()}</div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📅 Período")
    c1,c2=st.columns(2)
    with c1: anio=st.number_input("Año",value=2026,step=1,min_value=2000,max_value=2100)
    with c2: mes=st.selectbox("Mes",list(MESES.keys()),format_func=lambda x:MESES[x],index=0)
    st.markdown("---")
    st.markdown("### ✂️ Ley de Corte")
    VARS_CORTE = ['fem','fe','fedtt','dtt','p','s','sio2','al2o3','ue_fe']
    OPS = ['>=','<=','>','<']
    n_crit = st.number_input("Número de criterios", min_value=1, max_value=4, value=1, step=1)
    DEFAULTS = [
        {'var':'fem',  'op':'>=','val':28.0},
        {'var':'fe',   'op':'>=','val':0.0},
        {'var':'fedtt','op':'>=','val':0.0},
        {'var':'dtt',  'op':'>=','val':0.0},
    ]
    criterios = []
    for i in range(int(n_crit)):
        d = DEFAULTS[i] if i < len(DEFAULTS) else {'var':'fem','op':'>=','val':0.0}
        st.markdown(f"**Criterio {i+1}**")
        ca_c, cb_c, cc_c = st.columns([2,1,1.5])
        with ca_c:
            var = st.selectbox("Var", VARS_CORTE,
                               index=VARS_CORTE.index(d['var']) if d['var'] in VARS_CORTE else 0,
                               key=f"mlc_var_{i}", label_visibility="collapsed")
        with cb_c:
            op = st.selectbox("Op", OPS, index=OPS.index(d['op']),
                              key=f"mlc_op_{i}", label_visibility="collapsed")
        with cc_c:
            val = st.number_input("Val", value=d['val'], step=0.5,
                                  key=f"mlc_val_{i}", label_visibility="collapsed")
        criterios.append({'var':var,'op':op,'val':val})
    cutoff_str = " AND ".join([f"{c['var']} {c['op']} {c['val']}" for c in criterios])
    cutoff_fem = next((c['val'] for c in criterios if c['var']=='fem'), 28.0)
    st.markdown(f"<div class='cutoff-box'>🎯 {cutoff_str}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📐 Variable de calidad")
    var_cal=st.selectbox("Eje secundario",VARS_CALIDAD,format_func=str.upper)
    st.markdown("---")
    target_ton=st.number_input("🎯 Objetivo anual (kt)",value=0,step=100,min_value=0)
    st.markdown("---")

# ─── CARGA DE DATOS — separada de clasificación ───
@st.cache_data(show_spinner=False)
def cargar_raw(lp_b, mp_b, cp_b, modo_fase_k, fase_k, _cache_key=None):
    """
    Solo lee, mergea y libera memoria. NO clasifica ore.
    Así cambiar la ley de corte no recarga los 36 archivos.
    """
    def leer_grupo(files_b):
        out = pd.DataFrame()
        for name, content in files_b:
            try: dt = leer_csv(content)
            except: continue
            try: num = int(''.join(filter(str.isdigit, name.replace('.csv','').split('_')[-1])))
            except: num = 1
            dt['archivo'] = name
            dt['periodo'] = num
            if 'block_id' in dt.columns:
                dt['id'] = dt['block_id'].astype(str) + '_' + dt['periodo'].astype(str)
            out = pd.concat([out, dt], ignore_index=True)
        return out

    df_lp = leer_grupo(lp_b)
    df_mp  = leer_grupo(mp_b)
    df_cp  = leer_grupo(cp_b)
    if df_lp.empty: return pd.DataFrame()

    # Filtro por fase
    if modo_fase_k == "Por fase" and fase_k:
        for d in [df_lp, df_mp, df_cp]:
            if 'fase' in d.columns:
                d.drop(d[d['fase'].astype(str).str.lower() != fase_k].index, inplace=True)

    # MERGE EXACTO Cell 9
    df_mp_ren = df_mp.add_suffix('_mp'); df_mp_ren['id'] = df_mp['id']
    df_cp_ren = df_cp.add_suffix('_cp'); df_cp_ren['id'] = df_cp['id']
    df = df_lp.merge(df_mp_ren, on='id', how='outer')
    df = df.merge(df_cp_ren, on='id', how='outer')
    del df_lp, df_mp, df_cp, df_mp_ren, df_cp_ren

    # Drop columnas innecesarias post-merge
    cols_to_drop = [c for c in COLS_DROP_POST_MERGE if c in df.columns]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)

    # Tonelajes (Cell 14) — proportional_volume sin sufijo
    pv = 'proportional_volume'
    if 'densidad'    in df.columns and pv in df.columns:
        df['tonelaje']    = pd.to_numeric(df['densidad'],    errors='coerce') * pd.to_numeric(df[pv], errors='coerce')
    if 'densidad_mp' in df.columns and pv in df.columns:
        df['tonelaje_mp'] = pd.to_numeric(df['densidad_mp'], errors='coerce') * pd.to_numeric(df[pv], errors='coerce')
    if 'densidad_cp' in df.columns and pv in df.columns:
        df['tonelaje_cp'] = pd.to_numeric(df['densidad_cp'], errors='coerce') * pd.to_numeric(df[pv], errors='coerce')

    # nn → est en ocurrencia_cp (Cell 37)
    if 'ocurrencia_cp' in df.columns:
        df.loc[df['ocurrencia_cp'] == 'nn', 'ocurrencia_cp'] = 'est'

    return df


def clasificar(df, criterios):
    """
    Clasificación ore en tiempo real — no está en cache.
    Criterios dinámicos: lista de {'var','op','val'}.
    ue_fe >= 1 siempre requerido + todos los criterios adicionales.
    """
    df = df.copy()

    def _evaluar(row_ue, row_fem_dict, crit):
        """Evalúa todos los criterios sobre columnas del df."""
        resultados = []
        for c in crit:
            col = row_fem_dict.get(c['var'])
            if col is None: resultados.append(False); continue
            val = pd.to_numeric(col, errors='coerce').fillna(0)
            op = c['op']
            if   op == '>=': resultados.append(val >= c['val'])
            elif op == '<=': resultados.append(val <= c['val'])
            elif op == '>':  resultados.append(val >  c['val'])
            elif op == '<':  resultados.append(val <  c['val'])
        return resultados

    for suf, ue_col, ore_col in [('', 'ue_fe', 'ore_lp'),
                                  ('_mp', 'ue_fe_mp', 'ore_mp'),
                                  ('_cp', 'ue_fe_cp', 'ore_cp')]:
        ue = pd.to_numeric(df.get(ue_col, pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        ore = pd.Series('esteril', index=df.index)
        ore[ue >= 1] = 'marginal'

        # Evaluar todos los criterios
        mask_mineral = ue >= 1
        for c in criterios:
            col_name = f"{c['var']}{suf}"
            if col_name not in df.columns: mask_mineral = pd.Series(False, index=df.index); break
            val = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
            op = c['op']
            if   op == '>=': mask_mineral = mask_mineral & (val >= c['val'])
            elif op == '<=': mask_mineral = mask_mineral & (val <= c['val'])
            elif op == '>':  mask_mineral = mask_mineral & (val >  c['val'])
            elif op == '<':  mask_mineral = mask_mineral & (val <  c['val'])
        ore[mask_mineral] = 'mineral'
        df[ore_col] = ore

    df['conciliacion']    = df['ore_mp'] + '_' + df['ore_cp']
    df['conciliacion_lp'] = df['ore_lp'] + '_' + df['ore_cp']
    return df

# ─── MAIN ───
st.markdown("# Conciliación Mensual · Mina Los Colorados")

if not files_lp or not files_mp or not files_cp:
    st.markdown("<div class='warn-box'>⚠️ Carga los 36 archivos CSV (12 LP + 12 MP + 12 CP).</div>",
                unsafe_allow_html=True)
    st.stop()

lp_b = [(f.name, f.read()) for f in files_lp]
mp_b = [(f.name, f.read()) for f in files_mp]
cp_b = [(f.name, f.read()) for f in files_cp]

cache_key = (
    tuple((f.name, len(c)) for f,(_,c) in zip(files_lp, lp_b)),
    tuple((f.name, len(c)) for f,(_,c) in zip(files_mp, mp_b)),
    tuple((f.name, len(c)) for f,(_,c) in zip(files_cp, cp_b)),
)

with st.spinner("Cargando datos..."):
    df_raw = cargar_raw(lp_b, mp_b, cp_b, modo_fase, fase_sel, _cache_key=cache_key)

if df_raw.empty: st.error("No se pudieron cargar los datos."); st.stop()

# Clasificar ore en tiempo real — no afecta el cache de archivos
df = clasificar(df_raw, criterios)

mes_abr=MESES[mes]; mes_nombre=calendar.month_name[mes]
fase_lbl=f" · Fase {fase_sel.upper()}" if modo_fase=="Por fase" and fase_sel else ""
ocu_lbl =f" · {ocu_sel.upper()}" if modo_ocu=="Por ocurrencia" and ocu_sel else ""

# ─── TABLA MENSUAL (Cell 15) ───
def build_tabla(df, ocu_sel=None):
    """Replica exactamente Cell 15: resultado/resultado_mp/resultado_cp por separado, luego merge."""
    variable=ocu_sel

    def filtro(df,tipo):
        if tipo=="lp":   cond=df['ore_lp']=='mineral'
        elif tipo=="mp": cond=df['ore_mp']=='mineral'
        else:            cond=df['ore_cp']=='mineral'
        base=(df['periodo']>0)&cond
        if variable:
            base=base&(df['ocurrencia']==variable)
        return base

    # LP
    lp_cols={'fe':'fe','fem':'fem','fedtt':'fedtt','dtt':'dtt','p':'p','s':'s'}
    res_lp=(df[filtro(df,"lp")].groupby('periodo',group_keys=False)
            .apply(lambda x: pd.Series({
                **{k:(x[v]*x['tonelaje']).sum()/x['tonelaje'].sum()
                   for k,v in lp_cols.items() if v in x.columns},
                'volumen': x['proportional_volume'].sum() if 'proportional_volume' in x.columns else np.nan,
                'tonelaje': x['tonelaje'].sum()/1_000,
            }),include_groups=False).reset_index())

    # MP
    mp_cols={f'{k}_mp':f'{k}_mp' for k in ['fe','fem','fedtt','dtt','p','s']}
    res_mp=(df[filtro(df,"mp")].groupby('periodo',group_keys=False)
            .apply(lambda x: pd.Series({
                **{k:(x[v]*x['tonelaje_mp']).sum()/x['tonelaje_mp'].sum()
                   for k,v in mp_cols.items() if v in x.columns},
                'volumen_mp': x['proportional_volume'].sum() if 'proportional_volume' in x.columns else np.nan,
                'tonelaje_mp': x['tonelaje_mp'].sum()/1_000,
            }),include_groups=False).reset_index())

    # CP
    cp_cols={f'{k}_cp':f'{k}_cp' for k in ['fe','fem','fedtt','dtt','p','s']}
    res_cp=(df[filtro(df,"cp")].groupby('periodo',group_keys=False)
            .apply(lambda x: pd.Series({
                **{k:(x[v]*x['tonelaje_cp']).sum()/x['tonelaje_cp'].sum()
                   for k,v in cp_cols.items() if v in x.columns},
                'volumen_cp': x['proportional_volume'].sum() if 'proportional_volume' in x.columns else np.nan,
                'tonelaje_cp': x['tonelaje_cp'].sum()/1_000,
            }),include_groups=False).reset_index())

    # Merge final (Cell 15)
    tabla=res_lp.merge(res_mp,on='periodo',how='outer').merge(res_cp,on='periodo',how='outer')

    # Agregar meses (Cell 17)
    df_m=pd.DataFrame({'periodo':range(1,13),'mes':[MESES[i] for i in range(1,13)]})
    tabla=pd.merge(df_m,tabla,on='periodo',how='left')

    # Finos con DTT (Cell 17)
    for suf in ['','_mp','_cp']:
        dc,dt,df_=f'dtt{suf}',f'tonelaje{suf}',f'fino{suf}'
        if dc in tabla.columns and dt in tabla.columns:
            tabla[df_]=tabla[dc]*tabla[dt]/100

    # Fila Total (Cell 17)
    total={'mes':'Total','periodo':0}
    for suf in ['','_mp','_cp']:
        for col in ['fe','fem','fedtt','dtt','p','s']:
            cn,tn=f'{col}{suf}',f'tonelaje{suf}'
            if cn in tabla.columns and tn in tabla.columns:
                total[cn]=pond(tabla,cn,tn)
        for col in ['tonelaje','volumen','fino']:
            cn=f'{col}{suf}'
            if cn in tabla.columns: total[cn]=tabla[cn].sum()

    return pd.concat([tabla,pd.DataFrame([total])],ignore_index=True).round(2)

tabla_mensual=build_tabla(df,ocu_sel if modo_ocu=="Por ocurrencia" else None)

# tabla_plot sin Total
_df12=pd.DataFrame({'periodo':range(1,13),'mes':[MESES[i] for i in range(1,13)]})
tabla_plot=pd.merge(_df12,
                    tabla_mensual[tabla_mensual['mes']!='Total'].drop(columns=['mes'],errors='ignore'),
                    on='periodo',how='left').reset_index(drop=True)

mes_row  =tabla_mensual[tabla_mensual['periodo']==mes]
total_row=tabla_mensual[tabla_mensual['mes']=='Total']

# ─── CONCILIACIONES (Cell 30) ───
try:
    dm=df[df['periodo'].notna()&(df['periodo']!=-99)]
    conc_lp=(dm.groupby(['periodo','conciliacion_lp'],group_keys=False)
               .apply(lambda x: pd.Series({
                   'tonelaje': x['tonelaje'].sum(),
               }),include_groups=False).reset_index())
    conc_cp_v2=(dm.groupby(['periodo','conciliacion_lp'],group_keys=False)
                  .apply(lambda x: pd.Series({
                      'tonelaje_cp': x['tonelaje_cp'].sum(),
                  }),include_groups=False).reset_index())
    conc_mp_all=(dm.groupby(['periodo','conciliacion'],group_keys=False)
                   .apply(lambda x: pd.Series({
                       'tonelaje_mp': x['tonelaje_mp'].sum(),
                       'tonelaje_cp': x['tonelaje_cp'].sum(),
                   }),include_groups=False).reset_index())
except Exception: conc_lp=conc_cp_v2=conc_mp_all=pd.DataFrame()

# ─── TABS ───
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "📊 Balance","📈 Gráficos","🗺️ Visualización",
    "🌊 Cascadas","🔬 Dispersión","📐 Cuadrantes FeM","🧮 Matrices"])

# ── TAB 1: BALANCE ──
with tab1:
    st.markdown(f"### Balance Mensual — Mina Los Colorados{fase_lbl}{ocu_lbl}")
    st.markdown(f"*Mineral: {cutoff_str}*")
    st.markdown(f"#### Mes seleccionado — {mes_abr}")
    c1,c2,c3=st.columns(3)
    for w,mod,k_ton,k_fe,k_fem,k_dtt in [
        (c1,"LP","tonelaje",   "fe",    "fem",    "dtt"),
        (c2,"MP","tonelaje_mp","fe_mp", "fem_mp", "dtt_mp"),
        (c3,"CP","tonelaje_cp","fe_cp", "fem_cp", "dtt_cp"),
    ]:
        with w:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Tonelaje {mod} (kt)</div>
                <div class='metric-value'>{sv(mes_row,k_ton)}</div>
                <div class='metric-sub'>Fe:{sv(mes_row,k_fe,"{:.2f}")}% | FeM:{sv(mes_row,k_fem,"{:.2f}")}% | DTT:{sv(mes_row,k_dtt,"{:.2f}")}%</div>
            </div>""",unsafe_allow_html=True)

    st.markdown("#### Acumulado anual")
    c4,c5,c6=st.columns(3)
    for w,mod,k_ton,k_fe,k_fem,k_dtt in [
        (c4,"LP","tonelaje",   "fe",    "fem",    "dtt"),
        (c5,"MP","tonelaje_mp","fe_mp", "fem_mp", "dtt_mp"),
        (c6,"CP","tonelaje_cp","fe_cp", "fem_cp", "dtt_cp"),
    ]:
        with w:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Tonelaje {mod} acum. (kt)</div>
                <div class='metric-value'>{sv(total_row,k_ton)}</div>
                <div class='metric-sub'>Fe:{sv(total_row,k_fe,"{:.2f}")}% | FeM:{sv(total_row,k_fem,"{:.2f}")}% | DTT:{sv(total_row,k_dtt,"{:.2f}")}%</div>
            </div>""",unsafe_allow_html=True)

    # ── Desviaciones ──
    st.markdown("---")
    st.markdown("#### Desviaciones")
    dev_modelo = st.radio("Comparar CP contra:", ["LP","MP"], horizontal=True, key="dev_mod")
    k_ton_dev = "tonelaje"    if dev_modelo=="LP" else "tonelaje_mp"
    k_fe_dev  = "fe"          if dev_modelo=="LP" else "fe_mp"
    k_fem_dev = "fem"         if dev_modelo=="LP" else "fem_mp"
    k_dtt_dev = "dtt"         if dev_modelo=="LP" else "dtt_mp"

    def _dev(src, col_a, col_b, fmt_v="{:+,.1f}", fmt_p="{:+.1f}"):
        try:
            a = src[col_a].values[0]; b = src[col_b].values[0]
            if pd.isna(a) or pd.isna(b): return "—","—","neu"
            diff = float(b) - float(a)
            pct  = diff / abs(float(a)) * 100 if float(a) != 0 else 0
            cls  = "pos" if diff >= 0 else "neg"
            return fmt_v.format(diff), fmt_p.format(pct), cls
        except: return "—","—","neu"

    st.markdown(f"**Mes — {mes_abr}**")
    d1,d2,d3,d4 = st.columns(4)
    for w, lbl, ka, kb in [
        (d1, f"Ton. {dev_modelo} → CP (kt)", k_ton_dev, "tonelaje_cp"),
        (d2, f"Fe {dev_modelo} → CP (%)",    k_fe_dev,  "fe_cp"),
        (d3, f"FeM {dev_modelo} → CP (%)",   k_fem_dev, "fem_cp"),
        (d4, f"DTT {dev_modelo} → CP (%)",   k_dtt_dev, "dtt_cp"),
    ]:
        dv, dp, cls = _dev(mes_row, ka, kb)
        with w:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-dev {cls}'>{dv}</div>
                <div class='metric-dev {cls}' style='font-size:11px;'>{dp}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"**Acumulado anual**")
    d5,d6,d7,d8 = st.columns(4)
    for w, lbl, ka, kb in [
        (d5, f"Ton. {dev_modelo} → CP (kt)", k_ton_dev, "tonelaje_cp"),
        (d6, f"Fe {dev_modelo} → CP (%)",    k_fe_dev,  "fe_cp"),
        (d7, f"FeM {dev_modelo} → CP (%)",   k_fem_dev, "fem_cp"),
        (d8, f"DTT {dev_modelo} → CP (%)",   k_dtt_dev, "dtt_cp"),
    ]:
        dv, dp, cls = _dev(total_row, ka, kb)
        with w:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-dev {cls}'>{dv}</div>
                <div class='metric-dev {cls}' style='font-size:11px;'>{dp}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    cols_d=[c for c in ['mes','fe','fem','fedtt','dtt','p','s','tonelaje','fino',
                        'fe_mp','fem_mp','fedtt_mp','dtt_mp','p_mp','s_mp','tonelaje_mp','fino_mp',
                        'fe_cp','fem_cp','fedtt_cp','dtt_cp','p_cp','s_cp','tonelaje_cp','fino_cp']
            if c in tabla_mensual.columns]
    def _hl(row):
        if row['mes']=='Total': return ['background-color:#1e2235;font-weight:bold']*len(row)
        elif row['mes']==mes_abr: return ['font-weight:700;border-left:3px solid #E8650A']*len(row)
        return ['']*len(row)
    st.dataframe(tabla_mensual[cols_d].style
                 .format({c:'{:.2f}' for c in cols_d if c!='mes'})
                 .apply(_hl,axis=1),use_container_width=True,hide_index=True)
    st.download_button("⬇️ CSV",csv_bytes(tabla_mensual[cols_d]),"balance_mlc.csv","text/csv",key="dl_t")

# ── TAB 2: GRÁFICOS ──
with tab2:
    plt.rcParams['font.family']='DejaVu Sans'
    titulo_mlc=f"Mina Los Colorados{fase_lbl}{ocu_lbl}"
    st.markdown("#### ⚙️ Opciones")

    # ── Extracción Mensual ──
    st.markdown("**Extracción Mensual**")
    ca,cb,cc,cd=st.columns(4)
    with ca: ton_ymin_m=st.number_input("Ton. mín (kt)",value=0,   step=50, key="ty0m")
    with cb: ton_ymax_m=st.number_input("Ton. máx (kt)",value=2000,step=50, key="ty1m")
    with cc: ley_ymin_m=st.number_input("Ley mín (%)",  value=15.0,step=1.0,key="ly0m")
    with cd: ley_ymax_m=st.number_input("Ley máx (%)",  value=45.0,step=1.0,key="ly1m")

    # ── Extracción Trimestral ──
    st.markdown("**Extracción Trimestral**")
    ca2,cb2,cc2,cd2=st.columns(4)
    with ca2: ton_ymin_t=st.number_input("Ton. mín (kt)",value=0,   step=50, key="ty0t")
    with cb2: ton_ymax_t=st.number_input("Ton. máx (kt)",value=4500,step=50, key="ty1t")
    with cc2: ley_ymin_t=st.number_input("Ley mín (%)",  value=15.0,step=1.0,key="ly0t")
    with cd2: ley_ymax_t=st.number_input("Ley máx (%)",  value=45.0,step=1.0,key="ly1t")
    show_ann=st.checkbox("Etiquetas en barras",value=True,key="sann")
    with st.expander("🎨 Paleta"):
        _a,_b,_c=st.columns(3)
        with _a: col_lp=st.color_picker("LP barra","#D3D3D3",key="clp"); col_ll=st.color_picker("LP línea","#008000",key="cll")
        with _b: col_mp=st.color_picker("MP barra","#696969",key="cmp"); col_ml=st.color_picker("MP línea","#4682B4",key="cml")
        with _c: col_cp=st.color_picker("CP barra","#000000",key="ccp"); col_cl=st.color_picker("CP línea","#9370DB",key="ccl")

    # Mensual
    st.markdown(f"### Extracción Mensual · {var_cal.upper()}")
    mod_sel=st.radio("Comparar CP contra:",["LP","MP"],horizontal=True,key="rmod")
    c_ton=('tonelaje' if mod_sel=="LP" else 'tonelaje_mp')
    c_cal=(var_cal     if mod_sel=="LP" else f'{var_cal}_mp')
    c_calcp=f'{var_cal}_cp'
    bclr=(col_lp if mod_sel=="LP" else col_mp)
    lclr=(col_ll if mod_sel=="LP" else col_ml)

    pos=np.arange(12); ml=[MESES[i] for i in range(1,13)]
    def arr(d,c):
        if c not in d.columns: return np.full(12,np.nan)
        a=d[c].values[:12].astype(float)
        if len(a)<12: a=np.concatenate([a,np.full(12-len(a),np.nan)])
        return a
    tm=arr(tabla_plot,c_ton); cal=arr(tabla_plot,c_cal)
    tcp=arr(tabla_plot,'tonelaje_cp'); calcp=arr(tabla_plot,c_calcp)

    fig,ax1=make_fig((14,5)); bw=0.4
    b1=ax1.bar(pos-bw/2,tm, bw,color=bclr,label=f'Ton {mod_sel}')
    b2=ax1.bar(pos+bw/2,tcp,bw,color=col_cp,label='Ton CP')
    if show_ann: ann_bars(ax1,b1,pd.Series(cal)); ann_bars(ax1,b2,pd.Series(calcp))
    ax1.set_ylabel('Tonelaje (kt)',color='#444'); ax1.set_xticks(pos); ax1.set_xticklabels(ml,color='#444')
    ax1.set_xlim(-0.5,11.5); ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax1.tick_params(colors='#444'); ax1.set_ylim(ton_ymin_m,ton_ymax_m); style_ax(ax1)
    ax2=ax1.twinx(); ax2.set_facecolor('white')
    ax2.plot(pos,cal,  marker='.',color=lclr, lw=1.8,label=f'{var_cal.upper()} {mod_sel}')
    ax2.plot(pos,calcp,marker='.',color=col_cl,lw=1.8,label=f'{var_cal.upper()} CP')
    ax2.set_ylabel(f'Ley {var_cal.upper()} (%)',color='#444'); ax2.set_ylim(ley_ymin_m,ley_ymax_m)
    ax2.tick_params(colors='#444')
    for s in ['top','bottom']: ax2.spines[s].set_visible(False)
    for s in ['left','right']:  ax2.spines[s].set_color('#ccc')
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc='upper center',bbox_to_anchor=(0.5,-0.12),frameon=False,ncol=4,labelcolor='#333',fontsize=9)
    ax1.set_title(f'Extracción Mensual {anio} — {titulo_mlc}\n{cutoff_str}',color='#222',fontsize=11,pad=10)
    plt.tight_layout()
    png_m=save_png(fig); st.pyplot(fig); plt.close()
    st.download_button("⬇️ PNG Mensual",png_m,"ext_mensual_mlc.png","image/png",key="dlm")

    # Trimestral
    st.markdown(f"### Extracción Trimestral · {var_cal.upper()}")
    tp2=tabla_mensual[tabla_mensual['mes']!='Total'].copy()
    tp2['t']=np.select([tp2['mes'].isin(['Ene','Feb','Mar']),tp2['mes'].isin(['Abr','May','Jun']),
                        tp2['mes'].isin(['Jul','Ago','Sep']),tp2['mes'].isin(['Oct','Nov','Dic'])],
                       ['1T','2T','3T','4T'],default=pd.NA)
    td=[]
    for t in ['1T','2T','3T','4T']:
        g=tp2[tp2['t']==t]
        row={'trimestre':t}
        for sf,vc,tc in [('lp',var_cal,'tonelaje'),('mp',f'{var_cal}_mp','tonelaje_mp'),('cp',f'{var_cal}_cp','tonelaje_cp')]:
            row[f'ton_{sf}']=g[tc].sum(skipna=True) if tc in g.columns else np.nan
            row[f'cal_{sf}']=pond(g,vc,tc) if (vc in g.columns and tc in g.columns) else np.nan
        td.append(row)
    tt=pd.DataFrame(td)
    fig2,ax1=make_fig((12,5)); p2=np.arange(len(tt)); bw2=0.28
    for tc,cc_c,lbl,clr,idx in [('ton_lp','cal_lp','LP',col_lp,0),('ton_mp','cal_mp','MP',col_mp,1),('ton_cp','cal_cp','CP',col_cp,2)]:
        if tc in tt.columns:
            bars=ax1.bar(p2+(idx-1)*bw2,tt[tc],bw2,color=clr,label=f'Ton {lbl}')
            if show_ann: ann_bars(ax1,bars,tt[cc_c])
    ax1.set_ylabel('Tonelaje (kt)',color='#444'); ax1.set_xticks(p2); ax1.set_xticklabels(tt['trimestre'],color='#444')
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}')); ax1.tick_params(colors='#444')
    ax1.set_ylim(ton_ymin_t,ton_ymax_t); style_ax(ax1)
    ax2b=ax1.twinx(); ax2b.set_facecolor('white')
    for cc_c,lbl,lc in [('cal_lp',f'{var_cal.upper()} LP',col_ll),('cal_mp',f'{var_cal.upper()} MP',col_ml),('cal_cp',f'{var_cal.upper()} CP',col_cl)]:
        if cc_c in tt.columns: ax2b.plot(p2,tt[cc_c],marker='.',color=lc,lw=1.8,label=lbl)
    ax2b.set_ylabel(f'Ley {var_cal.upper()} (%)',color='#444'); ax2b.set_ylim(ley_ymin_t,ley_ymax_t); ax2b.tick_params(colors='#444')
    for s in ['top','bottom']: ax2b.spines[s].set_visible(False)
    for s in ['left','right']:  ax2b.spines[s].set_color('#ccc')
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc='upper center',bbox_to_anchor=(0.5,-0.1),frameon=False,ncol=6,labelcolor='#333',fontsize=9)
    ax1.set_title(f'Extracción Trimestral {anio} — {titulo_mlc}\n{cutoff_str}',color='#222',fontsize=11,pad=10)
    plt.tight_layout(); png_t=save_png(fig2); st.pyplot(fig2); plt.close()
    st.download_button("⬇️ PNG Trimestral",png_t,"ext_trim_mlc.png","image/png",key="dlt")

    # Acumulado
    st.markdown("### Extracción Acumulada")
    show_ann_acum=st.checkbox("Mostrar etiquetas en acumulado",value=True,key="sann_acum")
    fig3,ax3=make_fig((14,5))
    for col,lbl,clr,off in [('tonelaje','LP',col_lp,-18),('tonelaje_mp','MP',col_mp,-10),('tonelaje_cp','CP',col_cp,8)]:
        if col in tabla_plot.columns:
            cum=tabla_plot[col].cumsum()
            ax3.plot(tabla_plot['mes'],cum,marker='o',label=lbl,color=clr,lw=1.8)
            if show_ann_acum:
                for i,(m,v) in enumerate(zip(tabla_plot['mes'],cum)):
                    if not pd.isna(v): ax3.annotate(f'{v:,.1f}',xy=(i,v),xytext=(0,off),textcoords='offset points',ha='center',fontsize=7,color=clr)
    if target_ton>0:
        ax3.plot(tabla_plot['mes'],np.cumsum([target_ton/12]*12),'--',color='#cc4444',lw=1.5,label=f'Obj({target_ton:,.0f}kt)')
    ax3.set_ylabel('Tonelaje acumulado (kt)',color='#444'); ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax3.tick_params(colors='#444'); style_ax(ax3); ax3.legend(frameon=False,labelcolor='#333')
    ax3.set_title(f'Gráfico Acumulado {anio} — {titulo_mlc}\n{cutoff_str}',color='#222',fontsize=11,pad=10)
    plt.tight_layout(); png_a=save_png(fig3); st.pyplot(fig3); plt.close()
    st.download_button("⬇️ PNG Acumulado",png_a,"ext_acum_mlc.png","image/png",key="dla")

# ── TAB 3: VISUALIZACIÓN (Cell 27,28) ──
with tab3:
    st.markdown("### Visualización de Bloques por Banco")
    st.markdown("*Bloques 12.5×12.5 m — MLC*")
    tipo_m=st.selectbox("Visualizar por:",["Clasificación (Ore)","Ocurrencia"])
    cv1,cv2=st.columns(2)
    with cv1: panel_m=st.radio("Paneles:",["MP y CP","Solo MP","Solo CP"],horizontal=True,key="pm")
    with cv2: grilla=st.checkbox("Mostrar grilla",value=True)
    cg1,cg2=st.columns(2)
    with cg1: grid_x=st.number_input("Espaciado grilla X (m)",value=100,step=25,min_value=10,key="gx")
    with cg2: grid_y=st.number_input("Espaciado grilla Y (m)",value=100,step=25,min_value=10,key="gy")
    TAM=12.5
    # Cota (Cell 27): cota = df.loc[periodo==mes, centroid_z].min()
    try:
        z_col='centroid_z' if 'centroid_z' in df.columns else 'centroid_z_mp'
        z_disp=sorted(df.loc[df['periodo']==mes,z_col].dropna().unique())
    except: z_disp=[]

    if not z_disp:
        st.warning(f"Sin bloques para periodo {mes}.")
    else:
        n_b=min(3,len(z_disp))
        nb1,nb2,_=st.columns([1,1,3])
        with nb1: bidx=st.number_input("Banco (0=base,+1,+2)",min_value=0,max_value=n_b-1,value=0,step=1,key="bidx")
        cota=z_disp[bidx]; banco_real=cota-7.5
        with nb2: st.markdown(f"<div class='cutoff-box'>Banco<br><b>Z={banco_real:.1f}m</b></div>",unsafe_allow_html=True)
        try:
            # Cell 27: df_mp_cota = df[df['centroid_z_mp']==cota]
            df_mp_c=df[df.get('centroid_z_mp',df.get('centroid_z',''))==cota].copy()
            df_mp_c=(df_mp_c.sort_values('periodo',ascending=False)
                     .drop_duplicates(subset=['centroid_x_mp','centroid_y_mp'],keep='first'))
            df_cp_c=df[df.get('centroid_z_cp',df.get('centroid_z',''))==cota].copy()
            if 'id' in df_mp_c.columns and 'id' in df_cp_c.columns:
                df_cp_c=df_cp_c.merge(df_mp_c[['id']],on='id',how='inner')

            if panel_m=="Solo MP":   pnl=[(df_mp_c,'MP','centroid_x_mp','centroid_y_mp')]
            elif panel_m=="Solo CP": pnl=[(df_cp_c,'CP','centroid_x_cp','centroid_y_cp')]
            else:                    pnl=[(df_mp_c,'MP','centroid_x_mp','centroid_y_mp'),
                                          (df_cp_c,'CP','centroid_x_cp','centroid_y_cp')]

            fig,axs=plt.subplots(1,len(pnl),figsize=(7*len(pnl),6),facecolor='white',squeeze=False)
            axs=axs[0]
            for ax in axs: ax.set_facecolor('white')

            def sax(ax,tit):
                ax.set_title(tit,color='#222',fontsize=10,fontweight='bold')
                ax.set_xlabel('Este',color='#444'); ax.set_ylabel('Norte',color='#444')
                ax.tick_params(colors='#444')
                if grilla:
                    ax.grid(True,color='#ccc',linewidth=0.5)
                    ax.xaxis.set_major_locator(MultipleLocator(grid_x))
                    ax.yaxis.set_major_locator(MultipleLocator(grid_y))
                for s in ['top','right']: ax.spines[s].set_visible(False)
                for s in ['bottom','left']: ax.spines[s].set_color('#ccc')
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{int(y):,}'))

            if tipo_m=="Clasificación (Ore)":
                clrs={'mineral':(255/255,136/255,255/255),'marginal':(153/255,136/255,255/255),'esteril':(238/255,238/255,238/255)}
                ore_c={'MP':'ore_mp','CP':'ore_cp'}
                for ax,(df_i,lbl,cx,cy) in zip(axs,pnl):
                    for _,row in df_i.iterrows():
                        c=clrs.get(str(row.get(ore_c[lbl],'esteril')),(0.8,0.8,0.8))
                        a=1.0 if row['periodo']==mes else 0.25
                        ax.add_patch(Rectangle((row[cx]-TAM/2,row[cy]-TAM/2),TAM,TAM,facecolor=(*c[:3],a),edgecolor='none'))
                    ax.autoscale(); ax.set_aspect('equal'); sax(ax,f'{lbl} — Banco={banco_real:.1f}m')
                ley=[Patch(facecolor=c,label=k) for k,c in clrs.items()]
                for ax in axs: ax.legend(handles=ley,frameon=False,labelcolor='#333')
            else:
                co={'mac':(1.0,0.0,0.0),'bre':(0.0,0.0,1.0),'gyd':(102/255,153/255,255/255),'est':(0.85,0.85,0.85)}
                occ_c={'MP':'ocurrencia_mp','CP':'ocurrencia_cp'}
                for ax,(df_i,lbl,cx,cy) in zip(axs,pnl):
                    oc=occ_c.get(lbl,'ocurrencia_mp')
                    for _,row in df_i.iterrows():
                        c=co.get(str(row.get(oc,'est')).lower(),(0.8,0.8,0.8))
                        a=1.0 if row['periodo']==mes else 0.25
                        ax.add_patch(Rectangle((row[cx]-TAM/2,row[cy]-TAM/2),TAM,TAM,facecolor=(*c[:3],a),edgecolor='none'))
                    ax.autoscale(); ax.set_aspect('equal'); sax(ax,f'{lbl} — Banco={banco_real:.1f}m')
                ley=[Patch(facecolor=c,label=k.upper()) for k,c in co.items()]
                for ax in axs: ax.legend(handles=ley,frameon=False,labelcolor='#333')

            plt.suptitle(f'Mina Los Colorados{fase_lbl} — {mes_abr} {anio}',color='#222',fontsize=11,y=1.01)
            plt.tight_layout(); png_bl=save_png(fig)
            st.pyplot(fig); plt.close()
            st.download_button("⬇️ PNG Bloques",png_bl,"bloques_mlc.png","image/png",key="dlbl")
        except Exception as e: st.error(f"Error mapa: {e}")

# ── TAB 4: CASCADAS (Cell 32,33) ──
with tab4:
    st.markdown("### Gráficos de Cascada")

    def waterfall(x,y,titulo,yrange=None):
        fw=go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","total","relative","relative","relative","total"],
            x=x, text=[f"{v:,.1f}" for v in y], textposition="outside", y=y,
            textfont=dict(color='black', size=13),
            increasing={"marker":{"color":"#43A047"}},
            decreasing={"marker":{"color":"#EF5350"}},
            totals={"marker":{"color":"#42A5F5"}},
            connector={"line":{"color":"#999","width":1}},
        ))
        fw.update_layout(
            title=dict(text=f"<b>{titulo}</b>",
                       font=dict(size=20,color='black'),x=0,xanchor='left'),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black', size=13),
            height=420, margin=dict(t=80,b=60,l=60,r=30)
        )
        fw.update_yaxes(title_text="Tonelaje (kt)", range=yrange, tickformat=",d",
                        showgrid=False, zerolinecolor='#ccc',
                        tickfont=dict(color='black',size=12),
                        title_font=dict(color='black',size=12))
        fw.update_xaxes(tickfont=dict(color='black',size=12), showgrid=False)
        return fw

    X_CASC=["Proyectado","Mineral Marginal","Mineral Esteril","Mineral Mineral",
             "Ajuste por densidad","Marginal Mineral","Esteril Mineral","Mineral Real"]

    def _color_tipo(row):
        t=row.get('Tipo','')
        if t=='Pérdida': return ['color:#cc4444']*len(row)
        if t=='Ganancia': return ['color:#44aa66']*len(row)
        if t in ('Proyectado','Real'): return ['font-weight:bold']*len(row)
        return ['']*len(row)

    # ── Cascada Mensual ──
    st.markdown("#### Cascada Mensual")
    _cs0,_cm1,_cm2,_=st.columns([1,1,1,2])
    with _cs0: modelo_casc_mes=st.radio("Modelo vs CP:",["MP","LP"],horizontal=True,key="r_casc_mes_mlc")
    with _cm1: cy0=st.number_input("Ton. mín",value=0,   step=50, key="cmy0")
    with _cm2: cy1=st.number_input("Ton. máx",value=2000,step=50, key="cmy1")
    try:
        if modelo_casc_mes == "MP":
            # MP vs CP: usa conc_mp_all filtrado por mes
            dm_s = conc_mp_all[conc_mp_all['periodo'] == mes]
            def _mp(cv,col): return dm_s[dm_s['conciliacion']==cv][col].sum()/1000 if not dm_s.empty else 0
            proy  = dm_s[dm_s['conciliacion'].isin(['mineral_mineral','mineral_marginal','mineral_esteril'])]['tonelaje_mp'].sum()/1000
            mm    = _mp('mineral_marginal','tonelaje_mp')
            me    = _mp('mineral_esteril', 'tonelaje_mp')
            mmin  = _mp('mineral_mineral', 'tonelaje_mp')
            aj    = _mp('mineral_mineral', 'tonelaje_cp') - mmin
            marg  = _mp('marginal_mineral','tonelaje_cp')
            est   = _mp('esteril_mineral', 'tonelaje_cp')
            real  = dm_s[dm_s['conciliacion'].isin(['mineral_mineral','marginal_mineral','esteril_mineral'])]['tonelaje_cp'].sum()/1000
            lbl_proy_mes = f"Proyectado MP {mes_abr}"
        else:
            # LP vs CP: usa conc_lp y conc_cp_v2 filtrado por mes
            dl_m = conc_lp[conc_lp['periodo'] == mes]
            dc_m = conc_cp_v2[conc_cp_v2['periodo'] == mes]
            def _lp_m(cv): return dl_m[dl_m['conciliacion_lp']==cv]['tonelaje'].sum()/1000 if not dl_m.empty else 0
            def _cp_m(cv): return dc_m[dc_m['conciliacion_lp']==cv]['tonelaje_cp'].sum()/1000 if not dc_m.empty else 0
            proy  = dl_m[dl_m['conciliacion_lp'].isin(['mineral_mineral','mineral_marginal','mineral_esteril'])]['tonelaje'].sum()/1000 if not dl_m.empty else 0
            mm    = _lp_m('mineral_marginal'); me=_lp_m('mineral_esteril'); mmin=_lp_m('mineral_mineral')
            aj    = _cp_m('mineral_mineral') - mmin
            marg  = _cp_m('marginal_mineral'); est=_cp_m('esteril_mineral')
            real  = dc_m[dc_m['conciliacion_lp'].isin(['mineral_mineral','marginal_mineral','esteril_mineral'])]['tonelaje_cp'].sum()/1000 if not dc_m.empty else 0
            lbl_proy_mes = f"Budget LP {mes_abr}"

        fw1=waterfall(X_CASC,[proy,-mm,-me,mmin,aj,marg,est,real],
                      f"Mineral Proyectado/Real ({mes_abr}) — {modelo_casc_mes} vs CP: {cutoff_str}",
                      yrange=[cy0,cy1])
        st.plotly_chart(fw1,use_container_width=True)
        df_r1=pd.DataFrame({'Paso':X_CASC,'Tonelaje (kt)':[round(v,1) for v in [proy,-mm,-me,mmin,aj,marg,est,real]],
                             'Tipo':['Proyectado','Pérdida','Pérdida','Subtotal','Ajuste','Ganancia','Ganancia','Real']})
        with st.expander("📋 Tabla cascada mensual"):
            try: st.dataframe(df_r1.style.apply(_color_tipo,axis=1),hide_index=True,use_container_width=True)
            except: st.dataframe(df_r1,hide_index=True,use_container_width=True)
        d1,d2,_=st.columns([1,1,3])
        with d1: st.download_button("⬇️ HTML",html_bytes(fw1),f"casc_mes_mlc_{mes_abr}.html","text/html",key="dlcm")
        with d2: st.download_button("⬇️ CSV",csv_bytes(df_r1),f"casc_mes_mlc_{mes_abr}.csv","text/csv",key="dlcmc")
    except Exception as e: st.error(f"Error cascada mensual: {e}")

    st.markdown("---")

    # ── Cascada Acumulada ──
    st.markdown("#### Cascada Acumulada")
    _cas0,_ca1,_ca2,_=st.columns([1,1,1,2])
    with _cas0: modelo_casc_acum=st.radio("Modelo vs CP:",["MP","LP"],horizontal=True,key="r_casc_acum_mlc")
    with _ca1: cay0=st.number_input("Ton. mín",value=0,    step=100,key="cay0")
    with _ca2: cay1=st.number_input("Ton. máx",value=15000,step=500,key="cay1")
    try:
        if modelo_casc_acum == "MP":
            dm_a = conc_mp_all[conc_mp_all['periodo'] <= mes]
            def _mp_a(cv,col): return dm_a[dm_a['conciliacion']==cv][col].sum()/1000 if not dm_a.empty else 0
            proy2  = dm_a[dm_a['conciliacion'].isin(['mineral_mineral','mineral_marginal','mineral_esteril'])]['tonelaje_mp'].sum()/1000
            mm2    = _mp_a('mineral_marginal','tonelaje_mp'); me2=_mp_a('mineral_esteril','tonelaje_mp')
            mmin2  = _mp_a('mineral_mineral', 'tonelaje_mp')
            aj2    = _mp_a('mineral_mineral', 'tonelaje_cp') - mmin2
            marg2  = _mp_a('marginal_mineral','tonelaje_cp'); est2=_mp_a('esteril_mineral','tonelaje_cp')
            real2  = dm_a[dm_a['conciliacion'].isin(['mineral_mineral','marginal_mineral','esteril_mineral'])]['tonelaje_cp'].sum()/1000
            lbl_proy_acum = f"Proyectado MP (hasta {mes_abr})"
        else:
            dl = conc_lp[conc_lp['periodo'] <= mes]
            dc = conc_cp_v2[conc_cp_v2['periodo'] <= mes]
            def _lp(cv): return dl[dl['conciliacion_lp']==cv]['tonelaje'].sum()/1000 if not dl.empty else 0
            def _cp(cv): return dc[dc['conciliacion_lp']==cv]['tonelaje_cp'].sum()/1000 if not dc.empty else 0
            proy2  = dl[dl['conciliacion_lp'].isin(['mineral_mineral','mineral_marginal','mineral_esteril'])]['tonelaje'].sum()/1000 if not dl.empty else 0
            mm2    = _lp('mineral_marginal'); me2=_lp('mineral_esteril'); mmin2=_lp('mineral_mineral')
            aj2    = _cp('mineral_mineral') - mmin2
            marg2  = _cp('marginal_mineral'); est2=_cp('esteril_mineral')
            real2  = dc[dc['conciliacion_lp'].isin(['mineral_mineral','marginal_mineral','esteril_mineral'])]['tonelaje_cp'].sum()/1000 if not dc.empty else 0
            lbl_proy_acum = f"Budget LP (hasta {mes_abr})"

        fw2=waterfall(X_CASC,[proy2,-mm2,-me2,mmin2,aj2,marg2,est2,real2],
                      f"Mineral Proyectado/Real Acumulado (hasta {mes_abr}) — {modelo_casc_acum} vs CP: {cutoff_str}",
                      yrange=[cay0,cay1])
        st.plotly_chart(fw2,use_container_width=True)
        df_r2=pd.DataFrame({'Paso':X_CASC,'Tonelaje (kt)':[round(v,1) for v in [proy2,-mm2,-me2,mmin2,aj2,marg2,est2,real2]],
                             'Tipo':['Proyectado','Pérdida','Pérdida','Subtotal','Ajuste','Ganancia','Ganancia','Real']})
        with st.expander("📋 Tabla cascada acumulada"):
            try: st.dataframe(df_r2.style.apply(_color_tipo,axis=1),hide_index=True,use_container_width=True)
            except: st.dataframe(df_r2,hide_index=True,use_container_width=True)
        d3,d4,_=st.columns([1,1,3])
        with d3: st.download_button("⬇️ HTML",html_bytes(fw2),f"casc_acum_mlc_{mes_abr}.html","text/html",key="dlca")
        with d4: st.download_button("⬇️ CSV",csv_bytes(df_r2),f"casc_acum_mlc_{mes_abr}.csv","text/csv",key="dlcac")
    except Exception as e: st.error(f"Error cascada acumulada: {e}")

# ── TAB 5: DISPERSIÓN (Cell 35) ──
with tab5:
    st.markdown("### Análisis de Dispersión — Fe")

    _d1, _d2, _d3 = st.columns(3)
    with _d1:
        modelo_disp = st.radio("Comparar CP contra:", ["MP","LP"],
                               horizontal=True, key="modelo_disp")
    with _d2:
        periodo_disp = st.radio("Período:", ["Mes","Acumulado"],
                                horizontal=True, key="pdisp")
    with _d3:
        col_ue = 'ue_fe_mp' if 'ue_fe_mp' in df.columns else 'ue_fe'
        ue_vals = sorted([int(x) for x in df[col_ue].dropna().unique()
                          if str(x).replace('.0','').lstrip('-').isdigit()
                          and 1 <= float(x) < 10])
        ue_vals = ue_vals or [4, 5]
        ue_sel = st.multiselect("UE_FE:", ue_vals,
                                default=[u for u in [4,5] if u in ue_vals] or ue_vals[:2],
                                key="uesel")

    if not ue_sel:
        st.warning("Selecciona al menos una UE_FE.")
    else:
        try:
            import seaborn as _sns
            clr_p = _sns.color_palette()[0]

            df_d = (df[df['periodo'] <= mes].copy() if periodo_disp == "Acumulado"
                    else df[df['periodo'] == mes].copy())
            periodo_lbl = f"Ene–{mes_abr}" if periodo_disp == "Acumulado" else mes_abr

            # Columna Y según selector LP o MP
            if modelo_disp == "MP":
                y_col    = 'fe_mp'
                ue_y_col = 'ue_fe_mp'
                y_label  = 'Fe_mp (%)'
            else:
                y_col    = 'fe'
                ue_y_col = 'ue_fe'
                y_label  = 'Fe_lp (%)'

            fig, axes = plt.subplots(1, len(ue_sel),
                                     figsize=(6*len(ue_sel), 6),
                                     facecolor='white', squeeze=False)
            axes = axes[0]
            for ax in axes: ax.set_facecolor('white')

            for idx, ue in enumerate(sorted(ue_sel)):
                ax = axes[idx]

                # Filtro: ue_fe del modelo Y == ue AND ue_fe_cp == ue
                mask = ((df_d[ue_y_col] == ue) &
                        (df_d['ue_fe_cp'] == ue)) if ue_y_col in df_d.columns else pd.Series(False, index=df_d.index)
                df_ue = df_d[mask][[y_col, 'fe_cp']].dropna()

                if df_ue.empty:
                    ax.text(0.5, 0.5, f'UE_FE={ue}\nSin datos',
                            transform=ax.transAxes, ha='center', color='#888')
                    ax.set_title(f'Fe {modelo_disp} vs CP — UE_FE={ue}: {periodo_lbl}',
                                 color='#222')
                    continue

                _sns.scatterplot(x='fe_cp', y=y_col, data=df_ue,
                                 alpha=1, edgecolors='none', s=20,
                                 ax=ax, color=clr_p)
                ax.plot([0, 70], [0, 70], color='gray', linestyle='-', lw=1)
                ax.set_xlim(0, 70); ax.set_ylim(0, 70)
                ax.set_xlabel('Fe_cp (%)', color='#444')
                ax.set_ylabel(y_label, color='#444')
                ax.set_title(f'Fe {modelo_disp} vs CP — UE_FE={ue}: {periodo_lbl}',
                             color='#222', fontsize=11)
                style_ax(ax)

                pr, _ = pearsonr(df_ue['fe_cp'], df_ue[y_col])
                ax.text(0.05, 0.95, f'Coeficiente de Pearson: {pr:.2f}',
                        transform=ax.transAxes, fontsize=11, va='top', color='green')

                ax_t = ax.inset_axes([0, 1.02, 1, 0.2], sharex=ax)
                ax_t.hist(df_ue['fe_cp'], bins=20, color=clr_p, alpha=0.9, linewidth=0)
                ax_t.axis('off')
                ax_r = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
                ax_r.hist(df_ue[y_col], bins=20, orientation='horizontal',
                          color=clr_p, alpha=0.9, linewidth=0)
                ax_r.axis('off')

            plt.suptitle(f'Mina Los Colorados{fase_lbl} — {modelo_disp} vs CP',
                         color='#222', fontsize=11, y=1.04)
            plt.tight_layout()
            png_d = save_png(fig, dpi=300)
            st.pyplot(fig); plt.close()
            st.download_button("⬇️ PNG Dispersión", png_d,
                               f"disp_mlc_{modelo_disp}_{mes_abr}.png",
                               "image/png", key="dldisp")

        except Exception as e:
            st.error(f"Error dispersión: {e}")

# ── TAB 6: CUADRANTES FeM (Cell 41) ──
with tab6:
    st.markdown("### Conciliación por Cuadrantes FeM")
    _cq1,_cq2,_cq3=st.columns(3)
    with _cq1: cutoff_q=st.number_input("Corte FeM (%)",value=cutoff_fem,step=0.5,key="cq")
    with _cq2: periodo_q=st.radio("Período:",["Mes","Acumulado"],horizontal=True,key="pq")
    with _cq3:
        HP={"Blues":"Blues","YlOrBr":"YlOrBr","Greens":"Greens","Purples":"Purples",
            "Oranges":"Oranges","RdYlGn":"RdYlGn","viridis":"viridis","coolwarm":"coolwarm"}
        hq=st.selectbox("🎨 Paleta",list(HP.keys()),index=0,key="hq")
    hq_cmap=HP[hq]

    try:
        # Cell 41: df_base = df[periodo==mes].drop_duplicates('block_id').loc[vol>=0.75*dims]
        df_q=df[df['periodo']<=mes].copy() if periodo_q=="Acumulado" else df[df['periodo']==mes].copy()
        if 'block_id' in df_q.columns: df_q=df_q.drop_duplicates('block_id')
        if 'proportional_volume' in df_q.columns and 'dim_x' in df_q.columns:
            df_q=df_q[df_q['proportional_volume']>=0.75*df_q['dim_x']*df_q['dim_y']*df_q['dim_z']].copy()

        # Cell 41: df_base usa fem_mp (MP) vs fem_cp (CP)
        if 'fem_mp' not in df_q.columns or 'fem_cp' not in df_q.columns:
            st.warning("Columnas fem_mp o fem_cp no disponibles.")
        else:
            df_fem = df_q[['fem_mp', 'fem_cp']].dropna().copy()

            if df_fem.empty:
                st.warning("Sin datos.")
            else:
                pr_q,_=pearsonr(df_fem['fem_cp'],df_fem['fem_mp'])
                coin=(df_fem['fem_cp']>=cutoff_q)&(df_fem['fem_mp']>=cutoff_q)
                perd=(df_fem['fem_cp']< cutoff_q)&(df_fem['fem_mp']>=cutoff_q)
                ster=(df_fem['fem_cp']< cutoff_q)&(df_fem['fem_mp']< cutoff_q)
                gain=(df_fem['fem_cp']>=cutoff_q)&(df_fem['fem_mp']< cutoff_q)
                cnt={"I":int(coin.sum()),"II":int(perd.sum()),"III":int(ster.sum()),"IV":int(gain.sum())}

                r1=st.columns(4)
                for cw,(lbl,val),cls in zip(r1,[("Coincidencia Mineral (I)",cnt["I"]),("Pérdida Mineral (II)",cnt["II"]),
                                                 ("Coincidencia Estéril (III)",cnt["III"]),("Ganancia Mineral (IV)",cnt["IV"])],
                                            ['green','red','','blue']):
                    with cw: st.markdown(_metric_html(lbl,f'{val:,}',cls),unsafe_allow_html=True)

                df_fem['ore_mp_fem']=np.where(df_fem['fem_mp']>=cutoff_q,'mineral','esteril')
                df_fem['ore_cp_fem']=np.where(df_fem['fem_cp']>=cutoff_q,'mineral','esteril')

                sc_col,_=st.columns([1,1])
                with sc_col:
                    fig,ax=make_fig((7,6))
                    ax.plot([0,70],[0,70],color='black',lw=1)
                    ax.axvline(cutoff_q,ls='--',color='black',lw=1)
                    ax.axhline(cutoff_q,ls='--',color='black',lw=1)
                    ax.scatter(df_fem.loc[ster,'fem_cp'],df_fem.loc[ster,'fem_mp'],s=18,color='#d9d9d9',label='Estéril',alpha=0.8)
                    ax.scatter(df_fem.loc[coin,'fem_cp'],df_fem.loc[coin,'fem_mp'],s=18,color='#2ca02c',label='Coincidencia',alpha=0.8)
                    ax.scatter(df_fem.loc[perd,'fem_cp'],df_fem.loc[perd,'fem_mp'],s=18,color='#ff7f0e',label='Pérdida',alpha=0.8)
                    ax.scatter(df_fem.loc[gain,'fem_cp'],df_fem.loc[gain,'fem_mp'],s=18,color='#1f77b4',label='Ganancia',alpha=0.8)
                    ax.set_xlim(0,70); ax.set_ylim(0,70)
                    ax.set_xlabel('FeM CP (%)',color='#444'); ax.set_ylabel('FeM MP (%)',color='#444')
                    style_ax(ax)
                    hi=(cutoff_q+70)/2; lo=cutoff_q/2
                    # Cell 41 posición etiquetas: fija en texto (52,30),(10,30),(10,10),(52,10)
                    for txt,xp,yp in [('I',52,30),('II',10,30),('III',10,10),('IV',52,10)]:
                        ax.text(xp,yp,txt,fontsize=12,weight='bold',color='#444')
                    ax.text(0.02,0.98,
                            f'Coef. Pearson: {pr_q:.2f}\n'
                            f'Coincidencia Mineral (I): {cnt["I"]}\n'
                            f'Pérdida Mineral (II): {cnt["II"]}\n'
                            f'Coincidencia Estéril (III): {cnt["III"]}\n'
                            f'Ganancia Mineral (IV): {cnt["IV"]}',
                            transform=ax.transAxes,fontsize=9,color='#333',va='top',
                            bbox=dict(facecolor='white',alpha=1,edgecolor='none',boxstyle='round,pad=0.3'))
                    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.08),ncol=4,frameon=False,fontsize=9,labelcolor='#333')
                    ax.set_title(f'Dispersión (MP vs CP) - MLC – {mes_abr}',color='#222',fontsize=11)
                    plt.tight_layout(); png_cuad=save_png(fig)
                    st.pyplot(fig); plt.close()
                    st.download_button("⬇️ PNG Cuadrantes",png_cuad,f"cuad_mlc_{mes_abr}.png","image/png",key="dlcuad")

                mx_col,_=st.columns([1,1])
                with mx_col:
                    ow=['mineral','esteril']
                    df_fem['ore_mp_fem']=pd.Categorical(df_fem['ore_mp_fem'],categories=ow,ordered=True)
                    df_fem['ore_cp_fem']=pd.Categorical(df_fem['ore_cp_fem'],categories=ow,ordered=True)
                    ct_ow=(pd.crosstab(df_fem['ore_mp_fem'],df_fem['ore_cp_fem']).reindex(index=ow[::-1],columns=ow).fillna(0))
                    ct_pct=ct_ow.div(ct_ow.sum(axis=1),axis=0).mul(100).round(1).fillna(0)
                    ann=(ct_ow.astype(int).apply(lambda c:c.map('{:,}'.format)).astype(str)+'\n('+ct_pct.astype(str)+'%)')
                    fig2,ax2=plt.subplots(figsize=(6,5),facecolor='white'); ax2.set_facecolor('white')
                    sns.heatmap(ct_pct,cmap=hq_cmap,annot=ann,fmt='',square=True,linewidths=0.5,cbar=True,annot_kws={'size':10},ax=ax2)
                    prd_lbl=f"Ene–{mes_abr}" if periodo_q=="Acumulado" else mes_nombre
                    ax2.set_title(f'Cumplimiento Ore/Waste (%) - MLC: {prd_lbl}',color='#222',pad=10)
                    ax2.tick_params(colors='#333'); ax2.set_xlabel('Ore CP',color='#444'); ax2.set_ylabel('Ore MP',color='#444')
                    plt.tight_layout(); png_ow=save_png(fig2)
                    st.pyplot(fig2); plt.close()
                    st.download_button("⬇️ PNG Ore/Waste",png_ow,f"ow_mlc_{mes_abr}.png","image/png",key="dlow")

                if 'mineral' in ct_ow.index and 'mineral' in ct_ow.columns:
                    TP=int(ct_ow.loc['mineral','mineral']); TN=int(ct_ow.loc['esteril','esteril']) if 'esteril' in ct_ow.index else 0
                    FP=int(ct_ow.loc['esteril','mineral']) if 'esteril' in ct_ow.index else 0
                    FN=int(ct_ow.loc['mineral','esteril']) if 'esteril' in ct_ow.columns else 0
                    pv=FN/(TP+FN)*100 if (TP+FN)>0 else 0
                    r2=st.columns(3)
                    with r2[0]: st.markdown(_metric_html('Mineral MP',f'{TP+FN:,}'),unsafe_allow_html=True)
                    with r2[1]: st.markdown(_metric_html('Mineral CP',f'{TP+FP:,}'),unsafe_allow_html=True)
                    with r2[2]: st.markdown(_metric_html('Pérdida Min→Est',f'{FN:,} ({pv:.1f}%)','red'),unsafe_allow_html=True)
    except Exception as e: st.error(f"Error cuadrantes: {e}")

# ── TAB 7: MATRICES (Cell 38) ──
with tab7:
    st.markdown("### Matriz de Ocurrencia — Modelo Geológico")
    st.markdown("*MLC: mac, bre, gyd, est — Cell 38 del notebook*")
    _mo1,_mo2=st.columns(2)
    with _mo1:
        HO={"Blues":"Blues","YlOrBr":"YlOrBr","Greens":"Greens","Purples":"Purples",
            "Oranges":"Oranges","RdYlGn":"RdYlGn","viridis":"viridis","coolwarm":"coolwarm"}
        ho=st.selectbox("🎨 Paleta",list(HO.keys()),index=0,key="ho")
    with _mo2:
        pocc=st.radio("Período:",["Mes seleccionado",f"Acumulado (Ene–{mes_abr})"],horizontal=True,key="pocc")
    ho_cmap=HO[ho]; es_acum=pocc.startswith("Acum")

    try:
        orden=['mac','bre','gyd','est']
        cat_t=CategoricalDtype(categories=orden,ordered=True)
        # Cell 38: df_mes_matriz = df[(periodo==mes)&(vol>=0.75*dims)]
        df_occ=df[df['periodo']<=mes].copy() if es_acum else df[df['periodo']==mes].copy()
        if 'proportional_volume' in df_occ.columns and 'dim_x' in df_occ.columns:
            df_occ=df_occ[df_occ['proportional_volume']>=0.75*df_occ['dim_x']*df_occ['dim_y']*df_occ['dim_z']].copy()

        occ_mp='ocurrencia_mp' if 'ocurrencia_mp' in df_occ.columns else 'ocurrencia'
        occ_cp='ocurrencia_cp'
        if occ_cp not in df_occ.columns:
            st.warning("ocurrencia_cp no disponible.")
        else:
            for c in [occ_mp,occ_cp]:
                if c in df_occ.columns:
                    df_occ[c]=df_occ[c].astype(str).str.strip().str.lower().astype(cat_t)

            ct=(pd.crosstab(df_occ[occ_mp],df_occ[occ_cp])
                  .reindex(index=orden[::-1],columns=orden).fillna(0))
            ct_pct=ct.div(ct.sum(axis=1),axis=0).mul(100).round(1).fillna(0)
            lbl_occ=(ct.astype(int).apply(lambda c:c.map('{:,}'.format)).astype(str)+'\n('+ct_pct.astype(str)+'%)')

            prd_lbl=f"Ene–{mes_abr}" if es_acum else mes_nombre
            fig,ax=plt.subplots(figsize=(6,5),facecolor='white'); ax.set_facecolor('white')
            sns.heatmap(ct_pct,cmap=ho_cmap,annot=lbl_occ,fmt='',square=True,linewidths=0.5,
                        cbar=True,annot_kws={'size':10},ax=ax,
                        xticklabels=orden,yticklabels=orden[::-1])
            ax.set_title(f'Cumplimiento (%): {prd_lbl} - Ocurrencias MLC',fontsize=12,color='#222')
            ax.set_xlabel('Ocurrencia CP',fontsize=10,color='#444')
            ax.set_ylabel('Ocurrencia MP',fontsize=10,color='#444')
            ax.tick_params(colors='#333'); plt.tight_layout()

            co_col,_=st.columns([1,1])
            with co_col:
                png_occ=save_png(fig); st.pyplot(fig); plt.close()
                st.download_button("⬇️ PNG Ocurrencia",png_occ,f"occ_mlc_{mes_abr}.png","image/png",key="dlocc")

            with st.expander("📋 Tabla de conteos"):
                ct_raw=ct.astype(int).copy(); ct_raw.index.name='MP \\ CP'; ct_raw.columns.name=None
                st.dataframe(ct_raw,use_container_width=True)
            d1,d2,_=st.columns([1,1,3])
            with d1: st.download_button("⬇️ CSV (%)",csv_bytes(ct_pct.reset_index()),f"occ_pct_{mes_abr}.csv","text/csv",key="dlocp")
            with d2: st.download_button("⬇️ CSV Conteos",csv_bytes(ct_raw.reset_index()),f"occ_raw_{mes_abr}.csv","text/csv",key="dlocr")
    except Exception as e: st.error(f"Error matriz ocurrencia: {e}")
