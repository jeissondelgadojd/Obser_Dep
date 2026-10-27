# ============================================================
# OBSERVATORIO SOCIOECONÓMICO DEPARTAMENTAL - COLOMBIA
# Versión Streamlit - 35 indicadores (2010-2023)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unicodedata
import warnings

from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    SPATIAL_AVAILABLE = True
except Exception:
    SPATIAL_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Observatorio Socioeconómico Departamental",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NORMALIZACIÓN DE DEPARTAMENTOS
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "caldas": "Caldas", "caqueta": "Caquetá", "casanare": "Casanare", "cauca": "Cauca",
    "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba", "cundinamarca": "Cundinamarca",
    "guainia": "Guainía", "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo", "quindio": "Quindío",
    "risaralda": "Risaralda", "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia", "santander": "Santander",
    "sucre": "Sucre", "tolima": "Tolima", "valle": "Valle del Cauca",
    "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada",
}

def normalizar_departamento(nombre):
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(clave, str(nombre).strip().title() if nombre else None)

LISTA_DEPTOS = sorted(list(set(DEPARTAMENTOS_CANONICOS.values())))

# ============================================================
# GENERACIÓN DEL PANEL (35 INDICADORES)
# ============================================================
@st.cache_data(show_spinner="Generando panel de indicadores...")
def generar_panel_completo(seed=42):
    np.random.seed(seed)
    anios = list(range(2010, 2024))
    n_d = len(LISTA_DEPTOS)

    # Pesos económicos reales aproximados (para que el ranking tenga sentido)
    pesos_economicos = {
        "Bogotá D.C.": 25.5, "Antioquia": 14.8, "Valle del Cauca": 9.7,
        "Santander": 6.5, "Cundinamarca": 6.1, "Atlántico": 4.5,
        "Bolívar": 3.6, "Meta": 2.9, "Boyacá": 2.6, "Tolima": 2.2,
        "Córdoba": 1.9, "Huila": 1.8, "Cesar": 1.7, "Norte de Santander": 1.7,
        "Risaralda": 1.6, "Caldas": 1.5, "Nariño": 1.5, "Magdalena": 1.4,
        "Cauca": 1.4, "Quindío": 0.9, "Casanare": 1.3, "La Guajira": 1.0,
        "Sucre": 0.8, "Caquetá": 0.5, "Chocó": 0.45, "Putumayo": 0.4,
        "Arauca": 0.55, "San Andrés y Providencia": 0.15, "Guaviare": 0.12,
        "Amazonas": 0.10, "Vichada": 0.09, "Guainía": 0.07, "Vaupés": 0.05,
    }
    for dep in LISTA_DEPTOS:
        if dep not in pesos_economicos:
            pesos_economicos[dep] = 0.3

    f_econ   = np.random.normal(0, 0.6, n_d)
    f_social = np.random.normal(0, 1.0, n_d)
    f_seg    = -0.5 * f_econ + np.random.normal(0, 0.7, n_d)

    filas = []
    for i, dep in enumerate(LISTA_DEPTOS):
        peso = pesos_economicos.get(dep, 0.3)
        base_pib = peso * 18000 * np.exp(0.15 * f_econ[i])

        base_desemp  = 9.5 + 2.8 * f_social[i] - 0.8 * f_econ[i]
        base_ocup    = 57 - 0.7 * base_desemp
        base_inf     = 48 + 7.5 * f_social[i] - 3 * f_econ[i]
        base_hom     = np.exp(3.2 + 0.9 * f_seg[i])
        base_ipm     = 26 + 11 * f_social[i] - 4 * f_econ[i]
        base_exp     = peso * 1200 * np.exp(0.3 * f_econ[i])
        base_comp    = 45 + 16 * f_econ[i] - 3 * f_social[i]
        base_viol    = 160 + 45 * f_seg[i]
        base_brecha  = 4.5 + 1.8 * f_social[i]
        base_serv    = 70 + 14 * f_econ[i] - 4 * f_social[i]
        base_recaudo = peso * 900 * np.exp(0.25 * f_econ[i])
        base_edu     = 67 + 12 * f_econ[i] - 3 * f_social[i]
        base_salud   = 73 + 11 * f_econ[i] - 2.5 * f_social[i]
        base_internet= 40 + 18 * f_econ[i]
        base_pobreza = 30 + 10 * f_social[i] - 3 * f_econ[i]
        base_gini    = 0.47 + 0.035 * f_social[i]
        base_inv     = peso * 700 * np.exp(0.3 * f_econ[i])
        base_turismo = peso * 180 * np.exp(0.4 * f_econ[i] - 0.3 * f_seg[i])
        base_innov   = 32 + 19 * f_econ[i]
        base_part    = 61 + 3.5 * f_econ[i]
        base_subemp  = 11 + 4 * f_social[i]
        base_analf   = 7.5 + 4.5 * f_social[i] - 2 * f_econ[i]
        base_escol   = 7.6 + 1.9 * f_econ[i]
        base_mort    = 17 + 5.5 * f_social[i] - 2.5 * f_econ[i]
        base_agua    = 76 + 13 * f_econ[i]
        base_energia = 84 + 11 * f_econ[i]
        base_vias    = 42 + 16 * f_econ[i]
        base_seg     = 52 - 11 * f_seg[i] + 6 * f_econ[i]
        base_prod    = 38 + 17 * f_econ[i]
        base_com     = peso * 850 * np.exp(0.25 * f_econ[i])

        for t, anio in enumerate(anios):
            tend = 0.022 * t
            ruido = np.random.normal(0, 0.05)

            filas.append({
                "departamento": dep,
                "anio": anio,
                "Tasa_Desempleo": np.clip(base_desemp - 0.16*t + np.random.normal(0,0.55), 3.2, 26),
                "Tasa_Ocupacion": np.clip(base_ocup + 0.12*t + np.random.normal(0,1.2), 41, 73),
                "Informalidad": np.clip(base_inf - 0.40*t + np.random.normal(0,1.5), 21, 75),
                "Brecha_Genero_Desempleo": np.clip(base_brecha - 0.06*t + np.random.normal(0,0.45), 0.3, 13),
                "Tasa_Participacion": np.clip(base_part + 0.14*t + np.random.normal(0,1.3), 51, 74),
                "Subempleo": np.clip(base_subemp - 0.18*t + np.random.normal(0,1.1), 5, 24),
                "Empleo_Formal": np.clip(100 - (base_inf - 0.40*t), 22, 79),
                "Desempleo_Juvenil": np.clip(base_desemp*1.65 - 0.18*t + np.random.normal(0,1.4), 8, 38),
                "PIB": base_pib * np.exp(tend + ruido),
                "PIB_per_capita": (base_pib * np.exp(tend + ruido)) / (peso * 180 + 50),
                "Indice_Competitividad": np.clip(base_comp + 0.65*t + np.random.normal(0,2.0), 20, 93),
                "Exportaciones": base_exp * np.exp(0.025*t + np.random.normal(0,0.09)),
                "Recaudo_Fiscal": base_recaudo * np.exp(0.027*t + np.random.normal(0,0.08)),
                "Inversion_Publica": base_inv * np.exp(0.024*t + np.random.normal(0,0.09)),
                "Indice_Innovacion": np.clip(base_innov + 0.85*t + np.random.normal(0,2.3), 10, 90),
                "Productividad_Laboral": np.clip(base_prod + 0.75*t + np.random.normal(0,2.3), 16, 88),
                "Comercio_Minorista": base_com * np.exp(0.021*t + np.random.normal(0,0.09)),
                "Homicidios": max(3.5, base_hom * np.exp(-0.03*t + np.random.normal(0,0.09))),
                "Violencia_Intrafamiliar": max(35, base_viol * np.exp(-0.017*t + np.random.normal(0,0.07))),
                "IPM": np.clip(base_ipm - 0.58*t + np.random.normal(0,1.5), 5.5, 55),
                "Pobreza_Monetaria": np.clip(base_pobreza - 0.52*t + np.random.normal(0,1.4), 8, 65),
                "Gini": np.clip(base_gini - 0.0018*t + np.random.normal(0,0.006), 0.38, 0.61),
                "Indice_Seguridad": np.clip(base_seg + 0.55*t + np.random.normal(0,2.8), 22, 93),
                "Desplazamiento": max(40, 700 + 280*f_seg[i] - 18*t + np.random.normal(0,70)),
                "Cobertura_Educacion": np.clip(base_edu + 0.52*t + np.random.normal(0,1.8), 40, 98),
                "Anos_Escolaridad": np.clip(base_escol + 0.07*t + np.random.normal(0,0.25), 5.2, 11.8),
                "Analfabetismo": np.clip(base_analf - 0.22*t + np.random.normal(0,0.7), 1.2, 18),
                "Cobertura_Salud": np.clip(base_salud + 0.45*t + np.random.normal(0,1.6), 45, 99),
                "Mortalidad_Infantil": np.clip(base_mort - 0.38*t + np.random.normal(0,1.3), 5.5, 32),
                "Cobertura_Servicios": np.clip(base_serv + 0.52*t + np.random.normal(0,1.7), 32, 98),
                "Penetracion_Internet": np.clip(base_internet + 1.7*t + np.random.normal(0,2.1), 14, 95),
                "Cobertura_Agua": np.clip(base_agua + 0.48*t + np.random.normal(0,1.8), 42, 99),
                "Cobertura_Energia": np.clip(base_energia + 0.38*t + np.random.normal(0,1.4), 55, 100),
                "Indice_Vial": np.clip(base_vias + 0.65*t + np.random.normal(0,2.8), 18, 92),
                "Turismo_Internacional": base_turismo * np.exp(0.028*t + np.random.normal(0,0.10)),
            })
    return pd.DataFrame(filas)

# Cargar panel
DF_PANEL = generar_panel_completo()
INDICADORES = [c for c in DF_PANEL.columns if c not in ["departamento", "anio"]]

# ============================================================
# FUNCIONES DE ANÁLISIS
# ============================================================
def preparar(df, deptos=None, anio_min=2010, anio_max=2023):
    d = df[(df["anio"] >= anio_min) & (df["anio"] <= anio_max)].copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    return d

def mapa_ranking(indicador, anio):
    d = DF_PANEL[DF_PANEL["anio"] == int(anio)].copy()
    if d.empty or indicador not in d.columns:
        return go.Figure()
    fig = px.bar(
        d.sort_values(indicador, ascending=True),
        x=indicador, y="departamento", orientation="h",
        color=indicador, color_continuous_scale="YlOrRd",
        title=f"{indicador} ({anio}) – Ranking departamental"
    )
    fig.update_layout(height=650, yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.title("🇨🇴 Observatorio Socioeconómico Departamental (Colombia)")
st.markdown(f"**{len(INDICADORES)} indicadores** | Años: **2010 – 2023** | Disponibles en todas las pestañas")

# Sidebar
st.sidebar.header("Filtros generales")
deptos_sel = st.sidebar.multiselect("Departamentos", ["Todos"] + LISTA_DEPTOS, default=["Todos"])
anio_min, anio_max = st.sidebar.slider("Rango de años", 2010, 2023, (2010, 2023))

# Pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Exploración", "2. EDA & PCA", "3. Econometría (VAR)",
    "4. Espacial", "5. Machine Learning", "6. Simulador & SHAP"
])

# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Exploración de Indicadores")
    col1, col2 = st.columns([2, 1])
    with col1:
        indicador = st.selectbox("Indicador", ["Todos"] + INDICADORES, key="ind_tab1")
    with col2:
        st.write("")
        btn1 = st.button("Actualizar", key="btn1")

    if btn1 or True:
        d = preparar(DF_PANEL, deptos_sel, anio_min, anio_max)
        if indicador != "Todos":
            d_plot = d[["departamento", "anio", indicador]].rename(columns={indicador: "valor"})
            fig_t = px.line(d_plot, x="anio", y="valor", color="departamento", markers=True,
                            title=f"Tendencia – {indicador}")
            st.plotly_chart(fig_t, use_container_width=True)

            db = d[d["anio"] == anio_max]
            fig_b = px.bar(db.nlargest(15, indicador), x=indicador, y="departamento", orientation="h",
                           color=indicador, color_continuous_scale="Blues",
                           title=f"Top 15 – {indicador} ({anio_max})")
            fig_b.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Seleccione un indicador específico para ver tendencias y rankings.")

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Análisis Exploratorio + PCA")
    n_comp = st.slider("Número de componentes PCA", 2, 10, 5)

    d = preparar(DF_PANEL, deptos_sel, anio_min, anio_max)
    if not d.empty:
        # Correlación
        corr = d[INDICADORES].corr().round(2)
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Matriz de correlaciones")
        st.plotly_chart(fig_corr, use_container_width=True)

        # PCA
        X = d[INDICADORES].dropna()
        if len(X) >= 25:
            Xs = StandardScaler().fit_transform(X)
            n_comp = min(n_comp, Xs.shape[1], Xs.shape[0]-1)
            pca = PCA(n_components=n_comp).fit(Xs)
            var = pca.explained_variance_ratio_

            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(n_comp)], y=var, name="Varianza"))
            fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_comp)],
                                           y=np.cumsum(var), mode="lines+markers", name="Acumulada"))
            fig_scree.update_layout(title="Scree Plot")
            st.plotly_chart(fig_scree, use_container_width=True)

            st.markdown(f"**Varianza explicada:** {np.round(var, 3).tolist()}")
            st.markdown(f"**Acumulada:** {np.round(np.cumsum(var), 3).tolist()}")

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("VAR / Granger / IRF")
    vars_sel = st.multiselect("Variables", INDICADORES,
                              default=["PIB", "Tasa_Desempleo", "Homicidios", "IPM"])
    maxlags = st.slider("Máximo de lags", 1, 4, 2)

    if st.button("Estimar VAR") and len(vars_sel) >= 2:
        d = preparar(DF_PANEL, deptos_sel, anio_min, anio_max)
        ts = d.groupby("anio")[vars_sel].mean().sort_index().dropna()
        if len(ts) >= 12:
            model = VAR(ts)
            order = model.select_order(maxlags=min(maxlags, len(ts)//3))
            res = model.fit(order.aic)
            st.text(f"Lag seleccionado (AIC): {order.aic}")
            st.text(str(res.summary())[:2500])

            irf = res.irf(8)
            fig = go.Figure()
            if ts.shape[1] >= 2:
                fig.add_trace(go.Scatter(y=irf.irfs[:, 0, 1], mode="lines+markers",
                                         name=f"{ts.columns[0]} ← {ts.columns[1]}"))
            fig.update_layout(title="Función de Impulso-Respuesta (IRF)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Series demasiado cortas.")

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Análisis Espacial – Ranking Departamental")
    ind_esp = st.selectbox("Indicador", INDICADORES, index=INDICADORES.index("PIB") if "PIB" in INDICADORES else 0)
    anio_esp = st.slider("Año", 2010, 2023, 2022)
    k_nn = st.slider("K vecinos (Moran)", 3, 8, 5)

    fig_mapa = mapa_ranking(ind_esp, anio_esp)
    st.plotly_chart(fig_mapa, use_container_width=True)

    if SPATIAL_AVAILABLE and st.button("Calcular I de Moran"):
        d = DF_PANEL[DF_PANEL["anio"] == anio_esp].dropna(subset=[ind_esp]).sort_values("departamento").reset_index(drop=True)
        if len(d) >= 8:
            w = KNN.from_dataframe(d.assign(x=range(len(d)), y=0), k=min(k_nn, len(d)-1))
            w.transform = "r"
            mi = Moran(d[ind_esp].values, w)
            interp = "Autocorrelación positiva significativa" if (mi.p_sim < 0.05 and mi.I > 0) else "No significativa"
            st.markdown(f"**I de Moran = {mi.I:.4f}** | p-sim = {mi.p_sim:.4f} → {interp}")
        else:
            st.warning("Pocas observaciones.")

# -------------------- TAB 5 --------------------
with tab5:
    st.subheader("Machine Learning")
    target = st.selectbox("Variable objetivo (Target)", INDICADORES, index=INDICADORES.index("Tasa_Desempleo"))
    features = st.multiselect("Features", INDICADORES,
                              default=["PIB", "Informalidad", "IPM", "Indice_Competitividad"])
    modelo = st.radio("Modelo", ["Ridge", "LASSO", "SVR", "MLP"], horizontal=True)

    if st.button("Entrenar modelo") and features:
        d = preparar(DF_PANEL, anio_min=anio_min, anio_max=anio_max).dropna()
        feats = [f for f in features if f != target]
        X, y = d[feats], d[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        if modelo == "Ridge":
            pipe = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3, 3, 40)))])
        elif modelo == "LASSO":
            pipe = Pipeline([("sc", StandardScaler()), ("m", LassoCV(cv=5))])
        elif modelo == "SVR":
            pipe = Pipeline([("sc", StandardScaler()), ("m", SVR(C=10))])
        else:
            pipe = Pipeline([("sc", StandardScaler()),
                             ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42))])

        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)

        st.success(f"**{modelo}** | RMSE = {rmse:.3f} | R² = {r2:.3f}")

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=yte, y=pred, mode="markers", opacity=0.6, name="Predicciones"))
        lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
        fig_pred.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash"), name="Ideal"))
        fig_pred.update_layout(title=f"{modelo}: Predicho vs Real", xaxis_title="Real", yaxis_title="Predicho")
        st.plotly_chart(fig_pred, use_container_width=True)

# -------------------- TAB 6 --------------------
with tab6:
    st.subheader("Simulador de Política + SHAP")
    target6 = st.selectbox("Target", INDICADORES, index=INDICADORES.index("Tasa_Desempleo"), key="t6")
    features6 = st.multiselect("Variables de política", INDICADORES,
                               default=["PIB", "Informalidad", "IPM", "Indice_Competitividad"], key="f6")

    st.markdown("**Valores del escenario** (dejar vacío = media histórica)")
    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(features6[:4]):
        with cols[i]:
            vals[f] = st.number_input(f, value=None, key=f"val_{f}")

    if st.button("Simular escenario"):
        d = preparar(DF_PANEL).dropna()
        feats = [f for f in features6 if f in d.columns and f != target6]
        X, y = d[feats], d[target6]
        pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
        pipe.fit(X, y)
        base = X.mean().values.reshape(1, -1)
        pred_base = pipe.predict(base)[0]
        esc = base.copy()
        for i, f in enumerate(feats):
            if f in vals and vals[f] is not None:
                esc[0, i] = float(vals[f])
        pred_cf = pipe.predict(esc)[0]
        st.metric(label=target6, value=f"{pred_cf:.2f}", delta=f"{pred_cf - pred_base:+.2f}")

    if SHAP_AVAILABLE and st.button("Calcular SHAP"):
        d = preparar(DF_PANEL).dropna()
        feats = [f for f in features6 if f != target6][:8]
        X = StandardScaler().fit_transform(d[feats])
        model = Ridge(alpha=1.0).fit(X, d[target6])
        explainer = shap.LinearExplainer(model, X)
        sv = explainer.shap_values(X)
        mean_abs = np.abs(sv).mean(0)
        imp = pd.DataFrame({"feature": feats, "mean_|SHAP|": mean_abs}).sort_values("mean_|SHAP|")
        fig = px.bar(imp, x="mean_|SHAP|", y="feature", orientation="h",
                     title=f"Importancia SHAP – {target6}", color="mean_|SHAP|",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Observatorio Socioeconómico Departamental | {len(INDICADORES)} indicadores | 2010-2023 | Panel sintético calibrado")
