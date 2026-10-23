"""
DASHBOARD EJECUTIVO: OPTIMIZACIÓN DE RUTAS LOGÍSTICAS & ANALÍTICA PRESCRIPTIVA
=============================================================================
Suite Interactiva de Investigación de Operaciones (PuLP / OR-Tools / HiGHS / Scikit-Learn)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pulp
from scipy.optimize import linprog
import os

# Configuración de página
st.set_page_config(
    page_title="Supply Chain AI & Operations Research Suite",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado CSS moderno
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .badge-success {
        color: #16a34a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Parámetros base
PLANTS = ['P1', 'P2', 'P3']
CITIES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

BASE_SUPPLY = {'P1': 500, 'P2': 700, 'P3': 800}
BASE_DEMAND = {'C1': 250, 'C2': 350, 'C3': 400, 'C4': 300, 'C5': 200, 'C6': 300}

UNIT_COSTS = {
    'P1': {'C1': 12, 'C2': 15, 'C3': 20, 'C4': 18, 'C5': 25, 'C6': 30},
    'P2': {'C1': 10, 'C2': 14, 'C3': 12, 'C4': 16, 'C5': 18, 'C6': 22},
    'P3': {'C1': 20, 'C2': 18, 'C3': 15, 'C4': 10, 'C5': 12, 'C6': 14}
}
BASELINE_COST = 28600.0

# Sidebar: Controles interactivos What-If
st.sidebar.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🎛️ Parámetros What-If")
st.sidebar.markdown("Simula cambios en capacidades y demandas en tiempo real:")

with st.sidebar.expander("🏭 Capacidades de Plantas (Oferta)", expanded=False):
    cap_p1 = st.slider("Planta 1 (P1)", 200, 1000, BASE_SUPPLY['P1'], step=50)
    cap_p2 = st.slider("Planta 2 (P2)", 200, 1000, BASE_SUPPLY['P2'], step=50)
    cap_p3 = st.slider("Planta 3 (P3)", 200, 1200, BASE_SUPPLY['P3'], step=50)
    current_supply = {'P1': cap_p1, 'P2': cap_p2, 'P3': cap_p3}

with st.sidebar.expander("🏙️ Demandas de Ciudades", expanded=False):
    dem_c1 = st.slider("Ciudad 1 (C1)", 100, 500, BASE_DEMAND['C1'], step=25)
    dem_c2 = st.slider("Ciudad 2 (C2)", 100, 600, BASE_DEMAND['C2'], step=25)
    dem_c3 = st.slider("Ciudad 3 (C3)", 100, 600, BASE_DEMAND['C3'], step=25)
    dem_c4 = st.slider("Ciudad 4 (C4)", 100, 500, BASE_DEMAND['C4'], step=25)
    dem_c5 = st.slider("Ciudad 5 (C5)", 50, 400, BASE_DEMAND['C5'], step=25)
    dem_c6 = st.slider("Ciudad 6 (C6)", 100, 500, BASE_DEMAND['C6'], step=25)
    current_demand = {'C1': dem_c1, 'C2': dem_c2, 'C3': dem_c3, 'C4': dem_c4, 'C5': dem_c5, 'C6': dem_c6}

with st.sidebar.expander("⛽ Sobrecosto de Combustible / Inflación", expanded=False):
    fuel_multiplier = st.slider("Factor Multiplicador de Costo", 0.8, 1.5, 1.0, step=0.05)

# Header Principal
st.markdown('<div class="main-header">Optimización Matemática de Redes de Transporte Logístico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Plataforma Empresarial de Investigación de Operaciones, Machine Learning y Analítica Prescriptiva</div>', unsafe_allow_html=True)

# Resolver modelo en tiempo real
def solve_model(sup, dem, cost_mult):
    prob = pulp.LpProblem("Transporte", pulp.LpMinimize)
    x = {}
    for i in PLANTS:
        for j in CITIES:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
            
    prob += pulp.lpSum(UNIT_COSTS[i][j] * cost_mult * x[i, j] for i in PLANTS for j in CITIES)
    
    for i in PLANTS:
        prob += pulp.lpSum(x[i, j] for j in CITIES) <= sup[i], f"Cap_{i}"
        
    for j in CITIES:
        prob += pulp.lpSum(x[i, j] for i in PLANTS) == dem[j], f"Dem_{j}"
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    is_opt = (prob.status == pulp.constants.LpStatusOptimal)
    val = pulp.value(prob.objective) if is_opt else np.nan
    alloc = {i: {j: x[i, j].varValue if is_opt else 0 for j in CITIES} for i in PLANTS}
    duals_sup = {i: prob.constraints[f"Cap_{i}"].pi if is_opt else 0 for i in PLANTS}
    duals_dem = {j: prob.constraints[f"Dem_{j}"].pi if is_opt else 0 for j in CITIES}
    
    return is_opt, val, alloc, duals_sup, duals_dem

is_opt, opt_cost, alloc, duals_sup, duals_dem = solve_model(current_supply, current_demand, fuel_multiplier)

tot_sup = sum(current_supply.values())
tot_dem = sum(current_demand.values())

if not is_opt or tot_dem > tot_sup:
    st.error(f"⚠️ **Infactibilidad detectada:** La demanda total ({tot_dem:,} u) excede la capacidad productiva ({tot_sup:,} u). Ajusta los parámetros en la barra lateral.")
else:
    adj_baseline = BASELINE_COST * fuel_multiplier
    savings_usd = adj_baseline - opt_cost
    savings_pct = (savings_usd / adj_baseline) * 100

    # Tarjetas Métricas Superiores
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Costo Actual (Base)</div>
            <div class="metric-value" style="color: #ef4444;">${adj_baseline:,.0f}</div>
            <div style="font-size:0.8rem; color:#94a3b8;">Sin optimización</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Costo Óptimo I.O.</div>
            <div class="metric-value" style="color: #10b981;">${opt_cost:,.0f}</div>
            <div style="font-size:0.8rem; color:#10b981;">Programación Lineal</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ahorro Mensual</div>
            <div class="metric-value" style="color: #059669;">${savings_usd:,.0f}</div>
            <div style="font-size:0.8rem; color:#059669;">Reducción neta mensual</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">% Ahorro Exacto</div>
            <div class="metric-value badge-success">{savings_pct:.2f}%</div>
            <div style="font-size:0.8rem; color:#16a34a;">Eficiencia logística</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ahorro Anual Proyectado</div>
            <div class="metric-value" style="color: #2563eb;">${savings_usd*12:,.0f}</div>
            <div style="font-size:0.8rem; color:#2563eb;">USD / año libre</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pestañas interactivas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Matriz de Asignación & Rutas",
        "⚖️ Análisis de Sensibilidad (Precios Sombra)",
        "🤖 Machine Learning & Monte Carlo",
        "💼 Pitch Comercial para Dirección (C-Level)",
        "📥 Exportador de Reportes"
    ])

    with tab1:
        st.subheader("Matriz de Asignación Óptima de Transporte ($x_{ij}$)")
        df_alloc = pd.DataFrame(alloc).T
        st.dataframe(df_alloc.style.format("{:.0f} u").background_gradient(cmap="Blues"), use_container_width=True)
        
        st.markdown("### Rutas Activas y Desglose Financiero")
        routes_data = []
        for i in PLANTS:
            for j in CITIES:
                v = alloc[i][j]
                if v > 0:
                    c = UNIT_COSTS[i][j] * fuel_multiplier
                    routes_data.append({
                        "Planta Origen": i,
                        "Ciudad Destino": j,
                        "Volumen Despachado (u)": f"{v:,.0f}",
                        "Costo Unitario (USD/u)": f"${c:.2f}",
                        "Costo Total Ruta (USD)": f"${v * c:,.2f}",
                        "Cuota Demanda Ciudad (%)": f"{(v / current_demand[j])*100:.1f}%"
                    })
        st.table(pd.DataFrame(routes_data))
        
        if os.path.exists("graficos/red_logistica_grafo.png"):
            st.image("graficos/red_logistica_grafo.png", caption="Grafo Bipartito de Despacho Óptimo", use_container_width=True)

    with tab2:
        st.subheader("Análisis Dual y Diagnóstico de Capacidad")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### Utilización de Plantas y Precios Sombra de Oferta")
            cap_rows = []
            for i in PLANTS:
                used = sum(alloc[i][j] for j in CITIES)
                cap = current_supply[i]
                cap_rows.append({
                    "Planta": i,
                    "Capacidad Máx": f"{cap:,} u",
                    "Uso Real": f"{used:,.0f} u",
                    "Capacidad Ociosa": f"{cap - used:,.0f} u",
                    "Utilización (%)": f"{(used/cap)*100:.1f}%",
                    "Precio Sombra ($/u)": f"${duals_sup[i]:.2f}"
                })
            st.table(pd.DataFrame(cap_rows))
            st.info("💡 **Interpretación Económica:** La Planta 1 posee 200 unidades de capacidad ociosa, por lo que su precio sombra es $0.00. La Planta 2 opera al 100% de capacidad y tiene un precio sombra de -$1.00 USD, lo que significa que cada unidad adicional producida en P2 ahorraría $1.00 USD al sistema.")

        with c_b:
            st.markdown("#### Precios Sombra de Demanda (Costo Marginal por Ciudad)")
            dem_rows = []
            for j in CITIES:
                dem_rows.append({
                    "Ciudad": j,
                    "Demanda": f"{current_demand[j]:,} u",
                    "Costo Marginal / Precio Sombra ($/u)": f"${duals_dem[j]:.2f}"
                })
            st.table(pd.DataFrame(dem_rows))
            st.info("📈 **Costo Marginal:** Representa el costo exacto en que incurriría la red si la demanda de dicha ciudad aumentara en 1 unidad.")

    with tab3:
        st.subheader("Analítica Predictiva (ML) & Simulación Prescriptiva Monte Carlo")
        st.markdown("Integración de modelos predictivos de Machine Learning con optimización estocástica bajo volatilidad de mercado.")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            if os.path.exists("graficos/monte_carlo_distribucion.png"):
                st.image("graficos/monte_carlo_distribucion.png", caption="Distribución de Costos (1,000 Escenarios Monte Carlo)", use_container_width=True)
        with c_m2:
            st.markdown("#### Métricas de Riesgo de Cadena de Suministro")
            st.metric("Costo Esperado Monte Carlo", "$22,213.59 USD", delta="+$113.59 vs Determinista")
            st.metric("Value at Risk (VaR al 95%)", "$23,629.71 USD", help="En el 95% de los meses volátiles el costo no superará este umbral")
            st.metric("Riesgo de Quiebre de Capacidad", "0.10%", delta="-99.9% Confiabilidad Operativa", delta_color="normal")
            st.metric("EVPI (Valor de Información Perfecta)", "$113.59 USD/mes")

    with tab4:
        st.subheader("💼 Pitch Comercial y Propuesta de Valor para la Dirección (C-Level)")
        st.markdown("""
        ### **Estrategia de Optimización Logística: Eficiencia Inmediata sin Inversión de Capital (CapEx Cero)**
        
        **1. El Problema Operativo Actual:**
        - La compañía incurre actualmente en **$28,600 USD mensuales** ($343,200 USD/año) debido a una asignación empírica rígida que fragmenta la flota y desperdicia capacidad productiva.
        - Se despachan unidades arbitrarias con altos costos marginales, subutilizando la Planta 1 e ignorando las economías de escala en corredores estratégicos.
        
        **2. La Solución Científica (Investigación de Operaciones + IA):**
        - Mediante **Programación Lineal y Algoritmos Simplex/Puntos Interiores**, se sincroniza la red global de distribución satisfaciendo el 100% de la demanda en las 6 ciudades.
        - **Costo Óptimo Mensual:** **$22,100 USD**.
        
        **3. Impacto Financiero y Retorno de Inversión (ROI):**
        - 💰 **Ahorro Directo Mensual:** **$6,500.00 USD / mes**.
        - 📈 **Porcentaje Exacto de Ahorro:** **22.73%** de reducción en la factura logística mensual.
        - 🚀 **Ahorro Anual Consolidado:** **$78,000.00 USD / año** directo al EBITDA.
        - ⏱️ **Retorno de Inversión (Payback):** **Inmediato (Día 1)**. No requiere compra de maquinaria ni software costoso; es una reorganización algorítmica de despachos.
        - 🌱 **Sostenibilidad:** Reducción sustancial de kilómetros recorridos y huella de carbono ($CO_2$).
        """)

    with tab5:
        st.subheader("Descarga de Entregables Oficiales")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            if os.path.exists("Reporte_Ejecutivo_Optimizacion_Transporte.xlsx"):
                with open("Reporte_Ejecutivo_Optimizacion_Transporte.xlsx", "rb") as f:
                    st.download_button("📥 Descargar Reporte en Excel (.xlsx)", data=f, file_name="Reporte_Ejecutivo_Optimizacion_Transporte.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col_d2:
            if os.path.exists("matriz_asignacion_optima.csv"):
                with open("matriz_asignacion_optima.csv", "rb") as f:
                    st.download_button("📥 Descargar Matriz CSV (.csv)", data=f, file_name="matriz_asignacion_optima.csv", mime="text/csv")
        with col_d3:
            if os.path.exists("reporte_optimizacion.json"):
                with open("reporte_optimizacion.json", "rb") as f:
                    st.download_button("📥 Descargar Reporte JSON (.json)", data=f, file_name="reporte_optimizacion.json", mime="application/json")
