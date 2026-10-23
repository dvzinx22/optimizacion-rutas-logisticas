"""
MÓDULO DE INVESTIGACIÓN DE OPERACIONES: OPTIMIZACIÓN DE RUTAS LOGÍSTICAS
========================================================================
Formulaciones:
1. Modelo de Programación Lineal (LP) Clásico de Transporte (PuLP y Google OR-Tools).
2. Modelo Mixto Entero-Lineal (MILP) con variables binarias de activación de ruta (y_ij) y costos fijos.
3. Análisis de Sensibilidad: Precios Sombra (Duals), Costos Reducidos y Holguras.
4. Exportación a CSV, Excel y JSON.
"""

import pulp
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import json
import os

# -------------------------------------------------------------
# 1. PARÁMETROS DEL PROBLEMA
# -------------------------------------------------------------
PLANTS = ['P1', 'P2', 'P3']
CITIES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

SUPPLY = {
    'P1': 500,
    'P2': 700,
    'P3': 800
}

DEMAND = {
    'C1': 250,
    'C2': 350,
    'C3': 400,
    'C4': 300,
    'C5': 200,
    'C6': 300
}

UNIT_COSTS = {
    'P1': {'C1': 12, 'C2': 15, 'C3': 20, 'C4': 18, 'C5': 25, 'C6': 30},
    'P2': {'C1': 10, 'C2': 14, 'C3': 12, 'C4': 16, 'C5': 18, 'C6': 22},
    'P3': {'C1': 20, 'C2': 18, 'C3': 15, 'C4': 10, 'C5': 12, 'C6': 14}
}

BASELINE_COST = 28600.0  # USD reportado en escenario actual empírico

def solve_lp_pulp():
    """
    Resuelve el problema de transporte utilizando Programación Lineal en PuLP.
    Variables: x_ij >= 0 (cantidad transportada)
    """
    prob = pulp.LpProblem("Optimizacion_Transporte_PuLP", pulp.LpMinimize)
    
    # Variables de decisión continuas
    x = {}
    for i in PLANTS:
        for j in CITIES:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
            
    # Función Objetivo
    prob += pulp.lpSum(UNIT_COSTS[i][j] * x[i, j] for i in PLANTS for j in CITIES), "Minimizar_Costo_Total"
    
    # Restricciones de Capacidad (Oferta)
    for i in PLANTS:
        prob += pulp.lpSum(x[i, j] for j in CITIES) <= SUPPLY[i], f"Capacidad_{i}"
        
    # Restricciones de Demanda
    for j in CITIES:
        prob += pulp.lpSum(x[i, j] for i in PLANTS) == DEMAND[j], f"Demanda_{j}"
        
    # Resolver
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    total_cost = pulp.value(prob.objective)
    
    # Extraer asignación
    allocation = {i: {j: x[i, j].varValue for j in CITIES} for i in PLANTS}
    
    # Precios sombra y holguras
    duals_supply = {i: prob.constraints[f"Capacidad_{i}"].pi for i in PLANTS}
    slacks_supply = {i: prob.constraints[f"Capacidad_{i}"].slack for i in PLANTS}
    duals_demand = {j: prob.constraints[f"Demanda_{j}"].pi for j in CITIES}
    
    reduced_costs = {(i, j): x[i, j].dj for i in PLANTS for j in CITIES}
    
    return {
        "status": status,
        "total_cost": total_cost,
        "allocation": allocation,
        "duals_supply": duals_supply,
        "slacks_supply": slacks_supply,
        "duals_demand": duals_demand,
        "reduced_costs": reduced_costs,
        "raw_vars": x
    }

def solve_lp_ortools():
    """
    Resuelve el problema de transporte utilizando Google OR-Tools (Solver GLOP).
    """
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("No se pudo inicializar el solver GLOP de Google OR-Tools.")
        
    x = {}
    for i in PLANTS:
        for j in CITIES:
            x[i, j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")
            
    # Función Objetivo
    objective = solver.Objective()
    for i in PLANTS:
        for j in CITIES:
            objective.SetCoefficient(x[i, j], UNIT_COSTS[i][j])
    objective.SetMinimization()
    
    # Restricciones de Capacidad
    supply_constraints = {}
    for i in PLANTS:
        ct = solver.Constraint(0.0, float(SUPPLY[i]), f"Capacidad_{i}")
        for j in CITIES:
            ct.SetCoefficient(x[i, j], 1.0)
        supply_constraints[i] = ct
        
    # Restricciones de Demanda
    demand_constraints = {}
    for j in CITIES:
        ct = solver.Constraint(float(DEMAND[j]), float(DEMAND[j]), f"Demanda_{j}")
        for i in PLANTS:
            ct.SetCoefficient(x[i, j], 1.0)
        demand_constraints[j] = ct
        
    status = solver.Solve()
    status_str = "Optimal" if status == pywraplp.Solver.OPTIMAL else "Non-optimal"
    total_cost = solver.Objective().Value()
    
    allocation = {i: {j: x[i, j].solution_value() for j in CITIES} for i in PLANTS}
    duals_supply = {i: supply_constraints[i].dual_value() for i in PLANTS}
    duals_demand = {j: demand_constraints[j].dual_value() for j in CITIES}
    
    return {
        "status": status_str,
        "total_cost": total_cost,
        "allocation": allocation,
        "duals_supply": duals_supply,
        "duals_demand": duals_demand
    }

def solve_milp_with_fixed_routes(fixed_cost_per_route=150.0, max_routes=None):
    """
    Modelo Mixto Entero-Lineal (MILP) que incorpora:
    - Variables continuas x_ij (flujo)
    - Variables binarias y_ij (activación de ruta)
    - Restricción de acoplamiento: x_ij <= M_ij * y_ij, donde M_ij = min(Supply_i, Demand_j)
    - Costo fijo f_ij por ruta activa (ej. peajes, contratos con transportistas, overhead de despacho)
    - Límite opcional de rutas activas totales (para simplificar la logística)
    """
    prob = pulp.LpProblem("Transporte_MILP_Rutas", pulp.LpMinimize)
    
    x = {}
    y = {}
    for i in PLANTS:
        for j in CITIES:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
            y[i, j] = pulp.LpVariable(f"y_{i}_{j}", cat=pulp.LpBinary)
            
    # Función Objetivo: Costos variables + Costos fijos de activación de ruta
    prob += (
        pulp.lpSum(UNIT_COSTS[i][j] * x[i, j] for i in PLANTS for j in CITIES) +
        pulp.lpSum(fixed_cost_per_route * y[i, j] for i in PLANTS for j in CITIES)
    ), "Minimizar_Costo_Total_Con_Rutas"
    
    # Capacidad
    for i in PLANTS:
        prob += pulp.lpSum(x[i, j] for j in CITIES) <= SUPPLY[i], f"Capacidad_{i}"
        
    # Demanda
    for j in CITIES:
        prob += pulp.lpSum(x[i, j] for i in PLANTS) == DEMAND[j], f"Demanda_{j}"
        
    # Restricción Big-M ajustada de acoplamiento (Tight bound)
    for i in PLANTS:
        for j in CITIES:
            M_ij = min(SUPPLY[i], DEMAND[j])
            prob += x[i, j] <= M_ij * y[i, j], f"Acoplamiento_{i}_{j}"
            
    # Límite opcional de rutas
    if max_routes is not None:
        prob += pulp.lpSum(y[i, j] for i in PLANTS for j in CITIES) <= max_routes, "Max_Rutas_Permitidas"
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    total_cost = pulp.value(prob.objective)
    variable_cost = sum(UNIT_COSTS[i][j] * x[i, j].varValue for i in PLANTS for j in CITIES)
    active_routes_count = int(sum(y[i, j].varValue for i in PLANTS for j in CITIES))
    fixed_cost_total = active_routes_count * fixed_cost_per_route
    
    allocation = {i: {j: x[i, j].varValue for j in CITIES} for i in PLANTS}
    active_matrix = {i: {j: int(round(y[i, j].varValue)) for j in CITIES} for i in PLANTS}
    
    return {
        "status": pulp.LpStatus[prob.status],
        "total_cost": total_cost,
        "variable_cost": variable_cost,
        "fixed_cost_total": fixed_cost_total,
        "active_routes_count": active_routes_count,
        "allocation": allocation,
        "active_matrix": active_matrix
    }

def export_all_results(output_dir="."):
    """
    Ejecuta todos los modelos y exporta las matrices de asignación y comparativas a CSV, JSON y Excel.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Resolver modelo LP principal (PuLP)
    lp_res = solve_lp_pulp()
    ortools_res = solve_lp_ortools()
    milp_res = solve_milp_with_fixed_routes(fixed_cost_per_route=150.0)
    
    opt_cost = lp_res["total_cost"]
    savings_usd = BASELINE_COST - opt_cost
    savings_pct = (savings_usd / BASELINE_COST) * 100.0
    
    # 2. DataFrame de Matriz de Asignación Óptima
    df_alloc = pd.DataFrame(lp_res["allocation"]).T
    df_alloc.index.name = "Planta"
    df_alloc.to_csv(os.path.join(output_dir, "matriz_asignacion_optima.csv"))
    
    # 3. DataFrame de Matriz de Costos Unitarios
    df_costs = pd.DataFrame(UNIT_COSTS).T
    df_costs.index.name = "Planta"
    df_costs.to_csv(os.path.join(output_dir, "matriz_costos_unitarios.csv"))
    
    # 4. DataFrame Comparativo de Escenarios
    df_comp = pd.DataFrame([
        {
            "Escenario": "Escenario Actual Empírico (Sin Optimizar)",
            "Costo Total Mensual (USD)": BASELINE_COST,
            "Costo Anual Proyectado (USD)": BASELINE_COST * 12,
            "Ahorro Mensual vs Actual (USD)": 0.0,
            "Ahorro Porcentual (%)": 0.0,
            "Rutas Activas": "Asignación rígida no coordinada",
            "Estado": "Ineficiente"
        },
        {
            "Escenario": "Modelo Óptimo LP (PuLP / OR-Tools)",
            "Costo Total Mensual (USD)": opt_cost,
            "Costo Anual Proyectado (USD)": opt_cost * 12,
            "Ahorro Mensual vs Actual (USD)": savings_usd,
            "Ahorro Porcentual (%)": savings_pct,
            "Rutas Activas": "7 rutas activas coordinadas",
            "Estado": "Óptimo Global"
        },
        {
            "Escenario": "Modelo MILP con Costo Fijo por Ruta ($150/ruta)",
            "Costo Total Mensual (USD)": milp_res["total_cost"],
            "Costo Anual Proyectado (USD)": milp_res["total_cost"] * 12,
            "Ahorro Mensual vs Actual (USD)": BASELINE_COST - milp_res["total_cost"],
            "Ahorro Porcentual (%)": ((BASELINE_COST - milp_res["total_cost"]) / BASELINE_COST) * 100.0,
            "Rutas Activas": f"{milp_res['active_routes_count']} rutas activas",
            "Estado": "Óptimo con Fijos"
        }
    ])
    df_comp.to_csv(os.path.join(output_dir, "comparativa_escenarios.csv"), index=False)
    
    # 5. Detalle de Rutas Activas
    routes_detail = []
    for i in PLANTS:
        for j in CITIES:
            val = lp_res["allocation"][i][j]
            c = UNIT_COSTS[i][j]
            if val > 0:
                cost_tot = val * c
                routes_detail.append({
                    "Origen (Planta)": i,
                    "Destino (Ciudad)": j,
                    "Volumen (Unidades)": val,
                    "Costo Unitario (USD/u)": c,
                    "Costo Total Ruta (USD)": cost_tot,
                    "Participación en Demanda Ciudad (%)": (val / DEMAND[j]) * 100
                })
    df_routes = pd.DataFrame(routes_detail)
    df_routes.to_csv(os.path.join(output_dir, "detalle_rutas_optimas.csv"), index=False)
    
    # 6. Utilización de Capacidad por Planta
    capacity_detail = []
    for i in PLANTS:
        used = sum(lp_res["allocation"][i][j] for j in CITIES)
        cap = SUPPLY[i]
        slack = cap - used
        capacity_detail.append({
            "Planta": i,
            "Capacidad Máxima (u)": cap,
            "Producción Despachada (u)": used,
            "Capacidad Ociosa / Holgura (u)": slack,
            "Tasa de Utilización (%)": (used / cap) * 100.0,
            "Precio Sombra Dual (USD)": lp_res["duals_supply"][i]
        })
    df_cap = pd.DataFrame(capacity_detail)
    df_cap.to_csv(os.path.join(output_dir, "utilizacion_capacidad.csv"), index=False)
    
    # 7. Precios Sombra de Demanda (Costo Marginal de Abastecimiento)
    df_demand_duals = pd.DataFrame([
        {
            "Ciudad": j,
            "Demanda Requerida (u)": DEMAND[j],
            "Precio Sombra / Costo Marginal (USD/u)": lp_res["duals_demand"][j]
        }
        for j in CITIES
    ])
    df_demand_duals.to_csv(os.path.join(output_dir, "precios_sombra_demanda.csv"), index=False)
    
    # 8. Exportar reporte consolidado a JSON
    full_report = {
        "metadata": {
            "title": "Optimización Global de Rutas Logísticas de Distribución",
            "author": "Antigravity Operations Research & AI Suite",
            "version": "2.0-Enterprise"
        },
        "baseline": {
            "monthly_cost_usd": BASELINE_COST,
            "annual_cost_usd": BASELINE_COST * 12
        },
        "optimal_solution": {
            "solver_pulp": "CBC-2.9.0",
            "solver_ortools": "GLOP",
            "monthly_cost_usd": opt_cost,
            "annual_cost_usd": opt_cost * 12,
            "savings_monthly_usd": savings_usd,
            "savings_annual_usd": savings_usd * 12,
            "savings_percentage": round(savings_pct, 4),
            "allocation_matrix": lp_res["allocation"],
            "supply_usage": capacity_detail,
            "active_routes": routes_detail
        },
        "milp_fixed_routes_solution": {
            "fixed_cost_per_route": 150.0,
            "total_monthly_cost_usd": milp_res["total_cost"],
            "variable_cost_usd": milp_res["variable_cost"],
            "fixed_cost_usd": milp_res["fixed_cost_total"],
            "active_routes_count": milp_res["active_routes_count"],
            "active_routes_matrix": milp_res["active_matrix"]
        }
    }
    
    with open(os.path.join(output_dir, "reporte_optimizacion.json"), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)
        
    # 9. Exportar a Excel multi-hoja profesional con OpenPyXL
    excel_path = os.path.join(output_dir, "Reporte_Ejecutivo_Optimizacion_Transporte.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_comp.to_excel(writer, sheet_name="Comparativa_Escenarios", index=False)
        df_alloc.to_excel(writer, sheet_name="Matriz_Asignacion_Optima")
        df_costs.to_excel(writer, sheet_name="Matriz_Costos_Unitarios")
        df_routes.to_excel(writer, sheet_name="Detalle_Rutas_Activas", index=False)
        df_cap.to_excel(writer, sheet_name="Uso_Capacidad_Plantas", index=False)
        df_demand_duals.to_excel(writer, sheet_name="Precios_Sombra_Demanda", index=False)
        
    print(f"Exportación completa exitosa en directorio: {output_dir}")
    print(f"- Costo Óptimo: ${opt_cost:,.2f} USD")
    print(f"- Ahorro Mensual: ${savings_usd:,.2f} USD ({savings_pct:.2f}%)")
    print(f"- Archivo Excel generado: {excel_path}")

if __name__ == "__main__":
    export_all_results(".")
