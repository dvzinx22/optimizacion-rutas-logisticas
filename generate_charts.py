"""
GENERADOR DE GRÁFICOS EJECUTIVOS Y VISUALIZACIONES LOGÍSTICAS
=============================================================
Produce figuras de alta resolución (300 DPI) para el informe de Investigación de Operaciones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

# Configuración de estilo visual premium
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

output_dir = "."
charts_dir = os.path.join(output_dir, "graficos")
os.makedirs(charts_dir, exist_ok=True)

# 1. Gráfico Comparativo de Costos y Ahorro
def plot_cost_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    categories = ['Línea Base Actual\n(Sin Optimizar)', 'Modelo Óptimo LP\n(PuLP / OR-Tools)', 'Monte Carlo Esperado\n(Incertidumbre)', 'Peor Escenario\n(VaR 95%)']
    values = [28600.0, 22100.0, 22213.59, 23629.71]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22']
    
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor='#2c3e50', linewidth=1.2, zorder=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Etiquetas de valor
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f} USD",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
        
    # Anotación de Ahorro
    ax.annotate('AHORRO: -$6,500 / mes\n(-22.73% MENSUAL)',
                xy=(1, 22100), xytext=(0.5, 26000),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
                fontsize=11, fontweight='heavy', color='#27ae60',
                bbox=dict(boxstyle="round,pad=0.5", fc="#eafaf1", ec="#27ae60", lw=1.5),
                ha='center')
    
    ax.set_ylabel('Costo Total Mensual (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Impacto Financiero: Escenario Actual vs. Optimización Matemática (I.O.)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 33000)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "comparativa_costos.png"))
    plt.close(fig)

# 2. Matriz de Calor: Costos Unitarios vs Asignación Óptima
def plot_heatmaps():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    plants = ['P1', 'P2', 'P3']
    cities = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    
    costs_matrix = [
        [12, 15, 20, 18, 25, 30],
        [10, 14, 12, 16, 18, 22],
        [20, 18, 15, 10, 12, 14]
    ]
    
    alloc_matrix = [
        [0, 300, 0, 0, 0, 0],
        [250, 50, 400, 0, 0, 0],
        [0, 0, 0, 300, 200, 300]
    ]
    
    # Heatmap Costos
    sns.heatmap(costs_matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=cities, yticklabels=plants,
                cbar_kws={'label': 'USD / Unidad'}, ax=ax1, linewidths=1, linecolor='white')
    ax1.set_title('Matriz de Costos Unitarios de Transporte ($USD/u)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Planta Origen', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Ciudad Destino', fontsize=11, fontweight='bold')
    
    # Heatmap Asignación
    sns.heatmap(alloc_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=cities, yticklabels=plants,
                cbar_kws={'label': 'Unidades Despachadas'}, ax=ax2, linewidths=1, linecolor='white')
    ax2.set_title('Matriz de Asignación Óptima (x_ij)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Planta Origen', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Ciudad Destino', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "matrices_costo_y_asignacion.png"))
    plt.close(fig)

# 3. Diagrama de Red de Flujo Logístico (Bipartite Network Graph)
def plot_network_graph():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    
    plants = {'P1 (Cap: 500)': (0.1, 0.8), 'P2 (Cap: 700)': (0.1, 0.5), 'P3 (Cap: 800)': (0.1, 0.2)}
    cities = {
        'C1 (Dem: 250)': (0.9, 0.9),
        'C2 (Dem: 350)': (0.9, 0.74),
        'C3 (Dem: 400)': (0.9, 0.58),
        'C4 (Dem: 300)': (0.9, 0.42),
        'C5 (Dem: 200)': (0.9, 0.26),
        'C6 (Dem: 300)': (0.9, 0.10)
    }
    
    # Dibujar Nodos Plantas
    for name, (x, y) in plants.items():
        ax.scatter(x, y, s=1800, color='#2c3e50', edgecolors='#34495e', linewidth=2, zorder=4)
        ax.text(x - 0.02, y, name, fontsize=10, fontweight='bold', ha='right', va='center', color='#1a252f')
        
    # Dibujar Nodos Ciudades
    for name, (x, y) in cities.items():
        ax.scatter(x, y, s=1500, color='#2980b9', edgecolors='#1f618d', linewidth=2, zorder=4)
        ax.text(x + 0.02, y, name, fontsize=10, fontweight='bold', ha='left', va='center', color='#1a252f')
        
    # Flujos activos
    routes = [
        ('P1 (Cap: 500)', 'C2 (Dem: 350)', 300, 15, '#e67e22'),
        ('P2 (Cap: 700)', 'C1 (Dem: 250)', 250, 10, '#27ae60'),
        ('P2 (Cap: 700)', 'C2 (Dem: 350)', 50, 14, '#16a085'),
        ('P2 (Cap: 700)', 'C3 (Dem: 400)', 400, 12, '#2ecc71'),
        ('P3 (Cap: 800)', 'C4 (Dem: 300)', 300, 10, '#8e44ad'),
        ('P3 (Cap: 800)', 'C5 (Dem: 200)', 200, 12, '#9b59b6'),
        ('P3 (Cap: 800)', 'C6 (Dem: 300)', 300, 14, '#6c3483')
    ]
    
    for p_name, c_name, flow, cost, col in routes:
        x1, y1 = plants[p_name]
        x2, y2 = cities[c_name]
        lw = flow / 70.0
        ax.plot([x1, x2], [y1, y2], color=col, linewidth=lw, alpha=0.85, zorder=2)
        # Etiqueta en el centro del arco
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset_y = (y2 - y1) * 0.05
        ax.text(mx, my + offset_y, f"{flow}u @ ${cost}/u", fontsize=8.5, fontweight='bold',
                ha='center', va='center', color=col,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec=col, lw=1, alpha=0.9), zorder=5)
        
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')
    ax.set_title('Red Logística Óptima de Distribución: Flujos Activos (x_ij)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "red_logistica_grafo.png"))
    plt.close(fig)

# 4. Distribución Monte Carlo
def plot_monte_carlo():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    # Cargar datos de simulación
    with open("reporte_machine_learning.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    mc = data["monte_carlo_analytics"]
    
    # Generar distribución normal representativa de los resultados
    costs = np.random.normal(mc["expected_monthly_cost"], mc["standard_deviation"], 2000)
    
    sns.histplot(costs, kde=True, color='#3498db', bins=35, ax=ax, stat='density', alpha=0.6)
    
    ax.axvline(22100.0, color='#27ae60', linestyle='--', linewidth=2, label='Óptimo Determinista ($22,100)')
    ax.axvline(mc["expected_monthly_cost"], color='#2980b9', linestyle='-', linewidth=2.5, label=f'Media Monte Carlo (${mc["expected_monthly_cost"]:,.0f})')
    ax.axvline(mc["worst_case_var95"], color='#e74c3c', linestyle=':', linewidth=2, label=f'VaR 95% (${mc["worst_case_var95"]:,.0f})')
    
    ax.set_title('Simulación Prescriptiva Monte Carlo (1,000 Escenarios de Demanda)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Costo Total Mensual (USD)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Densidad de Probabilidad', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "monte_carlo_distribucion.png"))
    plt.close(fig)

if __name__ == "__main__":
    plot_cost_comparison()
    plot_heatmaps()
    plot_network_graph()
    plot_monte_carlo()
    print("Gráficos generados exitosamente en la carpeta:", charts_dir)
