# 🚚 Optimización de Rutas Logísticas de Distribución (I.O. & Machine Learning)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PuLP](https://img.shields.io/badge/Solver-PuLP%20(CBC)-green.svg)](https://coin-or.github.io/pulp/)
[![Google OR-Tools](https://img.shields.io/badge/Solver-Google%20OR--Tools-orange.svg)](https://developers.google.com/optimization)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Optimal%20Solution-brightgreen.svg)]()

> **Proyecto de Investigación de Operaciones, Programación Lineal Entera Mixta (MILP), Teoría Dual y Analítica Prescriptiva con Machine Learning y Simulación Monte Carlo.**

---

## 📌 1. Resumen Ejecutivo de Resultados

| Métrica Clave | Escenario Actual (Empírico) | Modelo Óptimo (I.O.) | Impacto / Beneficio |
| :--- | :---: | :---: | :---: |
| **Costo Logístico Mensual** | **$28,600.00 USD** | **$22,100.00 USD** | **-$6,500.00 USD / mes** |
| **Porcentaje de Ahorro** | *0.00% (Línea Base)* | **22.73% (Exacto: 22.7273%)** | **+22.73% Eficiencia** |
| **Ahorro Anual Proyectado** | $0.00 USD | **$78,000.00 USD / año** | Directo al EBITDA |
| **Demanda Abastecida** | 1,800 u (100%) | 1,800 u (100%) | SLA 100% garantizado |
| **Capacidad Ociosa** | Desperdicio disperso | 200 u en Planta 1 (P1) | Reserva estratégica |
| **Inversión Requerida (CapEx)**| - | **$0.00 USD (CapEx Cero)** | Retorno Inmediato (Día 1) |

---

## 🏭 2. Contexto Operativo & Balance de Red

La red logística consta de **3 plantas de manufactura** y **6 centros de consumo (ciudades)**:

- **Oferta de Plantas:**
  - Planta 1 ($P_1$): $500\text{ u/mes}$
  - Planta 2 ($P_2$): $700\text{ u/mes}$
  - Planta 3 ($P_3$): $800\text{ u/mes}$
  - **Capacidad Total:** **$2,000\text{ unidades/mes}$**
- **Demanda de Ciudades:**
  - $C_1 = 250, C_2 = 350, C_3 = 400, C_4 = 300, C_5 = 200, C_6 = 300\text{ u/mes}$.
  - **Demanda Total:** **$1,800\text{ unidades/mes}$**
- **Condición de Red:** Superávit estructural de oferta de **200 unidades**.

### Matriz de Costos de Transporte ($USD / Unidad)

| Planta \ Ciudad | C1 | C2 | C3 | C4 | C5 | C6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Planta 1 (P1)** | $12.00 | $15.00 | $20.00 | $18.00 | $25.00 | $30.00 |
| **Planta 2 (P2)** | $10.00 | $14.00 | $12.00 | $16.00 | $18.00 | $22.00 |
| **Planta 3 (P3)** | $20.00 | $18.00 | $15.00 | $10.00 | $12.00 | $14.00 |

---

## 📐 3. Formulación Matemática (Pregunta 1)

### Índices y Parámetros
- $i \in I = \{1, 2, 3\}$: Plantas de producción ($P_1, P_2, P_3$).
- $j \in J = \{1, \dots, 6\}$: Ciudades destino ($C_1, \dots, C_6$).
- $S_i$: Capacidad de oferta de la planta $i$.
- $D_j$: Demanda requerida por la ciudad $j$.
- $c_{ij}$: Costo unitario de flete desde $i$ hasta $j$.

### Variables de Decisión
- $x_{ij} \ge 0$: Unidades transportadas de planta $i$ a ciudad $j$.
- $y_{ij} \in \{0, 1\}$: Variable binaria de activación de ruta ($y_{ij} = 1$ si $x_{ij} > 0$; $0$ en caso contrario).

### Modelo Primal
$$\min Z = \sum_{i \in I} \sum_{j \in J} c_{ij} \, x_{ij}$$

Sujeto a:
1. **Capacidad de Plantas:**
   $$\sum_{j \in J} x_{ij} \le S_i \quad \forall i \in I$$
2. **Demanda de Ciudades:**
   $$\sum_{i \in I} x_{ij} = D_j \quad \forall j \in J$$
3. **Acoplamiento Lógico Big-M Ajustado con $y_{ij}$:**
   $$x_{ij} \le \min(S_i, D_j) \cdot y_{ij} \quad \forall i \in I, \forall j \in J$$
4. **No Negatividad e Integridad:**
   $$x_{ij} \ge 0, \quad y_{ij} \in \{0, 1\} \quad \forall i \in I, \forall j \in J$$

---

## 🎯 4. Matriz de Asignación Óptima ($x_{ij}$) (Pregunta 2)

| Planta \ Ciudad | C1 | C2 | C3 | C4 | C5 | C6 | Producción Real | Capacidad | Holgura (Ociosa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Planta 1 (P1)** | 0 | **300** | 0 | 0 | 0 | 0 | **300 u** | 500 u | **200 u (40% ociosa)** |
| **Planta 2 (P2)** | **250** | **50** | **400** | 0 | 0 | 0 | **700 u** | 700 u | **0 u (100% saturada)** |
| **Planta 3 (P3)** | 0 | 0 | 0 | **300** | **200** | **300** | **800 u** | 800 u | **0 u (100% saturada)** |
| **Demanda Total** | **250** | **350** | **400** | **300** | **200** | **300** | **1,800 u** | **2,000 u** | **200 u** |

### Cálculo Exacto del Porcentaje de Ahorro:
$$\text{Ahorro Mensual Absoluto} = \$28,600.00 - \$22,100.00 = \mathbf{\$6,500.00 \text{ USD/mes}}$$

$$\% \text{ Ahorro} = \frac{\$6,500.00}{\$28,600.00} \times 100\% = \frac{5}{22} \times 100\% = \mathbf{22.727272\dots\%} \approx \mathbf{22.73\%}$$

$$\text{Ahorro Anual Proyectado} = \$6,500.00 \times 12 = \mathbf{\$78,000.00 \text{ USD/año}}$$

---

## ⚖️ 5. Análisis de Sensibilidad y Precios Sombra (Teoría Dual)

- **Planta 1 ($\pi_1 = \$0.00$):** Al poseer 200 unidades de capacidad ociosa, un incremento de capacidad en P1 no genera ningún ahorro marginal.
- **Planta 2 ($\pi_2 = -\$1.00$):** Planta saturada al 100%. Por cada unidad adicional de capacidad que se habilite en P2, **el costo global del sistema disminuye en $1.00 USD** (desplaza despacho de P1 a C2 a $14 vs $15).
- **Planta 3 ($\pi_3 = \$0.00$):** Planta saturada dedicada exclusivamente a las ciudades C4, C5 y C6.
- **Costos Marginales por Ciudad ($\lambda_j$):** C1: $11 | C2: $15 | C3: $13 | C4: $10 | C5: $12 | C6: $14.

---

## 🤖 6. Machine Learning & Analítica Prescriptiva

1. **Modelos Predictivos (Random Forest Regressor):** Entrenados sobre series sintéticas macroeconómicas (estacionalidad mensual, precio del diésel y gasto publicitario), alcanzando un **$R^2 = 0.985$**.
2. **Simulación Prescriptiva Monte Carlo (1,000 iteraciones):**
   - **Costo Esperado Estocástico:** **$22,213.59 USD/mes** (+0.51% vs determinista).
   - **Value at Risk al 95% (VaR 95):** **$23,629.71 USD/mes** (en el 95% de los meses volátiles, el costo nunca superará este nivel).
   - **Confiabilidad del Sistema:** **99.90%** (riesgo de rotura de capacidad menor al 0.10%).
   - **EVPI (Expected Value of Perfect Information):** **$113.59 USD/mes**.

---

## 📁 7. Estructura de Entregables del Repositorio

```bash
├── Solucion_Optimizacion_Rutas_Logisticas.pdf  # Informe técnico oficial en PDF (10 páginas, sin autor)
├── Reporte_Ejecutivo_Optimizacion_Transporte.xlsx # Libro Excel con 6 hojas de análisis corporativo
├── dashboard.html                             # Dashboard interactivo autónomo en HTML5 / SVG
├── app.py                                     # Aplicación interactiva en Streamlit (What-If Simulator)
├── optimizer.py                               # Script principal de optimización (PuLP & Google OR-Tools)
├── ml_predictive_prescriptive.py              # Pipeline de Machine Learning y Monte Carlo
├── generate_charts.py                         # Generador de gráficos de alta resolución
├── matriz_asignacion_optima.csv               # Matriz de despacho x_ij para ERP (SAP/Oracle)
├── comparativa_escenarios.csv                 # Resumen comparativo en CSV
├── detalle_rutas_optimas.csv                  # Desglose de los 7 corredores activos
├── reporte_optimizacion.json                  # Payload JSON para integración en microservicios
└── graficos/                                  # Figuras en 300 DPI (Grafo de Red, Heatmaps, Monte Carlo)
```

---

## 🚀 8. Instrucciones de Ejecución

### 1. Clonar e Instalar Dependencias
```bash
git clone https://github.com/dvzinx22/optimizacion-rutas-logisticas.git
cd optimizacion-rutas-logisticas
pip install -r requirements.txt
```

### 2. Ejecutar la Optimización (PuLP & Google OR-Tools)
```bash
python optimizer.py
```

### 3. Ejecutar Pipeline de Machine Learning & Monte Carlo
```bash
python ml_predictive_prescriptive.py
```

### 4. Lanzar la Aplicación Interactiva (Streamlit)
```bash
streamlit run app.py
```

### 5. Abrir el Dashboard Autónomo
Doble clic en `dashboard.html` o abrirlo en cualquier navegador web.

---

## 💼 9. Pitch Ejecutivo C-Level

- **Problema:** Fuga recurrente de $6,500 USD mensuales ($78,000 USD/año) por asignación empírica rígida.
- **Solución:** Sincronización global mediante Programación Lineal y algoritmos de optimización.
- **Inversión:** **$0.00 USD (CapEx Cero)**.
- **Retorno:** **Payback Inmediato (Día 1)** y reducción de huella de carbono ($CO_2$).
