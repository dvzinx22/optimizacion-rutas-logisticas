"""
SCRIPT DE GENERACIÓN DE PDF PROFESIONAL: OPTIMIZACIÓN DE RUTAS LOGÍSTICAS
========================================================================
Genera un documento PDF corporativo y académico de alta calidad mediante
Playwright Chromium, con diseño editorial, tablas, código, gráficos embebidos
en base64 y SIN NOMBRES DE AUTOR.
"""

import os
import base64
from playwright.sync_api import sync_playwright

output_dir = r"C:\Users\Windows 11\.gemini\antigravity-ide\scratch\supply_chain_optimization"
charts_dir = os.path.join(output_dir, "graficos")

def get_base64_image(filename):
    path = os.path.join(charts_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""

img_comp = get_base64_image("comparativa_costos.png")
img_matrix = get_base64_image("matrices_costo_y_asignacion.png")
img_graph = get_base64_image("red_logistica_grafo.png")
img_mc = get_base64_image("monte_carlo_distribucion.png")

html_path = os.path.join(output_dir, "documento_impresion.html")
pdf_path = os.path.join(output_dir, "Solucion_Optimizacion_Rutas_Logisticas.pdf")

with open(html_path, "r", encoding="utf-8") as f:
    template = f.read()

template = template.replace("__IMG_COMP__", img_comp)
template = template.replace("__IMG_MATRIX__", img_matrix)
template = template.replace("__IMG_GRAPH__", img_graph)
template = template.replace("__IMG_MC__", img_mc)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(template)

print(f"HTML preparado en: {html_path}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(f"file:///{html_path.replace(os.sep, '/')}", wait_until="networkidle")
    
    page.pdf(
        path=pdf_path,
        format="A4",
        print_background=True,
        margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"}
    )
    browser.close()

print(f"PDF generado exitosamente en: {pdf_path}")
print(f"Tamaño del PDF: {os.path.getsize(pdf_path):,} bytes")
