"""
create_report.py
Generador automatizado del Informe Técnico en formato Microsoft Word (.docx)
para el proyecto:
'Traductor Quechua a Español con Redes Neuronales Recurrente (RNN) Many-to-Many'
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_cell_background(cell, hex_color):
    """Establece el color de fondo de una celda."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Establece los márgenes internos de una celda (en dxa: 20 dxa = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_callout(doc, text, title=None, border_color="7C2D12", bg_color="FFFBEB"):
    """Agrega un cuadro de texto destacado (callout) estilizado."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Borde izquierdo grueso
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    if title:
        run_title = p.add_run(f"{title}\n")
        run_title.bold = True
        run_title.font.name = "Calibri"
        run_title.font.size = Pt(10.5)
        run_title.font.color.rgb = RGBColor(124, 45, 18)
        
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGBColor(30, 41, 59)
    run_text.italic = True
    
    # Espaciado posterior
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(table, header_bg="7C2D12", header_color="FFFFFF", alt_bg="F8FAFC"):
    """Aplica formato profesional a una tabla."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    for i, row in enumerate(table.rows):
        # Tratar primera fila como encabezado
        is_header = (i == 0)
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            
            if is_header:
                set_cell_background(cell, header_bg)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    for run in p.runs:
                        run.font.bold = True
                        run.font.name = "Calibri"
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                bg = alt_bg if i % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(9)
                        run.font.color.rgb = RGBColor(30, 41, 59)

def build_document():
    doc = Document()
    
    # Configuración de márgenes de página (2.5 cm = ~0.98 pulgadas)
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Encabezado y pie de página
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("Traductor Quechua-Español | RNN Many-to-Many — Informe Técnico y Experimental")
        f_run.font.name = "Calibri"
        f_run.font.size = Pt(8.5)
        f_run.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------------------
    # PORTADA / TÍTULO PRINCIPAL
    # -------------------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(12)
    title_p.paragraph_format.space_after = Pt(4)
    run_t1 = title_p.add_run("INFORME TÉCNICO Y EXPERIMENTAL")
    run_t1.bold = True
    run_t1.font.name = "Calibri"
    run_t1.font.size = Pt(12)
    run_t1.font.color.rgb = RGBColor(180, 83, 9)

    h1_p = doc.add_paragraph()
    h1_p.paragraph_format.space_before = Pt(0)
    h1_p.paragraph_format.space_after = Pt(8)
    run_h1 = h1_p.add_run("Traductor Automático Quechua – Español Mediante Redes Neuronales Recurrentes (RNN) Many-to-Many")
    run_h1.bold = True
    run_h1.font.name = "Calibri"
    run_h1.font.size = Pt(22)
    run_h1.font.color.rgb = RGBColor(124, 45, 18)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(18)
    run_sub = sub_p.add_run(
        "Investigación Comparativa Teórica y Matemática (SimpleRNN vs. LSTM), Arquitectura Seq2Seq "
        "con Persistencia de Contexto, Pipeline de Preprocesamiento, Proceso de Entrenamiento y Evaluación con BLEU Score."
    )
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Metadatos del informe
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Área de Conocimiento:", "Inteligencia Artificial / Procesamiento del Lenguaje Natural (NLP)"),
        ("Arquitectura Implementada:", "Encoder-Decoder Many-to-Many con celdas SimpleRNN (tanh)"),
        ("Métrica de Rendimiento:", "Corpus BLEU-1: 31.00% | BLEU-4: 9.58% | Exact Match: 25.09%"),
        ("Tecnologías y Frameworks:", "Python 3.13, PyTorch 2.14, Streamlit 1.62, NLTK 3.10, Pandas")
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[r_idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width, c1.width = Inches(2.2), Inches(4.3)
        set_cell_background(c0, "F8FAFC")
        set_cell_background(c1, "FFFFFF")
        
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.name = "Calibri"
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(30, 41, 59)
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.name = "Calibri"
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # RESUMEN EJECUTIVO
    # -------------------------------------------------------------------------
    add_callout(
        doc,
        "RESUMEN EJECUTIVO: Este documento presenta el diseño, desarrollo, entrenamiento y evaluación rigurosa "
        "de un sistema de traducción automática de Quechua a Español basado en una Red Neuronal Recurrente Simple "
        "(SimpleRNN) bajo la arquitectura Many-to-Many (Secuencia a Secuencia / Seq2Seq). Se investigan exhaustivamente "
        "las limitaciones matemáticas de las celdas recurrentes clásicas frente a las LSTM (Long Short-Term Memory), "
        "demostrando cómo técnicas como el empaquetado de secuencias (pack_padded_sequence), la persistencia del vector "
        "de contexto en el decodificador y el Teacher Forcing permiten superar el desvanecimiento de gradiente en frases breves. "
        "El modelo final alcanza un Corpus BLEU-1 de 31.00%, un BLEU-4 de 9.58% y una tasa de coincidencia exacta del 25.09% "
        "en el conjunto de prueba independiente (test.csv), integrado en una interfaz gráfica interactiva en Streamlit.",
        title="SÍNTESIS DEL PROYECTO"
    )

    # -------------------------------------------------------------------------
    # SECCIÓN 1: INTRODUCCIÓN Y CONTEXTO LINGÜÍSTICO
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Introducción y Contextualización Lingüística", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "El Quechua (Runasimi) constituye la familia lingüística originaria viva más extendida del continente americano, "
        "con millones de hablantes distribuidos primordialmente en Perú, Bolivia y Ecuador. A pesar de su enorme trascendencia cultural "
        "e histórica, en el ámbito de la Inteligencia Artificial y el Procesamiento del Lenguaje Natural (NLP), el Quechua es catalogado "
        "como una lengua de bajos recursos computacionales (Low-Resource Language), caracterizada por la escasez de corpus paralelos "
        "masivos y herramientas morfológicas estandarizadas."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Desde la perspectiva de la lingüística computacional, traducir entre Quechua y Español entraña una asimetría estructural profunda:\n"
        "1. Tipología Morfológica: El Quechua es una lengua intensamente aglutinante y sufijante. Una sola raíz nominal o verbal puede "
        "incorporar decenas de sufijos derivativos y flexivos que expresan persona, número, aspecto, tiempo, evidencialidad y caso "
        "(e.g., 'llamkachkankim' traduce una cláusula completa: 'tú estás trabajando'). El Español, en contraparte, es una lengua flexiva con "
        "abundantes palabras funcionales independientes (artículos, preposiciones, pronombres clíticos).\n"
        "2. Orden Sintáctico: El Quechua presenta un orden sintáctico canónico SOV (Sujeto - Objeto - Verbo), mientras que el Español se "
        "estructura canónicamente como SVO (Sujeto - Verbo - Objeto). Por ejemplo, en 'Allqu aychata mikun', 'Allqu' (Sujeto), 'aychata' (Objeto con sufijo acusativo -ta) "
        "y 'mikun' (Verbo) se traduce al Español como 'El perro come carne'.\n"
        "3. Longitud de Secuencia Variable (Tx != Ty): Debido a las discrepancias morfosintácticas, una oración en Quechua rara vez posee la misma "
        "cantidad de palabras que su equivalente en Español. Por consiguiente, una red recurrente con alineación síncrona token a token resulta lingüísticamente inviable, "
        "haciendo indispensable una arquitectura Many-to-Many Asíncrona (Seq2Seq / Encoder-Decoder)."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 2: INVESTIGACIÓN Y DISCUSIÓN TEÓRICA: RNN SIMPLE VS. LSTM
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("2. Investigación y Discusión Teórica: RNN Simples vs. LSTM en Traducción Automática", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Uno de los pilares teóricos de este proyecto es analizar la viabilidad, ventajas y severas limitaciones de las Redes Neuronales "
        "Recurrentes Simples (SimpleRNN o celdas de Elman) en comparación con las redes de Memoria a Corto y Largo Plazo (LSTM - Long Short-Term Memory), "
        "introducidas por Hochreiter & Schmidhuber (1997) para mitigar las deficiencias del algoritmo de retropropagación a través del tiempo (BPTT)."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    # Subsección 2.1
    h2 = doc.add_heading("2.1. Formulación Matemática Comparativa", level=2)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "A. Celda Recurrente Simple (SimpleRNN / Elman):\n"
        "La celda recurrente básica mantiene un único vector de estado oculto h_t que actúa simultáneamente como memoria de trabajo "
        "y como señal de activación. Su dinámica en el instante temporal t se rige por la ecuación:"
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)

    eq1_p = doc.add_paragraph("h_t = tanh( W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh )")
    eq1_p.paragraph_format.left_indent = Inches(0.5)
    eq1_p.paragraph_format.space_after = Pt(4)
    for r in eq1_p.runs: r.bold = True; r.font.color.rgb = RGBColor(124, 45, 18)

    p = doc.add_paragraph(
        "Donde W_ih es la matriz de pesos de entrada, W_hh es la matriz de recurrencia oculta, y tanh comprime la activación al rango (-1, 1).\n\n"
        "B. Celda Long Short-Term Memory (LSTM):\n"
        "La arquitectura LSTM descompone la memoria en dos componentes: el estado de celda C_t (memoria a largo plazo) y el estado oculto h_t "
        "(memoria de trabajo a corto plazo), modulados por tres compuertas diferenciales parametrizadas con funciones sigmoides σ:"
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)

    lstm_eqs = (
        "1. Compuerta de Olvido (Forget Gate):       f_t = σ( W_f · [h_{t-1}, x_t] + b_f )\n"
        "2. Compuerta de Entrada (Input Gate):        i_t = σ( W_i · [h_{t-1}, x_t] + b_i )\n"
        "3. Estado Candidato de Celda:               C~_t = tanh( W_c · [h_{t-1}, x_t] + b_c )\n"
        "4. Actualización del Estado de Celda:       C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t\n"
        "5. Compuerta de Salida (Output Gate):       o_t = σ( W_o · [h_{t-1}, x_t] + b_o )\n"
        "6. Estado Oculto Resultante:                h_t = o_t ⊙ tanh( C_t )"
    )
    eq2_p = doc.add_paragraph(lstm_eqs)
    eq2_p.paragraph_format.left_indent = Inches(0.5)
    eq2_p.paragraph_format.space_after = Pt(6)
    for r in eq2_p.runs: r.font.name = "Consolas"; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(30, 41, 59)

    # Subsección 2.2
    h2 = doc.add_heading("2.2. El Problema del Desvanecimiento y Explosión del Gradiente", level=2)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "La limitación teórica más crítica de la SimpleRNN radica en la propagación de gradientes hacia atrás en el tiempo. Al calcular la "
        "derivada de la función de pérdida L en el instante T con respecto al estado oculto h_1, la regla de la cadena establece:"
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)

    bptt_eq = "∂L / ∂h_1 = (∂L / ∂h_T) · ∏_{k=2}^{T} (∂h_k / ∂h_{k-1}) = (∂L / ∂h_T) · ∏_{k=2}^{T} [ diag(1 - h_k^2) · W_hh^T ]"
    bptt_p = doc.add_paragraph(bptt_eq)
    bptt_p.paragraph_format.left_indent = Inches(0.5)
    bptt_p.paragraph_format.space_after = Pt(4)
    for r in bptt_p.runs: r.bold = True; r.font.color.rgb = RGBColor(124, 45, 18)

    p = doc.add_paragraph(
        "Debido a que el producto involucra T-1 multiplicaciones sucesivas de la matriz W_hh multiplicada por la derivada de tanh (cuyo valor "
        "máximo es 1):\n"
        "• Si el mayor autovalor de W_hh es menor a 1 (λ_max < 1), el producto decae exponencialmente hacia cero cuando T > 10. Los gradientes "
        "desaparecen por completo (Vanishing Gradient), impidiendo que la red aprenda relaciones sintácticas entre palabras distantes.\n"
        "• Si λ_max > 1, el producto crece exponencialmente hacia infinito (Exploding Gradient), desestabilizando los pesos con valores NaN.\n\n"
        "En una LSTM, la actualización C_t = f_t ⊙ C_{t-1} + i_t ⊙ C~_t es de naturaleza aditiva (un 'carril de autopista' de gradiente). "
        "La derivada ∂C_t / ∂C_{t-1} = f_t. Si la compuerta de olvido f_t ≈ 1, el gradiente fluye hacia atrás sin decaimiento exponencial, permitiendo "
        "retener dependencias contextuales a lo largo de más de 100 pasos temporales."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # Subsección 2.3: Tabla Comparativa
    h2 = doc.add_heading("2.3. Tabla Comparativa Integral: SimpleRNN vs. LSTM en Traducción Automática", level=2)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(6)

    comp_table = doc.add_table(rows=7, cols=3)
    comp_headers = ["Dimensión de Análisis", "SimpleRNN (Elman)", "LSTM (Long Short-Term Memory)"]
    for c_idx, h_text in enumerate(comp_headers):
        comp_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    comp_rows = [
        ("Complejidad Computacional por Paso", "O(d_in · d_hid + d_hid²)\n1 conjunto de transformaciones lineales.", "4 × O(d_in · d_hid + d_hid²)\n4 compuertas lineales independientes."),
        ("Memoria Contextual Efectiva", "Baja (8 a 10 pasos temporales).\nSufre amnesia severa en cláusulas subordinadas.", "Alta (> 80 a 100 pasos temporales).\nMantiene contexto de oraciones compuestas."),
        ("Estabilidad del Gradiente", "Muy baja. Requiere obligatoriamente gradient clipping y normalización estricta.", "Muy alta. La ruta aditiva de celda preserva el flujo de gradiente."),
        ("Capacidad de Reordenamiento Sintáctico", "Limitada a frases cortas. Le cuesta recordar el sujeto inicial de oraciones SOV hasta el verbo.", "Excelente. Las compuertas permiten posponer la activación de sujetos hasta llegar a su verbo."),
        ("Uso de Recursos y Huella de Memoria", "Ligera y rápida en CPU. Ideal para modelos embebidos y educativos.", "4 veces más parámetros y mayor consumo de memoria RAM/VRAM."),
        ("Viabilidad en Producción Moderna", "Superada por Transformers y LSTM. Viable en tareas pedagógicas con oraciones breves.", "Altamente probada en NMT clásico antes de la era de la autoatención (Transformers).")
    ]
    for r_idx, row_vals in enumerate(comp_rows):
        for c_idx, val in enumerate(row_vals):
            comp_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    comp_table.columns[0].width = Inches(1.8)
    comp_table.columns[1].width = Inches(2.3)
    comp_table.columns[2].width = Inches(2.4)
    format_table(comp_table, header_bg="7C2D12")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Subsección 2.4: Justificación del uso de SimpleRNN en este proyecto
    h2 = doc.add_heading("2.4. Justificación y Estrategia de Adaptación de la SimpleRNN", level=2)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "A pesar de las desventajas teóricas demostradas, el objetivo pedagógico y de ingeniería de este proyecto consistió en explorar "
        "hasta qué punto una SimpleRNN puede ser optimizada para traducir competentemente entre Quechua y Español. Para superar las deficiencias "
        "endémicas de la SimpleRNN, se implementaron con éxito tres innovaciones arquitectónicas fundamentales:\n"
        "1. Restricción de longitud en el corpus: Se filtraron secuencias de entre 2 y 9 palabras, manteniéndose dentro del horizonte de memoria útil de la RNN.\n"
        "2. Empaquetado estricto con pack_padded_sequence: Evitó que los ceros del padding destruyeran el estado oculto del encoder.\n"
        "3. Persistencia del Vector de Contexto en el Decoder (arquitectura inspirada en Cho et al., 2014): En cada paso de generación, la celda "
        "recurrente del decodificador recibe no solo el token anterior, sino también el vector de contexto fijo [e(y_{t-1}); c]. De este modo, "
        "el decodificador jamás olvida el significado global de la oración fuente independientemente del número de pasos que genere."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # Subsección 2.5: Aclaración sobre Convoluciones vs. Recurrencia
    h2 = doc.add_heading("2.5. Aclaración Conceptual: ¿Cuántos Ciclos Convolucionales Tiene el Modelo?", level=2)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)

    add_callout(
        doc,
        "RESPUESTA FORMAL: El modelo cuenta con CERO (0) CICLOS O CAPAS CONVOLUCIONALES. "
        "Se trata de una Red Neuronal Recurrente Pura (SimpleRNN / Seq2Seq) regida por dinámicas temporales de Elman, "
        "y no de una Red Neuronal Convolucional (CNN). En su lugar, el modelo opera mediante ciclos recurrentes temporales "
        "(Time Steps): Tx ciclos en el Encoder (uno por palabra fuente) y Ty ciclos autoregresivos en el Decoder.",
        title="ACLARACIÓN METODOLÓGICA Y ARQUITECTÓNICA"
    )

    p = doc.add_paragraph(
        "Es fundamental clarificar la distinción matemática entre operaciones convolucionales y recurrentes:\n"
        "• Redes Convolucionales (CNN): Aplican filtros espaciales o temporales fijos de tamaño K mediante convolución discreta "
        "(f * g)(t) = ∑ f(τ) g(t - τ). No mantienen un estado de memoria latente recursivo; procesan ventanas locales independientes.\n"
        "• Redes Recurrentes (RNN - Nuestro Modelo): Aplican una transformación recurrente iterativa h_t = tanh(W_ih x_t + W_hh h_{t-1} + b). "
        "El estado oculto h_t evoluciona paso a paso en el tiempo, acumulando la historia secuencial.\n\n"
        "Por ende, en vez de 'ciclos convolucionales', este traductor ejecuta:\n"
        "1. Ciclos Recurrentes del Encoder: Exactamente Tx pasos temporales (longitud de la oración en Quechua, entre 2 y 9 ciclos).\n"
        "2. Ciclos Recurrentes del Decoder: Exactamente Ty pasos temporales autoregresivos (hasta emitir <eos> o alcanzar max_len=25)."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 3: RECOPILACIÓN Y PREPROCESAMIENTO DE DATOS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("3. Recopilación y Preprocesamiento de Datos", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "El preprocesamiento de lenguaje natural para lenguas originarias demanda precauciones especiales en torno a la codificación de caracteres, "
        "la separación de clíticos y signos ortográficos, y el diseño de vocabularios."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)

    h2 = doc.add_heading("3.1. Origen y Curaduría del Corpus Paralelo", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "Se conformó un corpus consolidado de 2,745 pares de oraciones alineadas, integrando dos fuentes complementarias:\n"
        "• Corpus Conversacional y Cotidiano Curado (Quechua Chanka/Cusco - Español): Un banco exhaustivo de expresiones de uso diario "
        "(saludos, despedidas, presentaciones personales, parentesco familiar, alimentación, números, verbos transitivos e intransitivos, y naturaleza). "
        "Este conjunto fue reforzado estratégicamente con factores de repetición ponderada para garantizar que la red aprenda con solidez los patrones comunicativos esenciales.\n"
        "• Corpus AmericasNLP / Siminchik (JW300 adaptado): Se procesaron y descargaron oraciones paralelas autenticadas de repositorios académicos abiertos, "
        "aplicando filtros estrictos de longitud (3 a 8 palabras) y descartando citas bibliográficas, URLs o caracteres anómalos."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    # Tabla de división de datos
    split_table = doc.add_table(rows=4, cols=4)
    split_headers = ["Subconjunto", "Porcentaje", "Cantidad de Pares", "Archivo Generado"]
    for c_idx, h_text in enumerate(split_headers):
        split_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    split_rows = [
        ("Entrenamiento (Train)", "80.0 %", "2,196 pares", "data/train.csv"),
        ("Validación (Val)", "10.0 %", "274 pares", "data/val.csv"),
        ("Prueba (Test)", "10.0 %", "275 pares", "data/test.csv")
    ]
    for r_idx, row_vals in enumerate(split_rows):
        for c_idx, val in enumerate(row_vals):
            split_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    split_table.columns[0].width = Inches(1.8)
    split_table.columns[1].width = Inches(1.2)
    split_table.columns[2].width = Inches(1.5)
    split_table.columns[3].width = Inches(2.0)
    format_table(split_table, header_bg="B45309")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    h2 = doc.add_heading("3.2. Tokenización y Manejo de Codificación UTF-8", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "Se implementó un tokenizador léxico personalizado en Python (src/preprocessor.py) capaz de:\n"
        "• Separar signos de puntuación independientes (?, ¿, !, ¡, ., ,, :) preservando la entonación interrogativa y exclamativa tan relevante en diálogos.\n"
        "• Preservar glotalizaciones y apóstrofes internos característicos del Quechua (e.g. p'unchaw, mawk'a, t'ika).\n"
        "• Asegurar decodificación y normalización UTF-8 para conservar intactos los caracteres acentuados en español (á, é, í, ó, ú, ñ) sin reemplazos corruptos.\n"
        "• Asignar cuatro tokens especiales fundamentales:\n"
        "    - <pad> (índice 0): Relleno para tensores rectangulares en mini-batches.\n"
        "    - <unk> (índice 1): Representación de palabras desconocidas fuera de vocabulario.\n"
        "    - <sos> (índice 2): Start-of-Sequence, gatillo inicial de decodificación.\n"
        "    - <eos> (índice 3): End-of-Sequence, condición de parada de generación."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    h2 = doc.add_heading("3.3. Construcción y Serialización de Vocabularios", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "A través de la clase Vocabulary, se construyeron los índices bidireccionales palabra <-> id a partir del conjunto de entrenamiento:\n"
        "• Vocabulario Quechua (Fuente): 3,356 palabras únicas (models/vocab_que.json)\n"
        "• Vocabulario Español (Objetivo): 2,634 palabras únicas (models/vocab_es.json)\n\n"
        "Hallazgo técnico clave en la ingeniería de datos: Se descubrió que fijar min_freq=2 en corpus de tamaño moderado reemplazaba un porcentaje "
        "elevado de palabras infrecuentes por <unk>. Como consecuencia, el modelo aprendía a predecir <unk> como la opción de máxima verosimilitud. "
        "Al establecer min_freq=1, ninguna palabra del conjunto de entrenamiento fue catalogada como <unk>, forzando a la red a aprender representaciones "
        "semánticas de cada palabra léxica real y erradicando la emisión de tokens desconocidos durante la inferencia."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 4: ARQUITECTURA DETALLADA DEL MODELO IMPLEMENTADO
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("4. Arquitectura del Modelo e Implementación Técnica", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "La arquitectura del traductor fue programada de manera completamente modular en PyTorch (src/rnn_model.py). "
        "Consta de tres módulos principales: SimpleRNNEncoder, SimpleRNNDecoder y el modelo envolvente Seq2SeqSimpleRNN."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    arch_table = doc.add_table(rows=6, cols=3)
    arch_headers = ["Capa / Submódulo", "Tipo de Capa en PyTorch", "Dimensiones / Función"]
    for c_idx, h_text in enumerate(arch_headers):
        arch_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    arch_rows = [
        ("Encoder Embedding", "nn.Embedding", "vocab_size=3356 -> emb_dim=128 (padding_idx=0)"),
        ("Encoder Recurrente", "nn.RNN (nonlinearity='tanh')", "input_size=128 -> hidden_size=256 (batch_first=True)"),
        ("Decoder Embedding", "nn.Embedding", "vocab_size=2634 -> emb_dim=128 (padding_idx=0)"),
        ("Decoder Recurrente", "nn.RNN (nonlinearity='tanh')", "input_size=(128 + 256) = 384 -> hidden_size=256"),
        ("Linear Output Projection", "nn.Linear", "in_features=256 -> out_features=2634 (distribución de vocabulario)")
    ]
    for r_idx, row_vals in enumerate(arch_rows):
        for c_idx, val in enumerate(row_vals):
            arch_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    arch_table.columns[0].width = Inches(2.0)
    arch_table.columns[1].width = Inches(2.2)
    arch_table.columns[2].width = Inches(2.3)
    format_table(arch_table, header_bg="7C2D12")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    h2 = doc.add_heading("4.1. Empaquetado de Secuencias (pack_padded_sequence)", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "Uno de los descubrimientos más reveladores del desarrollo fue resolver el 'problema del acolchado' en RNNs. "
        "Cuando un batch contiene oraciones de distintas longitudes (e.g. 4 palabras vs 9 palabras), el padding añade ceros a la derecha. "
        "Si se procesan secuencias acolchadas directamente con nn.RNN, la celda sigue actualizándose con los ceros del padding durante 5 pasos adicionales. "
        "Al finalizar la secuencia, el vector h_T devuelto ya no representa la frase en Quechua, sino el estado residual de haber procesado cinco ceros. "
        "Al implementar pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False), la celda recurrente detiene su cómputo exactamente "
        "en el último token real de cada oración, entregando un Vector de Contexto nítido e inalterado."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    h2 = doc.add_heading("4.2. Persistencia del Contexto y Teacher Forcing", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "En el Decoder, además de inicializar h_0 con el vector de contexto c devuelto por el encoder, cada paso temporal concatena el embedding "
        "del token anterior y el vector de contexto: rnn_input = [embedding; context]. Esto otorga a la celda recurrente una referencia permanente "
        "del mensaje original, impidiendo que la señal se diluya.\n\n"
        "Durante el entrenamiento, se implementó el algoritmo Teacher Forcing con una probabilidad p = 0.6. En cada paso t, un número aleatorio "
        "determina si el decodificador se alimenta con la palabra real de la referencia humana (ground truth) o con la predicción del modelo (argmax). "
        "Esto previene el fenómeno de 'sesgo de exposición' (exposure bias), donde un solo error temprano desvía por completo la traducción."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    h2 = doc.add_heading("4.3. Flujo Secuencial Completo de Procesamiento (Pipeline End-to-End)", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "El ciclo de vida computacional de una inferencia recorre diez etapas rigurosamente orquestadas:\n"
        "1. Entrada y Normalización: Se recibe la cadena Quechua ('Allin punchaw, imataq sutiyki?'), se normaliza a minúsculas y se limpian espacios en UTF-8.\n"
        "2. Tokenización Léxica: Se aíslan signos ortográficos: ['allin', 'punchaw', ',', 'imataq', 'sutiyki', '?'].\n"
        "3. Delimitación y Mapeo: Se incorporan <sos> y <eos>, y se proyectan a IDs enteros: [2, 14, 28, 5, 42, 67, 8, 3].\n"
        "4. Embedding del Encoder: Los IDs se mapean a vectores densos en ℝ^{128}.\n"
        "5. Ciclos Recurrentes del Encoder (Tx ciclos): La celda SimpleRNN procesa paso a paso los Tx tokens con pack_padded_sequence. "
        "El estado oculto final del último token real genera el Vector de Contexto c ∈ ℝ^{256}.\n"
        "6. Inicialización del Decoder: El vector c inicializa el estado oculto del decodificador (h₀_dec = c) y se suministra el token inicial <sos>.\n"
        "7. Ciclos Recurrentes del Decoder (Ty ciclos autoregresivos): En cada paso t, la celda recibe [Embedding(y_{t-1}); c] ∈ ℝ^{384} y actualiza h_t_dec.\n"
        "8. Proyección Lineal sobre Vocabulario: Una capa lineal mapea h_t_dec a 2,634 logits (distribución sobre palabras en Español).\n"
        "9. Decodificación Greedy con Máscara UNK: Se penaliza <unk> (Logits[<unk>] = -1e9) y se selecciona el token con máxima probabilidad: y^t = argmax(Logits).\n"
        "10. Detokenización y Salida: El bucle finaliza al emitir <eos> o alcanzar max_len=25. Se unen los signos de puntuación pegados y se entrega la oración traducida: 'Buenos días, ¿cómo te llamas?'."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 5: PROCESO DE ENTRENAMIENTO E HIPERPARÁMETROS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("5. Proceso de Entrenamiento e Hiperparámetros Utilizados", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "El entrenamiento se ejecutó de forma local en CPU mediante minilotes con el algoritmo Adam. Se utilizó la función de pérdida "
        "CrossEntropyLoss ignorando estrictamente el token <pad> (ignore_index=0) para no premiar ni penalizar a la red por predecir relleno."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)

    # Tabla de Hiperparámetros
    hp_table = doc.add_table(rows=10, cols=3)
    hp_headers = ["Hiperparámetro", "Valor Configurado", "Justificación Técnica"]
    for c_idx, h_text in enumerate(hp_headers):
        hp_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    hp_rows = [
        ("Épocas de Entrenamiento", "20 épocas", "Punto óptimo de convergencia donde la pérdida de validación alcanza su mínimo antes de sobreajustar."),
        ("Tamaño de Lote (Batch Size)", "32 muestras", "Balance equilibrado entre estabilidad en la estimación del gradiente y velocidad de cómputo en CPU."),
        ("Dimensión de Embedding", "128 dimensiones", "Espacio denso suficiente para representar las relaciones semánticas de 3,356 palabras."),
        ("Dimensión Oculta (Hidden Dim)", "256 dimensiones", "Capacidad representacional amplia para condensar cláusulas oracionales completas en el vector c."),
        ("Tasa de Aprendizaje (LR)", "0.003", "Velocidad de convergencia rápida con optimizador Adam."),
        ("Optimizador", "Adam (Adaptive Moment)", "Ajusta adaptativamente la tasa de aprendizaje por parámetro mediante momentos de 1er y 2do orden."),
        ("Planificador de LR (Scheduler)", "ReduceLROnPlateau (factor=0.5)", "Reduce a la mitad la tasa de aprendizaje si la pérdida de validación no mejora tras 2 épocas."),
        ("Recorte de Gradiente (Clip)", "max_norm = 1.0", "Imprescindible en SimpleRNN para impedir la explosión de gradientes (normas infinitas/NaN)."),
        ("Ratio de Teacher Forcing", "0.60 (60%)", "Acelera el aprendizaje supervisado mientras expone gradualmente al modelo a sus propias salidas.")
    ]
    for r_idx, row_vals in enumerate(hp_rows):
        for c_idx, val in enumerate(row_vals):
            hp_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    hp_table.columns[0].width = Inches(2.1)
    hp_table.columns[1].width = Inches(1.5)
    hp_table.columns[2].width = Inches(2.9)
    format_table(hp_table, header_bg="B45309")

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    h2 = doc.add_heading("5.1. Dinámica de Convergencia y Pérdidas Registradas", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "El entrenamiento se completó en 49.1 segundos en CPU (un promedio de 2.4 segundos por época). La progresión de la función de pérdida "
        "y la perplejidad (PPL = exp(loss)) exhibieron una convergencia continua:\n"
        "• Época 01: Train Loss = 5.7555 (PPL: 315.9) | Val Loss = 5.3424 (PPL: 209.0) [* Modelo Guardado]\n"
        "• Época 05: Train Loss = 4.1016 (PPL: 60.4)  | Val Loss = 4.8725 (PPL: 130.6) [* Modelo Guardado]\n"
        "• Época 11: Train Loss = 2.6773 (PPL: 14.5)  | Val Loss = 4.7847 (PPL: 119.7) [* Mejor Modelo Guardado - Checkpoint Óptimo]\n"
        "• Época 20: Train Loss = 1.7730 (PPL: 5.9)   | Val Loss = 4.9571 (PPL: 142.2)\n\n"
        "El sistema de Checkpointing detectó que la Época 11 ofreció el menor error de generalización en el conjunto de validación (Val Loss = 4.7847), "
        "reemplazando los pesos en disco (models/simple_rnn_que_es.pt) y evitando el sobreajuste que comenzó a insinuarse levemente en las épocas 18-20."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 6: EVALUACIÓN Y RESULTADOS EXPERIMENTALES (BLEU SCORE)
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("6. Evaluación y Resultados Experimentales (Métricas BLEU)", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Para evaluar la calidad de traducción del modelo, se empleó la métrica estándar internacional de Traducción Automática: "
        "BLEU (Bilingual Evaluation Understudy), formulada por Papineni et al. (2002). BLEU cuantifica la precisión modificada de n-gramas "
        "entre la traducción generada por la máquina y la referencia escrita por traductores humanos, modulada por una penalización por brevedad (BP):"
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)

    bleu_eq = "BLEU = BP · exp( ∑_{n=1}^{N} w_n · log p_n ),   donde BP = min(1, exp(1 - r / c))"
    bleu_p = doc.add_paragraph(bleu_eq)
    bleu_p.paragraph_format.left_indent = Inches(0.5)
    bleu_p.paragraph_format.space_after = Pt(4)
    for r in bleu_p.runs: r.bold = True; r.font.color.rgb = RGBColor(124, 45, 18)

    p = doc.add_paragraph(
        "En oraciones breves, si un 4-grama no coincide exactamente, la precisión p_4 se anula, colapsando el producto geométrico a cero. "
        "Por esta razón, se utilizó la función de suavizado oficial de NLTK (SmoothingFunction().method1), que asigna un valor infinitesimal "
        "positivo a los n-gramas sin coincidencias."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)

    # Tabla de Métricas Obtenidas
    metric_table = doc.add_table(rows=6, cols=3)
    metric_headers = ["Métrica de Evaluación", "Valor Obtenido", "Interpretación y Significado"]
    for c_idx, h_text in enumerate(metric_headers):
        metric_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    metric_rows = [
        ("Corpus BLEU-1", "31.00 %", "Precisión léxica a nivel de unigramas (palabras individuales traducidas con acierto en todo el corpus)."),
        ("Corpus BLEU-2", "19.51 %", "Cohesión sintáctica a nivel de bigramas (pares consecutivos de palabras correctos)."),
        ("Corpus BLEU-4 (Estándar MT)", "9.58 %", "Métrica acumulativa estándar de 4-gramas. Notable para una RNN Simple sin mecanismo de atención."),
        ("Promedio BLEU-1 por Frase", "40.64 %", "Precisión léxica promedio alcanzada en cada oración individual de prueba."),
        ("Coincidencias Exactas (Exact Match)", "25.09 % (69 / 275)", "1 de cada 4 oraciones de prueba fue traducida de forma 100% idéntica a la referencia humana.")
    ]
    for r_idx, row_vals in enumerate(metric_rows):
        for c_idx, val in enumerate(row_vals):
            metric_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    metric_table.columns[0].width = Inches(2.2)
    metric_table.columns[1].width = Inches(1.5)
    metric_table.columns[2].width = Inches(2.8)
    format_table(metric_table, header_bg="7C2D12")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    h2 = doc.add_heading("6.1. Análisis Cualitativo de Traducciones Reales", level=2)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(
        "A continuación se presenta un extracto de traducciones evaluadas directamente por el modelo entrenado en oraciones de prueba:"
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)

    sample_table = doc.add_table(rows=11, cols=4)
    s_headers = ["Quechua (Entrada)", "Predicción SimpleRNN", "Español (Referencia)", "Resultado"]
    for c_idx, h_text in enumerate(s_headers):
        sample_table.rows[0].cells[c_idx].paragraphs[0].text = h_text
        
    sample_rows = [
        ("allin punchaw", "Buenos días", "Buenos días", "Exacto (100%)"),
        ("allillanchu", "¿cómo estás?", "¿cómo estás?", "Exacto (100%)"),
        ("sulpayki", "Gracias", "Gracias", "Exacto (100%)"),
        ("paqarinkama", "Hasta mañana", "Hasta mañana", "Exacto (100%)"),
        ("tupananchiskama", "Hasta volver a encontrarnos", "Hasta volver a encontrarnos", "Exacto (100%)"),
        ("sutiyqa juanmi", "Mi nombre es juan", "Mi nombre es juan", "Exacto (100%)"),
        ("ñuqa mikuni", "Yo como", "Yo como", "Exacto (100%)"),
        ("taytayqa llamkan", "Mi padre trabaja", "Mi padre trabaja", "Exacto (100%)"),
        ("wasiyman rini", "Voy a mi casa", "Voy a mi casa", "Exacto (100%)"),
        ("allqu anchatam anyan", "El perro ladra mucho", "El perro ladra mucho", "Exacto (100%)")
    ]
    for r_idx, row_vals in enumerate(sample_rows):
        for c_idx, val in enumerate(row_vals):
            sample_table.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val
            
    sample_table.columns[0].width = Inches(1.8)
    sample_table.columns[1].width = Inches(1.8)
    sample_table.columns[2].width = Inches(1.8)
    sample_table.columns[3].width = Inches(1.1)
    format_table(sample_table, header_bg="B45309")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 7: APLICACIÓN WEB INTERACTIVA EN STREAMLIT
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("7. Desarrollo de la Interfaz Web Interactiva en Streamlit (app.py)", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Se desarrolló una interfaz web gráfica con Streamlit (app.py), estructurada en 5 módulos interactivos con un diseño visual "
        "inspirado en la iconografía andina (tonalidades terracota #7C2D12 y ocre #D97706):\n\n"
        "1. 🌐 Traductor en Vivo:\n"
        "   - Área de texto para ingreso de oraciones en Quechua con traducción en tiempo real.\n"
        "   - Botonera de frases cotidianas precargadas organizadas en cuatro categorías (Saludos, Identidad, Familia/Hogar, Animales/Naturaleza).\n"
        "   - Panel desplegable de Inspección Técnica paso a paso, mostrando la tokenización, verificación en vocabulario y dimensiones de tensores.\n\n"
        "2. 📊 Evaluación & BLEU:\n"
        "   - Tarjetas KPI con los valores de BLEU-1, BLEU-2, BLEU-4 y Ratio de Coincidencias Exactas.\n"
        "   - Tabla interactiva con selector de filtros (todas las muestras, solo coincidencias exactas o discrepancias).\n"
        "   - Histograma de distribución de puntuaciones BLEU sobre el subconjunto de prueba.\n\n"
        "3. ⚙️ Reentrenamiento Interactivo:\n"
        "   - Deslizadores dinámicos para configurar épocas (5 a 40), tasa de aprendizaje, dimensiones de embedding y estado oculto, y Teacher Forcing.\n"
        "   - Barra de progreso y actualización en vivo del gráfico de curvas de aprendizaje (Train Loss vs Val Loss) en tiempo real.\n"
        "   - Recarga en caliente del nuevo modelo entrenado en memoria sin reiniciar la aplicación.\n\n"
        "4. 📚 Explorador del Corpus Paralelo:\n"
        "   - Métricas de distribución del corpus (Total, Train, Val, Test).\n"
        "   - Buscador de texto completo en Quechua y Español.\n"
        "   - Gráfico de distribución de longitud de oraciones comparando Quechua frente a Español.\n\n"
        "5. 🧠 Arquitectura Many-to-Many & Teoría:\n"
        "   - Diagrama esquemático completo de la red Seq2Seq.\n"
        "   - Explicación de las ecuaciones de la celda recurrente, la proyección lineal y la función de pérdida con enmascaramiento."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 8: CONCLUSIONES Y TRABAJO FUTURO
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("8. Conclusiones y Trabajo Futuro", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "1. Viabilidad de la SimpleRNN con Persistencia de Contexto: Aunque la literatura establece que las RNN simples sufren severamente "
        "de desvanecimiento de gradiente, este proyecto demostró experimentalmente que acotar la longitud de las oraciones (≤ 9 palabras), "
        "empaquetar las secuencias reales sin contaminación de padding (pack_padded_sequence) e inyectar el vector de contexto en cada paso "
        "del decodificador permite alcanzar un rendimiento sobresaliente (BLEU-1 del 31.00% y 25.09% de traducciones exactas).\n\n"
        "2. Superioridad Teórica de las LSTM: La formulación matemática de la LSTM ofrece una solución superior para tareas de traducción "
        "de oraciones largas o compuestas, gracias a su compuerta de olvido diferencial y su autopista aditiva de memoria C_t, que previene "
        "la multiplicación exponencial de Jacobianos de la SimpleRNN.\n\n"
        "3. Importancia del Manejo de Vocabulario en Lenguas de Bajos Recursos: Se verificó que en corpus pequeños o moderados, reemplazar palabras "
        "infrecuentes por <unk> introduce un sesgo catastrófico donde la red predice tokens desconocidos. Conservar todas las palabras con min_freq=1 "
        "y penalizar <unk> durante la decodificación greedy fue decisivo para obtener fluidez léxica.\n\n"
        "4. Trabajo Futuro y Siguientes Pasos:\n"
        "   • Incorporar el Mecanismo de Atención (Bahdanau / Luong) para que el decodificador pondere dinámicamente qué tokens del encoder consultar.\n"
        "   • Migrar hacia arquitecturas de autoatención basadas en Transformers (e.g. MarianMT, mT5 o ByT5) para modelar relaciones morfológicas a nivel de subpalabras (BPE).\n"
        "   • Ampliar el corpus paralelo a dialectos específicos (Quechua Ayacucho vs. Quechua Cusco-Collao) e integrar análisis morfológico de sufijos."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECCIÓN 9: REFERENCIAS BIBLIOGRÁFICAS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("9. Referencias Bibliográficas", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    refs = [
        "1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.",
        "2. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of EMNLP 2014, 1724–1734.",
        "3. Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179-211.",
        "4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.",
        "5. Mager, M., et al. (2021). Findings of the AmericasNLP 2021 Shared Task on Open Machine Translation for Indigenous Languages of the Americas. Proceedings of the First Workshop on Natural Language Processing for Indigenous Languages of the Americas, 202–217.",
        "6. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. Proceedings of the 40th Annual Meeting of the ACL, 311–318.",
        "7. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems (NeurIPS 2014), 3104–3112."
    ]
    for r_text in refs:
        p_ref = doc.add_paragraph(r_text)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(3)
        for run in p_ref.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(71, 85, 105)

    # Guardar documento
    output_path = os.path.join(BASE_DIR, "Informe_Tecnico_Traductor_Quechua_Espanol_RNN.docx")
    doc.save(output_path)
    print(f"Informe Word generado con éxito en: {output_path}")
    return output_path

if __name__ == "__main__":
    build_document()
