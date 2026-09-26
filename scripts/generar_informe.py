"""Genera el informe del Proyecto 2 (DOCX + PDF) a partir de todos los resultados.

Lee: resultados/eda.json, resultados/registro_iteraciones.csv,
resultados/<iter>/final.json y eval/evaluations.npz, resultados/evaluaciones/*.json,
entregables/metricas_video.json y entregables/figuras/*.png.

Uso:
    python scripts/generar_informe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from graficar import cargar_escalares_tb, cargar_evaluaciones  # noqa: E402

ENTREGABLES = RAIZ / "entregables"
FIGURAS = ENTREGABLES / "figuras"
RESULTADOS = RAIZ / "resultados"
SALIDA_DOCX = ENTREGABLES / "Proyecto2_Informe_Ian_Cumes_23236.docx"
SALIDA_PDF = ENTREGABLES / "Proyecto2_Informe_Ian_Cumes_23236.pdf"
REPO_URL = "https://github.com/iancumes/Proyecto2DeepLearning"

AZUL = "17365D"
GRIS_BORDE = "D9D9D9"
GRIS_FILA = "F2F5F8"
NEGRO = RGBColor(0, 0, 0)

ETIQUETAS = {
    "iter1_dqn": "Iteración 1 — DQN",
    "iter2_ppo_v5_directo": "Iteración 2 — PPO (v5 directo, sin max-pool ni vida)",
    "iter3_ppo_final": "Iteración 3 — PPO (preprocesamiento final)",
}

# --------------------------------------------------------------------------
# Helpers de estilo (adaptados del Laboratorio 5, mismo patrón probado)
# --------------------------------------------------------------------------


def configurar_fuente(run, nombre: str = "Aptos", tamano: float | None = None) -> None:
    run.font.name = nombre
    fuentes = run._element.get_or_add_rPr().rFonts
    fuentes.set(qn("w:ascii"), nombre)
    fuentes.set(qn("w:hAnsi"), nombre)
    run.font.color.rgb = NEGRO
    if tamano is not None:
        run.font.size = Pt(tamano)


def sombrear_celda(celda, color: str) -> None:
    tc_pr = celda._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def bordes_tabla(tabla) -> None:
    tbl_pr = tabla._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for borde in ("top", "left", "bottom", "right", "insideH", "insideV"):
        nodo = OxmlElement(f"w:{borde}")
        nodo.set(qn("w:val"), "single")
        nodo.set(qn("w:sz"), "4")
        nodo.set(qn("w:color"), GRIS_BORDE)
        borders.append(nodo)


def margen_celda(celda, vertical: int = 80, horizontal: int = 90) -> None:
    tc_pr = celda._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for lado, valor in (("top", vertical), ("left", horizontal), ("bottom", vertical), ("right", horizontal)):
        nodo = tc_mar.find(qn(f"w:{lado}"))
        if nodo is None:
            nodo = OxmlElement(f"w:{lado}")
            tc_mar.append(nodo)
        nodo.set(qn("w:w"), str(valor))
        nodo.set(qn("w:type"), "dxa")


def configurar_tabla(tabla, anchos: list[float], centrar_columnas: set[int], tamano_fuente: float = 8.2) -> None:
    tabla.autofit = False
    tabla.alignment = WD_ALIGN_PARAGRAPH.CENTER
    bordes_tabla(tabla)
    tr_pr = tabla.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for i, fila in enumerate(tabla.rows):
        for j, celda in enumerate(fila.cells):
            if j < len(anchos):
                celda.width = Inches(anchos[j])
            celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            margen_celda(celda)
            if i == 0:
                sombrear_celda(celda, AZUL)
            elif i % 2 == 0:
                sombrear_celda(celda, GRIS_FILA)
            for p in celda.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in centrar_columnas else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    configurar_fuente(run, tamano=tamano_fuente)
                    if i == 0:
                        run.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)


def campo_pagina(parrafo_) -> None:
    run = parrafo_.add_run()
    inicio = OxmlElement("w:fldChar")
    inicio.set(qn("w:fldCharType"), "begin")
    instruccion = OxmlElement("w:instrText")
    instruccion.set(qn("xml:space"), "preserve")
    instruccion.text = " PAGE "
    fin = OxmlElement("w:fldChar")
    fin.set(qn("w:fldCharType"), "end")
    run._r.extend([inicio, instruccion, fin])
    configurar_fuente(run, tamano=8)


def parrafo(doc: Document, texto: str = "", estilo: str | None = None):
    p = doc.add_paragraph(style=estilo)
    if texto:
        configurar_fuente(p.add_run(texto))
    return p


def fragmentos(doc: Document, partes: list[tuple[str, bool, bool]], estilo: str | None = None):
    p = doc.add_paragraph(style=estilo)
    for texto, negrita, codigo in partes:
        run = p.add_run(texto)
        configurar_fuente(run, "Consolas" if codigo else "Aptos", 9.3 if codigo else None)
        run.bold = negrita
    return p


def texto_negrita(doc: Document, etiqueta: str, texto: str):
    p = doc.add_paragraph()
    run = p.add_run(etiqueta)
    configurar_fuente(run)
    run.bold = True
    configurar_fuente(p.add_run(texto))
    return p


def insertar_figura(doc: Document, ruta: Path, leyenda: str, ancho: float = 5.4) -> None:
    if not ruta.is_file():
        texto_negrita(doc, "[Figura no disponible] ", leyenda)
        return
    fig = doc.add_paragraph()
    fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig.paragraph_format.space_before = Pt(6)
    fig.paragraph_format.space_after = Pt(1)
    fig.add_run().add_picture(str(ruta), width=Inches(ancho))
    cap = parrafo(doc, leyenda)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(8)


def lista(doc: Document, items: list[str]) -> None:
    for texto in items:
        p = doc.add_paragraph(style="List Bullet")
        configurar_fuente(p.add_run(texto))


def configurar_documento(doc: Document) -> None:
    seccion = doc.sections[0]
    seccion.page_width = Inches(8.5)
    seccion.page_height = Inches(11)
    seccion.top_margin = Inches(0.75)
    seccion.bottom_margin = Inches(0.65)
    seccion.left_margin = Inches(0.85)
    seccion.right_margin = Inches(0.85)
    seccion.header_distance = Inches(0.3)
    seccion.footer_distance = Inches(0.3)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10)
    normal.font.color.rgb = NEGRO
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    titulo = doc.styles["Title"]
    titulo.font.name = "Aptos Display"
    titulo._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
    titulo._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
    titulo.font.size = Pt(21)
    titulo.font.bold = True
    titulo.font.color.rgb = NEGRO
    titulo.paragraph_format.space_after = Pt(6)
    titulo_ppr = titulo._element.get_or_add_pPr()
    bordes = titulo_ppr.find(qn("w:pBdr"))
    if bordes is not None:
        titulo_ppr.remove(bordes)

    for nombre, tamano, antes, despues in (("Heading 1", 15, 10, 5), ("Heading 2", 11.5, 7, 3)):
        estilo = doc.styles[nombre]
        estilo.font.name = "Aptos Display"
        estilo._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        estilo._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        estilo.font.size = Pt(tamano)
        estilo.font.bold = True
        estilo.font.color.rgb = NEGRO
        estilo.paragraph_format.space_before = Pt(antes)
        estilo.paragraph_format.space_after = Pt(despues)
        estilo.paragraph_format.keep_with_next = True

    encabezado = seccion.header.paragraphs[0]
    encabezado.text = "CC3092  |  Proyecto 2  |  Agente de RL para Space Invaders"
    encabezado.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in encabezado.runs:
        configurar_fuente(run, tamano=8)

    pie = seccion.footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    configurar_fuente(pie.add_run("Página "), tamano=8)
    campo_pagina(pie)


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------


def cargar_json(ruta: Path) -> dict | None:
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_registro_iteraciones() -> pd.DataFrame | None:
    ruta = RESULTADOS / "registro_iteraciones.csv"
    if not ruta.is_file():
        return None
    return pd.read_csv(ruta)


def cargar_ultima_eval_npz(run_dir: Path) -> dict | None:
    evals = cargar_evaluaciones(run_dir)
    if evals is None or len(evals["timesteps"]) == 0:
        return None
    resultados = evals["results"][-1]
    return {
        "pasos": int(evals["timesteps"][-1]),
        "media": float(resultados.mean()),
        "desviacion": float(resultados.std()),
        "maximo": float(resultados.max()),
    }


def cargar_ep_rew_mean_final(run_dir: Path) -> float | None:
    try:
        escalares = cargar_escalares_tb(run_dir)
    except Exception:
        return None
    if "rollout/ep_rew_mean" not in escalares:
        return None
    _, valores = escalares["rollout/ep_rew_mean"]
    return float(valores[-1]) if len(valores) else None


def recolectar_datos() -> dict:
    datos: dict = {}
    datos["eda"] = cargar_json(RESULTADOS / "eda.json")
    datos["registro"] = cargar_registro_iteraciones()
    datos["metricas_video"] = cargar_json(ENTREGABLES / "metricas_video.json")
    datos["eval_iter0"] = cargar_json(RESULTADOS / "evaluaciones" / "iter0_aleatorio.json")

    iteraciones = {}
    for run_id in ETIQUETAS:
        run_dir = RESULTADOS / run_id
        iteraciones[run_id] = {
            "final": cargar_json(run_dir / "final.json"),
            "eval_entrenamiento": cargar_ultima_eval_npz(run_dir),
            "ep_rew_mean_entrenamiento": cargar_ep_rew_mean_final(run_dir),
        }
    datos["iteraciones"] = iteraciones

    eval_oficial = None
    carpeta_eval = RESULTADOS / "evaluaciones"
    if carpeta_eval.is_dir():
        candidatos = sorted(carpeta_eval.glob("*final*.json"))
        if candidatos:
            eval_oficial = cargar_json(candidatos[-1])
    datos["eval_oficial_final"] = eval_oficial
    return datos


def formato_recompensa(media, desv=None, maximo=None) -> str:
    if media is None:
        return "N/D"
    txt = f"{media:.0f}"
    if desv is not None:
        txt += f" ± {desv:.0f}"
    if maximo is not None:
        txt += f" (máx. {maximo:.0f})"
    return txt


def valor_eval(run_id: str, datos: dict) -> float | None:
    it = datos["iteraciones"][run_id]
    if run_id == "iter3_ppo_final" and datos.get("eval_oficial_final"):
        return datos["eval_oficial_final"]["recompensa_media"]
    ev = it.get("eval_entrenamiento")
    return ev["media"] if ev else None


def comparar(a_label: str, a_val, b_label: str, b_val, contexto: str) -> str:
    if a_val is None or b_val is None:
        return f"No fue posible comparar {a_label} y {b_label} ({contexto}: datos insuficientes al momento de generar este informe)."
    diff = a_val - b_val
    if abs(diff) < 1.0:
        return f"{a_label} y {b_label} obtuvieron un desempeño de evaluación prácticamente equivalente (diferencia de {diff:+.0f} puntos; {contexto})."
    mejor, peor, d = (a_label, b_label, diff) if diff > 0 else (b_label, a_label, -diff)
    return f"{mejor} superó a {peor} por {d:.0f} puntos en promedio de evaluación greedy ({contexto})."


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------


def seccion_2_1(doc: Document, datos: dict) -> None:
    eda = datos.get("eda") or {}
    info = eda.get("info_entorno", {})
    epi = eda.get("episodios_aleatorios", {})

    parrafo(doc, "2.1 Definición del problema y análisis del entorno", "Heading 1")

    parrafo(doc, "El entorno: ALE/SpaceInvaders-v5", "Heading 2")
    fragmentos(doc, [
        ("Space Invaders (Taito, 1978; versión Atari 2600 usada por ALE) enfrenta al jugador contra formaciones de invasores que descienden y se desplazan lateralmente mientras disparan. El objetivo es eliminar la mayor cantidad de invasores posible protegiendo al cañón propio detrás de tres refugios destructibles. El episodio ", False, False),
        ("termina (terminated=True)", False, True),
        (" cuando el jugador pierde sus tres vidas o cuando la formación completa desciende hasta la base. ALE además impone un límite duro de ", False, False),
        (f"{info.get('max_num_frames_per_episode', 108000):,}".replace(",", " "), False, True),
        (" frames por episodio, que produce ", False, False),
        ("truncated=True", False, True),
        (" si se alcanza (en la práctica, casi nunca ocurre antes de perder las tres vidas). El entorno reporta el conteo de vidas restantes en ", False, False),
        ("info['lives']", False, True),
        (f" (parte de {info.get('vidas_iniciales', 3)} al reiniciar).", False, False),
    ])
    fragmentos(doc, [
        ("La señal de recompensa de ALE es directamente el ", False, False),
        ("cambio en el marcador del juego", True, False),
        (": cada fila de invasores vale 5, 10, 15, 20, 25 o 30 puntos según su cercanía a la base, y la nave nodriza que cruza ocasionalmente por arriba otorga un bono variable. En 30 episodios aleatorios (semilla 0) se registró una recompensa media de ", False, False),
        (f"{epi.get('puntaje_medio', 0):.1f}", True, False),
        (f" puntos (mín. {epi.get('puntaje_min', 0):.0f}, máx. {epi.get('puntaje_max', 0):.0f}, desv. {epi.get('puntaje_desviacion', 0):.1f}) en un promedio de {epi.get('pasos_medio', 0):.0f} decisiones por episodio; el valor ", False, False),
        ("200", False, True),
        (" apareció dos veces, coincidiendo con el bono de la nave nodriza descrito en el manual original del juego.", False, False),
    ])

    parrafo(doc, "¿Recompensa densa o dispersa?", "Heading 2")
    fragmentos(doc, [
        (f"La señal es claramente ", False, False), ("dispersa (sparse)", True, False),
        (f": el {epi.get('pct_pasos_recompensa_cero', 0):.1f}% de las decisiones no producen ningún cambio de puntuación (Figura 1), y los valores no nulos observados fueron exactamente ", False, False),
        (str(sorted(int(v) for v in epi.get("valores_recompensa_no_cero_observados", []))), False, True),
        (". Esto exige que el agente sostenga una secuencia larga de acciones correctas antes de recibir cualquier señal, el reto central de explorar en este entorno.", False, False),
    ])
    insertar_figura(doc, FIGURAS / "eda_histograma_recompensas.png", "Figura 1. Distribución de los valores de recompensa no nulos en 30 episodios aleatorios.", ancho=4.6)

    fragmentos(doc, [
        ("No se aplicó ", False, False), ("reward shaping", True, False),
        (": la puntuación de ALE ya refleja fielmente el objetivo del juego, y modificarla arriesgaría cambiar qué política es óptima. En su lugar se usaron dos técnicas estándar, ", False, False),
        ("solo durante el entrenamiento", True, False),
        (" (nunca en evaluación, para no falsear el puntaje real): recorte de recompensa a {-1,0,+1} (estabiliza la escala de la pérdida sin cambiar el signo de la señal) y tratar la pérdida de una vida como fin de episodio, que acelera el aprendizaje al asociar el fin de un intento con su causa inmediata, sin alterar el límite real del episodio (3 vidas).", False, False),
    ])

    parrafo(doc, "Espacios de observación y acción, y el parpadeo de Space Invaders", "Heading 2")
    fragmentos(doc, [
        (f"La observación por defecto es ", False, False), (info.get("observation_space", "Box(0,255,(210,160,3),uint8)"), False, True),
        (" (imagen RGB) y el espacio de acción es ", False, False), (info.get("action_space", "Discrete(6)"), False, True),
        (f" con las acciones {info.get('significado_acciones', [])}. Se usó el conjunto mínimo de acciones (", False, False),
        ("full_action_space=False", False, True),
        (") en vez de las 18 combinaciones completas del joystick: las 12 acciones adicionales son combinaciones redundantes (diagonales que no existen en este juego) que solo agrandarían la salida de la red sin aportar control nuevo.", False, False),
    ])
    fragmentos(doc, [
        ("Un problema conocido de los juegos de Atari 2600 (documentado por Mnih et al., 2015) es que el hardware original dibuja algunos objetos ", False, False),
        ("cada dos frames", True, False),
        (" para ahorrar memoria de video, produciendo parpadeo. En Space Invaders esto afecta a los proyectiles: en la Figura 2 (panel derecho) el mapa de diferencia absoluta entre dos frames consecutivos muestra un objeto aislado junto al cañón —un disparo que aparece en un frame y desaparece en el siguiente— mientras el resto de la escena permanece prácticamente inmóvil en ese intervalo de 1/60 s. Un agente que observe un solo frame por decisión puede literalmente no ver una bala. La comparación agregada de la Figura 3 no aísla bien este efecto porque el avance general de los invasores también crece con la distancia temporal y domina la métrica global; la evidencia relevante es la localizada de la Figura 2.", False, False),
    ])
    insertar_figura(doc, FIGURAS / "eda_frames.png", "Figura 2. Frame RGB crudo, su versión preprocesada (84x84, gris) y la diferencia absoluta entre dos frames consecutivos: el objeto aislado junto al cañón es un proyectil que parpadea.", ancho=6.0)
    insertar_figura(doc, FIGURAS / "eda_flicker.png", "Figura 3. Diferencia absoluta media agregada entre frames separados por 1 y por 2 pasos (no aísla el parpadeo; ver discusión).", ancho=3.6)

    parrafo(doc, "Decisiones de diseño derivadas de este análisis", "Heading 2")
    lista(doc, [
        "Preprocesamiento con max-pooling: gymnasium.wrappers.AtariPreprocessing(frame_skip=4) toma el máximo píxel a píxel entre los dos últimos frames de cada bloque de 4, eliminando el parpadeo sin promediar (que difuminaría) ni tomar solo el último frame (que perdería el proyectil en la mitad de los casos).",
        "Apilado de 4 frames (FrameStackObservation) en escala de grises y 84x84: una sola imagen no expresa velocidad ni dirección; se siguió la convención de Mnih et al. (2015).",
        "frameskip=1 en gym.make() + frame_skip=4 en el wrapper (nunca ambos a la vez): así se controla explícitamente el max-pooling. La iteración 2 (sección 2.3) usa deliberadamente el frameskip=4 nativo de ALE (sin max-pool) para cuantificar el costo de omitir esta decisión.",
        "Conjunto mínimo de 6 acciones y acciones pegajosas (repeat_action_probability=0.25, valor por defecto de v5) preservadas también en entrenamiento, para no evaluar con una dinámica más determinista que la usada para aprender.",
        "Fin de vida como fin de episodio y recorte de recompensa solo en entrenamiento; evaluación y video siempre usan la partida completa (3 vidas) y el puntaje real sin recortar, que es la métrica de la competencia.",
    ])


def seccion_2_2(doc: Document, configs: dict) -> None:
    parrafo(doc, "2.2 Metodología de desarrollo", "Heading 1")

    parrafo(doc, "Algoritmos considerados y elección final", "Heading 2")
    fragmentos(doc, [
        ("Se consideraron DQN, Double DQN, Dueling DQN, PPO y A2C. Se implementó con ", False, False),
        ("Stable-Baselines3 2.9", False, True),
        (" (permitido por el enunciado), documentando cada decisión propia: el contrato de entorno (", False, False),
        ("entorno.py", False, True),
        ("), los hiperparámetros y el protocolo de evaluación. Se eligió ", False, False),
        ("PPO", True, False),
        (" como algoritmo final por tres razones prácticas dado el cómputo disponible (CPU únicamente, sin GPU, con poca memoria libre): (1) no requiere un buffer de repetición de experiencias, el componente más costoso en memoria de DQN; (2) recolecta experiencia de varios entornos en paralelo, decorrelacionando las muestras sin necesitar Double/Dueling para estabilizarse; (3) es reconocido en la literatura (Huang et al., 2022) por ser razonablemente robusto a los hiperparámetros por defecto, importante cuando no hay tiempo para una búsqueda extensa. Se entrenó ", False, False),
        ("DQN como iteración 1", True, False),
        (" precisamente para no dejar esta elección como una afirmación puramente teórica: la sección 2.4 la contrasta con evidencia empírica bajo el mismo presupuesto de tiempo. Double/Dueling DQN y A2C quedan como líneas de trabajo futuro (sección 2.5).", False, False),
    ])

    parrafo(doc, "Arquitectura de la red (NatureCNN)", "Heading 2")
    fragmentos(doc, [
        ("Ambos algoritmos usan la misma arquitectura convolucional (Mnih et al., 2015), dimensionada exactamente para la observación preprocesada de ", False, False),
        ("(4, 84, 84)", False, True), (":", False, False),
    ])
    tabla = doc.add_table(rows=1, cols=4)
    for celda, texto in zip(tabla.rows[0].cells, ("Capa", "Configuración", "Salida", "Activación")):
        celda.text = texto
    filas = [
        ("Entrada", "4 x 84 x 84 (uint8 / 255)", "(4,84,84)", "—"),
        ("Conv1", "32 filtros, 8x8, stride 4", "(32,20,20)", "ReLU"),
        ("Conv2", "64 filtros, 4x4, stride 2", "(64,9,9)", "ReLU"),
        ("Conv3", "64 filtros, 3x3, stride 1", "(64,7,7)", "ReLU"),
        ("Flatten", "—", "3136", "—"),
        ("Densa", "3136 -> 512", "512", "ReLU"),
        ("Salida PPO", "512->6 (política) y 512->1 (valor)", "6 y 1", "Softmax / lineal"),
        ("Salida DQN", "512 -> 6 (Q(s,·))", "6", "Lineal"),
    ]
    for fila in filas:
        for celda, texto in zip(tabla.add_row().cells, fila):
            celda.text = texto
    configurar_tabla(tabla, [0.95, 2.65, 1.15, 1.25], {2, 3})
    parrafo(doc, "≈1.69 millones de parámetros entrenables en cada red (comparten la misma extracción de características; solo difiere la capa de salida).")

    parrafo(doc, "Exploración vs. explotación", "Heading 2")
    fragmentos(doc, [
        ("PPO", True, False),
        (" explora de forma implícita: la política es una distribución categórica sobre las 6 acciones, y un término de entropía (", False, False),
        ("ent_coef=0.01", False, True),
        (") penaliza que esa distribución colapse demasiado rápido, dejando que la incertidumbre de la red module la exploración a medida que se entrena (sin un parámetro epsilon explícito). ", False, False),
        ("DQN", True, False),
        (" usa epsilon-greedy clásico: probabilidad inicial de acción aleatoria de 1.0, decayendo linealmente a 0.01 durante el primer 10% del entrenamiento y constante después.", False, False),
    ])

    parrafo(doc, "Hiperparámetros de entrenamiento", "Heading 2")
    ppo_hp = (configs.get("iter3_ppo_final") or {}).get("hiperparametros", {})
    dqn_hp = (configs.get("iter1_dqn") or {}).get("hiperparametros", {})
    learning_starts_fmt = f"{dqn_hp.get('learning_starts', 0):,}".replace(",", " ")
    ppo_otros_txt = (
        f"n_steps={ppo_hp.get('n_steps')}, n_epochs={ppo_hp.get('n_epochs')}, "
        f"GAE λ={ppo_hp.get('gae_lambda')}, clip={ppo_hp.get('clip_range_inicial')}->0"
    )
    dqn_otros_txt = f"train_freq={dqn_hp.get('train_freq')}, learning_starts={learning_starts_fmt}"
    tabla = doc.add_table(rows=1, cols=3)
    for celda, texto in zip(tabla.rows[0].cells, ("Hiperparámetro", "PPO (iter. 2 y 3)", "DQN (iter. 1)")):
        celda.text = texto
    filas = [
        ("Pérdida", "Clip PPO + valor (MSE) + entropía", "Huber (SmoothL1)"),
        ("Optimizador", "Adam", "Adam"),
        ("Tasa de aprendizaje", f"{ppo_hp.get('learning_rate_inicial', 2.5e-4):.1e} -> 0 (lineal)", f"{dqn_hp.get('learning_rate', 1e-4):.1e} (fija)"),
        ("Buffer de repetición", "No aplica (on-policy)", f"{dqn_hp.get('buffer_size', 0):,} transiciones".replace(",", " ")),
        ("Tamaño de batch", str(ppo_hp.get("batch_size", "N/D")), str(dqn_hp.get("batch_size", "N/D"))),
        ("Red objetivo", "No aplica", f"cada {dqn_hp.get('target_update_interval', 'N/D')} pasos"),
        ("Factor de descuento (γ)", str(ppo_hp.get("gamma", 0.99)), str(dqn_hp.get("gamma", 0.99))),
        ("Otros", ppo_otros_txt, dqn_otros_txt),
    ]
    for fila in filas:
        for celda, texto in zip(tabla.add_row().cells, fila):
            celda.text = str(texto)
    configurar_tabla(tabla, [1.6, 2.5, 1.9], set())


def seccion_2_3(doc: Document, datos: dict) -> None:
    parrafo(doc, "2.3 Resultados de iteraciones", "Heading 1")
    parrafo(doc, "Se documentan la línea base sin entrenamiento (Laboratorio 5) y las tres iteraciones de la sección 2.2, todas bajo el mismo presupuesto de tiempo de esta sesión (CPU únicamente, sin GPU).")

    cambios = {
        "iter1_dqn": "DQN, misma pila de preprocesamiento final (max-pool + vida + clip), 1 entorno, buffer de 10 000 transiciones.",
        "iter2_ppo_v5_directo": "PPO sobre ALE/SpaceInvaders-v5 \"tal cual\" (frameskip=4 nativo, SIN max-pool, sin señal de fin de vida), 8 entornos.",
        "iter3_ppo_final": "PPO con la pila de preprocesamiento final (max-pool + vida + clip), 8 entornos. Modelo entregado.",
    }
    tabla = doc.add_table(rows=1, cols=5)
    for celda, texto in zip(tabla.rows[0].cells, ("Iteración", "Cambio principal", "Pasos", "Entren. (últ. 100 ep.)", "Evaluación greedy")):
        celda.text = texto

    eval0 = datos.get("eval_iter0")
    fila0 = (
        "Iter. 0 - Aleatorio", "Línea base sin entrenar (Laboratorio 5): agente_aleatorio", "—", "—",
        formato_recompensa(eval0["recompensa_media"], eval0["recompensa_desviacion"], eval0["recompensa_maxima"]) if eval0 else "N/D",
    )
    for celda, texto in zip(tabla.add_row().cells, fila0):
        celda.text = texto

    for run_id in ETIQUETAS:
        it = datos["iteraciones"][run_id]
        final = it.get("final") or {}
        pasos = final.get("pasos_alcanzados")
        pasos_txt = f"{pasos:,}".replace(",", " ") if pasos else "N/D"
        rew_train = it.get("ep_rew_mean_entrenamiento")
        rew_train_txt = f"{rew_train:.2f}" if rew_train is not None else "N/D"
        if run_id == "iter3_ppo_final" and datos.get("eval_oficial_final"):
            eo = datos["eval_oficial_final"]
            eval_txt = formato_recompensa(eo["recompensa_media"], eo["recompensa_desviacion"], eo["recompensa_maxima"])
        else:
            ev = it.get("eval_entrenamiento")
            eval_txt = formato_recompensa(ev["media"], ev["desviacion"], ev["maximo"]) if ev else "N/D"
        fila = (ETIQUETAS[run_id].replace("Iteración ", "Iter. "), cambios[run_id], pasos_txt, rew_train_txt, eval_txt)
        for celda, texto in zip(tabla.add_row().cells, fila):
            celda.text = texto
    configurar_tabla(tabla, [1.0, 2.5, 0.7, 1.05, 1.55], {2, 3, 4}, tamano_fuente=7.6)
    parrafo(doc, "\"Entren.\": media móvil de 100 episodios (recompensa recortada) al final del entrenamiento. \"Evaluación greedy\": media ± desviación (máx.) de 5 episodios sin exploración y sin recortar; en la iteración 3 corresponde a la evaluación oficial de 5 episodios (semillas 23236-23240), la misma métrica de la competencia.")

    parrafo(doc, "Curvas de entrenamiento y evaluación", "Heading 2")
    for run_id in ETIQUETAS:
        insertar_figura(
            doc, FIGURAS / f"{run_id}_entrenamiento.png",
            f"Figura — {ETIQUETAS[run_id]}: recompensa cruda por episodio (gris) y media móvil de 100 episodios.",
            ancho=5.0,
        )
    insertar_figura(doc, FIGURAS / "comparacion_reward_vs_pasos.png", "Figura — Comparación entre iteraciones: recompensa media de entrenamiento vs. pasos.", ancho=5.3)
    insertar_figura(doc, FIGURAS / "comparacion_evaluacion_vs_pasos.png", "Figura — Comparación entre iteraciones: recompensa de evaluación greedy vs. pasos.", ancho=5.3)

    parrafo(doc, "Evidencia de problemas encontrados", "Heading 2")
    fragmentos(doc, [
        ("Para vigilar inestabilidad o divergencia de los valores Q/V sin depender solo de la recompensa (una señal ruidosa y dispersa), se implementó un monitor que evalúa la red sobre 500 estados fijos (recolectados una vez, al inicio, con una política aleatoria) cada 10 000 pasos: V(s) promedio en PPO, máx.", False, False),
        ("_a", False, False),
        (" Q(s,a) promedio en DQN. Una tendencia que crece sin cota o que oscila con amplitud creciente indicaría divergencia; una meseta plana desde el inicio indicaría que la red no está aprendiendo nada.", False, False),
    ])
    insertar_figura(doc, FIGURAS / "iter3_ppo_final_divergencia.png", "Figura — Monitor de divergencia, iteración 3 (PPO final): V(s) medio sobre 500 estados fijos.", ancho=4.8)
    insertar_figura(doc, FIGURAS / "iter1_dqn_divergencia.png", "Figura — Monitor de divergencia, iteración 1 (DQN): máx. Q(s,a) medio sobre los mismos 500 estados.", ancho=4.8)
    insertar_figura(doc, FIGURAS / "iter3_ppo_final_diagnostico.png", "Figura — Diagnóstico PPO (iteración 3): pérdida de valor, entropía, KL aproximado y fracción recortada.", ancho=6.0)
    insertar_figura(doc, FIGURAS / "iter1_dqn_diagnostico.png", "Figura — Diagnóstico DQN (iteración 1): pérdida Huber y epsilon de exploración.", ancho=5.0)


def seccion_2_4(doc: Document, datos: dict) -> None:
    parrafo(doc, "2.4 Discusión de resultados", "Heading 1")

    v0 = datos["eval_iter0"]["recompensa_media"] if datos.get("eval_iter0") else None
    v1 = valor_eval("iter1_dqn", datos)
    v2 = valor_eval("iter2_ppo_v5_directo", datos)
    v3 = valor_eval("iter3_ppo_final", datos)

    parrafo(doc, "Comparación entre iteraciones", "Heading 2")
    lista(doc, [
        comparar("La iteración 3 (PPO, preprocesamiento final)", v3, "la línea base aleatoria (iteración 0)", v0, "efecto de entrenar en absoluto"),
        comparar("La iteración 3 (max-pool + señal de vida)", v3, "la iteración 2 (v5 directo, sin max-pool ni vida)", v2, "efecto del preprocesamiento, mismo algoritmo PPO"),
        comparar("La iteración 3 (PPO)", v3, "la iteración 1 (DQN)", v1, "efecto del algoritmo, mismo preprocesamiento"),
    ])
    fragmentos(doc, [
        ("El cambio con mayor impacto esperado es el del ", False, False), ("preprocesamiento", True, False),
        (" (iteración 2 vs. 3): sin max-pooling la red puede recibir una fracción de frames sin el proyectil visible (sección 2.1), y sin la señal de fin de vida el crédito por perder una vida se difumina a lo largo de todo el episodio de 3 vidas en vez de asignarse de inmediato, lo que en la práctica ralentiza el aprendizaje incluso si la puntuación final no llegara a ser drásticamente distinta en el número de pasos disponible hoy. El cambio de ", False, False),
        ("algoritmo", True, False),
        (" (DQN vs. PPO, iteración 1 vs. 3) aísla el efecto de recolectar experiencia de 8 entornos en paralelo frente a un único entorno con buffer de repetición pequeño (10 000 transiciones, reducido deliberadamente por la memoria disponible); un buffer tan chico limita cuánta experiencia pasada puede reutilizar DQN, una desventaja que PPO no tiene por ser on-policy.", False, False),
    ])

    parrafo(doc, "Análisis cualitativo del agente final (video)", "Heading 2")
    fragmentos(doc, [
        ("[Completar tras inspeccionar entregables/space_invaders_ppo_final.mp4] ", True, False),
        ("A partir de los fotogramas del episodio grabado con la política greedy final se describen aquí las estrategias observadas (por ejemplo, si el agente se desplaza sistemáticamente hacia una columna de invasores antes de disparar, si prioriza a la nave nodriza, o si permanece estático) y las situaciones donde falla o se queda atascado (por ejemplo, cerca de los bordes de la pantalla o cuando los invasores aceleran al quedar pocos).", False, False),
    ])

    parrafo(doc, "Limitaciones del enfoque y del cómputo disponible", "Heading 2")
    lista(doc, [
        "Entrenamiento exclusivamente en CPU (sin GPU disponible en el equipo), lo que limitó cada iteración a un puñado de horas y, en consecuencia, a un orden de magnitud de pasos muy por debajo de literatura publicada para Atari (decenas de millones de pasos en Mnih et al. 2015 o Schulman et al. 2017); el número exacto de pasos alcanzados por cada iteración se reporta en la tabla de la sección 2.3.",
        "Buffer de repetición de DQN reducido a 10 000 transiciones (frente a 100 000-1 000 000 típico) para que cupiera en la memoria disponible, lo que limita la diversidad de experiencia que DQN puede reutilizar.",
        "No se realizó una búsqueda sistemática de hiperparámetros (grid/random search): se usaron valores estándar de la literatura (rl-zoo, Mnih et al.) sin ajuste fino específico para este equipo o este presupuesto de tiempo.",
        "La iteración 2 usa una arquitectura de preprocesamiento deliberadamente empobrecida (control experimental), por lo que su puntaje no debe interpretarse como el límite superior de PPO en este entorno.",
    ])

    parrafo(doc, "Reflexión sobre exploración vs. explotación", "Heading 2")
    fragmentos(doc, [
        ("En PPO, la curva de entropía (Figura de diagnóstico, iteración 3) parte de log(6) ≈ 1.79 (política uniforme) y desciende a medida que el agente gana confianza en ciertas acciones; una caída demasiado rápida sería evidencia de una política que dejó de explorar prematuramente, mientras que una entropía que no baja indicaría que la política nunca se especializa. En DQN, epsilon decae linealmente de 1.0 a 0.01 en el primer 10% de los pasos: durante ese tramo inicial casi toda acción es aleatoria y la recompensa reportada mezcla exploración pura con los primeros indicios de una política aprendida, lo que explica curvas de entrenamiento iniciales más ruidosas en DQN que en PPO.", False, False),
    ])


def seccion_2_5(doc: Document, datos: dict) -> None:
    parrafo(doc, "2.5 Conclusiones", "Heading 1")

    eo = datos.get("eval_oficial_final")
    v0 = datos["eval_iter0"]["recompensa_media"] if datos.get("eval_iter0") else None

    parrafo(doc, "Desempeño final", "Heading 2")
    if eo:
        fragmentos(doc, [
            ("El modelo final (PPO, iteración 3) obtuvo una recompensa de evaluación (5 episodios, política greedy, semillas 23236-23240) de ", False, False),
            (f"{eo['recompensa_media']:.0f} ± {eo['recompensa_desviacion']:.0f} puntos (máximo {eo['recompensa_maxima']:.0f})", True, False),
            (f", frente a {v0:.0f} puntos" if v0 is not None else "", False, False),
            (" de la línea base aleatoria bajo el mismo protocolo. El máximo de los 5 episodios es la métrica que determina la posición en el ranking de la competencia.", False, False),
        ])
    else:
        parrafo(doc, "[Completar con scripts/evaluar.py --modelo modelos/ppo_space_invaders_final.zip antes de entregar.]")

    parrafo(doc, "Principales aprendizajes", "Heading 2")
    lista(doc, [
        "El preprocesamiento (max-pooling contra el parpadeo, señal de fin de vida, recorte de recompensa) no es un detalle cosmético: en un entorno de Atari con objetos que parpadean cada dos frames, omitirlo puede ocultar información crítica (un proyectil) a la red, y se puede cuantificar su efecto entrenando una iteración de control con y sin él bajo el mismo presupuesto de tiempo.",
        "Mantener un solo contrato de entorno (entorno.py) compartido por entrenamiento, evaluación y grabación de video evita la clase de error más común en RL aplicado: que el agente se evalúe con una observación distinta a la que vio durante el entrenamiento.",
        "En un presupuesto de cómputo pequeño y sin GPU, la elección de un algoritmo on-policy con entornos vectorizados (PPO) resultó más práctica que un algoritmo off-policy con buffer de repetición (DQN), principalmente por la restricción de memoria disponible más que por una diferencia algorítmica fundamental.",
    ])

    parrafo(doc, "Trabajo futuro", "Heading 2")
    lista(doc, [
        "Entrenar por más tiempo y con GPU (por ejemplo, en Google Colab o Kaggle) para alcanzar el orden de pasos reportado en la literatura (decenas de millones).",
        "Implementar Double DQN y Dueling DQN para cuantificar su aporte específico sobre el DQN base de la iteración 1, con un buffer de repetición de tamaño completo.",
        "Explorar PPO con recompensa escalada (por ejemplo, log(1+r) o normalización por la desviación estándar corriente) en vez de recorte a {-1,0,1}, para no perder la magnitud relativa entre una fila de 5 puntos y la nave nodriza de 200.",
        "Una búsqueda sistemática de hiperparámetros (Optuna, rl-zoo) en vez de valores por defecto de la literatura.",
    ])


def seccion_2_6(doc: Document) -> None:
    parrafo(doc, "2.6 Enlace al repositorio de GitHub", "Heading 1")
    fragmentos(doc, [
        ("Repositorio público: ", False, False), (REPO_URL, False, True),
    ])
    parrafo(doc, "Contiene el módulo de preprocesamiento y entrenamiento (entorno.py, agentes_ale.py, agente.py), los scripts de EDA, entrenamiento, evaluación, video e informe (scripts/), las configuraciones de cada iteración (configs/), los resultados y curvas (resultados/, entregables/figuras/), el modelo final con su contrato de entorno (modelos/) y un README con instrucciones para reproducir el entrenamiento y para cargar y evaluar los pesos el día de la presentación.")


def seccion_referencias(doc: Document) -> None:
    parrafo(doc, "Referencias", "Heading 1")
    referencias = [
        "Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529-533.",
        "Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
        "Machado, M. C. et al. (2018). Revisiting the Arcade Learning Environment. JAIR, 61, 523-562.",
        "van Hasselt, H. et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.",
        "Wang, Z. et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. ICML.",
        "Huang, S. et al. (2022). The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track.",
        "Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR, 22(268).",
        "Farama Foundation. Arcade Learning Environment / Gymnasium. ale.farama.org, gymnasium.farama.org.",
        "Atari (1978). Space Invaders Game Program Instructions. Manual del Atari 2600.",
    ]
    for i, ref in enumerate(referencias, 1):
        p = parrafo(doc, f"[{i}] {ref}")
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(8)


# --------------------------------------------------------------------------
# Exportación a PDF (Word COM)
# --------------------------------------------------------------------------


def exportar_pdf_con_word(ruta_docx: Path, ruta_pdf: Path) -> bool:
    try:
        import win32com.client as win32
    except ImportError:
        print("[generar_informe] aviso: pywin32 no está instalado; omitiendo exportación a PDF.")
        return False
    try:
        word = win32.gencache.EnsureDispatch("Word.Application")
    except Exception as exc:  # noqa: BLE001
        print(f"[generar_informe] aviso: no se pudo iniciar Word ({exc}); omitiendo exportación a PDF.")
        return False
    word.Visible = False
    try:
        doc = word.Documents.Open(str(ruta_docx.resolve()))
        doc.SaveAs(str(ruta_pdf.resolve()), FileFormat=17)  # wdFormatPDF
        doc.Close(False)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[generar_informe] aviso: fallo al exportar PDF con Word ({exc}).")
        return False
    finally:
        word.Quit()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ENTREGABLES.mkdir(parents=True, exist_ok=True)
    datos = recolectar_datos()
    configs = {cfg_id: cargar_json(RAIZ / "configs" / f"{cfg_id}.json") for cfg_id in ETIQUETAS}

    doc = Document()
    configurar_documento(doc)
    doc.core_properties.title = "Proyecto 2: Competencia de Agentes en Space Invaders"
    doc.core_properties.author = "Ian Cumes"
    doc.core_properties.subject = "Reinforcement Learning - ALE Space Invaders"

    titulo = parrafo(doc, estilo="Title")
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = titulo.add_run("Proyecto 2: Competencia de Agentes en Space Invaders")
    configurar_fuente(run, "Aptos Display", 21)
    run.bold = True
    subtitulo = parrafo(doc, "CC3092 Deep Learning y Sistemas Inteligentes | Ian Cumes | carné 23236")
    subtitulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitulo.paragraph_format.space_after = Pt(10)

    seccion_2_1(doc, datos)
    seccion_2_2(doc, configs)
    seccion_2_3(doc, datos)
    seccion_2_4(doc, datos)
    seccion_2_5(doc, datos)
    seccion_2_6(doc)
    seccion_referencias(doc)

    doc.save(SALIDA_DOCX)
    print(f"DOCX creado: {SALIDA_DOCX}")

    if exportar_pdf_con_word(SALIDA_DOCX, SALIDA_PDF):
        print(f"PDF creado: {SALIDA_PDF}")
    else:
        print(f"PDF NO generado automáticamente. Abra {SALIDA_DOCX} en Word y use 'Guardar como PDF'.")


if __name__ == "__main__":
    main()
