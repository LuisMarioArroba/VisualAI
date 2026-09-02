"""
interpretacion_arte.py
Modulo compartido: todo el vocabulario y la logica que traduce metricas
tecnicas (delta-E, SSIM, entropia, coincidencia de bordes) a explicaciones
en terminos de oficio pictorico y teoria del arte.

Lo usan:
  - reporte_educativo.py     (explica obras YA etiquetadas del dataset)
  - inferencia_obra.py       (explica una foto NUEVA que sube un alumno)

Asi el vocabulario y los criterios son siempre los mismos en ambos lados.
"""
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab

# ==========================================================
# 0. ESTRUCTURA DE LAS 5 CAPAS
# ==========================================================

LAYER_INFO = [
    ("lab",     "Espacio de color",        "01_Lab",         "_lab.png"),
    ("quant",   "Paleta simplificada",     "02_Quantization", "_quant.png"),
    ("entropy", "Mapa de detalle",         "03_Entropy",      "_entropy.png"),
    ("brush",   "Simulacion de pincelada", "04_Brush",        "_brush.png"),
    ("edges",   "Contornos",               "05_Edges",        "_edges.png"),
]

LAYER_METRICS = {
    "lab":     ["lab_deltaE", "lab_ssim", "lab_entropy", "lab_edge"],
    "quant":   ["quant_deltaE", "quant_ssim", "quant_entropy", "quant_edge"],
    "entropy": ["entropy_deltaE", "entropy_ssim", "entropy_entropy", "entropy_edge"],
    "brush":   ["brush_deltaE", "brush_ssim", "brush_entropy", "brush_edge"],
    "edges":   ["edges_deltaE", "edges_ssim", "edges_entropy", "edges_edge"],
}

# Por que esta capa importa, explicado para alguien que viene del taller,
# no del laboratorio. Se muestra como introduccion de cada capa en el reporte.
GLOSARIO_DESC = {
    "lab": (
        "Aisla el color de la forma. Sirve para ver si una obra se sostiene "
        "por el color en si mismo (como en el impresionismo) o si necesita "
        "del dibujo para leerse (como en el realismo academico)."
    ),
    "quant": (
        "Reduce la obra a un puñado de colores dominantes. Sirve para ver "
        "que tan rica o que tan acotada es la paleta real que uso el pintor, "
        "mas alla de los miles de tonos que percibe el ojo."
    ),
    "entropy": (
        "Mide en donde se concentra la complejidad visual. Sirve para ver "
        "si el pintor reparte el detalle parejo por toda la tela o lo "
        "concentra en un foco (un rostro, un objeto) y aboceta el resto."
    ),
    "brush": (
        "Simula como se veria la superficie si se resaltara la textura del "
        "trazo. Sirve para distinguir una pincelada visible y gestual "
        "(impresionismo, post-impresionismo) de un acabado liso y pulido "
        "(realismo academico)."
    ),
    "edges": (
        "Detecta el dibujo que sostiene la escena. Sirve para ver si la "
        "obra depende de un contorno firme y preciso, o si los bordes se "
        "disuelven porque la forma se construye con manchas de color."
    ),
}

TOP_K_RAZONES = 3

# ==========================================================
# 1. DICCIONARIO DE INTERPRETACION PICTORICA
# Cada metrica tecnica se traduce a un rasgo que un pintor reconoce en su
# propio oficio, segun si la obra esta por encima ("alto") o por debajo
# ("bajo") del promedio del dataset.
# ==========================================================

ART_INTERPRETATION = {
    "lab_deltaE": {
        "alto": "transiciones de color abruptas, con saltos de contraste en vez de gradaciones suaves",
        "bajo": "transiciones de color suaves y graduales, casi sin saltos de contraste",
    },
    "lab_ssim": {
        "alto": "una figura que se reconoce aunque se aisle solo el color: es el color mismo el que construye la forma",
        "bajo": "una figura que depende del dibujo y del contorno para leerse, mas que del color",
    },
    "lab_entropy": {
        "alto": "una modulacion cromatica rica dentro de cada zona, el llamado 'color roto': muchos matices conviviendo en una misma area",
        "bajo": "zonas de color planas y uniformes, con poca variacion tonal interna",
    },
    "lab_edge": {
        "alto": "bordes que nacen del propio contraste entre colores, sin necesidad de una linea que los delimite",
        "bajo": "bordes que no bastan por si solos: necesitan una linea o contorno explicito para existir",
    },
    "quant_deltaE": {
        "alto": "una paleta amplia y variada, que se transforma notablemente en cuanto se reduce el numero de colores",
        "bajo": "una paleta ya de por si acotada, cercana a la de alguien que trabaja con pocos pigmentos",
    },
    "quant_ssim": {
        "alto": "una composicion que se sostiene aunque se simplifique drasticamente el color: es la estructura y el dibujo lo que ordena la escena",
        "bajo": "una composicion que pierde legibilidad al simplificar el color: es el color, no el dibujo, lo que sostiene la forma",
    },
    "quant_entropy": {
        "alto": "una gama de colores dominantes amplia y diversa",
        "bajo": "un numero reducido de colores dominantes, una paleta mas restringida",
    },
    "quant_edge": {
        "alto": "contornos dibujados con firmeza, que sobreviven aunque se simplifique el color",
        "bajo": "contornos que dependen del color para existir y se difuminan en cuanto este se simplifica",
    },
    "entropy_deltaE": {
        "alto": "las zonas de mayor detalle son tambien las de mayor contraste de toda la obra",
        "bajo": "el detalle no coincide necesariamente con las zonas de mas contraste",
    },
    "entropy_ssim": {
        "alto": "la estructura de la composicion se mantiene clara incluso en las zonas mas trabajadas",
        "bajo": "las zonas de mayor detalle tienden a desdibujar la estructura general de la composicion",
    },
    "entropy_entropy": {
        "alto": "el detalle se concentra en zonas puntuales (un rostro, un objeto central) mientras el resto se resuelve de forma mas abocetada",
        "bajo": "el nivel de detalle esta repartido de manera pareja en toda la superficie",
    },
    "entropy_edge": {
        "alto": "el detalle coincide con los contornos: el dibujo y la textura trabajan juntos",
        "bajo": "el detalle es mas disperso y atmosferico, sin anclarse estrictamente en los contornos",
    },
    "brush_deltaE": {
        "alto": "pinceladas con fuerte contraste entre si, visibles como marcas individuales sobre la superficie",
        "bajo": "pinceladas con poco contraste entre ellas, fundidas en una superficie mas continua",
    },
    "brush_ssim": {
        "alto": "la composicion conserva su claridad aunque se simule la textura de la pincelada: el dibujo de base es solido",
        "bajo": "la textura de la pincelada domina sobre la forma, propio de una ejecucion mas gestual y suelta",
    },
    "brush_entropy": {
        "alto": "una pincelada rica y variada, de trazo suelto y visible: la mano del pintor queda a la vista",
        "bajo": "una pincelada uniforme y controlada, casi imperceptible, propia de un acabado pulido",
    },
    "brush_edge": {
        "alto": "los bordes se mantienen firmes incluso despues de simular la textura de la pincelada",
        "bajo": "los bordes se disuelven bajo la textura de la pincelada: la forma se funde con el trazo",
    },
    "edges_deltaE": {
        "alto": "contornos de fuerte contraste, marcados con claridad",
        "bajo": "contornos suaves y de bajo contraste, mas sugeridos que dibujados",
    },
    "edges_ssim": {
        "alto": "una red de contornos que reproduce con fidelidad la estructura de la obra original",
        "bajo": "contornos que se alejan de la estructura literal de la obra, mas selectivos o esquematicos",
    },
    "edges_entropy": {
        "alto": "una densidad alta de lineas y contornos, una composicion apoyada en la definicion lineal",
        "bajo": "pocos contornos dominantes, una composicion que se apoya mas en masas de color que en lineas",
    },
    "edges_edge": {
        "alto": "una coincidencia muy precisa entre los contornos detectados y la obra original: un dibujo firme y exacto",
        "bajo": "contornos mas sueltos o aproximados respecto de la obra original",
    },
}

# ==========================================================
# 2. FICHA DE CADA CORRIENTE
# Texto largo (para cerrar la explicacion de una obra puntual) y señas de
# identidad cortas (para la ficha de referencia al inicio de cada seccion).
# ==========================================================

MOVEMENT_KEYNOTES = {
    "impresionismo": (
        "el impresionismo privilegia la pincelada visible y el color roto por sobre el dibujo preciso, "
        "buscando capturar la luz y la sensacion del instante mas que el contorno exacto de las cosas."
    ),
    "realismo": (
        "el realismo se apoya en un dibujo firme, contornos precisos y una paleta mas contenida, "
        "priorizando la fidelidad descriptiva de la escena por sobre el gesto pictorico."
    ),
    "postimpresionismo": (
        "el postimpresionismo retoma el color y la pincelada del impresionismo pero los pone al servicio "
        "de una estructura o una carga expresiva mas deliberada, ya sea por construccion geometrica o por trazo emocional."
    ),
    "simbolismo": (
        "el simbolismo subordina la fidelidad visual a la sugestion: color y forma funcionan como signos "
        "de una idea o un estado de animo, mas que como descripcion literal del mundo visible."
    ),
}

MOVEMENT_SENAS = {
    "impresionismo": [
        "Pincelada visible, suelta, a menudo empastada",
        "Color roto: manchas de color puro yuxtapuestas, no mezcladas de antemano en la paleta",
        "Contornos disueltos por la luz, poco dibujo lineal explicito",
        "Interes en el instante, la luz cambiante y la pintura al aire libre",
    ],
    "realismo": [
        "Pincelada controlada, poco visible, acabado pulido",
        "Paleta mas terrosa y contenida, colores mezclados con cuidado",
        "Contornos y dibujo firmes, precision descriptiva",
        "Interes en representar la vida cotidiana tal como es, sin idealizarla",
    ],
    "postimpresionismo": [
        "Pincelada estructurada (por ejemplo Cezanne, Seurat) o gestual y expresiva (por ejemplo Van Gogh)",
        "Color intensificado, a veces simbolico mas que naturalista",
        "Formas que pueden simplificarse o distorsionarse de forma deliberada",
        "Interes en la construccion formal o en la carga emocional, mas alla de la impresion optica",
    ],
    "simbolismo": [
        "Pincelada al servicio de la atmosfera, no del registro optico fiel",
        "Color subjetivo, onirico, cargado de connotacion",
        "Formas estilizadas o distorsionadas para sugerir una idea",
        "Interes en lo psicologico, lo mitico o lo espiritual, no en la descripcion del mundo visible",
    ],
}


# ==========================================================
# 3. EXPLICACION POR CAPA (dos rasgos: el mas marcado y el segundo)
# ==========================================================

def layer_caption(row, layer_key, global_mean, global_std, incluir_intro=True):
    """
    Explica una capa con los DOS rasgos pictoricos mas marcados de esa obra
    en esa capa (antes solo se mostraba uno), sin exponer numeros. Los
    z-scores se usan solo por dentro para decidir que rasgos destacar.
    """
    metrics = LAYER_METRICS[layer_key]
    zscores = {}
    for feat in metrics:
        valor = row[feat]
        prom = global_mean[feat]
        z = (valor - prom) / global_std[feat]
        zscores[feat] = z

    orden = sorted(metrics, key=lambda f: -abs(zscores[f]))

    frases = []
    for feat in orden[:2]:
        z = zscores[feat]
        direccion = "alto" if z > 0 else "bajo"
        frase = ART_INTERPRETATION.get(feat, {}).get(direccion)
        if not frase:
            continue
        intensidad = "marcadamente " if abs(z) > 1 else ("levemente " if abs(z) < 0.3 else "")
        frases.append(f"{intensidad}{frase}")

    if not frases:
        cuerpo = "un comportamiento cercano al promedio del conjunto en esta capa."
    elif len(frases) == 1:
        cuerpo = frases[0].capitalize() + "."
    else:
        cuerpo = (frases[0].capitalize() + ". Ademas, " + frases[1] + ".")

    intro = (GLOSARIO_DESC.get(layer_key, "") + " ") if incluir_intro else ""
    return intro + cuerpo


# ==========================================================
# 4. EXPLICACION GENERAL DE LA OBRA
# Version generalizada: no depende de tener la etiqueta real (movimiento
# verdadero). Sirve tanto para obras del dataset (con etiqueta) como para
# una foto nueva que solo tiene movimiento PREDICHO.
# ==========================================================

def _distancias_a_movimientos(row_features_z, movement_profile_z, feature_cols):
    dists = ((movement_profile_z[feature_cols] - row_features_z[feature_cols]) ** 2).sum(axis=1) ** 0.5
    return dists.sort_values()


def _confianza(ranking_mov):
    """Qualifica que tan clara es la cercania al primer movimiento vs. el segundo."""
    valores = ranking_mov.values
    if len(valores) < 2 or valores[0] == 0:
        return "muy clara"
    brecha_relativa = (valores[1] - valores[0]) / valores[1]
    if brecha_relativa > 0.35:
        return "muy clara"
    elif brecha_relativa > 0.15:
        return "clara"
    elif brecha_relativa > 0.05:
        return "moderada"
    else:
        return "ambigua (el perfil queda a medio camino entre dos corrientes)"


def explain_features(z, feature_cols, movement_profile_z, ranking, movement_predicho, movement_real=None):
    """
    z: pandas Series de z-scores (una obra) ya calculada por el llamador.
    ranking: pandas Series (feature -> f-value) que ordena que features
        son mas discriminantes entre movimientos (viene del bundle entrenado).
    movement_predicho: el movimiento que el modelo asigna a esta obra.
    movement_real: si se conoce (obras del dataset de entrenamiento), se
        usa para comentar coincidencias/discrepancias; si es None (foto
        nueva de un alumno) se omite esa parte.

    Devuelve (parrafo_html, ficha_comparativa_html).
    """
    top_feats = sorted(ranking.index.tolist(), key=lambda f: -abs(z[f]))[:TOP_K_RAZONES]

    frases = []
    for feat in top_feats:
        direccion = "alto" if z[feat] > 0 else "bajo"
        frase = ART_INTERPRETATION.get(feat, {}).get(direccion)
        if frase:
            frases.append(frase)

    if not frases:
        razon = "un perfil visual sin rasgos que se aparten demasiado del promedio del conjunto"
    elif len(frases) == 1:
        razon = frases[0]
    else:
        razon = "; ademas, ".join(frases[:-1]) + "; y tambien " + frases[-1]

    ranking_mov = _distancias_a_movimientos(z, movement_profile_z, feature_cols)
    mas_parecido = ranking_mov.index[0]
    mas_distinto = ranking_mov.index[-1]
    confianza = _confianza(ranking_mov)

    if movement_real is not None:
        apertura = f"Esta obra fue catalogada como <b>{movement_real}</b>. "
    else:
        apertura = f"El modelo clasifica esta obra como <b>{movement_predicho}</b>. "

    parrafo = (
        f"{apertura}"
        f"Lo que mas define su lenguaje visual es {razon}. "
        f"Comparando ese comportamiento con el perfil tipico de cada corriente en este conjunto de obras, "
        f"el resultado se acerca mas a <b>{mas_parecido}</b> y se aleja mas de <b>{mas_distinto}</b> "
        f"(la cercania al movimiento mas afin es {confianza}). "
    )

    movimiento_para_cerrar = mas_parecido
    if movement_real is not None and mas_parecido != movement_real:
        parrafo += (
            f"Vale la pena notarlo: aunque la obra pertenece a {movement_real}, su color, su pincelada y su dibujo "
            f"se comportan de forma mas cercana a lo que suele caracterizar al {mas_parecido} — "
            f"{MOVEMENT_KEYNOTES.get(mas_parecido, '')}"
        )
    elif movement_real is not None:
        parrafo += (
            f"Esto coincide con lo que la historia del arte asocia al {movement_real}: "
            f"{MOVEMENT_KEYNOTES.get(movement_real, '')}"
        )
    else:
        parrafo += (
            f"Esto es consistente con lo que caracteriza al {mas_parecido}: "
            f"{MOVEMENT_KEYNOTES.get(mas_parecido, '')}"
        )

    # Ficha comparativa: que tan cerca queda de CADA corriente, no solo la mas y la menos parecida.
    orden_texto = " &middot; ".join(
        f"{mov}: {'el mas cercano' if mov == mas_parecido else ('el mas lejano' if mov == mas_distinto else 'intermedio')}"
        for mov in ranking_mov.index
    )
    ficha = f'<p class="ficha-comparativa"><i>Cercania a cada corriente (de mas a menos afin): {orden_texto}.</i></p>'

    return parrafo, ficha


def build_movement_ficha(movement):
    """Ficha corta de 'señas de identidad' para mostrar al inicio de cada seccion/corriente."""
    senas = "".join(f"<li>{s}</li>" for s in MOVEMENT_SENAS.get(movement, []))
    keynote = MOVEMENT_KEYNOTES.get(movement, "")
    return (
        f'<div class="mov-ficha">'
        f'<p>{keynote.capitalize()}</p>'
        f'<p><b>Señas de identidad a buscar en la obra:</b></p>'
        f'<ul>{senas}</ul>'
        f'</div>'
    )


# ==========================================================
# 5. COLOROMETRIA
# Todo lo que faltaba: colores dominantes, temperatura (calido/frio/
# neutro), balance cromatico, peso visual y su relacion con el
# movimiento detectado.
#
# No hace falta tocar el pipeline que genera las capas: la imagen de la
# capa "02_Quantization" ya fue reducida por k-means en espacio Lab a un
# puñado de colores. Aqui simplemente se cuentan los pixeles de cada
# color unico en esa imagen -> eso da a la vez la proporcion de cada
# color Y su ubicacion (fila/columna), sin tener que reabrir el kmeans
# original ni tocar el script que genera las capas.
#
# La redaccion de este bloque sigue el mismo criterio que el resto del
# modulo: el vocabulario tecnico (angulo de tono, espacio Lab, croma) se
# usa solo por dentro para calcular; lo que se muestra en el reporte se
# traduce siempre a terminos de taller (calido/frio, dominante/secundario,
# saturado/apagado), sin exponer numeros ni nombres de espacios de color.
# ==========================================================

UMBRAL_CHROMA_NEUTRO = 8     # por debajo de este croma en Lab, se considera "neutro" (gris)
PROPORCION_MINIMA = 0.03     # colores que ocupan menos de este % de la obra se ignoran (ruido de cuantizacion)

TEORIA_COLOR = {
    "temperatura": (
        "La division entre colores calidos y frios sigue el criterio clasico del circulo cromatico de Itten: "
        "una diagonal imaginaria que va del amarillo-verdoso al rojo-violeta separa el semicirculo calido "
        "(rojo, naranja, amarillo) del semicirculo frio (verde, azul, violeta). Aqui se estima hacia que lado "
        "de esa rueda se inclina cada color dominante de la obra."
    ),
    "balance": (
        "El balance cromatico se apoya en la ley del contraste simultaneo de Chevreul: dos colores de tono opuesto "
        "(complementarios) se intensifican mutuamente al aparecer juntos, mientras que tonos cercanos en el "
        "circulo generan armonias mas estables. Aqui se estima comparando que tan cerca o que tan opuestos son "
        "los colores dominantes en esa misma rueda cromatica, y que tan repartido o concentrado esta el peso de cada uno."
    ),
    "peso_visual": (
        "El peso visual retoma la idea, desarrollada por Arnheim al analizar la composicion pictorica, de que un "
        "elemento pesa mas en la balanza compositiva cuanto mas oscuro, mas saturado y mas alejado del centro "
        "geometrico se encuentra. Aqui se aproxima combinando la proporcion de area, la luminosidad y la "
        "saturacion de cada color dominante, junto con su posicion respecto del centro del lienzo."
    ),
}

# Comportamiento cromatico esperado de cada corriente, para contrastarlo con lo medido en la obra.
COLOR_MOVIMIENTO = {
    "impresionismo": (
        "colores puros yuxtapuestos sin mezclar demasiado en la paleta, contrastes calido-frio marcados "
        "(sombras azuladas o violaceas junto a luces calidas) y una saturacion media-alta, con muy poco uso del negro"
    ),
    "realismo": (
        "una paleta terrosa y de saturacion baja, dominada por ocres, pardos y grises, con contrastes calido-frio "
        "suaves y un uso deliberado de negros y sombras oscuras para modelar el volumen"
    ),
    "postimpresionismo": (
        "colores intensificados y a veces no naturalistas, de saturacion alta, que pueden mostrar una fuerte "
        "dominancia calida y gestual (como en Van Gogh) o construirse a partir de pequeños contrastes "
        "complementarios yuxtapuestos (como en el puntillismo de Seurat)"
    ),
    "simbolismo": (
        "una paleta subjetiva y de saturacion media-baja, con tendencia a violetas, azules y verdes oniricos: "
        "una temperatura fria o ambigua que refuerza la atmosfera psicologica antes que la descripcion literal"
    ),
}

_NOMBRES_HUE = ["rojo", "naranja", "amarillo", "verde-amarillento", "verde", "azul-verdoso", "azul", "violeta"]


def _lab_hue_angle(a, b):
    """Angulo de tono en el plano a*-b* de CIELAB, en grados [0, 360)."""
    return math.degrees(math.atan2(b, a)) % 360


def _lab_chroma(a, b):
    """Croma (saturacion) en CIELAB: distancia al origen en el plano a*-b*."""
    return math.sqrt(a ** 2 + b ** 2)


def clasificar_temperatura(a, b, umbral_chroma=UMBRAL_CHROMA_NEUTRO):
    """Devuelve 'calido', 'frio' o 'neutro' segun el criterio de Itten (ver TEORIA_COLOR['temperatura'])."""
    chroma = _lab_chroma(a, b)
    if chroma < umbral_chroma:
        return "neutro"
    hue = _lab_hue_angle(a, b)
    # Semicirculo frio: de amarillo-verde (135) a rojo-violeta (315). El resto es calido.
    if 135 <= hue < 315:
        return "frio"
    return "calido"


def nombrar_color(L, a, b, umbral_chroma=UMBRAL_CHROMA_NEUTRO):
    """Nombre aproximado del color a partir de sus coordenadas Lab, para uso en el reporte."""
    chroma = _lab_chroma(a, b)
    if chroma < umbral_chroma:
        if L > 85:
            return "blanco"
        elif L > 60:
            return "gris claro"
        elif L > 35:
            return "gris medio"
        elif L > 15:
            return "gris oscuro"
        else:
            return "negro"

    hue = _lab_hue_angle(a, b)
    base = _NOMBRES_HUE[int(((hue + 22.5) % 360) // 45)]
    partes = [base]
    if chroma > 45:
        partes.append("vivo")
    elif chroma < 20:
        partes.append("apagado")
    if L > 70:
        partes.append("claro")
    elif L < 30:
        partes.append("oscuro")
    return " ".join(partes)


def extraer_paleta_dominante(quant_image_path, proporcion_minima=PROPORCION_MINIMA):
    """
    Lee la imagen ya cuantizada (capa 02_Quantization) y agrupa sus pixeles
    por color exacto. Como esa imagen ya fue reducida a un puñado de
    colores por k-means en Lab, no hace falta un segundo clustering: basta
    con contar pixeles por color unico para obtener, de una sola pasada,
    tanto la proporcion de cada color como su ubicacion en el lienzo.

    Devuelve una lista de dicts ordenada de mayor a menor proporcion:
      {rgb, L, a, b, proporcion, centro_xy: (x, y) normalizado 0-1}
    """
    img = Image.open(quant_image_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    pixeles = arr.reshape(-1, 3)
    total = pixeles.shape[0]

    colores_unicos, inverso, conteos = np.unique(pixeles, axis=0, return_inverse=True, return_counts=True)
    lab_unicos = rgb2lab(colores_unicos.reshape(1, -1, 3) / 255.0).reshape(-1, 3)

    filas, cols = np.indices((h, w))
    filas = filas.reshape(-1)
    cols = cols.reshape(-1)
    inverso = inverso.reshape(-1)

    paleta = []
    for i, (rgb, lab, cuenta) in enumerate(zip(colores_unicos, lab_unicos, conteos)):
        proporcion = cuenta / total
        if proporcion < proporcion_minima:
            continue
        mask = inverso == i
        paleta.append({
            "rgb": tuple(int(c) for c in rgb),
            "L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2]),
            "proporcion": float(proporcion),
            "centro_xy": (float(cols[mask].mean() / w), float(filas[mask].mean() / h)),
        })

    paleta.sort(key=lambda c: -c["proporcion"])
    return paleta


def describir_paleta_dominante(paleta, top_n=5):
    """Identificacion especifica de los colores dominantes, con su nombre aproximado y su peso en la obra."""
    if not paleta:
        return "No se identificaron colores dominantes claros en esta obra."

    items = []
    for c in paleta[:top_n]:
        nombre = nombrar_color(c["L"], c["a"], c["b"])
        temp = clasificar_temperatura(c["a"], c["b"])
        items.append(f"{nombre} ({temp}, {c['proporcion'] * 100:.0f}% de la superficie)")

    if len(items) == 1:
        listado = items[0]
    else:
        listado = ", ".join(items[:-1]) + " y " + items[-1]

    pct_calido = sum(c["proporcion"] for c in paleta if clasificar_temperatura(c["a"], c["b"]) == "calido") * 100
    pct_frio = sum(c["proporcion"] for c in paleta if clasificar_temperatura(c["a"], c["b"]) == "frio") * 100
    pct_neutro = sum(c["proporcion"] for c in paleta if clasificar_temperatura(c["a"], c["b"]) == "neutro") * 100

    return (
        f"{TEORIA_COLOR['temperatura']} "
        f"La paleta dominante de esta obra esta compuesta por {listado}. "
        f"En conjunto, los tonos calidos ocupan aproximadamente el {pct_calido:.0f}% de la superficie analizada, "
        f"los frios el {pct_frio:.0f}% y los neutros (grises, blancos, negros) el {pct_neutro:.0f}%."
    )


def describir_balance_cromatico(paleta):
    """Balance cromatico: relacion de dominancia y de tono entre los colores principales."""
    if len(paleta) < 2:
        return (
            "La obra esta dominada casi por completo por un unico color, sin un segundo tono que compita "
            "en peso visual con el: un balance cromatico extremadamente polarizado."
        )

    principal, segundo = paleta[0], paleta[1]
    dominancia = principal["proporcion"] - segundo["proporcion"]

    if dominancia > 0.3:
        forma = "claramente polarizada, con un color que domina sobre el resto"
    elif dominancia > 0.1:
        forma = "moderadamente jerarquizada, con un color principal que sobresale sin eliminar a los demas"
    else:
        forma = "repartida entre varios colores de peso similar, sin una unica nota dominante"

    hue1 = _lab_hue_angle(principal["a"], principal["b"])
    hue2 = _lab_hue_angle(segundo["a"], segundo["b"])
    diff = abs(hue1 - hue2)
    diff = min(diff, 360 - diff)

    cromaticos = (
        clasificar_temperatura(principal["a"], principal["b"]) != "neutro"
        and clasificar_temperatura(segundo["a"], segundo["b"]) != "neutro"
    )
    if not cromaticos:
        relacion = "con al menos uno de los dos colores principales de caracter neutro, lo que amortigua el contraste entre ellos"
    elif diff > 150:
        relacion = "en una relacion cercana a la complementariedad, el tipo de contraste que mas intensifica cada color por efecto del contraste simultaneo"
    elif diff < 60:
        relacion = "en una relacion analoga, de tonos cercanos entre si, lo que genera una armonia mas estable y menos tensa"
    else:
        relacion = "en una relacion de contraste moderado, ni claramente complementaria ni claramente analoga"

    return (
        f"{TEORIA_COLOR['balance']} "
        f"En esta obra, la relacion entre los colores dominantes es {forma}, y los dos colores principales se ubican {relacion}."
    )


def describir_peso_visual(paleta):
    """Peso visual y distribucion de los elementos en la composicion, a partir de la ubicacion de cada color en el lienzo."""
    if not paleta:
        return "No fue posible estimar el peso visual de la composicion por falta de una paleta dominante."

    pesos = []
    for c in paleta:
        chroma_norm = min(_lab_chroma(c["a"], c["b"]) / 60, 1.0)
        oscuridad = 1 - (c["L"] / 100)
        multiplicador_calido = 1.1 if clasificar_temperatura(c["a"], c["b"]) == "calido" else 1.0
        peso = c["proporcion"] * (0.5 + 0.3 * oscuridad + 0.2 * chroma_norm) * multiplicador_calido
        pesos.append(peso)

    peso_total = sum(pesos) or 1.0
    idx_max = max(range(len(paleta)), key=lambda i: pesos[i])
    color_pesado = paleta[idx_max]
    nombre_pesado = nombrar_color(color_pesado["L"], color_pesado["a"], color_pesado["b"])

    cx, cy = color_pesado["centro_xy"]
    horiz = "izquierda" if cx < 0.4 else ("derecha" if cx > 0.6 else "centro")
    vert = "superior" if cy < 0.4 else ("inferior" if cy > 0.6 else "central")
    if horiz == "centro" and vert == "central":
        ubicacion = "en la zona central del lienzo"
    else:
        ubicacion = f"hacia la zona {vert} {'del lienzo' if horiz == 'centro' else horiz + ' del lienzo'}"

    cx_masa = sum(p * c["centro_xy"][0] for p, c in zip(pesos, paleta)) / peso_total
    cy_masa = sum(p * c["centro_xy"][1] for p, c in zip(pesos, paleta)) / peso_total
    desplazamiento = math.sqrt((cx_masa - 0.5) ** 2 + (cy_masa - 0.5) ** 2)

    if desplazamiento < 0.08:
        equilibrio = "el centro de masa cromatico queda muy cerca del centro geometrico del lienzo, lo que produce una composicion visualmente equilibrada"
    elif desplazamiento < 0.18:
        equilibrio = "el centro de masa cromatico se desplaza levemente del centro geometrico, generando una composicion con una leve tension direccional"
    else:
        equilibrio = "el centro de masa cromatico se desplaza con claridad del centro geometrico, generando una composicion asimetrica con una tension marcada hacia un lado del lienzo"

    return (
        f"{TEORIA_COLOR['peso_visual']} "
        f"El color que mas peso visual aporta a esta obra es el {nombre_pesado}, ubicado {ubicacion}. "
        f"A nivel global, {equilibrio}."
    )


def explicar_color_por_movimiento(paleta, movimiento):
    """Relaciona lo medido en la paleta con el comportamiento cromatico esperado del movimiento detectado."""
    referencia = COLOR_MOVIMIENTO.get(movimiento)
    if not referencia or not paleta:
        return ""

    chroma_prom = sum(c["proporcion"] * _lab_chroma(c["a"], c["b"]) for c in paleta)
    saturacion_txt = "alta" if chroma_prom > 40 else ("media" if chroma_prom > 20 else "baja")

    return (
        f"En el {movimiento}, suele predominar {referencia}. "
        f"En esta obra la saturacion cromatica promedio de la paleta (ponderada por el area que ocupa cada color) "
        f"es {saturacion_txt}, un dato que conviene contrastar con esa caracterizacion tipica de la corriente "
        f"al momento de justificar la clasificacion."
    )


def color_caption(quant_image_path, movement_predicho=None, proporcion_minima=PROPORCION_MINIMA):
    """
    Punto de entrada unico para el bloque de colorometria del reporte.
    Se apoya en la misma imagen que ya usa la capa 'quant' (LAYER_INFO),
    asi que puede insertarse junto a esa capa sin cambios en el pipeline
    de generacion de imagenes.
    """
    paleta = extraer_paleta_dominante(quant_image_path, proporcion_minima)
    if not paleta:
        return "<p>No fue posible extraer una paleta de color representativa de esta obra.</p>"

    partes = [
        describir_paleta_dominante(paleta),
        describir_balance_cromatico(paleta),
        describir_peso_visual(paleta),
    ]
    if movement_predicho:
        extra = explicar_color_por_movimiento(paleta, movement_predicho)
        if extra:
            partes.append(extra)

    cuerpo = " ".join(f"<p>{p}</p>" for p in partes)
    return f'<div class="colorometria">{cuerpo}</div>'