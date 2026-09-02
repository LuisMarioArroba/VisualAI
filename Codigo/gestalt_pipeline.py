"""
gestalt_pipeline.py
Nucleo del pipeline "Gestalt Perceptual Pipeline" (color Lab, cuantizacion,
entropia, simulacion de pincelada, bordes) extraido tal cual del script de
entrenamiento, para poder reusarlo tanto ahi como en la inferencia sobre
una foto nueva.

IMPORTANTE: K_COLORS, BLOCK_SIZE y los parametros de Canny/bilateralFilter
deben quedar IDENTICOS a los que se usaron para generar metrics.csv y
entrenar el modelo. Si los cambias, las metricas de las fotos nuevas dejan
de ser comparables con el dataset de entrenamiento.

Requiere:
    pip install -q opencv-python scikit-image numpy
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

K_COLORS = 4
BLOCK_SIZE = 16


# ==========================================================
# ENTROPÍA LOCAL
# ==========================================================

def entropy_map(gray):

    gray = gray.astype(np.float32)

    mean = cv2.blur(gray, (9, 9))

    diff = (gray - mean) ** 2

    ent = cv2.blur(diff, (9, 9))

    ent /= (np.max(ent) + 1e-6)

    return ent


# ==========================================================
# ENTROPÍA SHANNON
# ==========================================================

def shannon_entropy(gray):

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    hist = hist / (hist.sum() + 1e-6)

    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))


# ==========================================================
# CUANTIZACIÓN LAB
# ==========================================================

def lab_quantization(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        K_COLORS,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    quant = centers[labels.flatten()]

    quant = quant.reshape(lab.shape)

    return lab, quant


# ==========================================================
# BRUSH STROKES
# ==========================================================

def brush_strokes(img, entropy):

    h, w = img.shape[:2]

    output = np.zeros_like(img)

    for y in range(0, h, BLOCK_SIZE):

        for x in range(0, w, BLOCK_SIZE):

            y2 = min(y + BLOCK_SIZE, h)

            x2 = min(x + BLOCK_SIZE, w)

            block = img[y:y2, x:x2]

            e = np.mean(entropy[y:y2, x:x2])

            k = int(3 + (1 - e) * 11)

            if k % 2 == 0:
                k += 1

            if k < 3:
                k = 3

            output[y:y2, x:x2] = cv2.bilateralFilter(
                block,
                d=k,
                sigmaColor=75,
                sigmaSpace=75
            )

    return output


# ==========================================================
# BORDES
# ==========================================================

def edge_map(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 150)

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, 1)

    return edges


# ==========================================================
# MÉTRICAS
# ==========================================================

def edge_score(original, transformed):

    e1 = cv2.Canny(original, 80, 150)

    e2 = cv2.Canny(transformed, 80, 150)

    inter = np.sum((e1 > 0) & (e2 > 0))

    union = np.sum(e1 > 0) + 1e-6

    return inter / union


def deltaE(img1, img2):

    lab1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    lab2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    diff = np.sqrt(
        np.sum((lab1 - lab2) ** 2, axis=2)
    )

    return np.mean(diff)


def evaluate_layer(original, transformed):

    gray_original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    gray_transformed = cv2.cvtColor(
        transformed,
        cv2.COLOR_BGR2GRAY
    )

    return {

        "entropy":
            shannon_entropy(gray_transformed),

        "ssim":
            ssim(
                gray_original,
                gray_transformed
            ),

        "edge":
            edge_score(
                original,
                transformed
            ),

        "deltaE":
            deltaE(
                original,
                transformed
            )
    }


# ==========================================================
# PROCESAMIENTO DE UNA SOLA IMAGEN EN MEMORIA
# (misma logica que process_image() del script de entrenamiento, pero sin
# escribir carpetas a disco ni usar multiprocessing -- pensado para
# inferencia sobre una foto nueva, una a la vez)
# ==========================================================

def process_single_image(image_bgr):
    """
    image_bgr: imagen ya leida con cv2.imread (formato BGR).

    Devuelve (features_dict, capas_dict):
      features_dict: {"entropy_original": ..., "lab_entropy": ...,
          "lab_ssim": ..., ..., "edges_deltaE": ...} -- las mismas 21
          columnas que metrics.csv (menos "image").
      capas_dict: {"lab": img, "quant": img, "entropy": img_gris,
          "brush": img, "edges": img_binaria} -- las mismas imagenes que
          se guardan en 01_Lab...05_Edges durante el entrenamiento, listas
          para mostrarse en la tarjeta educativa.
    """
    image = image_bgr

    # ---------------- CAPA 1 - LAB ----------------
    lab, quant = lab_quantization(image)
    lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ---------------- CAPA 2 - CUANTIZACIÓN ----------------
    quant_bgr = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    # ---------------- CAPA 3 - ENTROPÍA ----------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = entropy_map(gray)
    entropy_img = (entropy * 255).astype(np.uint8)
    entropy_bgr = cv2.cvtColor(entropy_img, cv2.COLOR_GRAY2BGR)

    # ---------------- CAPA 4 - BRUSH ----------------
    brush = brush_strokes(quant_bgr, entropy)

    # ---------------- CAPA 5 - BORDES ----------------
    edges = edge_map(image)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ---------------- EVALUACIÓN DE CADA CAPA ----------------
    metrics_lab = evaluate_layer(image, lab_bgr)
    metrics_quant = evaluate_layer(image, quant_bgr)
    metrics_entropy = evaluate_layer(image, entropy_bgr)
    metrics_brush = evaluate_layer(image, brush)
    metrics_edges = evaluate_layer(image, edges_bgr)

    features_dict = {
        "entropy_original": shannon_entropy(gray)
    }

    for prefix, values in [
        ("lab", metrics_lab),
        ("quant", metrics_quant),
        ("entropy", metrics_entropy),
        ("brush", metrics_brush),
        ("edges", metrics_edges),
    ]:
        features_dict[f"{prefix}_entropy"] = values["entropy"]
        features_dict[f"{prefix}_ssim"] = values["ssim"]
        features_dict[f"{prefix}_edge"] = values["edge"]
        features_dict[f"{prefix}_deltaE"] = values["deltaE"]

    capas_dict = {
        "lab": lab_bgr,
        "quant": quant_bgr,
        "entropy": entropy_img,
        "brush": brush,
        "edges": edges,
    }

    return features_dict, capas_dict
