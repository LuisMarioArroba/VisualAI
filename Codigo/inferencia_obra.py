"""
inferencia_obra.py
Carga el modelo ya entrenado (modelo_entrenado.joblib) y UNA foto nueva
(por ejemplo, la obra de un alumno) para generar la misma tarjeta
explicativa que el reporte educativo, pero para una imagen que el modelo
nunca vio.

La extraccion de features usa gestalt_pipeline.py, que es el mismo
pipeline con el que se genero metrics.csv durante el entrenamiento
(mismo K_COLORS, mismo BLOCK_SIZE, mismas formulas), asi que las metricas
de la foto nueva son comparables con las del dataset de entrenamiento.

Requiere:
    pip install -q opencv-python scikit-image pandas numpy joblib
"""

import os
import sys
import base64
import numpy as np
import pandas as pd
import cv2
import joblib

from interpretacion_arte import (
    LAYER_INFO,
    GLOSARIO_DESC,
    layer_caption,
    explain_features,
    build_movement_ficha,
    color_caption,
)
from gestalt_pipeline import process_single_image

ROOT_DATASET = "dataset_movimientos"
MODEL_PATH = os.path.join(ROOT_DATASET, "explainable_report", "modelo_entrenado.joblib")
OUTPUT_DIR = os.path.join(ROOT_DATASET, "inferencia")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 1. CARGA DEL MODELO ENTRENADO
# ==========================================================

def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontro {MODEL_PATH}. Corre primero entrenamiento_no_supervisado.py."
        )
    bundle = joblib.load(MODEL_PATH)
    for key in ("global_mean", "global_std", "feature_cols", "scaler", "kmeans", "clf",
                "movement_profile", "ranking", "cluster_majority"):
        if key not in bundle:
            raise KeyError(f"El modelo guardado no tiene '{key}'. Vuelve a entrenar con la version actualizada.")
    return bundle


# ==========================================================
# 2. EXTRACCION DE FEATURES DE UNA FOTO NUEVA
# ==========================================================

def extraer_features_de_imagen(image_path, feature_cols):
    """
    Corre el mismo pipeline Gestalt del entrenamiento (gestalt_pipeline.py)
    sobre una sola foto nueva.

    Devuelve (features_dict, capas_dict):
      features_dict: {"entropy_original": ..., "lab_deltaE": ...,
          "lab_ssim": ..., ...} -- una entrada por cada nombre en
          feature_cols.
      capas_dict: {"lab": img, "quant": img, "entropy": img_gris,
          "brush": img, "edges": img_binaria}
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    features_dict, capas_dict = process_single_image(image)

    faltantes = [f for f in feature_cols if f not in features_dict]
    if faltantes:
        raise KeyError(
            f"El pipeline no genero estas columnas que el modelo espera: {faltantes}. "
            "Revisa que gestalt_pipeline.py tenga los mismos K_COLORS/BLOCK_SIZE "
            "que se usaron para entrenar el modelo."
        )

    return features_dict, capas_dict


# ==========================================================
# 3. PREDICCION DEL MOVIMIENTO
# ==========================================================

def predecir_movimiento(features_dict, bundle):
    """
    Devuelve (movement_predicho, z) usando el mismo camino que el
    entrenamiento: escalar con el scaler entrenado, pedirle al cluster
    (KMeans/RandomForest) su cluster, y traducir ese cluster a movimiento
    con cluster_majority (el mapeo cluster -> movimiento mayoritario que
    ya se calculo en el entrenamiento).
    """
    feature_cols = bundle["feature_cols"]
    global_mean = bundle["global_mean"]
    global_std = bundle["global_std"]
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    cluster_majority = bundle["cluster_majority"]

    fila = pd.Series(features_dict)[feature_cols]

    # z-score respecto al dataset de entrenamiento (para las explicaciones)
    z = (fila - global_mean[feature_cols]) / global_std[feature_cols]

    # cluster predicho con el mismo pipeline scaler + clasificador del entrenamiento
    X_scaled = scaler.transform(fila.values.reshape(1, -1))
    cluster_id = clf.predict(X_scaled)[0]
    movement_predicho = cluster_majority.get(cluster_id, "desconocido")

    return movement_predicho, z


# ==========================================================
# 4. CONSTRUCCION DE LA TARJETA HTML
# ==========================================================

def image_to_data_uri(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def save_quant_image(img_bgr, output_dir, image_name):
    """
    Guarda temporalmente la capa cuantizada para que pueda ser
    utilizada por color_caption(), que necesita una RUTA de archivo
    (abre la imagen con PIL) y no un array en memoria.
    """
    os.makedirs(output_dir, exist_ok=True)

    quant_path = os.path.join(
        output_dir,
        f"{image_name}_quant_temp.png"
    )

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cv2.imwrite(
        quant_path,
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )

    return quant_path


HTML_HEAD = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Interpretacion de tu obra</title>
<style>
  body { font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 24px; color: #222; background: #fafaf7; }
  h1 { font-size: 1.6em; }
  .intro, .mov-ficha { background: #fff; border-left: 4px solid #b08d57; padding: 16px 20px; margin-bottom: 24px; border-radius: 4px; }
  .mov-ficha ul { margin: 6px 0 0; padding-left: 20px; }
  .mov-ficha li { margin-bottom: 4px; }
  .obra { background: #fff; border: 1px solid #e2ddd0; border-radius: 8px; padding: 20px; margin-bottom: 28px; }
  .encabezado { display: flex; gap: 20px; align-items: flex-start; }
  .encabezado img.original { width: 260px; border-radius: 6px; flex-shrink: 0; }
  .ficha-comparativa { color: #6b5a3e; font-size: 0.9em; }
  .capas { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 16px; }
  .capa { width: 210px; font-size: 0.8em; }
  .capa img { width: 210px; height: 210px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }
  .capa h4 { margin: 6px 0 2px; font-size: 0.95em; }
  .capa p { margin: 0; color: #555; line-height: 1.35; }
  .colorometria-wrap { display: flex; gap: 20px; align-items: flex-start; margin-top: 18px; padding-top: 16px; border-top: 1px dashed #d8d0bd; }
  .colorometria-wrap img.colorometria-img { width: 180px; border-radius: 6px; flex-shrink: 0; }
  .colorometria-texto h3 { margin: 0 0 8px; font-size: 1.05em; color: #6b5a3e; }
  .colorometria-texto p { margin: 0 0 10px; line-height: 1.45; }
</style>
</head>
<body>
<h1>Interpretacion de tu obra</h1>
<div class="intro">
<p>Esta lectura compara tu obra, capa por capa, contra el perfil estadistico de
cientos de obras de cuatro corrientes (impresionismo, realismo, postimpresionismo
y simbolismo). No es un veredicto definitivo -- es una lectura formal, pensada
para que puedas contrastarla con tu propia intencion al pintar.</p>
</div>
"""

HTML_TAIL = "\n</body>\n</html>\n"


def build_card_inferencia(image_path, features_dict, capas_dict, bundle):
    feature_cols = bundle["feature_cols"]
    global_mean = bundle["global_mean"]
    global_std = bundle["global_std"]
    ranking = bundle["ranking"]

    movement_profile_z = (
        (bundle["movement_profile"][feature_cols] - global_mean[feature_cols]) / global_std[feature_cols]
    )

    movement_predicho, z = predecir_movimiento(features_dict, bundle)

    original_img = cv2.imread(image_path)
    original_uri = image_to_data_uri(original_img) if original_img is not None else ""

    # Para reusar layer_caption necesitamos una "fila" tipo pandas.Series con
    # los valores crudos de las metricas (no los z-scores).
    fila = pd.Series(features_dict)

    capas_html = ""
    for key, titulo, _, _ in LAYER_INFO:
        img = capas_dict.get(key)
        if img is None:
            continue
        uri = image_to_data_uri(img)
        caption = layer_caption(fila, key, global_mean, global_std, incluir_intro=True)
        capas_html += f"""
        <div class="capa">
          <img src="{uri}" alt="{titulo}">
          <h4>{titulo}</h4>
          <p>{caption}</p>
        </div>"""

    parrafo, ficha = explain_features(
        z, feature_cols, movement_profile_z, ranking,
        movement_predicho=movement_predicho, movement_real=None,
    )

    nombre = os.path.splitext(os.path.basename(image_path))[0]

    # --- Colorometria: identificacion de la paleta dominante, temperatura,
    # balance cromatico, peso visual y su relacion con el movimiento
    # detectado. Se apoya en la capa "quant" ya generada por el pipeline,
    # por eso primero hay que volcarla a disco (color_caption necesita una
    # ruta de archivo, no el array en memoria).
    colorometria_bloque = ""
    quant_img = capas_dict.get("quant")
    if quant_img is not None:
        quant_path = save_quant_image(quant_img, OUTPUT_DIR, nombre)
        try:
            colorometria_html = color_caption(quant_path, movement_predicho=movement_predicho)
        finally:
            if os.path.exists(quant_path):
                os.remove(quant_path)
        # color_caption() ya devuelve su propio <div class="colorometria">, asi
        # que no lo volvemos a envolver con la misma clase: solo lo metemos
        # junto a la foto original dentro de un contenedor a dos columnas.
        colorometria_bloque = f"""
      <div class="colorometria-wrap">
        <img class="colorometria-img" src="{original_uri}" alt="{nombre}">
        <div class="colorometria-texto">
          <h3>Colorometria</h3>
          {colorometria_html}
        </div>
      </div>"""

    html = HTML_HEAD
    html += f"""
    <section class="obra">
      <div class="encabezado">
        <img class="original" src="{original_uri}" alt="{nombre}">
        <div>
          <h3>{nombre}</h3>
          <p>{parrafo}</p>
          {ficha}
        </div>
      </div>
      <div class="capas">
        {capas_html}
      </div>
      {colorometria_bloque}
    </section>
    """
    html += build_movement_ficha(movement_predicho)
    html += HTML_TAIL
    return html


# ==========================================================
# 5. MAIN
# ==========================================================

def interpretar_obra(image_path):
    bundle = load_model_bundle()
    features_dict, capas_dict = extraer_features_de_imagen(image_path, bundle["feature_cols"])
    html = build_card_inferencia(image_path, features_dict, capas_dict, bundle)

    nombre = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"interpretacion_{nombre}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Interpretacion guardada en: {out_path}")
    return out_path


def subir_y_interpretar():
    """
    Pensada para Google Colab: abre el selector de archivos del navegador
    para que subas una foto desde tu computadora, corre la interpretacion
    y la muestra directamente en el notebook -- sin escribir ninguna ruta
    a mano.

    Uso en una celda de Colab:
        from inferencia_obra import subir_y_interpretar
        subir_y_interpretar()
    """
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError(
            "subir_y_interpretar() esta pensada para Google Colab (usa "
            "google.colab.files para el selector de archivos). Fuera de "
            "Colab, usa interpretar_obra('ruta/a/tu/foto.jpg') directamente."
        )
    from IPython.display import HTML, display

    subida = files.upload()
    if not subida:
        print("No se subio ninguna imagen.")
        return None

    nombre_archivo = list(subida.keys())[0]
    out_path = interpretar_obra(nombre_archivo)

    with open(out_path, encoding="utf-8") as f:
        display(HTML(f.read()))

    return out_path


def main():
    if len(sys.argv) < 2:
        print("Uso: python inferencia_obra.py ruta/a/la/foto.jpg")
        sys.exit(1)
    interpretar_obra(sys.argv[1])


if __name__ == "__main__":
    main()