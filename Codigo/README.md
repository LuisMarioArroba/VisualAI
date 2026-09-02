# Módulo de Inferencia e Interpretación de Estilos Artísticos (`inferencia_obra`)

Este módulo implementa un pipeline para la **carga, análisis e interpretación automatizada de obras pictóricas** en **Google Colab**.

El sistema utiliza técnicas de **Inteligencia Artificial no supervisada** y un conjunto de características visuales y compositivas para analizar una obra y asociarla con uno de cuatro movimientos artísticos: **Impresionismo, Postimpresionismo, Simbolismo y Realismo**.

Además de la clasificación, el módulo genera una interpretación basada en características formales de la obra y en principios de la **teoría del arte y la psicología de la Gestalt**.

---

## 🎨 Principios de Análisis y Metodología

El sistema analiza la obra a partir de diferentes dimensiones visuales y compositivas:

1. **Composición y armonía del color:** análisis cuantitativo de la interacción cromática, saturación y contraste presentes en la obra.

2. **Distribución y peso visual:** identificación de centros de interés, equilibrio espacial y distribución de la masa visual dentro del lienzo.

3. **Análisis por capas:** descomposición de la estructura visual en primer plano, plano medio y fondo, con el objetivo de analizar la profundidad y la construcción espacial.

4. **Paleta de color utilizada:** extracción de colores dominantes y secundarios para caracterizar las principales tendencias cromáticas de la obra.

5. **Principios de Gestalt:** análisis de principios de percepción visual como **Figura-Fondo, Ley de Prägnanz (Buena Forma), Cierre, Proximidad, Semejanza y Continuidad**, utilizados como elementos complementarios para la interpretación de la composición.

---

## 🖼️ Movimientos Artísticos Soportados

El sistema utiliza características visuales y compositivas asociadas a cuatro movimientos artísticos principales:

* **Realismo:** se caracteriza, dentro del modelo, mediante el predominio de paletas naturales o terrosas, una distribución estructurada del peso visual, una delimitación clara de los planos y relaciones definidas entre figura y fondo.

* **Impresionismo:** se caracteriza mediante el análisis de la distribución de la luz entre las diferentes capas de la composición, el uso de paletas luminosas y saturadas y la presencia de relaciones visuales asociadas a la Ley de Semejanza, particularmente en la representación de pinceladas y variaciones cromáticas.

* **Postimpresionismo:** se caracteriza mediante el uso expresivo y estructural del color, una marcada organización del peso visual a través de formas simplificadas y la presencia de relaciones asociadas a las leyes de Cierre y Continuidad.

* **Simbolismo:** se caracteriza mediante una distribución del peso visual asociada a componentes emocionales o simbólicos, relaciones Figura-Fondo potencialmente ambiguas y el uso de paletas cromáticas que pueden alejarse de la representación directa de la realidad observada.

> **Nota:** estas características representan los criterios visuales utilizados por el sistema para diferenciar los movimientos artísticos y no constituyen una definición exhaustiva de cada movimiento dentro de la historia del arte.

---

## 📋 Requisitos e Instalación

El módulo está diseñado para ejecutarse en **Google Colab**.

Antes de utilizarlo, instala las dependencias necesarias ejecutando la siguiente instrucción en una celda de código:
```bash
    !pip install -q opencv-python-headless scikit-image joblib pandas numpy scikit-learn
```

Una vez instaladas las dependencias, importa el módulo:
```bash
    import inferencia_obra
```
---

## 🚀 Ejecución

Para iniciar el flujo interactivo de carga e interpretación de una obra, ejecuta:

```bash
    inferencia_obra.subir_y_interpretar()
```

A continuación, selecciona la imagen de la obra mediante el botón de carga que aparecerá en la interfaz de Google Colab.

El módulo procesará automáticamente la imagen y ejecutará el pipeline de análisis e inferencia.

---

## 📊 Salida y Visualización de Reportes

Al finalizar el proceso de inferencia, el sistema genera un reporte con los resultados obtenidos.

### En Google Colab

Se renderiza automáticamente una interfaz en formato **HTML interactivo** que presenta:

- El movimiento artístico estimado.
- Las características visuales extraídas.
- Las métricas utilizadas durante el análisis.
- La interpretación teórica de los resultados.

### Almacenamiento

El reporte HTML generado se guarda automáticamente en la carpeta:

    /inferencias

dentro del espacio de trabajo de Google Colab.

De esta manera, cada inferencia puede conservarse para su posterior consulta y análisis.