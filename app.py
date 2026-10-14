"app"
from pathlib import Path
import re
import uuid

import cv2
import easyocr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from reconocimiento_placas import (
    cargar_imagen,
    buscar_regiones_candidatas,
    analizar_region,
)


# ==========================================================
# CONFIGURACION GENERAL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Limite de subida: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

EXTENSIONES = {"jpg", "jpeg", "png"}

# Inicializar OCR una sola vez.
# La primera ejecucion puede descargar modelos.
lector_ocr = easyocr.Reader(["en"], gpu=False)


# ==========================================================
# VALIDAR ARCHIVOS
# ==========================================================

def extension_valida(nombre):
    return (
        "." in nombre
        and nombre.rsplit(".", 1)[1].lower() in EXTENSIONES
    )


# ==========================================================
# GUARDAR IMAGENES
# ==========================================================

def guardar_imagen(nombre, imagen):
    ruta = RESULT_DIR / nombre

    guardada = cv2.imwrite(str(ruta), imagen)

    if not guardada:
        raise RuntimeError(
            f"No se pudo guardar la imagen: {nombre}"
        )

    return f"results/{nombre}"


def guardar_espectro(nombre, espectro):
    ruta = RESULT_DIR / nombre

    fig, ax = plt.subplots(figsize=(7, 3))

    try:
        ax.imshow(espectro, cmap="inferno")
        ax.set_title("Transformada de Fourier")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(
            ruta,
            dpi=140,
            bbox_inches="tight"
        )
    finally:
        plt.close(fig)

    return f"results/{nombre}"


# ==========================================================
# RECONOCIMIENTO OCR DE LA PLACA
# ==========================================================

def formatear_placa(texto):
    """Normaliza y agrega guiones a formatos comunes de placas mexicanas."""
    limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
    # Formatos comunes: PCH-96-04, JCZ-245-A y ABC-123.
    if re.fullmatch(r"[A-Z]{3}\d{4}", limpio):
        return f"{limpio[:3]}-{limpio[3:5]}-{limpio[5:]}"
    if re.fullmatch(r"[A-Z]{3}\d{3}[A-Z]", limpio):
        return f"{limpio[:3]}-{limpio[3:6]}-{limpio[6]}"
    # También admite el formato de seis caracteres ABC-123.
    if re.fullmatch(r"[A-Z]{3}\d{3}", limpio):
        return f"{limpio[:3]}-{limpio[3:]}"
    return limpio or "No reconocida"


def leer_placa(imagen_placa):
    """Lee la región en varias versiones y combina los fragmentos OCR."""
    if imagen_placa is None or imagen_placa.size == 0:
        return "No reconocida", 0.0

    if len(imagen_placa.shape) == 3:
        gris = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_placa.copy()

    ampliada = cv2.resize(gris, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ampliada = cv2.bilateralFilter(ampliada, 9, 75, 75)
    _, binaria = cv2.threshold(ampliada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    mejores = []
    for version in (ampliada, binaria, cv2.bitwise_not(binaria)):
        resultados = lector_ocr.readtext(
            version, detail=1, paragraph=False, allowlist=allowlist
        )
        # EasyOCR puede dividir una matrícula en varios fragmentos; ordenarlos
        # por posición horizontal permite reconstruir el texto completo.
        resultados = sorted(
            resultados,
            key=lambda item: (sum(p[1] for p in item[0]) / len(item[0]),
                              sum(p[0] for p in item[0]) / len(item[0]))
        )
        fragmentos = []
        confianzas = []
        for _, texto, confianza in resultados:
            limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
            if limpio:
                fragmentos.append(limpio)
                confianzas.append(float(confianza))
        if fragmentos:
            combinado = "".join(fragmentos)
            confianza = sum(confianzas) / len(confianzas)
            # Favorece lecturas largas y con patrón de matrícula habitual.
            formato = formatear_placa(combinado)
            puntos = confianza + min(len(combinado), 7) * 0.025
            if re.fullmatch(r"[A-Z]{3}-\d{2}-\d{2}|[A-Z]{3}-\d{3}-[A-Z]|[A-Z]{3}-\d{3}", formato):
                puntos += 1.0
            mejores.append((puntos, formato, confianza))

    if not mejores:
        return "No reconocida", 0.0
    _, texto_final, confianza_media = max(mejores, key=lambda x: x[0])
    return texto_final, confianza_media

def leer_placa_foto_completa(imagen):
    """Busca una matrícula de cuatro dígitos agrupando OCR por renglón."""
    ampliada = cv2.resize(imagen, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    detecciones = lector_ocr.readtext(ampliada, detail=1, paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    filas = []
    for caja, texto, confianza in detecciones:
        limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
        if not limpio:
            continue
        ys, xs = [p[1] for p in caja], [p[0] for p in caja]
        cy = sum(ys) / len(ys)
        fila = next((r for r in filas if abs(r[0] - cy) < 30), None)
        if fila is None:
            fila = [cy, []]
            filas.append(fila)
        fila[1].append((sum(xs) / len(xs), limpio, float(confianza)))
    lecturas = []
    for _, piezas in filas:
        piezas.sort(key=lambda pieza: pieza[0])
        bruto = "".join(pieza[1] for pieza in piezas)
        if re.fullmatch(r"[A-Z]{3}\d{4}", bruto):
            certeza = sum(pieza[2] for pieza in piezas) / len(piezas)
            lecturas.append((formatear_placa(bruto), certeza))
    return max(lecturas, key=lambda item: item[1]) if lecturas else ("No reconocida", 0.0)

def analizar_todas_las_candidatas(imagen, gris, candidatas):
    alto, ancho = gris.shape[:2]

    mejor_texto = ""
    mejor_confianza = 0.0
    mejor_recorte = None
    mejor_caja = None

    for x, y, w, h in candidatas:
        # Añadir margen para conservar los caracteres de los extremos
        margen_x = max(4, int(w * 0.08))
        margen_y = max(3, int(h * 0.15))

        x1 = max(0, x - margen_x)
        y1 = max(0, y - margen_y)
        x2 = min(ancho, x + w + margen_x)
        y2 = min(alto, y + h + margen_y)

        recorte = imagen[y1:y2, x1:x2].copy()

        if recorte.size == 0:
            continue

        texto, confianza = leer_placa(recorte)

        if texto and texto != "No reconocida":
            # Preferir confianza; desempatar con más caracteres
            if (
                confianza > mejor_confianza
                or (
                    confianza == mejor_confianza
                    and len(texto) > len(mejor_texto)
                )
            ):
                mejor_texto = texto
                mejor_confianza = confianza
                mejor_recorte = recorte
                mejor_caja = (x1, y1, x2 - x1, y2 - y1)

    return (
        mejor_texto,
        mejor_confianza,
        mejor_recorte,
        mejor_caja
    )



# ==========================================================
# PROCESAMIENTO PRINCIPAL
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None
    error = None

    if request.method == "POST":

        archivo = request.files.get("imagen")

        if not archivo or not archivo.filename:
            error = "Selecciona una imagen primero."

        elif not extension_valida(archivo.filename):
            error = "Utiliza una imagen JPG, JPEG o PNG."

        else:
            nombre_original = secure_filename(
                archivo.filename
            )

            identificador = uuid.uuid4().hex[:10]

            nombre_subida = (
                identificador + "_" + nombre_original
            )

            ruta_subida = UPLOAD_DIR / nombre_subida

            archivo.save(ruta_subida)

            try:
                # ------------------------------------------
                # 1. CARGAR IMAGEN
                # ------------------------------------------

                imagen, gris = cargar_imagen(ruta_subida)

                if imagen is None or gris is None:
                    raise ValueError(
                        "No se pudo cargar la imagen."
                    )

                # ------------------------------------------
                # 2. BUSCAR REGIONES CANDIDATAS
                # ------------------------------------------

                candidatas, bordes, vista = (
                    buscar_regiones_candidatas(
                        imagen,
                        gris
                    )
                )

                prefijo = identificador

                # Diccionario de imagenes para la pagina.
                imagenes = {
                    "original": guardar_imagen(
                        f"{prefijo}_original.png",
                        imagen
                    ),
                    "gris": guardar_imagen(
                        f"{prefijo}_gris.png",
                        gris
                    ),
                    "bordes": guardar_imagen(
                        f"{prefijo}_bordes.png",
                        bordes
                    ),
                    "candidatas": guardar_imagen(
                        f"{prefijo}_candidatas.png",
                        vista
                    ),
                }

                # Valores predeterminados.
                descriptores = {}
                texto_placa = "No se encontro una placa"
                confianza_ocr = 0.0

                # ------------------------------------------
                # 3. ANALIZAR LAS CANDIDATAS Y ELEGIR LA MEJOR LECTURA
                # ------------------------------------------

                if candidatas:
                    # Revisar todas las regiones candidatas y elegir la que
                    # produzca la lectura más completa y confiable.
                    _, _, _, caja = analizar_todas_las_candidatas(
                        imagen, gris, candidatas
                    )
                    if caja is None:
                        caja = candidatas[0]
                    x, y, w, h = caja

                    # Evitar coordenadas fuera de la imagen.
                    alto, ancho = gris.shape[:2]

                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(ancho, int(x + w))
                    y2 = min(alto, int(y + h))

                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(
                            "La region candidata no tiene "
                            "dimensiones validas."
                        )

                    # Recortar la region en color y gris.
                    placa_color = imagen[
                        y1:y2,
                        x1:x2
                    ].copy()

                    region_gris = gris[
                        y1:y2,
                        x1:x2
                    ].copy()

                    # --------------------------------------
                    # 4. MARCAR LA PLACA EN LA FOTO
                    # --------------------------------------

                    marcada = imagen.copy()

                    cv2.rectangle(
                        marcada,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 80),
                        4
                    )

                    cv2.putText(
                        marcada,
                        "PLACA CANDIDATA",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 80),
                        2,
                        cv2.LINE_AA
                    )

                    imagenes["marcada"] = guardar_imagen(
                        f"{prefijo}_marcada.png",
                        marcada
                    )

                    # --------------------------------------
                    # 5. GUARDAR EL RECORTE
                    # --------------------------------------

                    imagenes["placa"] = guardar_imagen(
                        f"{prefijo}_placa.png",
                        placa_color
                    )

                    # --------------------------------------
                    # 6. LEER CARACTERES CON OCR
                    # --------------------------------------

                    texto_placa, confianza_ocr = leer_placa(placa_color)
                    texto_completo, confianza_completa = leer_placa_foto_completa(imagen)
                    if texto_completo != "No reconocida":
                        texto_placa, confianza_ocr = texto_completo, confianza_completa

                    # --------------------------------------
                    # 7. ANALISIS MATEMATICO
                    # --------------------------------------

                    (
                        normalizada,
                        binaria,
                        espectro,
                        descriptores
                    ) = analizar_region(region_gris)

                    imagenes["normalizada"] = guardar_imagen(
                        f"{prefijo}_normalizada.png",
                        normalizada
                    )

                    imagenes["binaria"] = guardar_imagen(
                        f"{prefijo}_binaria.png",
                        binaria
                    )

                    imagenes["fourier"] = guardar_espectro(
                        f"{prefijo}_fourier.png",
                        espectro
                    )

                # ------------------------------------------
                # 8. PREPARAR RESULTADOS PARA HTML
                # ------------------------------------------

                resultado = {
                    "nombre": nombre_original,
                    "cantidad": len(candidatas),
                    "imagenes": imagenes,
                    "descriptores": descriptores,
                    "texto_placa": texto_placa,
                    "confianza_ocr": confianza_ocr,
                }

            except Exception as exc:
                app.logger.exception(
                    "Error durante el procesamiento"
                )

                error = (
                    "No se pudo analizar la imagen. "
                    f"Detalle: {exc}"
                )

    return render_template(
        "index.html",
        resultado=resultado,
        error=error
    )


# ==========================================================
# MANEJO DE IMAGENES DEMASIADO GRANDES
# ==========================================================

@app.errorhandler(413)
def imagen_demasiado_grande(_error):

    return render_template(
        "index.html",
        resultado=None,
        error="La imagen supera el limite de 10 MB."
    ), 413


# ==========================================================
# INICIAR SERVIDOR
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)







