"""Procesamiento de imágenes y lectura OCR para la vista de Django."""

import base64
from io import BytesIO
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from django.core.files.uploadedfile import UploadedFile
from PIL import Image

from reconocimiento_placas import analizar_region, buscar_regiones_candidatas


MAX_IMAGE_BYTES = 10 * 1024 * 1024
_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    """Carga EasyOCR al recibir la primera imagen."""
    global _reader

    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def _image_data_uri(image, *, color_map=None):
    """Convierte una imagen a un formato que se pueda mostrar en la página."""
    if color_map is not None:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = cv2.applyColorMap(normalized.astype(np.uint8), color_map)

    height, width = image.shape[:2]

    if width > 1400:
        new_height = max(1, int(height * 1400 / width))
        image = cv2.resize(
            image,
            (1400, new_height),
            interpolation=cv2.INTER_AREA,
        )

    if image.ndim == 3:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        mime_type = "image/jpeg"
    else:
        ok, encoded = cv2.imencode(".png", image)
        mime_type = "image/png"

    if not ok:
        raise ValueError("No se pudo preparar una imagen del resultado.")

    content = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{content}"


def _plate_format(text):
    """Limpia y formatea algunos formatos comunes de matrícula."""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())

    if re.fullmatch(r"[A-Z]{3}\d{4}", clean):
        return f"{clean[:3]}-{clean[3:5]}-{clean[5:]}"

    if re.fullmatch(r"[A-Z]{3}\d{3}[A-Z]", clean):
        return f"{clean[:3]}-{clean[3:6]}-{clean[6]}"

    if re.fullmatch(r"[A-Z]{3}\d{3}", clean):
        return f"{clean[:3]}-{clean[3:]}"

    return clean or "No reconocida"


def _read_plate(image, reader):
    """Lee texto en un recorte de placa detectado automáticamente."""
    if image is None or image.size == 0:
        return "No reconocida", 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    enlarged = cv2.bilateralFilter(enlarged, 9, 75, 75)

    _, binary = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    results = []

    for version in (enlarged, binary, cv2.bitwise_not(binary)):
        detections = reader.readtext(
            version,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
        )

        detections.sort(
            key=lambda item: (
                sum(point[1] for point in item[0]) / len(item[0]),
                sum(point[0] for point in item[0]) / len(item[0]),
            )
        )

        parts = []
        confidences = []

        for _, detected_text, confidence in detections:
            clean = re.sub(r"[^A-Z0-9]", "", detected_text.upper())
            if clean:
                parts.append(clean)
                confidences.append(float(confidence))

        if parts:
            raw = "".join(parts)
            confidence = sum(confidences) / len(confidences)
            formatted = _plate_format(raw)

            score = confidence + min(len(raw), 7) * 0.025

            if re.fullmatch(
                r"[A-Z]{3}-\d{2}-\d{2}|[A-Z]{3}-\d{3}-[A-Z]|[A-Z]{3}-\d{3}",
                formatted,
            ):
                score += 1.0

            results.append((score, formatted, confidence))

    if not results:
        return "No reconocida", 0.0

    _, text, confidence = max(results, key=lambda item: item[0])
    return text, confidence


def _read_full_photo(image, reader):
    """Busca una matrícula en la foto completa aunque no haya recorte automático."""
    if image is None or image.size == 0:
        return "No reconocida", 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    enlarged = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(enlarged)

    _, binary = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    readings = []

    for version in (enlarged, binary, cv2.bitwise_not(binary)):
        detections = reader.readtext(
            version,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
            decoder="beamsearch",
        )

        # Junta en filas los fragmentos de texto detectados por EasyOCR.
        rows = []

        for box, detected_text, confidence in detections:
            clean = re.sub(r"[^A-Z0-9]", "", detected_text.upper())
            if not clean:
                continue

            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)

            row_tolerance = max(25, enlarged.shape[0] * 0.06)

            row = next(
                (
                    candidate
                    for candidate in rows
                    if abs(candidate["y"] - center_y) < row_tolerance
                ),
                None,
            )

            if row is None:
                row = {"y": center_y, "parts": []}
                rows.append(row)

            row["parts"].append(
                (center_x, clean, float(confidence))
            )

        # Prueba formatos como JCZ-263-A y ABC1234.
        for row in rows:
            parts = sorted(row["parts"], key=lambda item: item[0])
            raw = "".join(item[1] for item in parts)

            match = re.search(
                r"[A-Z]{3}\d{3}[A-Z]|[A-Z]{3}\d{4}|[A-Z]{3}\d{3}",
                raw,
            )

            if match:
                text = _plate_format(match.group(0))
                confidence = sum(item[2] for item in parts) / len(parts)
                readings.append((text, confidence))

    if not readings:
        return "No reconocida", 0.0

    return max(readings, key=lambda item: item[1])


def _best_candidate(image, gray, boxes, reader):
    """Ejecuta OCR en las regiones candidatas encontradas."""
    height, width = gray.shape[:2]
    best_text = ""
    best_confidence = 0.0
    best_box = None

    for x, y, box_width, box_height in boxes[:5]:
        margin_x = max(4, int(box_width * 0.08))
        margin_y = max(3, int(box_height * 0.15))

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + box_width + margin_x)
        y2 = min(height, y + box_height + margin_y)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        text, confidence = _read_plate(crop, reader)

        if text != "No reconocida" and confidence > best_confidence:
            best_text = text
            best_confidence = confidence
            best_box = (x1, y1, x2 - x1, y2 - y1)

    return best_text, best_confidence, best_box


def analizar_imagen(uploaded_file: UploadedFile):
    """Valida, procesa y analiza la imagen enviada desde Django."""
    if uploaded_file.size > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el límite de 10 MB.")

    raw = uploaded_file.read()

    with Image.open(BytesIO(raw)) as source_image:
        if source_image.width * source_image.height > 25_000_000:
            raise ValueError("La imagen tiene demasiados píxeles para procesarse.")

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("El archivo no contiene una imagen JPG o PNG válida.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes, edges, candidates_view = buscar_regiones_candidatas(image, gray)

    gallery = [
        {
            "clave": "original",
            "etiqueta": "Fotografía original",
            "src": _image_data_uri(image),
        },
        {
            "clave": "gris",
            "etiqueta": "Escala de grises",
            "src": _image_data_uri(gray),
        },
        {
            "clave": "bordes",
            "etiqueta": "Bordes Canny",
            "src": _image_data_uri(edges),
        },
        {
            "clave": "candidatas",
            "etiqueta": "Regiones candidatas",
            "src": _image_data_uri(candidates_view),
        },
    ]

    plate_text = "No se encontro una placa"
    confidence = 0.0
    descriptors = {}
    reader = _get_reader()

    if boxes:
        recognized_text, recognized_confidence, selected_box = _best_candidate(
            image,
            gray,
            boxes,
            reader,
        )

        # También intenta leer la foto entera, por si el recorte automático falla.
        full_text, full_confidence = _read_full_photo(image, reader)

        if full_text != "No reconocida":
            plate_text = full_text
            confidence = full_confidence
        elif recognized_text:
            plate_text = recognized_text
            confidence = recognized_confidence
        else:
            plate_text = "No reconocida"

        if selected_box is None:
            selected_box = boxes[0]

        x, y, box_width, box_height = selected_box
        height, width = gray.shape[:2]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(width, int(x + box_width))
        y2 = min(height, int(y + box_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("La región candidata no tiene dimensiones válidas.")

        plate_color = image[y1:y2, x1:x2].copy()
        gray_crop = gray[y1:y2, x1:x2].copy()
        marked = image.copy()

        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 220, 80), 4)
        cv2.putText(
            marked,
            "PLACA CANDIDATA",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

        normalized, binary_crop, spectrum, descriptors = analizar_region(gray_crop)

        gallery.extend(
            [
                {
                    "clave": "marcada",
                    "etiqueta": "Placa localizada",
                    "src": _image_data_uri(marked),
                },
                {
                    "clave": "placa",
                    "etiqueta": "Recorte de la placa",
                    "src": _image_data_uri(plate_color),
                },
                {
                    "clave": "normalizada",
                    "etiqueta": "Región normalizada",
                    "src": _image_data_uri(normalized),
                },
                {
                    "clave": "binaria",
                    "etiqueta": "Binarización Otsu",
                    "src": _image_data_uri(binary_crop),
                },
                {
                    "clave": "fourier",
                    "etiqueta": "Transformada de Fourier",
                    "src": _image_data_uri(
                        spectrum,
                        color_map=cv2.COLORMAP_INFERNO,
                    ),
                },
            ]
        )

    else:
        # CAMBIO: aunque no se detecten regiones, intenta OCR en toda la foto.
        plate_text, confidence = _read_full_photo(image, reader)

        normalized, binary_full, spectrum, descriptors = analizar_region(gray)

        gallery.extend(
            [
                {
                    "clave": "normalizada",
                    "etiqueta": "Imagen normalizada",
                    "src": _image_data_uri(normalized),
                },
                {
                    "clave": "binaria",
                    "etiqueta": "Binarización Otsu",
                    "src": _image_data_uri(binary_full),
                },
                {
                    "clave": "fourier",
                    "etiqueta": "Transformada de Fourier",
                    "src": _image_data_uri(
                        spectrum,
                        color_map=cv2.COLORMAP_INFERNO,
                    ),
                },
            ]
        )

    # Si el OCR leyó la matrícula en la foto completa, cuenta esa foto
    # como una región analizada para que la página no muestre cero.
    cantidad = len(boxes)
    if cantidad == 0 and plate_text != "No reconocida":
        cantidad = 1

    return {
        "nombre": Path(uploaded_file.name).name,
        "cantidad": cantidad,
        "galeria": gallery,
        "descriptores": descriptors,
        "texto_placa": plate_text,
        "confianza_porcentaje": confidence * 100.0,
        "revisar": plate_text in {
            "No reconocida",
            "No se encontro una placa",
        },
    }
