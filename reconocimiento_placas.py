import sys


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops







def cargar_imagen(ruta):
    """Carga una imagen en color y devuelve tambpython -c "from skimage.feature import graycomatrix, graycoprops; print('Todo funciona correctamente')"ién su versión en gris."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {ruta}")
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen, gris


def normalizar_gris(gris, tamano=(256, 96)):
    """Normaliza tamaño y contraste para comparar regiones."""
    redimensionada = cv2.resize(gris, tamano, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(redimensionada)


def dimension_fractal_box_counting(imagen_binaria):
    """
    Estima la dimensión fractal por conteo de cajas.
    Se espera una imagen binaria con valores 0 y 255.
    Es una estimación descriptiva, no una prueba de que la imagen sea fractal.
    """
    datos = imagen_binaria > 0
    alto, ancho = datos.shape
    escalas = [2, 4, 8, 16, 32]
    log_escalas, log_conteos = [], []

    for escala in escalas:
        if escala >= min(alto, ancho):
            continue
        conteo = 0
        for y in range(0, alto, escala):
            for x in range(0, ancho, escala):
                bloque = datos[y:y + escala, x:x + escala]
                if np.any(bloque):
                    conteo += 1
        if conteo > 0:
            log_escalas.append(np.log(1.0 / escala))
            log_conteos.append(np.log(conteo))

    if len(log_escalas) < 2:
        return 0.0
    pendiente = np.polyfit(log_escalas, log_conteos, 1)[0]
    return float(pendiente)


def descriptor_fourier(gris):
    """Obtiene energía de baja/alta frecuencia usando la magnitud de Fourier."""
    imagen = gris.astype(np.float32) / 255.0
    espectro = np.fft.fftshift(np.fft.fft2(imagen))
    magnitud = np.log1p(np.abs(espectro))

    alto, ancho = magnitud.shape
    yy, xx = np.ogrid[:alto, :ancho]
    radio = np.sqrt((yy - alto / 2) ** 2 + (xx - ancho / 2) ** 2)
    mascara_baja = radio < min(alto, ancho) * 0.12
    mascara_alta = ~mascara_baja

    energia_total = float(np.sum(magnitud ** 2) + 1e-12)
    energia_baja = float(np.sum(magnitud[mascara_baja] ** 2) / energia_total)
    energia_alta = float(np.sum(magnitud[mascara_alta] ** 2) / energia_total)
    return energia_baja, energia_alta, magnitud


def descriptor_textura(gris):
    """Calcula contraste, homogeneidad, energía y correlación GLCM."""
    cuantizada = (gris // 4).astype(np.uint8)  # 64 niveles
    glcm = graycomatrix(
        cuantizada,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    return {
        "contraste": float(graycoprops(glcm, "contrast").mean()),
        "homogeneidad": float(graycoprops(glcm, "homogeneity").mean()),
        "energia_textura": float(graycoprops(glcm, "energy").mean()),
        "correlacion_glcm": float(graycoprops(glcm, "correlation").mean()),
    }


def correlacion_espacial(imagen, plantilla):
    """
    Compara una plantilla con la imagen usando correlación normalizada.
    Útil para localizar una plantilla conocida; no identifica por sí sola
    cualquier placa desconocida.
    """
    if imagen.shape[0] < plantilla.shape[0] or imagen.shape[1] < plantilla.shape[1]:
        raise ValueError("La plantilla debe ser menor o igual que la imagen.")
    resultado = cv2.matchTemplate(imagen, plantilla, cv2.TM_CCOEFF_NORMED)
    _, maximo, _, ubicacion = cv2.minMaxLoc(resultado)
    return float(maximo), ubicacion


def buscar_regiones_candidatas(imagen_color, gris):
    """
    Propone regiones rectangulares candidatas a placa usando bordes y contornos.
    Es un filtro inicial; puede producir falsos positivos.
    """
    suavizada = cv2.bilateralFilter(gris, 11, 17, 17)
    bordes = cv2.Canny(suavizada, 40, 160)
    contornos, _ = cv2.findContours(
        bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    candidatas = []
    alto_img, ancho_img = gris.shape
    for contorno in contornos[:80]:
        x, y, w, h = cv2.boundingRect(contorno)
        area = w * h
        if h == 0:
            continue
        proporcion = w / h
        area_relativa = area / (ancho_img * alto_img)
        # Rangos heurísticos para una placa horizontal; deben ajustarse al dataset.
        if 2.0 <= proporcion <= 6.5 and 0.005 <= area_relativa <= 0.35:
            candidatas.append((x, y, w, h))

    # Eliminar duplicados muy similares y conservar hasta 10 regiones.
    resultado = []
    for caja in candidatas:
        x, y, w, h = caja
        if not any(
            abs(x - xx) < 10 and abs(y - yy) < 10
            and abs(w - ww) < 15 and abs(h - hh) < 15
            for xx, yy, ww, hh in resultado
        ):
            resultado.append(caja)
        if len(resultado) == 10:
            break

    vista = imagen_color.copy()
    for x, y, w, h in resultado:
        cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 220, 0), 2)
    return resultado, bordes, vista


def analizar_region(region_gris):
    """Calcula los cuatro grupos de descriptores solicitados."""
    normalizada = normalizar_gris(region_gris)
    _, binaria = cv2.threshold(
        normalizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    fractal = dimension_fractal_box_counting(binaria)
    energia_baja, energia_alta, espectro = descriptor_fourier(normalizada)
    textura = descriptor_textura(normalizada)

    descriptores = {
        "dimension_fractal_aproximada": fractal,
        "energia_fourier_baja": energia_baja,
        "energia_fourier_alta": energia_alta,
        **textura,
    }
    return normalizada, binaria, espectro, descriptores


def mostrar_resultados(imagen, gris, bordes, vista, region=None):
    """Muestra la imagen original, bordes, candidatas y análisis de región."""
    cantidad = 4 if region is None else 7
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(bordes, cmap="gray")
    plt.title("Bordes Canny")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(vista, cv2.COLOR_BGR2RGB))
    plt.title("Regiones candidatas")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(gris, cmap="gray")
    plt.title("Imagen en escala de grises")
    plt.axis("off")

    if region is not None:
        normalizada, binaria, espectro, _ = analizar_region(region)
        plt.figure(figsize=(13, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(normalizada, cmap="gray")
        plt.title("Región normalizada")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(binaria, cmap="gray")
        plt.title("Binarización Otsu")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(espectro, cmap="inferno")
        plt.title("Magnitud de Fourier")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Uso: python reconocimiento_placas.py ruta/a/imagen.jpg")
        raise SystemExit(1)

    ruta = Path(sys.argv[1])
    imagen, gris = cargar_imagen(ruta)
    candidatas, bordes, vista = buscar_regiones_candidatas(imagen, gris)

    print("\n=== RESULTADO DEL ANÁLISIS ===")
    print(f"Imagen: {ruta}")
    print(f"Regiones candidatas encontradas: {len(candidatas)}")

    region_para_analizar = None
    if candidatas:
        # Se toma la primera candidata por ahora; debe verificarse visualmente.
        x, y, w, h = candidatas[0]
        region_para_analizar = gris[y:y + h, x:x + w]
        _, _, _, descriptores = analizar_region(region_para_analizar)
        print("\nDescriptores de la primera región candidata:")
        for nombre, valor in descriptores.items():
            print(f"  {nombre}: {valor:.6f}")
    else:
        print("No se detectaron candidatas con los umbrales actuales.")
        print("Prueba otra imagen o ajusta los rangos de proporción y área.")

    mostrar_resultados(imagen, gris, bordes, vista, region_para_analizar)


if __name__ == "__main__":
    main()
