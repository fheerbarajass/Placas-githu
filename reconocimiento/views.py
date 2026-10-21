import logging

from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.shortcuts import render

from .processing import analizar_imagen


logger = logging.getLogger(__name__)


def inicio(request):
    contexto = {"resultado": None, "error": None}

    if request.method == "POST":
        try:
            archivo = request.FILES.get("imagen")

            if archivo is None:
                contexto["error"] = "Selecciona una imagen primero."
            elif archivo.size > settings.MAX_UPLOAD_SIZE:
                contexto["error"] = "La imagen supera el límite de 10 MB."
            elif not archivo.name.lower().endswith((".jpg", ".jpeg", ".png")):
                contexto["error"] = "Utiliza una imagen JPG, JPEG o PNG."
            else:
                contexto["resultado"] = analizar_imagen(archivo)

        except RequestDataTooBig:
            contexto["error"] = "La imagen supera el límite de 10 MB."
        except Exception:
            logger.exception("No se pudo analizar la imagen subida")
            contexto["error"] = (
                "No se pudo analizar la imagen. Intenta con una foto nítida "
                "y en formato JPG o PNG."
            )

    return render(request, "index.html", contexto)
