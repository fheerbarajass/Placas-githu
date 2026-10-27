# PlateVision

PlateVision recibe una fotografía JPG, JPEG o PNG, busca regiones candidatas a placa, aplica OCR y muestra una galería con las etapas del procesamiento y los descriptores de imagen.

## Ejecutar Django en Windows

Desde la carpeta que contiene `manage.py`, crea y activa un entorno virtual, instala dependencias y ejecuta el servidor:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Abre `http://127.0.0.1:8000`. Si no tienes un `.env` local, copia `.env.example` como `.env` y sustituye `DJANGO_SECRET_KEY` por una clave local. No publiques `.env`.

## Publicar en Render

Conecta el repositorio de GitHub que contiene `manage.py` en su raíz y configura el servicio como **Web Service**, con estos comandos:

```text
Build: pip install -r requirements.txt && python manage.py collectstatic --no-input
Start: python manage.py migrate && gunicorn config.wsgi:application --workers 1 --timeout 180
```

Agrega estas variables en **Environment** de Render:

```text
DJANGO_SECRET_KEY=<clave nueva y privada>
DEBUG=False
```

EasyOCR carga sus modelos la primera vez que se solicita un análisis. En el plan gratuito, el servicio puede 
tardar y tiene recursos limitados. Los archivos procesados se conservan solo mientras el proceso responde;
Render puede borrar los archivos locales al reiniciar o suspender el servicio.

LIGA PARA ABRIR LA PAGINA https://placas-githu.onrender.com/
