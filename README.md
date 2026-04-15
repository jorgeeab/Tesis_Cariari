# Tesis_Cariari

Codigo para realizar mapas tecnicos sobre el comportamiento del drenaje pluvial.

## Guia SBNS con FastAPI

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar local:

```bash
uvicorn app:app --reload
```

Despliegue con Gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app:app
```

Ruta principal:

```text
/
```

La aplicacion redirige a:

```text
/resultados_refinados/Guia/index.html
```
