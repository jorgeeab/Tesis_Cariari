import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados_refinados"
_GUIA_PATH  = "/resultados_refinados/Guia/index.html"
_MAPA3D_PATH = "/resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html"

# ROOT_PATH allows running under a subpath proxy (e.g. nginx /sbn → this app)
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")
GUIA_INDEX  = ROOT_PATH + _GUIA_PATH
MAPA3D_URL  = ROOT_PATH + _MAPA3D_PATH

app = FastAPI(title="Guia SBNS Cariari")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url=GUIA_INDEX, status_code=307)


@app.get("/guia", include_in_schema=False)
@app.get("/guia/", include_in_schema=False)
def guia() -> RedirectResponse:
    return RedirectResponse(url=GUIA_INDEX, status_code=307)


@app.get("/mapa3d", include_in_schema=False)
@app.get("/mapa3d/", include_in_schema=False)
def mapa3d() -> RedirectResponse:
    return RedirectResponse(url=MAPA3D_URL, status_code=307)


app.mount(
    "/resultados_refinados",
    StaticFiles(directory=RESULTADOS_DIR, html=True),
    name="resultados_refinados",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
