import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados_refinados"
_GUIA_PATH = "/resultados_refinados/Guia/index.html"

# ROOT_PATH allows running under a subpath proxy (e.g. nginx /sbn → this app)
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")
GUIA_INDEX = ROOT_PATH + _GUIA_PATH

app = FastAPI(title="Guia SBNS Cariari", root_path=ROOT_PATH)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url=GUIA_INDEX, status_code=307)


@app.get("/guia", include_in_schema=False)
@app.get("/guia/", include_in_schema=False)
def guia() -> RedirectResponse:
    return RedirectResponse(url=GUIA_INDEX, status_code=307)


app.mount(
    "/resultados_refinados",
    StaticFiles(directory=RESULTADOS_DIR, html=True),
    name="resultados_refinados",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
