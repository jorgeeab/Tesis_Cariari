from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados_refinados"
GUIA_INDEX = "/resultados_refinados/Guia/index.html"


app = FastAPI(title="Guia SBNS Cariari")


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
