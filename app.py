import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from swmm_integration import PySwmmUnavailableError, SwmmSimulationService, generate_hydrographs
from swmm_integration.schemas import (
    HydrographResult,
    HydrologyRunRequest,
    ScenarioRequest,
    ScenarioResponse,
    SwmmRunRequest,
    SwmmRunResult,
    SwmmValidateRequest,
    ValidationReport,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados_refinados"
_GUIA_PATH  = "/resultados_refinados/Guia/index.html"
_MAPA3D_PATH = "/resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html"

# ROOT_PATH allows running under a subpath proxy (e.g. nginx /sbn → this app)
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")
GUIA_INDEX  = ROOT_PATH + _GUIA_PATH
MAPA3D_URL  = ROOT_PATH + _MAPA3D_PATH

app = FastAPI(title="Guia SBNS Cariari")
swmm_service = SwmmSimulationService()

SCENARIOS: dict[str, ScenarioRequest] = {}
HYDROLOGY_RESULTS: dict[str, HydrographResult] = {}
SWMM_RESULTS: dict[str, SwmmRunResult] = {}


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


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


@app.post("/api/scenarios", response_model=ScenarioResponse)
def create_scenario(request: ScenarioRequest) -> ScenarioResponse:
    scenario_id = str(uuid4())
    SCENARIOS[scenario_id] = request
    return ScenarioResponse(scenario_id=scenario_id, scenario=request)


@app.post("/api/hydrology/run", response_model=HydrographResult)
def run_hydrology(request: HydrologyRunRequest) -> HydrographResult:
    result = generate_hydrographs(request)
    HYDROLOGY_RESULTS[result.hydrology_id] = result
    return result


@app.post("/api/swmm/validate", response_model=ValidationReport)
def validate_swmm(request: SwmmValidateRequest) -> ValidationReport:
    return swmm_service.validate_model(_resolve_project_path(request.inp_path))


@app.post("/api/swmm/run", response_model=SwmmRunResult)
def run_swmm(request: SwmmRunRequest) -> SwmmRunResult:
    hydrographs = request.hydrographs
    start_time = request.start_time
    duration_min = request.duration_min

    if request.hydrology_id:
        hydrology = HYDROLOGY_RESULTS.get(request.hydrology_id)
        if hydrology is None:
            raise HTTPException(status_code=404, detail=f"No existe hydrology_id: {request.hydrology_id}")
        hydrographs = hydrology.hydrographs
        start_time = start_time or hydrology.rainfall.start_time
        duration_min = duration_min or hydrology.rainfall.duration_min

    resolved_request = request.model_copy(
        update={
            "inp_path": str(_resolve_project_path(request.inp_path)),
            "hydrographs": hydrographs,
            "start_time": start_time,
            "duration_min": duration_min,
        }
    )
    try:
        result = swmm_service.run_simulation(resolved_request)
    except PySwmmUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    SWMM_RESULTS[result.simulation_id] = result
    return result


def _get_swmm_result(simulation_id: str) -> SwmmRunResult:
    result = SWMM_RESULTS.get(simulation_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No existe simulation_id: {simulation_id}")
    return result


@app.get("/api/swmm/results/{simulation_id}", response_model=SwmmRunResult)
def get_swmm_result(simulation_id: str) -> SwmmRunResult:
    return _get_swmm_result(simulation_id)


@app.get("/api/swmm/results/{simulation_id}/nodes")
def get_swmm_nodes(simulation_id: str):
    return _get_swmm_result(simulation_id).node_results


@app.get("/api/swmm/results/{simulation_id}/links")
def get_swmm_links(simulation_id: str):
    return _get_swmm_result(simulation_id).link_results


@app.get("/api/swmm/results/{simulation_id}/summary")
def get_swmm_summary(simulation_id: str):
    return _get_swmm_result(simulation_id).summary


app.mount(
    "/resultados_refinados",
    StaticFiles(directory=RESULTADOS_DIR, html=True),
    name="resultados_refinados",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
