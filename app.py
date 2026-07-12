import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from preview_server import render_preview_html
from swmm_integration import PySwmmUnavailableError, SwmmSimulationService, generate_hydrographs
from swmm_integration.schemas import (
    CaptureConfig,
    HydrographPoint,
    HydrographResult,
    HydrologyRunRequest,
    InfiltrationConfig,
    RainfallEvent,
    ScenarioRequest,
    ScenarioResponse,
    SwmmRunRequest,
    SwmmRunResult,
    SwmmValidateRequest,
    ValidationReport,
)
from swmm_integration.simulation import build_simulation_frames
from swmm_integration.surface_grid import (
    SurfaceGridUnavailableError,
    build_surface_grid,
    load_surface_grid_payload,
    tributaries_for,
)
from swmm_integration.ui_config import (
    deep_merge,
    load_ui_scenario,
    normalize_ui_scenario,
    save_ui_scenario,
)
from swmm_integration.validation import read_inp_node_ids


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados_refinados"
UI_SCENARIO_PATH = BASE_DIR / "results" / "swmm" / "ui_scenario.json"
_GUIA_PATH  = "/resultados_refinados/Guia/index.html"
_MAPA3D_PATH = "/resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html"
MAPA3D_FILE = RESULTADOS_DIR / "Refinado_30_ThreeJS_Cortina_Ajustada.html"

# ROOT_PATH allows running under a subpath proxy (e.g. nginx /sbn → this app)
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")
GUIA_INDEX  = ROOT_PATH + _GUIA_PATH
MAPA3D_URL  = ROOT_PATH + _MAPA3D_PATH
EXCAVATION_SCRIPT = BASE_DIR / "generar_dem_3d_threejs_satelital.py"

app = FastAPI(title="Guia SBNS Cariari")
swmm_service = SwmmSimulationService()
excavation_run_lock = threading.Lock()

SCENARIOS: dict[str, ScenarioRequest] = {}
HYDROLOGY_RESULTS: dict[str, HydrographResult] = {}
SWMM_RESULTS: dict[str, SwmmRunResult] = {}

SURFACE_GRID_PATH = BASE_DIR / "results" / "hydrology" / "surface_grid.json"
SIMULATION_FRAMES: dict[str, dict[str, Any]] = {}
LATEST_SIMULATION_ID: str | None = None


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _read_map_terrain_params() -> dict[str, Any]:
    if not MAPA3D_FILE.exists():
        return {}
    text = MAPA3D_FILE.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"const\s+TERRAIN_PARAMS\s*=\s*(\{.*?\});", text)
    if not match:
        return {}
    parsed = json.loads(match.group(1))
    return parsed if isinstance(parsed, dict) else {}


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


@app.get("/ajustar-excavacion", response_class=HTMLResponse, include_in_schema=False)
def ajustar_excavacion() -> HTMLResponse:
    page = render_preview_html(
        run_url=ROOT_PATH + "/api/excavation/run",
        map_url=MAPA3D_URL,
    )
    return HTMLResponse(page, headers={"Cache-Control": "no-cache"})


@app.get("/api/excavation/run", include_in_schema=False)
def run_excavation_adjustment(
    street_depth: float = Query(1.5, ge=0.3, le=3.0),
    street_width: float = Query(1.0, ge=0.4, le=3.0),
    street_ramp: float = Query(7.0, ge=1.0, le=20.0),
    river_depth: float = Query(2.5, ge=0.5, le=6.0),
    river_width: float = Query(4.0, ge=1.0, le=15.0),
) -> StreamingResponse:
    command = [
        sys.executable,
        str(EXCAVATION_SCRIPT),
        "--street-depth", f"{street_depth:.1f}",
        "--street-width", f"{street_width:.1f}",
        "--street-ramp", f"{street_ramp:.1f}",
        "--river-depth", f"{river_depth:.1f}",
        "--river-width", f"{river_width:.1f}",
    ]

    def event_stream():
        if not excavation_run_lock.acquire(blocking=False):
            yield f"data: {json.dumps('ERROR: Ya hay un ajuste en ejecución.', ensure_ascii=False)}\n\n"
            yield "data: __done__\n\n"
            return

        process = None
        try:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(BASE_DIR),
            )
            assert process.stdout is not None
            for line in process.stdout:
                message = json.dumps(line.rstrip(), ensure_ascii=False)
                yield f"data: {message}\n\n"

            return_code = process.wait()
            if return_code == 0:
                expected = {
                    "street_depth": street_depth,
                    "street_width": street_width,
                    "street_ramp": street_ramp,
                    "river_depth": river_depth,
                    "river_width": river_width,
                }
                actual = _read_map_terrain_params()
                mismatches = []
                for key, expected_value in expected.items():
                    try:
                        actual_value = float(actual[key])
                    except (KeyError, TypeError, ValueError):
                        mismatches.append(f"{key}=sin dato")
                        continue
                    if abs(actual_value - float(expected_value)) > 0.06:
                        mismatches.append(f"{key}={actual_value:.1f} esperado {expected_value:.1f}")
                if mismatches:
                    error = "ERROR: El HTML generado no coincide con los parametros solicitados: " + ", ".join(mismatches)
                    yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                else:
                    message = (
                        f"VERIFICADO: mapa principal actualizado "
                        f"(rio {river_depth:.1f} m, calles {street_depth:.1f} m)."
                    )
                    yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
            if return_code != 0:
                error = f"ERROR: El generador terminó con código {return_code}."
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error = f"ERROR: {exc}"
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
            excavation_run_lock.release()

        yield "data: __done__\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ui/scenario")
def get_ui_scenario() -> dict[str, Any]:
    return load_ui_scenario(UI_SCENARIO_PATH)


@app.post("/api/ui/scenario")
def update_ui_scenario(config: dict[str, Any]) -> dict[str, Any]:
    normalized = save_ui_scenario(UI_SCENARIO_PATH, config)
    return {
        "ok": True,
        "path": _display_path(UI_SCENARIO_PATH),
        "config": normalized,
    }


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


@app.get("/api/simulation/grid-status")
def simulation_grid_status() -> dict[str, Any]:
    try:
        payload = load_surface_grid_payload(SURFACE_GRID_PATH)
    except SurfaceGridUnavailableError as exc:
        return {"available": False, "detail": str(exc)}
    return {
        "available": True,
        "path": _display_path(SURFACE_GRID_PATH),
        "generated_at": payload.get("generated_at"),
        "nx": payload.get("nx"),
        "ny": payload.get("ny"),
        "sink_count": len(payload.get("sinks", [])),
        "params": payload.get("params", {}),
        "warnings": payload.get("warnings", []),
    }


@app.post("/api/simulation/run")
def run_rain_simulation(overrides: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    """Orquesta hidrología → SWMM sobre el grid D8 del DEM excavado y devuelve
    frames listos para el visor 3D (timeline + valores por tragante).

    Si PySWMM no está disponible, responde 200 en modo degradado
    (swmm_available=false) con los hidrogramas de superficie.
    """
    global LATEST_SIMULATION_ID
    cfg = normalize_ui_scenario(deep_merge(load_ui_scenario(UI_SCENARIO_PATH), overrides or {}))
    try:
        payload = load_surface_grid_payload(SURFACE_GRID_PATH)
    except SurfaceGridUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    capture = CaptureConfig(**cfg["capture"])
    grid = build_surface_grid(payload, capture)
    rainfall = RainfallEvent(
        scenario_name=cfg["rainfall"]["scenario_name"],
        duration_min=cfg["rainfall"]["duration_min"],
        interval_min=cfg["rainfall"]["interval_min"],
        intensity_mm_h=cfg["rainfall"]["intensity_mm_h"],
        safety_factor=cfg["rainfall"]["safety_factor"],
    )
    infiltration = InfiltrationConfig(
        **{key: value / 100.0 for key, value in cfg["infiltration_pct"].items()}
    )
    hydro = generate_hydrographs(
        HydrologyRunRequest(rainfall=rainfall, surface=grid, infiltration=infiltration),
        precomputed_tributaries=tributaries_for(SURFACE_GRID_PATH, grid),
    )
    HYDROLOGY_RESULTS[hydro.hydrology_id] = hydro

    swmm_result = None
    swmm_error = None
    inp_path = _resolve_project_path(cfg["swmm"]["inp_path"])
    # Mapeo efectivo: identidad para los tragantes que existen como nodo en el
    # INP, más el mapeo explícito del escenario (gana), filtrado a sinks y
    # nodos reales para no abortar SWMM por entradas obsoletas.
    inp_nodes = read_inp_node_ids(inp_path)
    node_mapping = {
        sink_id: sink_id for sink_id in hydro.hydrographs if sink_id in inp_nodes
    }
    for sink_id, node_id in cfg["swmm"]["node_mapping"].items():
        if sink_id in hydro.hydrographs and node_id in inp_nodes:
            node_mapping[sink_id] = node_id
    cfg["swmm"]["node_mapping"] = node_mapping  # los frames usan el mapeo efectivo

    if not node_mapping:
        swmm_error = (
            "Ningún tragante del grid existe como nodo en el INP "
            f"({cfg['swmm']['inp_path']}); se usa solo hidrología superficial."
        )
    else:
        # flow_at_minute retiene el último valor del hidrograma; se cierra con
        # un punto cero al fin de la lluvia para que la cola de drenaje no
        # reciba lluvia fantasma.
        zero_tail = HydrographPoint(
            minute=rainfall.duration_min, volume_m3=0.0, flow_m3s=0.0,
            captured_volume_m3=0.0, captured_flow_m3s=0.0,
            bypass_volume_m3=0.0, bypass_flow_m3s=0.0,
        )
        swmm_hydrographs = {
            sink_id: list(points) + [zero_tail]
            for sink_id, points in hydro.hydrographs.items()
        }
        try:
            swmm_result = swmm_service.run_simulation(
                SwmmRunRequest(
                    inp_path=str(inp_path),
                    hydrographs=swmm_hydrographs,
                    node_mapping=node_mapping,
                    start_time=rainfall.start_time,
                    # cola de drenaje tras la lluvia para la visualización
                    duration_min=rainfall.duration_min + 30.0,
                    report_step_min=cfg["swmm"]["report_step_min"],
                )
            )
            SWMM_RESULTS[swmm_result.simulation_id] = swmm_result
        except (PySwmmUnavailableError, KeyError, ValueError) as exc:
            swmm_error = str(exc)

    frames = build_simulation_frames(hydro, swmm_result, cfg, swmm_error=swmm_error)
    simulation_id = str(uuid4())
    frames["simulation_id"] = simulation_id
    frames["hydrology_id"] = hydro.hydrology_id
    frames["swmm_simulation_id"] = swmm_result.simulation_id if swmm_result else None
    SIMULATION_FRAMES[simulation_id] = frames
    LATEST_SIMULATION_ID = simulation_id
    return frames


@app.get("/api/simulation/latest/frames")
def get_latest_simulation_frames() -> dict[str, Any]:
    if LATEST_SIMULATION_ID is None:
        raise HTTPException(status_code=404, detail="No hay simulaciones ejecutadas.")
    return SIMULATION_FRAMES[LATEST_SIMULATION_ID]


@app.get("/api/simulation/{simulation_id}/frames")
def get_simulation_frames(simulation_id: str) -> dict[str, Any]:
    frames = SIMULATION_FRAMES.get(simulation_id)
    if frames is None:
        raise HTTPException(status_code=404, detail=f"No existe simulation_id: {simulation_id}")
    return frames


@app.get(_MAPA3D_PATH, include_in_schema=False)
def mapa3d_html() -> FileResponse:
    if not MAPA3D_FILE.exists():
        raise HTTPException(status_code=404, detail="Mapa 3D no generado.")
    return FileResponse(
        MAPA3D_FILE,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


app.mount(
    "/resultados_refinados",
    StaticFiles(directory=RESULTADOS_DIR, html=True),
    name="resultados_refinados",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
