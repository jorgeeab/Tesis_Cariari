from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


CoverageClass = Literal["base", "vegetation", "building", "street"]
DestinationKind = Literal["sink", "surface_outlet", "border", "no_valid_destination", "loop"]
CaptureMode = Literal["complete", "limited"]


class RainfallPoint(BaseModel):
    minute: float = Field(ge=0)
    intensity_mm_h: float = Field(ge=0)


class RainfallEvent(BaseModel):
    scenario_name: str = "base"
    start_time: datetime = Field(default_factory=datetime.utcnow)
    duration_min: float = Field(gt=0)
    interval_min: float = Field(gt=0)
    intensity_mm_h: float | None = Field(default=None, ge=0)
    time_series: list[RainfallPoint] | None = None
    safety_factor: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_event(self) -> "RainfallEvent":
        has_uniform = self.intensity_mm_h is not None
        has_series = bool(self.time_series)
        if has_uniform == has_series:
            raise ValueError("Configure lluvia uniforme o serie temporal, no ambas.")
        if has_series:
            points = self.time_series or []
            minutes = [p.minute for p in points]
            if minutes != sorted(minutes):
                raise ValueError("Los tiempos de lluvia deben estar ordenados.")
            if len(minutes) != len(set(minutes)):
                raise ValueError("La serie de lluvia contiene minutos duplicados.")
            if not minutes or minutes[0] != 0:
                raise ValueError("La serie de lluvia debe iniciar en minute=0.")
            if minutes[-1] > self.duration_min:
                raise ValueError("La serie de lluvia excede la duracion del evento.")
        return self


class RainfallInterval(BaseModel):
    start_min: float
    end_min: float
    duration_s: float
    intensity_mm_h: float
    depth_mm: float


class InfiltrationConfig(BaseModel):
    base: float = Field(default=0.22, ge=0, le=1)
    vegetation: float = Field(default=0.90, ge=0, le=1)
    building: float = Field(default=0.02, ge=0, le=1)
    street: float = Field(default=0.06, ge=0, le=1)

    def runoff_fraction(self, coverage: CoverageClass) -> float:
        return 1.0 - float(getattr(self, coverage))

    def as_runoff_config(self) -> dict[str, float]:
        return {
            "base": 1.0 - self.base,
            "vegetation": 1.0 - self.vegetation,
            "building": 1.0 - self.building,
            "street": 1.0 - self.street,
        }


class CaptureConfig(BaseModel):
    active: bool = True
    capture_mode: CaptureMode = "complete"
    nominal_capacity_l_s: float | None = Field(default=None, ge=0)
    capture_efficiency: float = Field(default=1.0, ge=0, le=1)
    obstruction_pct: float = Field(default=0.0, ge=0, le=100)
    surface_storage_m3: float = Field(default=0.0, ge=0)
    allow_bypass: bool = True

    def effective_capacity_m3s(self) -> float | None:
        if self.capture_mode == "complete":
            return None
        if self.nominal_capacity_l_s is None:
            raise ValueError("nominal_capacity_l_s es requerido en captura limitada.")
        return (
            self.nominal_capacity_l_s
            / 1000.0
            * self.capture_efficiency
            * (1.0 - self.obstruction_pct / 100.0)
        )


class SurfaceSink(BaseModel):
    id: str
    cell_index: int = Field(ge=0)
    lon: float | None = None
    lat: float | None = None
    capture: CaptureConfig = Field(default_factory=CaptureConfig)


class SurfaceGrid(BaseModel):
    nx: int = Field(gt=0)
    ny: int = Field(gt=0)
    cell_area_m2: float = Field(gt=0)
    receivers: list[int]
    coverage: list[CoverageClass] | None = None
    sinks: list[SurfaceSink] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grid(self) -> "SurfaceGrid":
        total = self.nx * self.ny
        if len(self.receivers) != total:
            raise ValueError("receivers debe tener nx*ny valores.")
        for idx, receiver in enumerate(self.receivers):
            if receiver < -1 or receiver >= total:
                raise ValueError(f"receiver invalido en celda {idx}: {receiver}")
        if self.coverage is not None and len(self.coverage) != total:
            raise ValueError("coverage debe tener nx*ny valores.")
        sink_cells = [sink.cell_index for sink in self.sinks]
        if any(cell >= total for cell in sink_cells):
            raise ValueError("Hay tragantes fuera del grid.")
        if len(sink_cells) != len(set(sink_cells)):
            raise ValueError("Hay multiples tragantes dentro de una misma celda.")
        return self


class TributaryArea(BaseModel):
    sink_id: str
    cell_count: int
    area_m2: float
    cell_indices: list[int]


class HydrographPoint(BaseModel):
    minute: float
    volume_m3: float
    flow_m3s: float
    captured_volume_m3: float
    captured_flow_m3s: float
    bypass_volume_m3: float
    bypass_flow_m3s: float


class HydrographResult(BaseModel):
    hydrology_id: str
    rainfall: RainfallEvent
    infiltration: InfiltrationConfig
    hydrographs: dict[str, list[HydrographPoint]]
    tributary_areas: list[TributaryArea]
    destination_counts: dict[DestinationKind, int]
    mass_balance: dict[str, float]
    warnings: list[str] = Field(default_factory=list)


class ScenarioRequest(BaseModel):
    rainfall: RainfallEvent
    infiltration: InfiltrationConfig = Field(default_factory=InfiltrationConfig)
    sink_capture: dict[str, CaptureConfig] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioResponse(BaseModel):
    scenario_id: str
    scenario: ScenarioRequest


class HydrologyRunRequest(BaseModel):
    rainfall: RainfallEvent
    surface: SurfaceGrid
    infiltration: InfiltrationConfig = Field(default_factory=InfiltrationConfig)


class SwmmValidateRequest(BaseModel):
    inp_path: str = "models/cariari_base.inp"


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    code: str
    message: str
    element_id: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class SwmmRunRequest(BaseModel):
    inp_path: str = "models/cariari_base.inp"
    hydrology_id: str | None = None
    hydrographs: dict[str, list[HydrographPoint]] | None = None
    node_mapping: dict[str, str] = Field(default_factory=dict)
    start_time: datetime | None = None
    duration_min: float | None = Field(default=None, gt=0)
    report_step_min: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def validate_source(self) -> "SwmmRunRequest":
        if not self.hydrology_id and not self.hydrographs:
            raise ValueError("Indique hydrology_id o hydrographs.")
        return self


class NodeResultPoint(BaseModel):
    id: str
    timestamp: datetime
    total_inflow_m3s: float
    lateral_inflow_m3s: float
    depth_m: float
    hydraulic_head_m: float
    flooding_m3s: float
    ponded_volume_m3: float
    status: str


class LinkResultPoint(BaseModel):
    id: str
    timestamp: datetime
    flow_m3s: float
    depth_m: float
    velocity_m_s: float
    capacity_ratio: float | None = None
    direction: str
    status: str


class SwmmRunResult(BaseModel):
    simulation_id: str
    inp_path: str
    node_results: dict[str, list[NodeResultPoint]]
    link_results: dict[str, list[LinkResultPoint]]
    summary: dict[str, Any]
