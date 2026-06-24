from datetime import datetime

from swmm_integration.hydrographs import generate_hydrographs
from swmm_integration.schemas import (
    CaptureConfig,
    HydrologyRunRequest,
    InfiltrationConfig,
    RainfallEvent,
    SurfaceGrid,
    SurfaceSink,
)


def _rainfall(intensity_mm_h: float = 60.0) -> RainfallEvent:
    return RainfallEvent(
        scenario_name="test",
        start_time=datetime(2026, 1, 1),
        duration_min=10,
        interval_min=10,
        intensity_mm_h=intensity_mm_h,
    )


def _surface(capture: CaptureConfig | None = None) -> SurfaceGrid:
    return SurfaceGrid(
        nx=2,
        ny=1,
        cell_area_m2=100.0,
        receivers=[1, -1],
        coverage=["base", "base"],
        sinks=[SurfaceSink(id="T1", cell_index=1, capture=capture or CaptureConfig())],
    )


def test_zero_rain_produces_zero_flow():
    result = generate_hydrographs(
        HydrologyRunRequest(
            rainfall=_rainfall(0.0),
            surface=_surface(),
            infiltration=InfiltrationConfig(base=0.0),
        )
    )

    assert result.mass_balance["rainfall_m3"] == 0.0
    assert result.hydrographs["T1"][0].flow_m3s == 0.0
    assert result.mass_balance["error_pct"] == 0.0


def test_full_infiltration_removes_surface_runoff():
    result = generate_hydrographs(
        HydrologyRunRequest(
            rainfall=_rainfall(),
            surface=_surface(),
            infiltration=InfiltrationConfig(base=1.0),
        )
    )

    assert result.mass_balance["rainfall_m3"] == 2.0
    assert result.mass_balance["infiltrated_m3"] == 2.0
    assert result.mass_balance["runoff_m3"] == 0.0
    assert result.hydrographs["T1"][0].captured_flow_m3s == 0.0


def test_runoff_volume_and_flow_are_physical_units():
    result = generate_hydrographs(
        HydrologyRunRequest(
            rainfall=_rainfall(),
            surface=_surface(),
            infiltration=InfiltrationConfig(base=0.0),
        )
    )

    point = result.hydrographs["T1"][0]
    assert result.tributary_areas[0].cell_count == 2
    assert result.tributary_areas[0].area_m2 == 200.0
    assert point.volume_m3 == 2.0
    assert point.flow_m3s == 2.0 / 600.0
    assert result.mass_balance["captured_m3"] == 2.0
    assert result.mass_balance["error_pct"] == 0.0


def test_limited_capture_creates_bypass_without_double_counting():
    result = generate_hydrographs(
        HydrologyRunRequest(
            rainfall=_rainfall(),
            surface=_surface(
                CaptureConfig(
                    capture_mode="limited",
                    nominal_capacity_l_s=1.0,
                )
            ),
            infiltration=InfiltrationConfig(base=0.0),
        )
    )

    point = result.hydrographs["T1"][0]
    assert point.captured_flow_m3s == 0.001
    assert point.captured_volume_m3 == 0.6
    assert round(point.bypass_volume_m3, 10) == 1.4
    assert result.mass_balance["error_pct"] == 0.0
