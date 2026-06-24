from __future__ import annotations

from .schemas import RainfallEvent, RainfallInterval, RainfallPoint


def _intensity_at(points: list[RainfallPoint], minute: float) -> float:
    current = points[0].intensity_mm_h
    for point in points:
        if point.minute > minute:
            break
        current = point.intensity_mm_h
    return current


def _integrate_piecewise_intensity(points: list[RainfallPoint], start_min: float, end_min: float) -> float:
    """Return average intensity in mm/h for [start_min, end_min).

    The time series is interpreted as stepwise-constant. A value remains active
    until the next point. This keeps SWMM inflows synchronized with the external
    runoff calculation and documents the interpolation strategy explicitly.
    """
    if end_min <= start_min:
        return 0.0

    breakpoints = [start_min, end_min]
    for point in points:
        if start_min < point.minute < end_min:
            breakpoints.append(point.minute)
    breakpoints = sorted(set(breakpoints))

    weighted = 0.0
    for left, right in zip(breakpoints, breakpoints[1:]):
        intensity = _intensity_at(points, left)
        weighted += intensity * (right - left)
    return weighted / (end_min - start_min)


def build_rainfall_intervals(event: RainfallEvent) -> list[RainfallInterval]:
    """Convert rainfall configuration into physical interval depths.

    Units:
    - intensity_mm_h: millimeters per hour
    - depth_mm: millimeters during the interval
    - duration_s: seconds
    """
    intervals: list[RainfallInterval] = []
    t = 0.0
    while t < event.duration_min - 1e-9:
        end = min(event.duration_min, t + event.interval_min)
        dt_min = end - t
        if event.time_series:
            intensity = _integrate_piecewise_intensity(event.time_series, t, end)
        else:
            intensity = float(event.intensity_mm_h or 0.0)
        intensity *= event.safety_factor
        depth_mm = intensity * dt_min / 60.0
        intervals.append(
            RainfallInterval(
                start_min=t,
                end_min=end,
                duration_s=dt_min * 60.0,
                intensity_mm_h=intensity,
                depth_mm=depth_mm,
            )
        )
        t = end
    return intervals

