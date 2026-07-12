from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_UI_SCENARIO: dict[str, Any] = {
    "infiltration_pct": {
        "base": 22.0,
        "vegetation": 90.0,
        "building": 2.0,
        "street": 6.0,
    },
    "rainfall": {
        "scenario_name": "tormenta-base",
        "duration_min": 30.0,
        "interval_min": 5.0,
        "intensity_mm_h": 60.0,
        "safety_factor": 1.0,
    },
    "capture": {
        "active": True,
        "capture_mode": "complete",
        "nominal_capacity_l_s": 100.0,
        "capture_efficiency": 1.0,
        "obstruction_pct": 0.0,
        "surface_storage_m3": 0.0,
        "allow_bypass": True,
    },
    "swmm": {
        "inp_path": "models/cariari_base.inp",
        "report_step_min": 5.0,
        "node_mapping_text": "T1:T1\nT2:T2",
        "node_mapping": {"T1": "T1", "T2": "T2"},
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return max(lo, min(hi, resolved))


def parse_node_mapping(text: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in str(text or "").replace(";", "\n").replace(",", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
        elif "=" in line:
            left, right = line.split("=", 1)
        else:
            left, right = line, line
        left, right = left.strip(), right.strip()
        if left and right:
            mapping[left] = right
    return mapping


def normalize_ui_scenario(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = deep_merge(DEFAULT_UI_SCENARIO, config or {})

    infiltration = merged["infiltration_pct"]
    for key, default_value in DEFAULT_UI_SCENARIO["infiltration_pct"].items():
        infiltration[key] = clamp_float(infiltration.get(key), default_value, 0.0, 100.0)

    rainfall = merged["rainfall"]
    rainfall["scenario_name"] = str(rainfall.get("scenario_name") or "tormenta-base")
    rainfall["duration_min"] = clamp_float(rainfall.get("duration_min"), 30.0, 1.0, 1440.0)
    rainfall["interval_min"] = clamp_float(rainfall.get("interval_min"), 5.0, 0.5, 240.0)
    rainfall["intensity_mm_h"] = clamp_float(rainfall.get("intensity_mm_h"), 60.0, 0.0, 500.0)
    rainfall["safety_factor"] = clamp_float(rainfall.get("safety_factor"), 1.0, 0.1, 5.0)

    capture = merged["capture"]
    capture["active"] = bool(capture.get("active", True))
    if capture.get("capture_mode") not in {"complete", "limited"}:
        capture["capture_mode"] = "complete"
    capture["nominal_capacity_l_s"] = clamp_float(capture.get("nominal_capacity_l_s"), 100.0, 0.0, 5000.0)
    capture["capture_efficiency"] = clamp_float(capture.get("capture_efficiency"), 1.0, 0.0, 1.0)
    capture["obstruction_pct"] = clamp_float(capture.get("obstruction_pct"), 0.0, 0.0, 100.0)
    capture["surface_storage_m3"] = clamp_float(capture.get("surface_storage_m3"), 0.0, 0.0, 100000.0)
    capture["allow_bypass"] = bool(capture.get("allow_bypass", True))

    swmm = merged["swmm"]
    swmm["inp_path"] = str(swmm.get("inp_path") or "models/cariari_base.inp").strip()
    swmm["report_step_min"] = clamp_float(swmm.get("report_step_min"), 5.0, 0.5, 240.0)
    swmm["node_mapping_text"] = str(swmm.get("node_mapping_text") or "")
    swmm["node_mapping"] = parse_node_mapping(swmm["node_mapping_text"])
    return merged


def load_ui_scenario(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return normalize_ui_scenario(None)
    scenario_path = Path(path)
    if not scenario_path.exists():
        return normalize_ui_scenario(None)
    try:
        data = json.loads(scenario_path.read_text(encoding="utf-8"))
    except Exception:
        return normalize_ui_scenario(None)
    return normalize_ui_scenario(data)


def save_ui_scenario(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_ui_scenario(config)
    scenario_path = Path(path)
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized
