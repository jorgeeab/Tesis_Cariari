from swmm_integration.ui_config import normalize_ui_scenario


def test_ui_scenario_normalizes_editable_values():
    config = normalize_ui_scenario(
        {
            "infiltration_pct": {
                "base": 30,
                "vegetation": 85,
                "building": 3,
                "street": 8,
            },
            "rainfall": {
                "scenario_name": "prueba",
                "intensity_mm_h": 75,
                "duration_min": 45,
                "interval_min": 5,
                "safety_factor": 1.2,
            },
            "capture": {
                "capture_mode": "limited",
                "nominal_capacity_l_s": 120,
                "capture_efficiency": 0.85,
                "obstruction_pct": 10,
                "surface_storage_m3": 25,
            },
            "swmm": {
                "inp_path": "models/cariari_base.inp",
                "report_step_min": 5,
                "node_mapping_text": "T1:T1\nT2=T2",
            },
        }
    )

    assert config["infiltration_pct"]["vegetation"] == 85
    assert config["rainfall"]["scenario_name"] == "prueba"
    assert config["capture"]["capture_mode"] == "limited"
    assert config["capture"]["capture_efficiency"] == 0.85
    assert config["swmm"]["node_mapping"] == {"T1": "T1", "T2": "T2"}


def test_ui_scenario_clamps_ranges():
    config = normalize_ui_scenario(
        {
            "infiltration_pct": {"base": -10, "street": 900},
            "rainfall": {"intensity_mm_h": 900, "duration_min": -1},
            "capture": {"capture_efficiency": 9, "obstruction_pct": -5},
            "swmm": {"report_step_min": 0},
        }
    )

    assert config["infiltration_pct"]["base"] == 0
    assert config["infiltration_pct"]["street"] == 100
    assert config["rainfall"]["intensity_mm_h"] == 500
    assert config["rainfall"]["duration_min"] == 1
    assert config["capture"]["capture_efficiency"] == 1
    assert config["capture"]["obstruction_pct"] == 0
    assert config["swmm"]["report_step_min"] == 0.5
