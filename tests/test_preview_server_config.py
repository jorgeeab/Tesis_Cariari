from urllib.parse import parse_qs

from preview_server import build_swmm_config_from_query


def test_preview_swmm_config_parses_editable_values():
    config = build_swmm_config_from_query(
        parse_qs(
            "scenario_name=prueba"
            "&rain_intensity_mm_h=75"
            "&rain_duration_min=45"
            "&rain_interval_min=5"
            "&rain_safety_factor=1.2"
            "&capture_active=1"
            "&capture_mode=limited"
            "&nominal_capacity_l_s=120"
            "&capture_efficiency=0.85"
            "&obstruction_pct=10"
            "&surface_storage_m3=25"
            "&allow_bypass=1"
            "&inp_path=models/cariari_base.inp"
            "&report_step_min=5"
            "&node_mapping=T1:T1%0AT2%3DT2"
        )
    )

    assert config["rainfall"]["scenario_name"] == "prueba"
    assert config["rainfall"]["intensity_mm_h"] == 75
    assert config["capture"]["capture_mode"] == "limited"
    assert config["capture"]["capture_efficiency"] == 0.85
    assert config["swmm"]["node_mapping"] == {"T1": "T1", "T2": "T2"}


def test_preview_swmm_config_clamps_ranges():
    config = build_swmm_config_from_query(
        parse_qs(
            "rain_intensity_mm_h=900"
            "&rain_duration_min=-1"
            "&capture_efficiency=9"
            "&obstruction_pct=-5"
            "&report_step_min=0"
        )
    )

    assert config["rainfall"]["intensity_mm_h"] == 500
    assert config["rainfall"]["duration_min"] == 1
    assert config["capture"]["capture_efficiency"] == 1
    assert config["capture"]["obstruction_pct"] == 0
    assert config["swmm"]["report_step_min"] == 0.5
