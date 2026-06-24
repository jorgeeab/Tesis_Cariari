from pathlib import Path

from swmm_integration.validation import validate_inp_file


def test_base_inp_is_valid():
    report = validate_inp_file("models/cariari_base.inp")

    assert report.valid
    assert report.issues == []


def test_subcatchments_are_rejected_to_avoid_double_runoff(tmp_path: Path):
    inp = tmp_path / "double_runoff.inp"
    inp.write_text(
        """
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
J1 0 1 0 0 0
[OUTFALLS]
O1 0 FREE NO
[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 0.6 0 0 0 1
[SUBCATCHMENTS]
S1 RG1 J1 1.0 10 100 0.0 0.0
""",
        encoding="utf-8",
    )

    report = validate_inp_file(inp)

    assert not report.valid
    assert any(issue.code == "duplicate_runoff_risk" for issue in report.issues)


def test_invalid_conduit_geometry_is_reported(tmp_path: Path):
    inp = tmp_path / "bad_conduit.inp"
    inp.write_text(
        """
[OPTIONS]
FLOW_UNITS CMS
[JUNCTIONS]
J1 0 1 0 0 0
[OUTFALLS]
O1 0 FREE NO
[CONDUITS]
C1 J1 O1 0 0.013 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 0.6 0 0 0 1
""",
        encoding="utf-8",
    )

    report = validate_inp_file(inp)

    assert not report.valid
    assert any(issue.code == "zero_length_conduit" for issue in report.issues)
