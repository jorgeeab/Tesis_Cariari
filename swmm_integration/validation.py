from __future__ import annotations

from pathlib import Path

from .schemas import ValidationIssue, ValidationReport


REQUIRED_SECTIONS = {"OPTIONS", "JUNCTIONS", "OUTFALLS", "CONDUITS", "XSECTIONS"}


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            current = line.split("]", 1)[0][1:].upper()
            sections.setdefault(current, [])
            continue
        if current:
            clean = _strip_comment(raw)
            if clean:
                sections[current].append(clean)
    return sections


def read_inp_node_ids(inp_path: str | Path) -> set[str]:
    """Ids de nodos declarados en el INP (para auto-mapear tragantes)."""
    path = Path(inp_path)
    if not path.exists():
        return set()
    sections = _parse_sections(path.read_text(encoding="utf-8", errors="replace"))
    nodes: set[str] = set()
    for section in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS"):
        for line in sections.get(section, []):
            parts = line.split()
            if parts:
                nodes.add(parts[0])
    return nodes


def validate_inp_file(inp_path: str | Path) -> ValidationReport:
    path = Path(inp_path)
    issues: list[ValidationIssue] = []
    if not path.exists():
        return ValidationReport(
            valid=False,
            issues=[
                ValidationIssue(
                    level="error",
                    code="inp_missing",
                    message=f"No existe el archivo INP: {path}",
                )
            ],
        )

    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _parse_sections(text)

    for section in sorted(REQUIRED_SECTIONS - set(sections)):
        issues.append(
            ValidationIssue(
                level="error",
                code="missing_section",
                message=f"Falta la seccion [{section}].",
            )
        )

    subcatchments = sections.get("SUBCATCHMENTS", [])
    if subcatchments:
        issues.append(
            ValidationIssue(
                level="error",
                code="duplicate_runoff_risk",
                message="El INP contiene [SUBCATCHMENTS]. Para este flujo MVP SWMM no debe recalcular escorrentia.",
            )
        )

    nodes = set()
    for section in ("JUNCTIONS", "OUTFALLS"):
        for line in sections.get(section, []):
            parts = line.split()
            if parts:
                nodes.add(parts[0])

    for line in sections.get("CONDUITS", []):
        parts = line.split()
        if len(parts) < 5:
            issues.append(
                ValidationIssue(level="error", code="invalid_conduit", message=f"Conduit invalido: {line}")
            )
            continue
        conduit_id, from_node, to_node = parts[0], parts[1], parts[2]
        try:
            length = float(parts[3])
        except ValueError:
            length = 0.0
        if from_node not in nodes:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_from_node",
                    message=f"El conducto {conduit_id} referencia nodo inicial inexistente {from_node}.",
                    element_id=conduit_id,
                )
            )
        if to_node not in nodes:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_to_node",
                    message=f"El conducto {conduit_id} referencia nodo final inexistente {to_node}.",
                    element_id=conduit_id,
                )
            )
        if length <= 0:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="zero_length_conduit",
                    message=f"El conducto {conduit_id} tiene longitud cero o invalida.",
                    element_id=conduit_id,
                )
            )

    xsection_ids = {line.split()[0] for line in sections.get("XSECTIONS", []) if line.split()}
    for line in sections.get("CONDUITS", []):
        parts = line.split()
        if parts and parts[0] not in xsection_ids:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="missing_xsection",
                    message=f"El conducto {parts[0]} no tiene geometria en [XSECTIONS].",
                    element_id=parts[0],
                )
            )

    return ValidationReport(valid=not any(issue.level == "error" for issue in issues), issues=issues)

