#!/usr/bin/env python3
"""Convert a georeferenced TIFF/GeoTIFF raster to MBTiles with GDAL."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_OVERVIEWS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
TILE_FORMATS = ("PNG", "PNG8", "JPEG", "WEBP")
ZOOM_STRATEGIES = ("AUTO", "LOWER", "UPPER")
RESAMPLING = ("NEAREST", "BILINEAR", "CUBIC", "CUBICSPLINE", "LANCZOS", "MODE", "AVERAGE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte un TIFF/GeoTIFF georreferenciado a MBTiles raster. "
            "Usa GDAL CLI o rasterio."
        )
    )
    parser.add_argument("input", type=Path, help="Archivo .tif o .tiff de entrada.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Archivo .mbtiles de salida. Por defecto usa el mismo nombre del TIFF.",
    )
    parser.add_argument(
        "--gdal-bin",
        type=Path,
        help="Carpeta donde estan gdal_translate/gdaladdo, por ejemplo C:\\OSGeo4W\\bin.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "gdal", "rasterio"),
        default="auto",
        help="Motor de conversion. auto usa GDAL CLI si existe; si no, usa rasterio.",
    )
    parser.add_argument(
        "--tile-format",
        choices=TILE_FORMATS,
        default="PNG",
        help="Formato de cada tile. PNG conserva transparencia; JPEG suele pesar menos.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="Calidad para JPEG/WEBP, de 1 a 100.",
    )
    parser.add_argument(
        "--zoom-strategy",
        choices=ZOOM_STRATEGIES,
        default="AUTO",
        help="AUTO usa el zoom mas cercano a la resolucion real; UPPER fuerza mas detalle.",
    )
    parser.add_argument(
        "--resampling",
        choices=RESAMPLING,
        default="BILINEAR",
        help="Remuestreo usado al crear los tiles.",
    )
    parser.add_argument(
        "--overview-resampling",
        choices=RESAMPLING,
        default="AVERAGE",
        help="Remuestreo usado al crear niveles de zoom inferiores.",
    )
    parser.add_argument(
        "--overview-levels",
        default=",".join(str(level) for level in DEFAULT_OVERVIEWS),
        help="Factores de overviews separados por coma. Ejemplo: 2,4,8,16,32.",
    )
    parser.add_argument(
        "--no-overviews",
        action="store_true",
        help="No crea niveles de zoom inferiores con gdaladdo.",
    )
    parser.add_argument(
        "--name",
        help="Nombre que se guarda en la metadata del MBTiles.",
    )
    parser.add_argument(
        "--description",
        help="Descripcion que se guarda en la metadata del MBTiles.",
    )
    parser.add_argument(
        "--type",
        choices=("overlay", "baselayer"),
        default="overlay",
        help="Tipo de capa guardado en la metadata del MBTiles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el .mbtiles si ya existe.",
    )
    return parser


def candidate_gdal_dirs() -> list[Path]:
    dirs: list[Path] = []
    for raw_dir in (
        "C:/OSGeo4W/bin",
        "C:/Program Files/QGIS/bin",
        "C:/Program Files (x86)/QGIS/bin",
    ):
        dirs.append(Path(raw_dir))

    for parent in (Path("C:/Program Files"), Path("C:/Program Files (x86)")):
        if not parent.exists():
            continue
        dirs.extend(path / "bin" for path in parent.glob("QGIS*") if path.is_dir())

    return dirs


def resolve_tool(name: str, gdal_bin: Path | None) -> str:
    executable_names = [name]
    if not name.lower().endswith(".exe"):
        executable_names.append(f"{name}.exe")

    if gdal_bin:
        for executable in executable_names:
            candidate = gdal_bin / executable
            if candidate.exists():
                return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    for directory in candidate_gdal_dirs():
        for executable in executable_names:
            candidate = directory / executable
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(name)


def try_resolve_tool(name: str, gdal_bin: Path | None) -> str | None:
    try:
        return resolve_tool(name, gdal_bin)
    except FileNotFoundError:
        return None


def parse_overview_levels(raw_levels: str) -> list[str]:
    levels: list[str] = []
    for raw_level in raw_levels.split(","):
        level = raw_level.strip()
        if not level:
            continue
        if not level.isdigit() or int(level) < 2:
            raise ValueError(f"Nivel de overview invalido: {level!r}")
        levels.append(level)
    if not levels:
        raise ValueError("Debe indicar al menos un nivel de overview.")
    return levels


def run_command(args: list[str]) -> None:
    printable = " ".join(f'"{arg}"' if " " in arg else arg for arg in args)
    print(f"\n> {printable}")
    subprocess.run(args, check=True)


def output_path_for(input_path: Path, output_path: Path | None) -> Path:
    if output_path:
        return output_path
    return input_path.with_suffix(".mbtiles")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")
    if input_path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("El archivo de entrada debe terminar en .tif o .tiff.")

    output_path = output_path_for(input_path, args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".mbtiles":
        output_path = output_path.with_suffix(".mbtiles")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Ya existe {output_path}. Use --overwrite o indique otro --output."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality debe estar entre 1 y 100.")

    levels = [] if args.no_overviews else parse_overview_levels(args.overview_levels)
    return input_path, output_path, levels


def convert_with_gdal(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    overview_levels: list[str],
    gdal_translate: str,
    gdaladdo: str | None,
) -> None:
    translate_cmd = [
        gdal_translate,
        "-of",
        "MBTILES",
        "-co",
        f"TILE_FORMAT={args.tile_format}",
        "-co",
        f"ZOOM_LEVEL_STRATEGY={args.zoom_strategy}",
        "-co",
        f"RESAMPLING={args.resampling}",
        "-co",
        f"TYPE={args.type}",
    ]

    if args.tile_format in ("JPEG", "WEBP"):
        translate_cmd.extend(["-co", f"QUALITY={args.quality}"])
    if args.name:
        translate_cmd.extend(["-co", f"NAME={args.name}"])
    if args.description:
        translate_cmd.extend(["-co", f"DESCRIPTION={args.description}"])

    translate_cmd.extend([str(input_path), str(output_path)])
    run_command(translate_cmd)

    if overview_levels and gdaladdo:
        run_command(
            [
                gdaladdo,
                "-r",
                args.overview_resampling.lower(),
                str(output_path),
                *overview_levels,
            ]
        )


def rasterio_resampling(name: str):
    from rasterio.enums import Resampling

    names = {
        "NEAREST": Resampling.nearest,
        "BILINEAR": Resampling.bilinear,
        "CUBIC": Resampling.cubic,
        "CUBICSPLINE": Resampling.cubic_spline,
        "LANCZOS": Resampling.lanczos,
        "MODE": Resampling.mode,
        "AVERAGE": Resampling.average,
    }
    return names[name]


def convert_with_rasterio(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    overview_levels: list[str],
) -> None:
    try:
        import rasterio
        from rasterio.shutil import copy as rasterio_copy
    except ImportError as exc:
        raise RuntimeError(
            "No se encontro rasterio. Instale rasterio o use GDAL CLI con --backend gdal."
        ) from exc

    creation_options = {
        "TILE_FORMAT": args.tile_format,
        "ZOOM_LEVEL_STRATEGY": args.zoom_strategy,
        "RESAMPLING": args.resampling,
        "TYPE": args.type,
    }
    if args.tile_format in ("JPEG", "WEBP"):
        creation_options["QUALITY"] = str(args.quality)
    if args.name:
        creation_options["NAME"] = args.name
    if args.description:
        creation_options["DESCRIPTION"] = args.description

    print("\n> rasterio copy MBTiles")
    with rasterio.open(input_path) as dataset:
        rasterio_copy(
            dataset,
            output_path,
            driver="MBTiles",
            copy_src_overviews=False,
            **creation_options,
        )

    if overview_levels:
        int_levels = [int(level) for level in overview_levels]
        print(f"\n> rasterio build_overviews {','.join(overview_levels)}")
        with rasterio.open(output_path, "r+") as dataset:
            dataset.build_overviews(
                int_levels,
                rasterio_resampling(args.overview_resampling),
            )
            dataset.update_tags(
                ns="rio_overview",
                resampling=args.overview_resampling.lower(),
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path, output_path, overview_levels = validate_args(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        gdal_translate = None
        gdaladdo = None
        if args.backend in ("auto", "gdal"):
            gdal_translate = try_resolve_tool("gdal_translate", args.gdal_bin)
            gdaladdo = try_resolve_tool("gdaladdo", args.gdal_bin) if overview_levels else None
            if args.backend == "gdal" and (not gdal_translate or (overview_levels and not gdaladdo)):
                raise RuntimeError(
                    "No se encontraron gdal_translate/gdaladdo. Instale GDAL, QGIS u OSGeo4W "
                    "y agregue la carpeta bin al PATH, o use --gdal-bin \"C:\\OSGeo4W\\bin\"."
                )

        if output_path.exists() and args.overwrite:
            output_path.unlink()

        if gdal_translate and (gdaladdo or not overview_levels):
            convert_with_gdal(args, input_path, output_path, overview_levels, gdal_translate, gdaladdo)
        else:
            if args.backend == "auto":
                print("\nGDAL CLI no disponible; usando backend rasterio.")
            convert_with_rasterio(args, input_path, output_path, overview_levels)
    except subprocess.CalledProcessError as exc:
        print(f"\nGDAL fallo con codigo {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"\nListo: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
