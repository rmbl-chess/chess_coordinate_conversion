"""Batch-transform NAD83 coordinates or WKT geometries in CSV files to ITRF2000."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.errors import WKTReadingError
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

PIPELINE_DEF = """
+proj=pipeline
+step +proj=cart +ellps=GRS80
+step +proj=helmert +inv
        +x=0.99343 +y=-1.90331 +z=-0.52655
        +rx=0.02591467 +ry=0.00942645 +rz=0.01159935
        +s=0.00171504
        +dx=0.00079 +dy=-0.00060 +dz=-0.00134
        +drx=0.00006667 +dry=-0.00075744 +drz=-0.00005133
        +ds=-0.00010201
        +t_epoch=2025.5
        +convention=coordinate_frame
+step +proj=helmert +inv
        +x=0.0019 +y=0.0017 +z=0.0105
        +rx=0 +ry=0 +rz=0
        +s=-0.00000134
        +dx=-0.0001 +dy=-0.0001 +dz=0.0018
        +drx=0 +dry=0 +drz=0
        +ds=-0.00000008
        +t_epoch=2025.5
        +convention=position_vector
+step +proj=cart +ellps=GRS80 +inv
"""

TRANSFORMER = Transformer.from_pipeline(PIPELINE_DEF)


def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(
            description=(
                  "Transform NAD83 longitude/latitude/height columns or WKT geometries stored in a CSV "
                  "into the ITRF2000 reference frame via the configured PROJ pipeline."
            )
      )
      parser.add_argument("input_csv", type=Path, help="Path to the source CSV file")
      parser.add_argument("output_csv", type=Path, help="Destination path for the transformed CSV")
      parser.add_argument(
            "--lon-col",
            default=None,
            help="Name of the longitude column in degrees. Requires --lat-col when provided.",
      )
      parser.add_argument(
            "--lat-col",
            default=None,
            help="Name of the latitude column in degrees. Requires --lon-col when provided.",
      )
      parser.add_argument(
            "--height-col",
            default=None,
            help="Optional ellipsoidal height column (meters). Defaults to --default-height when missing.",
      )
      parser.add_argument(
            "--epoch-col",
            default=None,
            help="Optional epoch column (decimal year). Defaults to --default-epoch when missing.",
      )
      parser.add_argument(
            "--default-height",
            type=float,
            default=0.0,
            help="Fallback ellipsoidal height in meters when no height column is supplied.",
      )
      parser.add_argument(
            "--default-epoch",
            type=float,
            default=2024.5,
            help="Fallback epoch (decimal year) when no epoch column is supplied.",
      )
      parser.add_argument(
            "--wkt-col",
            default=None,
            help="Column containing WKT geometries (e.g., POLYGON, MULTIPOLYGON) to transform.",
      )
      parser.add_argument(
            "--wkt-out",
            default=None,
            help="Output column name for transformed WKT geometries. Defaults to '<wkt-col>_itrf2000'.",
      )
      parser.add_argument(
            "--lon-out",
            default="lon_itrf2000",
            help="Name for the transformed longitude column (degrees).",
      )
      parser.add_argument(
            "--lat-out",
            default="lat_itrf2000",
            help="Name for the transformed latitude column (degrees).",
      )
      parser.add_argument(
            "--height-out",
            default="height_itrf2000",
            help="Name for the transformed height column (meters).",
      )
      parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow overwriting an existing output CSV.",
      )
      return parser.parse_args()


def _required_float_series(df: pl.DataFrame, column: str) -> np.ndarray:
      if column not in df.columns:
            raise ValueError(f"Column '{column}' is required but not present in the input CSV.")
      series = df.get_column(column).cast(pl.Float64, strict=False)
      if series.null_count() > 0:
            raise ValueError(f"Column '{column}' contains non-numeric or missing values that must be fixed.")
      values = series.to_numpy(zero_copy_only=False)
      if np.isnan(values).any():
            raise ValueError(f"Column '{column}' contains NaN values that must be addressed before transforming.")
      return values


def _optional_float_series(
      df: pl.DataFrame,
      column: Optional[str],
      default_value: float,
) -> np.ndarray:
      if column is None:
            return np.full(df.height, default_value, dtype=float)
      if column not in df.columns:
            raise ValueError(f"Column '{column}' was provided but is missing from the input CSV.")
      series = df.get_column(column).cast(pl.Float64, strict=False)
      if series.null_count() > 0:
            series = series.fill_null(default_value)
      series = series.fill_nan(default_value)
      return series.to_numpy(zero_copy_only=False)


def _transform_geometry(
      geom: BaseGeometry,
      height_value: float,
      epoch_value: float,
) -> BaseGeometry:
      """Transform a single geometry while respecting height and epoch defaults."""

      def _apply(xs, ys, zs=None):
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            if zs is None:
                  z_arr = np.full_like(x_arr, height_value)
                  t_arr = np.full_like(x_arr, epoch_value, dtype=float)
                  x_new, y_new, _, _ = TRANSFORMER.transform(x_arr, y_arr, z_arr, t_arr)
                  return x_new, y_new
            z_raw = np.asarray(zs, dtype=float)
            if np.isnan(z_raw).all():
                  z_arr = np.full_like(z_raw, height_value)
            else:
                  z_arr = np.nan_to_num(z_raw, nan=height_value)
            t_arr = np.full_like(z_arr, epoch_value, dtype=float)
            x_new, y_new, z_new, _ = TRANSFORMER.transform(x_arr, y_arr, z_arr, t_arr)
            return x_new, y_new, z_new

      return shapely_transform(_apply, geom)


def _transform_wkt_column(
      df: pl.DataFrame,
      wkt_col: str,
      wkt_out: str,
      height_vals: np.ndarray,
      epoch_vals: np.ndarray,
) -> pl.Series:
      original = df.get_column(wkt_col)
      transformed = []
      for idx, value in enumerate(original.to_list()):
            if value is None:
                  transformed.append(None)
                  continue
            wkt_text = str(value).strip()
            if not wkt_text:
                  transformed.append(None)
                  continue
            try:
                  geom = shapely_wkt.loads(wkt_text)
            except (WKTReadingError, AttributeError, TypeError) as exc:
                  raise ValueError(
                        f"Unable to parse WKT in row {idx} of column '{wkt_col}'."
                  ) from exc
            geom_itrf = _transform_geometry(
                  geom,
                  float(height_vals[idx]),
                  float(epoch_vals[idx]),
            )
            transformed.append(geom_itrf.wkt)
      return pl.Series(wkt_out, transformed)


def transform_dataframe(
      df: pl.DataFrame,
      lon_col: Optional[str],
      lat_col: Optional[str],
      height_col: Optional[str],
      epoch_col: Optional[str],
      default_height: float,
      default_epoch: float,
      lon_out: str,
      lat_out: str,
      height_out: str,
      wkt_col: Optional[str],
      wkt_out: Optional[str],
) -> pl.DataFrame:
      result = df
      height_vals = _optional_float_series(df, height_col, default_height)
      epoch_vals = _optional_float_series(df, epoch_col, default_epoch)

      if lon_col is not None and lat_col is not None:
            lon_vals = _required_float_series(df, lon_col)
            lat_vals = _required_float_series(df, lat_col)
            lon_new, lat_new, height_new, _ = TRANSFORMER.transform(
                  lon_vals, lat_vals, height_vals, epoch_vals
            )
            result = result.with_columns(
                  [
                        pl.Series(lon_out, lon_new),
                        pl.Series(lat_out, lat_new),
                        pl.Series(height_out, height_new),
                  ]
            )

      if wkt_col is not None and wkt_out is not None:
            wkt_series = _transform_wkt_column(result, wkt_col, wkt_out, height_vals, epoch_vals)
            result = result.with_columns(wkt_series)

      return result


def main() -> None:
      args = parse_args()

      if args.output_csv.exists() and not args.overwrite:
            raise FileExistsError(
                  f"Output path '{args.output_csv}' already exists. Pass --overwrite to replace it."
            )

      if (args.lon_col is None) ^ (args.lat_col is None):
            raise ValueError("Both --lon-col and --lat-col must be provided together.")

      if args.lon_col is None and args.wkt_col is None:
            raise ValueError(
                  "Supply either both --lon-col/--lat-col for point data or --wkt-col for geometry data."
            )

      wkt_out = args.wkt_out
      if args.wkt_col is not None and wkt_out is None:
            wkt_out = f"{args.wkt_col}_itrf2000"

      df = pl.read_csv(args.input_csv)

      transformed_df = transform_dataframe(
            df=df,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            height_col=args.height_col,
            epoch_col=args.epoch_col,
            default_height=args.default_height,
            default_epoch=args.default_epoch,
            lon_out=args.lon_out,
            lat_out=args.lat_out,
            height_out=args.height_out,
            wkt_col=args.wkt_col,
            wkt_out=wkt_out,
      )

      args.output_csv.parent.mkdir(parents=True, exist_ok=True)
      transformed_df.write_csv(args.output_csv)


if __name__ == "__main__":
      main()
