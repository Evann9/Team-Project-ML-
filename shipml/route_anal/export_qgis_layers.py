from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_AIS_DATA = Path(__file__).resolve().parent / "ais_data_10day.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export route analysis CSV outputs as QGIS-ready GeoJSON layers."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that contains route analysis CSV outputs.",
    )
    parser.add_argument(
        "--ais-data",
        type=Path,
        default=DEFAULT_AIS_DATA,
        help="Clean AIS point CSV used to build anomaly track lines.",
    )
    parser.add_argument(
        "--qgis-dir",
        type=Path,
        default=None,
        help="Output directory for GeoJSON files. Defaults to outputs/qgis_layers.",
    )
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    collection = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def point_feature(row: pd.Series, lon_col: str, lat_col: str) -> dict[str, Any] | None:
    lon = pd.to_numeric(row.get(lon_col), errors="coerce")
    lat = pd.to_numeric(row.get(lat_col), errors="coerce")
    if pd.isna(lon) or pd.isna(lat):
        return None

    props = {
        col: clean_value(value)
        for col, value in row.items()
        if col not in {lon_col, lat_col}
    }
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": props,
    }


def export_point_layer(
    csv_path: Path,
    geojson_path: Path,
    lon_col: str,
    lat_col: str,
) -> int:
    df = pd.read_csv(csv_path)
    features = []
    for _, row in df.iterrows():
        feature = point_feature(row, lon_col, lat_col)
        if feature is not None:
            features.append(feature)
    write_geojson(geojson_path, features)
    return len(features)


def export_route_center_lines(outputs_dir: Path, qgis_dir: Path) -> int:
    df = pd.read_csv(outputs_dir / "route_centers_long.csv")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["route_label", "step", "Latitude", "Longitude"])

    features: list[dict[str, Any]] = []
    for route_label, group in df.sort_values(["route_label", "step"]).groupby("route_label"):
        coords = [
            [float(row.Longitude), float(row.Latitude)]
            for row in group.itertuples(index=False)
        ]
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "route_label": str(route_label),
                    "route_point_count": len(coords),
                },
            }
        )

    write_geojson(qgis_dir / "route_center_lines.geojson", features)
    return len(features)


def export_anomaly_track_lines(outputs_dir: Path, ais_data: Path, qgis_dir: Path) -> int:
    anomaly_path = outputs_dir / "anomaly_ships.csv"
    if not anomaly_path.exists() or not ais_data.exists():
        return 0

    anomalies = pd.read_csv(anomaly_path)
    if anomalies.empty:
        write_geojson(qgis_dir / "anomaly_track_lines.geojson", [])
        return 0

    anomaly_mmsi = set(anomalies["MMSI"].astype(str))
    anomaly_props = anomalies.set_index(anomalies["MMSI"].astype(str)).to_dict("index")

    df = pd.read_csv(ais_data)
    df.columns = df.columns.astype(str).str.strip()
    df["MMSI"] = df["MMSI"].astype(str)
    df = df.loc[df["MMSI"].isin(anomaly_mmsi)].copy()
    if df.empty:
        write_geojson(qgis_dir / "anomaly_track_lines.geojson", [])
        return 0

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["MMSI", "Timestamp", "Latitude", "Longitude"])

    features: list[dict[str, Any]] = []
    for mmsi, group in df.sort_values(["MMSI", "Timestamp"]).groupby("MMSI"):
        coords = [
            [float(row.Longitude), float(row.Latitude)]
            for row in group.itertuples(index=False)
        ]
        if len(coords) < 2:
            continue
        props = {
            key: clean_value(value)
            for key, value in anomaly_props.get(str(mmsi), {}).items()
        }
        props["MMSI"] = str(mmsi)
        props["track_point_count"] = len(coords)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )

    write_geojson(qgis_dir / "anomaly_track_lines.geojson", features)
    return len(features)


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    qgis_dir = (args.qgis_dir or outputs_dir / "qgis_layers").resolve()
    qgis_dir.mkdir(parents=True, exist_ok=True)

    created = {
        "route_center_lines.geojson": export_route_center_lines(outputs_dir, qgis_dir),
        "anchorage_clusters.geojson": export_point_layer(
            outputs_dir / "anchorage_clusters.csv",
            qgis_dir / "anchorage_clusters.geojson",
            "center_lon",
            "center_lat",
        ),
        "route_prediction_endpoints.geojson": export_point_layer(
            outputs_dir / "route_predictions.csv",
            qgis_dir / "route_prediction_endpoints.geojson",
            "end_lon",
            "end_lat",
        ),
        "route_prediction_startpoints.geojson": export_point_layer(
            outputs_dir / "route_predictions.csv",
            qgis_dir / "route_prediction_startpoints.geojson",
            "start_lon",
            "start_lat",
        ),
        "anomaly_ship_endpoints.geojson": export_point_layer(
            outputs_dir / "anomaly_ships.csv",
            qgis_dir / "anomaly_ship_endpoints.geojson",
            "end_lon",
            "end_lat",
        ),
        "anomaly_track_lines.geojson": export_anomaly_track_lines(
            outputs_dir,
            args.ais_data.resolve(),
            qgis_dir,
        ),
    }

    print(f"QGIS layers saved: {qgis_dir}")
    for name, count in created.items():
        print(f"- {name}: {count:,} features")


if __name__ == "__main__":
    main()
