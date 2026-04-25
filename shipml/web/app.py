from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


WEB_DIR = Path(__file__).resolve().parent
SHIPML_DIR = WEB_DIR.parent
ROUTE_OUTPUTS = SHIPML_DIR / "route_anal" / "outputs"
DEFAULT_TYPED_PREDICTIONS = ROUTE_OUTPUTS / "route_predictions_with_types.csv"
DEFAULT_RAW_PREDICTIONS = ROUTE_OUTPUTS / "route_predictions.csv"
DEFAULT_ROUTE_CENTERS = ROUTE_OUTPUTS / "route_centers_long.csv"
DEFAULT_AIS_POINTS = SHIPML_DIR / "route_anal" / "ais_data_10day.csv"
DEFAULT_TYPE_SUMMARY = ROUTE_OUTPUTS / "route_type_summary.json"
DEFAULT_MODEL_METRICS = SHIPML_DIR / "type_anal" / "outputs" / "best_ship_type_model_metrics.json"
ROUTE_CENTER_POINTS = 24
MAX_TRACK_POINTS_PER_SHIP = 90


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/summary")
    def summary() -> Any:
        predictions = load_predictions()
        centers = load_route_centers()
        ship_types = (
            predictions["predicted_shiptype"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("name")
            .reset_index(name="count")
            .to_dict("records")
        )
        routes = (
            predictions["predicted_route"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("name")
            .reset_index(name="count")
            .to_dict("records")
        )
        return jsonify(
            {
                "shipTypes": ship_types,
                "routes": routes,
                "bounds": dataframe_bounds(predictions, centers),
                "model": load_model_summary(),
                "totalShips": int(len(predictions)),
            }
        )

    @app.get("/api/map-data")
    def map_data() -> Any:
        ship_type = request.args.get("ship_type", "").strip()
        route_label = request.args.get("route", "").strip()
        anomaly_only = request.args.get("anomaly", "").lower() in {"1", "true", "yes"}
        show_tracks = request.args.get("tracks", "").lower() in {"1", "true", "yes"}
        max_ships = parse_int(request.args.get("max_ships"), default=800)

        predictions = filter_predictions(
            load_predictions(),
            ship_type=ship_type,
            route_label=route_label,
            anomaly_only=anomaly_only,
        )
        centers = load_route_centers()

        route_counts = summarize_routes(predictions)
        route_labels = set(route_counts["route_label"].astype(str)) if not route_counts.empty else set()
        route_features = route_center_features(centers, route_counts, route_labels)

        ships = predictions
        if max_ships > 0 and len(ships) > max_ships:
            ships = ships.sort_values(
                ["is_anomaly", "predicted_route_probability"],
                ascending=[False, False],
            ).head(max_ships)

        track_features: list[dict[str, Any]] = []
        latest_points: dict[str, dict[str, Any]] = {}
        if show_tracks:
            track_features, latest_points = ship_actual_track_features(ships, load_ais_points())
            if not track_features:
                track_features = ship_straight_track_features(ships)

        return jsonify(
            {
                "filters": {
                    "shipType": ship_type or "__all__",
                    "route": route_label or "__all__",
                    "anomalyOnly": anomaly_only,
                    "showTracks": show_tracks,
                },
                "shipCount": int(len(predictions)),
                "shownShipCount": int(len(ships)),
                "routes": feature_collection(route_features),
                "shipTracks": feature_collection(track_features),
                "ships": feature_collection(ship_point_features(ships, latest_points)),
                "routeSummary": route_counts.to_dict("records"),
                "bounds": dataframe_bounds(ships, centers),
            }
        )

    return app


def data_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).resolve() if value else default.resolve()


def predictions_path() -> Path:
    configured = data_path("ROUTE_PREDICTIONS_CSV", DEFAULT_TYPED_PREDICTIONS)
    if configured.exists():
        return configured
    if DEFAULT_TYPED_PREDICTIONS.exists():
        return DEFAULT_TYPED_PREDICTIONS.resolve()
    return DEFAULT_RAW_PREDICTIONS.resolve()


@lru_cache(maxsize=4)
def read_csv_cached(path_text: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path_text)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv_cached(str(path), path.stat().st_mtime).copy()


def load_predictions() -> pd.DataFrame:
    df = read_csv(predictions_path())
    for col in [
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "predicted_route_probability",
        "predicted_shiptype_probability",
        "anomaly_score",
        "mean_sog",
        "width",
        "length",
        "draught",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "predicted_shiptype" not in df.columns:
        df["predicted_shiptype"] = "Unknown"
        df["predicted_shiptype_probability"] = float("nan")
    df["predicted_shiptype"] = df["predicted_shiptype"].fillna("Unknown").astype(str)
    df["predicted_route"] = df["predicted_route"].fillna("Unknown").astype(str)
    df["MMSI"] = df["MMSI"].astype(str)
    if "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].map(parse_bool)
    else:
        df["is_anomaly"] = False
    df["bearing"] = df.apply(
        lambda row: bearing_degrees(
            row.get("start_lat"),
            row.get("start_lon"),
            row.get("end_lat"),
            row.get("end_lon"),
        ),
        axis=1,
    )
    return df


def load_route_centers() -> pd.DataFrame:
    path = data_path("ROUTE_CENTERS_CSV", DEFAULT_ROUTE_CENTERS)
    df = read_csv(path)
    df["route_label"] = df["route_label"].astype(str)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df.dropna(subset=["route_label", "step", "Latitude", "Longitude"])


def load_ais_points() -> pd.DataFrame:
    path = data_path("AIS_POINTS_CSV", DEFAULT_AIS_POINTS)
    df = read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    rename = {
        "mmsi": "MMSI",
        "timestamp": "Timestamp",
        "latitude": "Latitude",
        "lat": "Latitude",
        "longitude": "Longitude",
        "lon": "Longitude",
        "lng": "Longitude",
        "sog": "SOG",
        "cog": "COG",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})
    keep_cols = [col for col in ["MMSI", "Timestamp", "Latitude", "Longitude", "SOG", "COG"] if col in df.columns]
    df = df[keep_cols].copy()
    df["MMSI"] = df["MMSI"].astype(str)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["Latitude", "Longitude", "SOG", "COG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["MMSI", "Timestamp", "Latitude", "Longitude"])
    df = df.loc[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    return df.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(drop=True)


def load_model_summary() -> dict[str, Any]:
    for path in [
        data_path("ROUTE_TYPE_SUMMARY_JSON", DEFAULT_TYPE_SUMMARY),
        data_path("TYPE_MODEL_METRICS_JSON", DEFAULT_MODEL_METRICS),
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if "model" in data:
                return compact_model_summary(data["model"])
            return compact_model_summary(data)
    return {
        "displayName": "Unknown",
        "modelName": "unknown",
        "accuracy": None,
        "macroF1": None,
    }


def compact_model_summary(data: dict[str, Any]) -> dict[str, Any]:
    best = data.get("best_metrics", {})
    return {
        "displayName": data.get("display_name") or data.get("displayName") or "Unknown",
        "modelName": data.get("model_name") or data.get("modelName") or "unknown",
        "accuracy": best.get("test_accuracy"),
        "macroF1": best.get("macro_f1"),
        "trainedAt": data.get("trained_at"),
    }


def filter_predictions(
    df: pd.DataFrame,
    ship_type: str,
    route_label: str,
    anomaly_only: bool,
) -> pd.DataFrame:
    filtered = df
    if ship_type and ship_type != "__all__":
        filtered = filtered.loc[filtered["predicted_shiptype"] == ship_type]
    if route_label and route_label != "__all__":
        filtered = filtered.loc[filtered["predicted_route"] == route_label]
    if anomaly_only:
        filtered = filtered.loc[filtered["is_anomaly"]]
    return filtered.copy()


def summarize_routes(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "route_label",
                "vessel_count",
                "anomaly_count",
                "avg_route_probability",
                "avg_shiptype_probability",
            ]
        )
    grouped = predictions.groupby("predicted_route", dropna=False)
    summary = grouped.agg(
        vessel_count=("MMSI", "count"),
        anomaly_count=("is_anomaly", "sum"),
        avg_route_probability=("predicted_route_probability", "mean"),
        avg_shiptype_probability=("predicted_shiptype_probability", "mean"),
    ).reset_index()
    return summary.rename(columns={"predicted_route": "route_label"}).sort_values(
        "vessel_count",
        ascending=False,
        ignore_index=True,
    )


def route_center_features(
    centers: pd.DataFrame,
    route_counts: pd.DataFrame,
    route_labels: set[str],
) -> list[dict[str, Any]]:
    if centers.empty or not route_labels:
        return []

    count_props = {
        str(row.route_label): {
            "vessel_count": int(row.vessel_count),
            "anomaly_count": int(row.anomaly_count),
            "avg_route_probability": clean_number(row.avg_route_probability),
            "avg_shiptype_probability": clean_number(row.avg_shiptype_probability),
        }
        for row in route_counts.itertuples(index=False)
    }

    features: list[dict[str, Any]] = []
    subset = centers.loc[centers["route_label"].isin(route_labels)]
    for route_label, group in subset.sort_values(["route_label", "step"]).groupby("route_label"):
        coords = [
            [float(row.Longitude), float(row.Latitude)]
            for row in group.itertuples(index=False)
        ]
        if len(coords) < 2:
            continue
        props = {"route_label": str(route_label)}
        props.update(count_props.get(str(route_label), {}))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )
    return features


def selected_route_center_features(
    predictions: pd.DataFrame,
    ais_points: pd.DataFrame,
    route_counts: pd.DataFrame,
    fallback_centers: pd.DataFrame,
) -> list[dict[str, Any]]:
    if predictions.empty or route_counts.empty:
        return []

    count_props = {
        str(row.route_label): {
            "vessel_count": int(row.vessel_count),
            "anomaly_count": int(row.anomaly_count),
            "avg_route_probability": clean_number(row.avg_route_probability),
            "avg_shiptype_probability": clean_number(row.avg_shiptype_probability),
        }
        for row in route_counts.itertuples(index=False)
    }

    route_labels = set(route_counts["route_label"].astype(str))
    route_to_mmsi = {
        str(route): set(group["MMSI"].astype(str))
        for route, group in predictions.groupby("predicted_route", dropna=False)
    }
    features: list[dict[str, Any]] = []

    for route_label in sorted(route_labels):
        mmsi_values = route_to_mmsi.get(route_label, set())
        group_points = ais_points.loc[ais_points["MMSI"].isin(mmsi_values)]
        coords = averaged_track_centerline(group_points)
        if len(coords) < 2:
            coords = fallback_route_coords(fallback_centers, route_label)
        if len(coords) < 2:
            continue

        props = {"route_label": route_label, "center_source": "selected_ais_tracks"}
        props.update(count_props.get(route_label, {}))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )
    return features


def averaged_track_centerline(points: pd.DataFrame) -> list[list[float]]:
    if points.empty:
        return []

    sampled_tracks: list[list[list[float]]] = []
    for _, group in points.groupby("MMSI", sort=False):
        coords = sample_group_track(group, ROUTE_CENTER_POINTS)
        if len(coords) >= 2:
            sampled_tracks.append(coords)

    if not sampled_tracks:
        return []

    centerline: list[list[float]] = []
    for idx in range(ROUTE_CENTER_POINTS):
        lons = [coords[idx][0] for coords in sampled_tracks if idx < len(coords)]
        lats = [coords[idx][1] for coords in sampled_tracks if idx < len(coords)]
        if not lons or not lats:
            continue
        centerline.append([float(sum(lons) / len(lons)), float(sum(lats) / len(lats))])
    return dedupe_coords(centerline)


def fallback_route_coords(centers: pd.DataFrame, route_label: str) -> list[list[float]]:
    group = centers.loc[centers["route_label"] == route_label].sort_values("step")
    return [
        [float(row.Longitude), float(row.Latitude)]
        for row in group.itertuples(index=False)
        if valid_lonlat(row.Longitude, row.Latitude)
    ]


def ship_actual_track_features(
    predictions: pd.DataFrame,
    ais_points: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if predictions.empty or ais_points.empty:
        return [], {}

    prediction_by_mmsi = {
        str(row.MMSI): row
        for row in predictions.itertuples(index=False)
    }
    wanted_mmsi = set(prediction_by_mmsi)
    subset = ais_points.loc[ais_points["MMSI"].isin(wanted_mmsi)]
    if subset.empty:
        return [], {}

    features: list[dict[str, Any]] = []
    latest_points: dict[str, dict[str, Any]] = {}

    for mmsi, group in subset.groupby("MMSI", sort=False):
        sorted_group = group.sort_values("Timestamp", kind="mergesort")
        coords = [
            [float(row.Longitude), float(row.Latitude)]
            for row in sorted_group.itertuples(index=False)
            if valid_lonlat(row.Longitude, row.Latitude)
        ]
        coords = downsample_coords(dedupe_coords(coords), MAX_TRACK_POINTS_PER_SHIP)
        if not coords:
            continue

        row = prediction_by_mmsi[str(mmsi)]
        latest = sorted_group.iloc[-1]
        latest_bearing = latest_bearing_degrees(sorted_group, coords)
        latest_points[str(mmsi)] = {
            "lat": clean_number(latest["Latitude"]),
            "lon": clean_number(latest["Longitude"]),
            "bearing": clean_number(latest_bearing),
        }

        if len(coords) < 2:
            continue
        props = ship_properties(row)
        props["bearing"] = clean_number(latest_bearing)
        props["track_point_count"] = int(len(coords))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )
    return features, latest_points


def ship_straight_track_features(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        if not valid_lonlat(row.start_lon, row.start_lat) or not valid_lonlat(row.end_lon, row.end_lat):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(row.start_lon), float(row.start_lat)],
                        [float(row.end_lon), float(row.end_lat)],
                    ],
                },
                "properties": ship_properties(row),
            }
        )
    return features


def ship_point_features(
    predictions: pd.DataFrame,
    latest_points: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    latest_points = latest_points or {}
    features: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        latest = latest_points.get(str(row.MMSI), {})
        lon = latest.get("lon", row.end_lon)
        lat = latest.get("lat", row.end_lat)
        if not valid_lonlat(lon, lat):
            continue
        props = ship_properties(row)
        if latest.get("bearing") is not None:
            props["bearing"] = latest["bearing"]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": props,
            }
        )
    return features


def ship_properties(row: Any) -> dict[str, Any]:
    return {
        "mmsi": str(row.MMSI),
        "shiptype": str(row.predicted_shiptype),
        "shiptype_probability": clean_number(getattr(row, "predicted_shiptype_probability", None)),
        "route": str(row.predicted_route),
        "route_probability": clean_number(getattr(row, "predicted_route_probability", None)),
        "is_anomaly": bool(row.is_anomaly),
        "anomaly_score": clean_number(getattr(row, "anomaly_score", None)),
        "bearing": clean_number(getattr(row, "bearing", None)),
        "mean_sog": clean_number(getattr(row, "mean_sog", None)),
        "width": clean_number(getattr(row, "width", None)),
        "length": clean_number(getattr(row, "length", None)),
        "draught": clean_number(getattr(row, "draught", None)),
    }


def sample_group_track(group: pd.DataFrame, point_count: int) -> list[list[float]]:
    group = group.sort_values("Timestamp", kind="mergesort")
    group = group.dropna(subset=["Timestamp", "Latitude", "Longitude"])
    coords = group[["Longitude", "Latitude"]].to_numpy(dtype=float)
    if len(coords) == 0:
        return []
    if len(coords) == 1:
        lon, lat = coords[0]
        return [[float(lon), float(lat)]]

    seconds = (group["Timestamp"] - group["Timestamp"].iloc[0]).dt.total_seconds()
    x = seconds.to_numpy(dtype=float)
    if float(np.nanmax(x)) == 0.0:
        x = np.arange(len(group), dtype=float)

    _, unique_idx = np.unique(x, return_index=True)
    unique_idx = np.sort(unique_idx)
    x = x[unique_idx]
    coords = coords[unique_idx]
    if len(coords) == 1 or float(x.max() - x.min()) == 0.0:
        lon, lat = coords[0]
        return [[float(lon), float(lat)]]

    targets = np.linspace(float(x.min()), float(x.max()), point_count)
    sampled_lon = np.interp(targets, x, coords[:, 0])
    sampled_lat = np.interp(targets, x, coords[:, 1])
    return [
        [float(lon), float(lat)]
        for lon, lat in zip(sampled_lon, sampled_lat)
        if valid_lonlat(lon, lat)
    ]


def downsample_coords(coords: list[list[float]], max_points: int) -> list[list[float]]:
    if len(coords) <= max_points or max_points <= 1:
        return coords
    indexes = {
        round(idx * (len(coords) - 1) / (max_points - 1))
        for idx in range(max_points)
    }
    return [coords[idx] for idx in sorted(indexes)]


def dedupe_coords(coords: list[list[float]]) -> list[list[float]]:
    deduped: list[list[float]] = []
    for coord in coords:
        if not deduped or deduped[-1] != coord:
            deduped.append(coord)
    return deduped


def latest_bearing_degrees(group: pd.DataFrame, coords: list[list[float]]) -> float:
    if "COG" in group.columns:
        cog = pd.to_numeric(group["COG"], errors="coerce").dropna()
        if not cog.empty and 0 <= float(cog.iloc[-1]) <= 360:
            return float(cog.iloc[-1])

    if len(coords) >= 2:
        start_lon, start_lat = coords[-2]
        end_lon, end_lat = coords[-1]
        return bearing_degrees(start_lat, start_lon, end_lat, end_lon)
    return 0.0


def dataframe_bounds(predictions: pd.DataFrame, centers: pd.DataFrame) -> list[list[float]] | None:
    lat_values = []
    lon_values = []
    for lat_col, lon_col in [("start_lat", "start_lon"), ("end_lat", "end_lon")]:
        if lat_col in predictions and lon_col in predictions:
            lat_values.extend(pd.to_numeric(predictions[lat_col], errors="coerce").dropna().tolist())
            lon_values.extend(pd.to_numeric(predictions[lon_col], errors="coerce").dropna().tolist())
    if not centers.empty:
        lat_values.extend(centers["Latitude"].dropna().tolist())
        lon_values.extend(centers["Longitude"].dropna().tolist())
    if not lat_values or not lon_values:
        return None
    return [
        [float(min(lat_values)), float(min(lon_values))],
        [float(max(lat_values)), float(max(lon_values))],
    ]


def feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def valid_lonlat(lon: Any, lat: Any) -> bool:
    try:
        lon_float = float(lon)
        lat_float = float(lat)
    except (TypeError, ValueError):
        return False
    return -180 <= lon_float <= 180 and -90 <= lat_float <= 90


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return int(number) if number.is_integer() else number
    return None


def bearing_degrees(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    try:
        lat1_float = float(lat1)
        lon1_float = float(lon1)
        lat2_float = float(lat2)
        lon2_float = float(lon2)
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in [lat1_float, lon1_float, lat2_float, lon2_float]):
        return 0.0

    lat1_rad = math.radians(lat1_float)
    lat2_rad = math.radians(lat2_float)
    dlon = math.radians(lon2_float - lon1_float)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
