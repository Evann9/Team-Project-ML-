"""항로 분류, 이상 탐지, 정박지 분석을 학습하고 실행한다.

이 스크립트는 원본 AIS 포인트 track을 선박 단위 특징으로 변환한다. 기존
항로 컬럼이 있으면 그 라벨을 사용하고, 없으면 전체 궤적 signature에 대한
KMeans cluster로 항로 라벨을 학습한다. 이후 초기 track 증거로 RandomForest
분류기를 학습하고, 거리 기반 이상 임계값을 계산하며, 정박 stop event를
탐지한다. 마지막으로 downstream 보고서와 지도 레이어에 쓸 모델 번들과 CSV
출력을 내보낸다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EARTH_RADIUS_KM = 6371.0088
DEFAULT_TRAIN_DATA = Path(__file__).resolve().parent / "ais_data_10day.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ship_anal_model.joblib"
ROUTE_LABEL_CANDIDATES = (
    "Route",
    "route",
    "route_id",
    "route_label",
    "RouteLabel",
    "ROUTE",
)


@dataclass
class FeatureBuildResult:
    """AIS 포인트 데이터셋에서 만들어진 모든 특징 테이블을 담는 컨테이너.

    ``vessels``는 MMSI당 하나의 집계 행을 담고, ``signature``는 전체 정규화
    궤적을 설명한다. ``early_signature``는 예측에 쓰는 초기 구간만 설명하며,
    ``clean_points``는 이후 정박지 분석과 내보내기에 사용할 정제된 포인트
    단위 입력을 보관한다.
    """

    vessels: pd.DataFrame
    signature: pd.DataFrame
    early_signature: pd.DataFrame
    clean_points: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """항로 모델 학습과 일괄 예측을 위한 CLI 옵션을 파싱한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Train and run AIS route prediction, anomaly analysis, and "
            "anchorage prediction."
        )
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_DATA,
        help="Historical AIS CSV used to train the route model.",
    )
    parser.add_argument(
        "--predict-data",
        type=Path,
        default=None,
        help=(
            "New AIS CSV to predict. If omitted, the script predicts the "
            "training data after fitting."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result CSV files are saved.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained model bundle is saved.",
    )
    parser.add_argument(
        "--route-clusters",
        type=int,
        default=0,
        help="Number of route clusters. 0 chooses a conservative value automatically.",
    )
    parser.add_argument(
        "--route-points",
        type=int,
        default=24,
        help="Number of normalized trajectory points used to describe a route.",
    )
    parser.add_argument(
        "--early-fraction",
        type=float,
        default=0.35,
        help=(
            "Fallback fraction of each track used by the classifier when "
            "--early-window-hours is 0."
        ),
    )
    parser.add_argument(
        "--early-window-hours",
        type=float,
        default=6.0,
        help=(
            "Fixed initial observation window used by the route classifier. "
            "Use 0 to fall back to --early-fraction."
        ),
    )
    parser.add_argument(
        "--early-eval-windows",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 6.0, 12.0],
        help="Observation windows, in hours, to compare for route classifier diagnostics.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Optional existing route label column. If absent, route labels are clustered.",
    )
    parser.add_argument(
        "--anomaly-quantile",
        type=float,
        default=0.95,
        help="Training quantile used as the route-distance anomaly threshold.",
    )
    parser.add_argument(
        "--slow-sog",
        type=float,
        default=1.0,
        help="SOG threshold in knots for anchorage/stop detection.",
    )
    parser.add_argument(
        "--anchorage-eps-km",
        type=float,
        default=2.0,
        help="DBSCAN radius in kilometers for anchorage clustering.",
    )
    parser.add_argument(
        "--anchorage-min-samples",
        type=int,
        default=4,
        help="Minimum stop events needed to form an anchorage cluster.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """흔한 AIS 컬럼명 변형을 스크립트 스키마로 정규화한다.

    입력 파일은 같은 필드에 대해 대소문자나 약어가 다를 때가 많다. 이 함수는
    헤더의 공백을 제거하고 알려진 변형을 표준 이름으로 바꿔, 이후 파이프라인이
    안정적인 컬럼 식별자 집합에 의존할 수 있게 한다.
    """

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        "# Timestamp": "Timestamp",
        "timestamp": "Timestamp",
        "TimeStamp": "Timestamp",
        "datetime": "Timestamp",
        "time": "Timestamp",
        "mmsi": "MMSI",
        "Mmsi": "MMSI",
        "latitude": "Latitude",
        "lat": "Latitude",
        "LAT": "Latitude",
        "longitude": "Longitude",
        "lon": "Longitude",
        "lng": "Longitude",
        "LON": "Longitude",
        "sog": "SOG",
        "speed": "SOG",
        "cog": "COG",
        "course": "COG",
        "width": "Width",
        "length": "Length",
        "draught": "Draught",
        "draft": "Draught",
        "Ship type": "shiptype",
        "Ship Type": "shiptype",
        "ship_type": "shiptype",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_and_clean_ais(path: Path) -> pd.DataFrame:
    """원본 AIS CSV 데이터를 읽고 모델링용 유효성 검사를 적용한다.

    정제 단계는 필수 항해 컬럼, 선택적 선박 메타데이터, 가능한 항로 라벨
    후보만 유지한다. 타입을 변환하고, 잘못된 MMSI와 좌표 값을 제거하며,
    반복 포인트를 중복 제거한다. 또한 선박별 track을 timestamp 순으로 정렬하고,
    누락된 정적 선박 속성은 MMSI별 중앙값과 전역 fallback으로 채운다.
    """

    if not path.exists():
        raise FileNotFoundError(f"AIS CSV not found: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)

    required = ["MMSI", "Timestamp", "Latitude", "Longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    keep_cols = [
        "MMSI",
        "Timestamp",
        "Latitude",
        "Longitude",
        "SOG",
        "COG",
        "Width",
        "Length",
        "Draught",
        "shiptype",
        *[col for col in ROUTE_LABEL_CANDIDATES if col in df.columns],
    ]
    keep_cols = list(dict.fromkeys([col for col in keep_cols if col in df.columns]))
    df = df[keep_cols].copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")

    numeric_cols = [
        "Latitude",
        "Longitude",
        "SOG",
        "COG",
        "Width",
        "Length",
        "Draught",
        "shiptype",
    ]
    for col in [col for col in numeric_cols if col in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[~df["MMSI"].between(100000000, 999999998), "MMSI"] = np.nan
    df.loc[~df["Latitude"].between(-90, 90), "Latitude"] = np.nan
    df.loc[~df["Longitude"].between(-180, 180), "Longitude"] = np.nan

    if "SOG" in df.columns:
        df.loc[(df["SOG"] < 0) | (df["SOG"] > 70), "SOG"] = np.nan
    if "COG" in df.columns:
        df.loc[~df["COG"].between(0, 360), "COG"] = np.nan
    for col in ["Width", "Length", "Draught"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    essential = [
        col
        for col in ["MMSI", "Timestamp", "Latitude", "Longitude"]
        if col in df.columns
    ]
    df = df.dropna(subset=essential)

    df["MMSI"] = df["MMSI"].astype("int64").astype(str)
    df = df.drop_duplicates(subset=["MMSI", "Timestamp", "Latitude", "Longitude"])
    df = df.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(drop=True)

    for col in ["SOG", "COG", "Width", "Length", "Draught", "shiptype"]:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["Width", "Length", "Draught", "shiptype"]:
        group_values = df.groupby("MMSI", observed=True)[col].transform("median")
        df[col] = df[col].fillna(group_values)
        global_median = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(global_median) else global_median)

    df["SOG"] = df["SOG"].fillna(0.0)
    df["COG"] = df["COG"].fillna(0.0)
    return df


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """좌표쌍 사이의 대권거리를 킬로미터 단위로 계산한다."""

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """시작점에서 끝점으로 향하는 초기 방위각을 계산한다."""

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def safe_median(values: pd.Series) -> float:
    """숫자 중앙값을 반환하고, 모든 값이 비어 있으면 0.0으로 fallback한다."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return float(numeric.median())
    return 0.0


def sample_track_signature(group: pd.DataFrame, points: int) -> np.ndarray:
    """선박 track을 고정 길이 위도/경도 signature로 표현한다.

    track마다 지속 시간과 포인트 개수가 다르므로, clustering과 거리 계산에는
    정규화된 표현이 필요하다. 이 함수는 각 track을 ``points``개의 균등한
    timestamp 위치로 보간하고, 샘플링된 위도/경도 쌍을 하나의 벡터로 펼친다.
    """

    group = group.sort_values("Timestamp", kind="mergesort")
    lat = group["Latitude"].to_numpy(dtype=float)
    lon = group["Longitude"].to_numpy(dtype=float)

    if len(group) == 0:
        return np.full(points * 2, np.nan)
    if len(group) == 1:
        return np.tile(np.array([lat[0], lon[0]], dtype=float), points)

    seconds = (group["Timestamp"] - group["Timestamp"].iloc[0]).dt.total_seconds()
    x = seconds.to_numpy(dtype=float)
    if float(np.nanmax(x)) == 0.0:
        x = np.arange(len(group), dtype=float)

    _, unique_idx = np.unique(x, return_index=True)
    unique_idx = np.sort(unique_idx)
    x = x[unique_idx]
    lat = lat[unique_idx]
    lon = lon[unique_idx]

    if len(x) == 1 or float(np.nanmax(x) - np.nanmin(x)) == 0.0:
        return np.tile(np.array([lat[0], lon[0]], dtype=float), points)

    target = np.linspace(float(x.min()), float(x.max()), points)
    sampled_lat = np.interp(target, x, lat)
    sampled_lon = np.interp(target, x, lon)
    return np.column_stack([sampled_lat, sampled_lon]).reshape(-1)


def early_track(
    group: pd.DataFrame,
    early_fraction: float,
    early_window_hours: float = 0.0,
) -> pd.DataFrame:
    """항로 예측에 사용할 track의 초기 구간을 반환한다.

    분류기는 전체 궤적을 알기 전에 선박의 가능성 높은 항로를 예측할 수
    있도록 초기 증거로 학습된다. 기본값은 실제 운영 시점에 맞추기 쉬운
    고정 시간창이며, 필요할 때만 전체 track 길이 기준 fraction을 fallback으로
    사용할 수 있다. 매우 짧거나 지속 시간이 0인 track은 가능한 한 최소 두
    포인트를 보존하도록 처음 몇 행을 fallback으로 사용한다.
    """

    if len(group) <= 2:
        return group

    start = group["Timestamp"].iloc[0]
    if early_window_hours and early_window_hours > 0:
        cutoff = start + pd.to_timedelta(float(early_window_hours), unit="h")
        early = group.loc[group["Timestamp"] <= cutoff]
        if len(early) < 2:
            early = group.iloc[:2]
        return early

    early_fraction = min(max(early_fraction, 0.05), 1.0)
    end = group["Timestamp"].iloc[-1]
    duration = (end - start).total_seconds()
    if duration <= 0:
        keep = max(2, int(math.ceil(len(group) * early_fraction)))
        return group.iloc[:keep]

    cutoff = start + pd.to_timedelta(duration * early_fraction, unit="s")
    early = group.loc[group["Timestamp"] <= cutoff]
    if len(early) < 2:
        early = group.iloc[:2]
    return early


def signature_columns(prefix: str, points: int) -> list[str]:
    """펼쳐진 궤적 signature 벡터에 사용할 안정적인 컬럼명을 만든다."""

    cols: list[str] = []
    for idx in range(points):
        cols.append(f"{prefix}_lat_{idx:02d}")
        cols.append(f"{prefix}_lon_{idx:02d}")
    return cols


def build_features(
    df: pd.DataFrame,
    route_points: int,
    early_fraction: float,
    early_window_hours: float = 0.0,
) -> FeatureBuildResult:
    """포인트 단위 AIS track을 선박 단위 모델 입력으로 집계한다.

    각 MMSI에 대해 시간, 공간, 속도, 방향, 선박 크기, 항로 형태 요약을
    계산한다. 또한 고정 길이 궤적 signature 두 개를 저장한다. 하나는
    clustering 및 이상 거리 계산에 쓰는 전체 항로이고, 다른 하나는 예측 시
    분류기 증거로 쓰는 초기 항로다.
    """

    vessel_rows: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []
    early_signatures: list[np.ndarray] = []
    full_sig_cols = signature_columns("route", route_points)
    early_sig_cols = signature_columns("early", route_points)

    for mmsi, group in df.groupby("MMSI", sort=False, observed=True):
        group = group.sort_values("Timestamp", kind="mergesort")
        lat = group["Latitude"].to_numpy(dtype=float)
        lon = group["Longitude"].to_numpy(dtype=float)
        sog = group["SOG"].to_numpy(dtype=float)
        cog = group["COG"].to_numpy(dtype=float)
        timestamps = group["Timestamp"]

        segment_distance = 0.0
        if len(group) > 1:
            segment_distance = float(
                np.nansum(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]))
            )

        displacement = float(haversine_km(lat[0], lon[0], lat[-1], lon[-1]))
        bearing = bearing_degrees(lat[0], lon[0], lat[-1], lon[-1])
        duration_hours = max(
            (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600.0,
            0.0,
        )
        cog_rad = np.radians(cog)
        route_target_cols = [col for col in ROUTE_LABEL_CANDIDATES if col in group.columns]

        early = early_track(group, early_fraction, early_window_hours)
        early_lat = early["Latitude"].to_numpy(dtype=float)
        early_lon = early["Longitude"].to_numpy(dtype=float)
        early_sog = early["SOG"].to_numpy(dtype=float)
        early_cog = early["COG"].to_numpy(dtype=float)
        early_timestamps = early["Timestamp"]

        early_segment_distance = 0.0
        if len(early) > 1:
            early_segment_distance = float(
                np.nansum(
                    haversine_km(
                        early_lat[:-1],
                        early_lon[:-1],
                        early_lat[1:],
                        early_lon[1:],
                    )
                )
            )

        early_displacement = float(
            haversine_km(early_lat[0], early_lon[0], early_lat[-1], early_lon[-1])
        )
        early_bearing = bearing_degrees(
            early_lat[0],
            early_lon[0],
            early_lat[-1],
            early_lon[-1],
        )
        early_duration_hours = max(
            (early_timestamps.iloc[-1] - early_timestamps.iloc[0]).total_seconds()
            / 3600.0,
            0.0,
        )
        early_cog_rad = np.radians(early_cog)

        row: dict[str, Any] = {
            "MMSI": mmsi,
            "first_timestamp": timestamps.iloc[0],
            "last_timestamp": timestamps.iloc[-1],
            "point_count": int(len(group)),
            "duration_hours": duration_hours,
            "start_lat": float(lat[0]),
            "start_lon": float(lon[0]),
            "end_lat": float(lat[-1]),
            "end_lon": float(lon[-1]),
            "mean_lat": float(np.nanmean(lat)),
            "mean_lon": float(np.nanmean(lon)),
            "std_lat": float(np.nanstd(lat)),
            "std_lon": float(np.nanstd(lon)),
            "min_lat": float(np.nanmin(lat)),
            "max_lat": float(np.nanmax(lat)),
            "min_lon": float(np.nanmin(lon)),
            "max_lon": float(np.nanmax(lon)),
            "mean_sog": float(np.nanmean(sog)),
            "std_sog": float(np.nanstd(sog)),
            "max_sog": float(np.nanmax(sog)),
            "slow_point_ratio": float(np.nanmean(sog <= 1.0)),
            "mean_cog_sin": float(np.nanmean(np.sin(cog_rad))),
            "mean_cog_cos": float(np.nanmean(np.cos(cog_rad))),
            "width": safe_median(group["Width"]),
            "length": safe_median(group["Length"]),
            "draught": safe_median(group["Draught"]),
            "shiptype": safe_median(group["shiptype"]),
            "total_distance_km": segment_distance,
            "displacement_km": displacement,
            "straightness_ratio": float(displacement / segment_distance)
            if segment_distance > 0
            else 0.0,
            "bearing_sin": float(math.sin(math.radians(bearing))),
            "bearing_cos": float(math.cos(math.radians(bearing))),
            "early_point_count": int(len(early)),
            "early_duration_hours": early_duration_hours,
            "early_start_lat": float(early_lat[0]),
            "early_start_lon": float(early_lon[0]),
            "early_observed_lat": float(early_lat[-1]),
            "early_observed_lon": float(early_lon[-1]),
            "early_mean_lat": float(np.nanmean(early_lat)),
            "early_mean_lon": float(np.nanmean(early_lon)),
            "early_std_lat": float(np.nanstd(early_lat)),
            "early_std_lon": float(np.nanstd(early_lon)),
            "early_min_lat": float(np.nanmin(early_lat)),
            "early_max_lat": float(np.nanmax(early_lat)),
            "early_min_lon": float(np.nanmin(early_lon)),
            "early_max_lon": float(np.nanmax(early_lon)),
            "early_mean_sog": float(np.nanmean(early_sog)),
            "early_std_sog": float(np.nanstd(early_sog)),
            "early_max_sog": float(np.nanmax(early_sog)),
            "early_slow_point_ratio": float(np.nanmean(early_sog <= 1.0)),
            "early_mean_cog_sin": float(np.nanmean(np.sin(early_cog_rad))),
            "early_mean_cog_cos": float(np.nanmean(np.cos(early_cog_rad))),
            "early_total_distance_km": early_segment_distance,
            "early_displacement_km": early_displacement,
            "early_straightness_ratio": float(
                early_displacement / early_segment_distance
            )
            if early_segment_distance > 0
            else 0.0,
            "early_bearing_sin": float(math.sin(math.radians(early_bearing))),
            "early_bearing_cos": float(math.cos(math.radians(early_bearing))),
        }

        for target_col in route_target_cols:
            mode = group[target_col].dropna().mode()
            if len(mode):
                row[target_col] = mode.iloc[0]

        vessel_rows.append(row)
        signatures.append(sample_track_signature(group, route_points))
        early_signatures.append(sample_track_signature(early, route_points))

    vessels = pd.DataFrame(vessel_rows)
    signature = pd.DataFrame(signatures, columns=full_sig_cols)
    early_signature = pd.DataFrame(early_signatures, columns=early_sig_cols)
    return FeatureBuildResult(vessels, signature, early_signature, df)


def choose_cluster_count(vessel_count: int, requested: int) -> int:
    """보수적인 KMeans 항로 cluster 개수를 선택한다."""

    if requested and requested > 1:
        return min(requested, max(vessel_count, 1))
    if vessel_count < 3:
        return max(1, vessel_count)
    return min(12, max(3, int(round(math.sqrt(vessel_count) / 3.0))))


def resolve_target_column(df: pd.DataFrame, requested: str | None) -> str | None:
    """사용할 항로 라벨 컬럼을 찾는다. 요청값이 있거나 후보 컬럼이 있을 때 반환한다."""

    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested target column not found: {requested}")
        return requested

    for col in ROUTE_LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def make_feature_matrix(
    vessels: pd.DataFrame,
    early_signature: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """초기 관측 특징과 초기 형태 특징으로 분류기 특징 행렬을 조립한다.

    항로 분류기는 전체 궤적이 끝난 뒤에야 알 수 있는 end_lat/end_lon,
    전체 duration, 전체 이동거리, 전체 위경도 범위 등을 입력으로 쓰지
    않는다. 정적 선박 치수와 초기 관측 구간 요약만 초기 궤적 signature와
    이어 붙여, 실시간/부분 track 예측 주장에 맞는 feature scope를 유지한다.
    """

    base_cols = [
        "early_point_count",
        "early_duration_hours",
        "early_start_lat",
        "early_start_lon",
        "early_observed_lat",
        "early_observed_lon",
        "early_mean_lat",
        "early_mean_lon",
        "early_std_lat",
        "early_std_lon",
        "early_min_lat",
        "early_max_lat",
        "early_min_lon",
        "early_max_lon",
        "early_mean_sog",
        "early_std_sog",
        "early_max_sog",
        "early_slow_point_ratio",
        "early_mean_cog_sin",
        "early_mean_cog_cos",
        "width",
        "length",
        "draught",
        "shiptype",
        "early_total_distance_km",
        "early_displacement_km",
        "early_straightness_ratio",
        "early_bearing_sin",
        "early_bearing_cos",
    ]
    feature_df = pd.concat(
        [
            vessels[base_cols].reset_index(drop=True),
            early_signature.reset_index(drop=True),
        ],
        axis=1,
    )
    return feature_df, feature_df.columns.tolist()


def train_route_labels(
    vessels: pd.DataFrame,
    signature: pd.DataFrame,
    target_col: str | None,
    requested_clusters: int,
    random_state: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """기존 컬럼이나 궤적 clustering에서 항로 라벨을 만든다.

    라벨이 있는 항로 데이터가 있으면 해당 라벨을 그대로 사용하고, 이상 점수
    계산을 위해 scaled signature 공간에서 centroid를 계산한다. 라벨이 없으면
    KMeans가 전체 궤적 signature를 cluster하고 cluster ID를 안정적인
    ``route_XX`` 라벨로 바꾼다. 반환되는 ``route_info``에는 이후 필요한
    scaler, 선택적 cluster 모델, 항로 centroid가 저장된다.
    """

    route_info: dict[str, Any] = {
        "target_col": target_col,
        "label_source": "existing_column" if target_col else "kmeans_signature",
    }

    if target_col:
        y = vessels[target_col].astype(str).fillna("unknown_route")
        route_info["signature_scaler"] = StandardScaler()
        scaled_signature = route_info["signature_scaler"].fit_transform(signature)
        route_info["route_centroids"] = {
            str(label): scaled_signature[y.to_numpy() == label].mean(axis=0)
            for label in sorted(y.unique())
        }
        return y, route_info

    clusters = choose_cluster_count(len(vessels), requested_clusters)
    scaler = StandardScaler()
    scaled_signature = scaler.fit_transform(signature)
    cluster_model = KMeans(
        n_clusters=clusters,
        random_state=random_state,
        n_init=20,
        max_iter=500,
    )
    labels = cluster_model.fit_predict(scaled_signature)
    y = pd.Series([f"route_{label:02d}" for label in labels], name="route_label")
    route_info.update(
        {
            "cluster_count": clusters,
            "signature_scaler": scaler,
            "cluster_model": cluster_model,
            "route_centroids": {
                f"route_{idx:02d}": center
                for idx, center in enumerate(cluster_model.cluster_centers_)
            },
        }
    )
    return y, route_info


def train_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """초기 track 항로 분류기와 선택적 홀드아웃 지표를 학습한다.

    분류기는 집계 특징과 초기 궤적 증거만 사용해 전체 항로 라벨을 예측한다.
    데이터셋이 충분히 크고 균형 잡혀 있으면 stratified 홀드아웃으로
    accuracy/F1 진단을 수행한다. 홀드아웃 가능 여부와 관계없이, 최종 반환
    분류기는 배포를 위해 전체 행에 학습된다.
    """

    classifier = make_route_classifier(random_state)

    metrics: dict[str, Any] = {
        "train_vessels": int(len(y)),
        "route_classes": int(y.nunique()),
        "classifier_feature_scope": "early_track_only",
        "excluded_full_track_features": [
            "point_count",
            "end_lat",
            "end_lon",
            "duration_hours",
            "mean_lat",
            "mean_lon",
            "min_lat",
            "max_lat",
            "min_lon",
            "max_lon",
            "total_distance_km",
            "displacement_km",
            "straightness_ratio",
        ],
    }

    counts = y.value_counts()
    can_split = len(y) >= 10 and y.nunique() > 1 and int(counts.min()) >= 2
    if can_split:
        test_size = 0.2 if len(y) >= 50 else 0.3
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test)
        metrics["holdout_accuracy"] = float(accuracy_score(y_test, pred))
        metrics["holdout_f1_macro"] = float(f1_score(y_test, pred, average="macro"))
        metrics["classification_report"] = classification_report(
            y_test,
            pred,
            zero_division=0,
            output_dict=True,
        )

    classifier.fit(x, y)
    return classifier, metrics


def make_route_classifier(random_state: int) -> Pipeline:
    """항로 분류용 RandomForest 파이프라인을 생성한다."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def assign_kmeans_route_labels(
    signature: pd.DataFrame,
    route_info: dict[str, Any],
) -> pd.Series:
    """학습된 KMeans route catalog 기준으로 signature를 route label에 배정한다."""

    scaler = route_info["signature_scaler"]
    cluster_model = route_info.get("cluster_model")
    if cluster_model is None:
        raise ValueError("KMeans route model is required for route assignment.")
    scaled = scaler.transform(signature)
    cluster_ids = cluster_model.predict(scaled)
    return pd.Series([f"route_{idx:02d}" for idx in cluster_ids], index=signature.index)


def evaluate_route_holdout(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> dict[str, Any]:
    """주어진 train/test 특징과 라벨로 항로 분류기를 평가한다."""

    classifier = make_route_classifier(random_state)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    return {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "route_classes_train": int(pd.Series(y_train).nunique()),
        "route_classes_test": int(pd.Series(y_test).nunique()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_strict_route_holdout(
    vessels: pd.DataFrame,
    signature: pd.DataFrame,
    x: pd.DataFrame,
    target_col: str | None,
    requested_clusters: int,
    random_state: int,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """KMeans route catalog까지 train fold 안에서만 fit하는 엄격 평가를 수행한다."""

    if len(vessels) < 10:
        return {"available": False, "reason": "not_enough_vessels"}

    indices = np.arange(len(vessels))
    if target_col:
        y_all = vessels[target_col].astype(str).fillna("unknown_route")
        counts = y_all.value_counts()
        stratify = y_all if y_all.nunique() > 1 and int(counts.min()) >= 2 else None
    else:
        y_all = None
        stratify = None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=min(max(test_size, 0.05), 0.5),
        random_state=random_state,
        stratify=stratify,
    )

    if target_col:
        y_train = y_all.iloc[train_idx].reset_index(drop=True)
        y_test = y_all.iloc[test_idx].reset_index(drop=True)
        label_source = "existing_column"
    else:
        y_train, fold_route_info = train_route_labels(
            vessels.iloc[train_idx].reset_index(drop=True),
            signature.iloc[train_idx].reset_index(drop=True),
            target_col=None,
            requested_clusters=requested_clusters,
            random_state=random_state,
        )
        y_test = assign_kmeans_route_labels(
            signature.iloc[test_idx].reset_index(drop=True),
            fold_route_info,
        ).reset_index(drop=True)
        label_source = "train_fold_kmeans_signature"

    metrics = evaluate_route_holdout(
        x.iloc[train_idx].reset_index(drop=True),
        x.iloc[test_idx].reset_index(drop=True),
        y_train,
        y_test,
        random_state,
    )
    metrics.update(
        {
            "available": True,
            "method": "random_vessel_holdout_train_fold_route_catalog",
            "label_source": label_source,
        }
    )
    return metrics


def evaluate_temporal_route_holdout(
    vessels: pd.DataFrame,
    signature: pd.DataFrame,
    x: pd.DataFrame,
    target_col: str | None,
    requested_clusters: int,
    random_state: int,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    """과거 출항 vessel을 train, 이후 vessel을 test로 두는 시간 기준 평가."""

    if "first_timestamp" not in vessels.columns or len(vessels) < 10:
        return {"available": False, "reason": "missing_timestamp_or_not_enough_vessels"}

    ordered = vessels.sort_values("first_timestamp", kind="mergesort").index.to_numpy()
    split_at = int(round(len(ordered) * min(max(train_ratio, 0.5), 0.9)))
    train_idx = ordered[:split_at]
    test_idx = ordered[split_at:]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"available": False, "reason": "empty_temporal_split"}

    if target_col:
        y_all = vessels[target_col].astype(str).fillna("unknown_route")
        y_train = y_all.loc[train_idx].reset_index(drop=True)
        y_test = y_all.loc[test_idx].reset_index(drop=True)
        label_source = "existing_column"
    else:
        y_train, fold_route_info = train_route_labels(
            vessels.loc[train_idx].reset_index(drop=True),
            signature.loc[train_idx].reset_index(drop=True),
            target_col=None,
            requested_clusters=requested_clusters,
            random_state=random_state,
        )
        y_test = assign_kmeans_route_labels(
            signature.loc[test_idx].reset_index(drop=True),
            fold_route_info,
        ).reset_index(drop=True)
        label_source = "past_train_kmeans_signature"

    metrics = evaluate_route_holdout(
        x.loc[train_idx].reset_index(drop=True),
        x.loc[test_idx].reset_index(drop=True),
        y_train,
        y_test,
        random_state,
    )
    train_times = vessels.loc[train_idx, "first_timestamp"]
    test_times = vessels.loc[test_idx, "first_timestamp"]
    metrics.update(
        {
            "available": True,
            "method": "temporal_vessel_holdout_train_fold_route_catalog",
            "label_source": label_source,
            "train_start": train_times.min().isoformat(),
            "train_end": train_times.max().isoformat(),
            "test_start": test_times.min().isoformat(),
            "test_end": test_times.max().isoformat(),
        }
    )
    return metrics


def evaluate_early_windows(
    clean_points: pd.DataFrame,
    route_labels_by_mmsi: pd.Series,
    route_points: int,
    early_fraction: float,
    windows: list[float],
    random_state: int,
) -> pd.DataFrame:
    """초기 관측 시간창별 항로 분류 성능을 표 형태로 계산한다."""

    rows: list[dict[str, Any]] = []
    for window in sorted({float(value) for value in windows if float(value) > 0}):
        built = build_features(
            clean_points,
            route_points=route_points,
            early_fraction=early_fraction,
            early_window_hours=window,
        )
        labels = built.vessels["MMSI"].map(route_labels_by_mmsi).astype(str)
        x_window, feature_cols = make_feature_matrix(built.vessels, built.early_signature)
        _, metrics = train_classifier(x_window, labels, random_state)
        rows.append(
            {
                "early_window_hours": window,
                "vessels": int(len(labels)),
                "feature_count": int(len(feature_cols)),
                "route_classes": int(labels.nunique()),
                "holdout_accuracy": metrics.get("holdout_accuracy"),
                "holdout_f1_macro": metrics.get("holdout_f1_macro"),
                "evaluation_label_source": "full_route_catalog_for_window_sensitivity",
            }
        )
    return pd.DataFrame(rows)


def route_distances(
    signature: pd.DataFrame,
    route_labels: pd.Series | np.ndarray,
    signature_scaler: StandardScaler,
    route_centroids: dict[str, np.ndarray],
) -> np.ndarray:
    """각 전체 궤적이 배정된 항로 centroid에서 얼마나 떨어져 있는지 측정한다."""

    scaled = signature_scaler.transform(signature)
    distances: list[float] = []
    for idx, label in enumerate(route_labels):
        centroid = route_centroids[str(label)]
        distances.append(float(np.linalg.norm(scaled[idx] - centroid)))
    return np.asarray(distances, dtype=float)


def train_anomaly_thresholds(
    signature: pd.DataFrame,
    route_labels: pd.Series,
    route_info: dict[str, Any],
    anomaly_quantile: float,
) -> dict[str, Any]:
    """비정상 궤적을 표시하는 데 쓸 항로 거리 임계값을 학습한다.

    거리는 항로 라벨에 사용한 것과 같은 scaled signature 공간에서 계산한다.
    이 함수는 전역 임계값과 항로별 임계값을 모두 기록하여, 흔한 항로와
    드물거나 변동성이 큰 항로가 더 적절한 이상 cutoff를 갖도록 한다.
    """

    distances = route_distances(
        signature,
        route_labels,
        route_info["signature_scaler"],
        route_info["route_centroids"],
    )
    labels = route_labels.astype(str).to_numpy()
    quantile = min(max(anomaly_quantile, 0.5), 0.999)
    per_route: dict[str, float] = {}

    for label in sorted(np.unique(labels)):
        route_distance = distances[labels == label]
        if len(route_distance) == 0:
            continue
        per_route[str(label)] = float(np.quantile(route_distance, quantile))

    global_threshold = float(np.quantile(distances, quantile))
    return {
        "global_threshold": global_threshold,
        "per_route_threshold": per_route,
        "train_distance_mean": float(np.mean(distances)),
        "train_distance_std": float(np.std(distances)),
    }


def detect_stop_events(df: pd.DataFrame, slow_sog: float) -> pd.DataFrame:
    """선박 track에서 연속된 저속 stop event를 추출한다."""

    rows: list[dict[str, Any]] = []

    for mmsi, group in df.groupby("MMSI", sort=False, observed=True):
        group = group.sort_values("Timestamp", kind="mergesort")
        slow = group.loc[group["SOG"] <= slow_sog].copy()
        if slow.empty:
            continue

        time_gap = slow["Timestamp"].diff().dt.total_seconds().div(3600.0)
        new_event = time_gap.isna() | (time_gap > 3.0)
        event_ids = new_event.cumsum()

        for event_id, event in slow.groupby(event_ids, sort=False):
            start = event["Timestamp"].iloc[0]
            end = event["Timestamp"].iloc[-1]
            duration = max((end - start).total_seconds() / 3600.0, 0.0)
            rows.append(
                {
                    "MMSI": mmsi,
                    "stop_event_id": f"{mmsi}_{int(event_id):04d}",
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "duration_hours": duration,
                    "point_count": int(len(event)),
                    "center_lat": float(event["Latitude"].mean()),
                    "center_lon": float(event["Longitude"].mean()),
                    "mean_sog": float(event["SOG"].mean()),
                }
            )

    return pd.DataFrame(rows)


def cluster_anchorages(
    stop_events: pd.DataFrame,
    eps_km: float,
    min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DBSCAN으로 stop-event 중심을 가능한 정박지 영역으로 cluster한다."""

    if stop_events.empty:
        return stop_events.assign(anchorage_id=pd.Series(dtype="object")), pd.DataFrame()

    coords = stop_events[["center_lat", "center_lon"]].to_numpy(dtype=float)
    labels = DBSCAN(
        eps=max(eps_km, 0.1) / EARTH_RADIUS_KM,
        min_samples=max(min_samples, 1),
        metric="haversine",
    ).fit_predict(np.radians(coords))

    events = stop_events.copy()
    events["anchorage_cluster"] = labels
    events["anchorage_id"] = np.where(labels >= 0, [f"anchorage_{x:03d}" for x in labels], "noise")

    clusters: list[dict[str, Any]] = []
    for label, group in events.loc[events["anchorage_cluster"] >= 0].groupby(
        "anchorage_cluster"
    ):
        weights = np.maximum(group["point_count"].to_numpy(dtype=float), 1.0)
        center_lat = float(np.average(group["center_lat"], weights=weights))
        center_lon = float(np.average(group["center_lon"], weights=weights))
        clusters.append(
            {
                "anchorage_id": f"anchorage_{int(label):03d}",
                "center_lat": center_lat,
                "center_lon": center_lon,
                "event_count": int(len(group)),
                "vessel_count": int(group["MMSI"].nunique()),
                "mean_duration_hours": float(group["duration_hours"].mean()),
                "total_stop_points": int(group["point_count"].sum()),
            }
        )

    anchorage_clusters = pd.DataFrame(clusters).sort_values(
        ["event_count", "vessel_count"],
        ascending=False,
        ignore_index=True,
    )
    return events, anchorage_clusters


def assign_nearest_anchorage(
    vessels: pd.DataFrame,
    anchorage_clusters: pd.DataFrame,
    eps_km: float,
) -> pd.DataFrame:
    """각 선박의 끝점을 가장 가까운 학습된 정박지 cluster에 배정한다."""

    base_cols = ["MMSI", "end_lat", "end_lon"]
    result = vessels[base_cols].copy()

    if anchorage_clusters.empty:
        result["predicted_anchorage_id"] = pd.NA
        result["predicted_anchorage_lat"] = np.nan
        result["predicted_anchorage_lon"] = np.nan
        result["anchorage_distance_km"] = np.nan
        result["anchorage_confidence"] = np.nan
        return result

    centers = anchorage_clusters[["center_lat", "center_lon"]].to_numpy(dtype=float)
    ids = anchorage_clusters["anchorage_id"].to_numpy()

    chosen_ids: list[str] = []
    chosen_lat: list[float] = []
    chosen_lon: list[float] = []
    chosen_distance: list[float] = []
    chosen_confidence: list[float] = []

    for row in vessels.itertuples(index=False):
        distances = haversine_km(row.end_lat, row.end_lon, centers[:, 0], centers[:, 1])
        best_idx = int(np.argmin(distances))
        distance = float(distances[best_idx])
        chosen_ids.append(str(ids[best_idx]))
        chosen_lat.append(float(centers[best_idx, 0]))
        chosen_lon.append(float(centers[best_idx, 1]))
        chosen_distance.append(distance)
        chosen_confidence.append(float(1.0 / (1.0 + distance / max(eps_km, 0.1))))

    result["predicted_anchorage_id"] = chosen_ids
    result["predicted_anchorage_lat"] = chosen_lat
    result["predicted_anchorage_lon"] = chosen_lon
    result["anchorage_distance_km"] = chosen_distance
    result["anchorage_confidence"] = chosen_confidence
    return result


def predict_routes(
    built: FeatureBuildResult,
    bundle: dict[str, Any],
    anchorage_clusters: pd.DataFrame,
    eps_km: float,
) -> pd.DataFrame:
    """선박별 항로, 이상 점수, 정박 목적지를 예측한다.

    항로 분류는 초기 관측 구간 특징만 사용한다. 이상 점수는 항적을 충분히
    관측한 뒤 쓰는 전체 궤적 signature와 예측 항로 centroid 사이의 거리,
    그리고 분류기 불확실성을 결합한다. 정박지 예측은 이후 병합되어 최종
    CSV가 선박당 한 행으로 항로, 이상 여부, 목적지 맥락을 모두 담게 한다.
    """

    x, _ = make_feature_matrix(built.vessels, built.early_signature)
    classifier: Pipeline = bundle["classifier"]
    predicted = pd.Series(classifier.predict(x), name="predicted_route").astype(str)

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(x)
        route_probability = proba.max(axis=1)
    else:
        route_probability = np.full(len(predicted), np.nan)

    distances = route_distances(
        built.signature,
        predicted,
        bundle["route_info"]["signature_scaler"],
        bundle["route_info"]["route_centroids"],
    )

    thresholds = bundle["anomaly_thresholds"]
    global_threshold = max(float(thresholds["global_threshold"]), 1e-9)
    per_route_threshold = thresholds["per_route_threshold"]
    label_threshold = np.array(
        [
            max(float(per_route_threshold.get(str(label), global_threshold)), 1e-9)
            for label in predicted
        ],
        dtype=float,
    )
    distance_ratio = distances / label_threshold
    anomaly_score = (0.75 * distance_ratio) + (0.25 * (1.0 - route_probability))
    is_anomaly = (distance_ratio > 1.0) | (route_probability < 0.35)

    results = built.vessels[
        [
            "MMSI",
            "first_timestamp",
            "last_timestamp",
            "point_count",
            "duration_hours",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "total_distance_km",
            "mean_sog",
            "width",
            "length",
            "draught",
        ]
    ].copy()
    results["predicted_route"] = predicted.to_numpy()
    results["predicted_route_probability"] = route_probability
    results["route_distance"] = distances
    results["route_distance_threshold"] = label_threshold
    results["route_distance_ratio"] = distance_ratio
    results["anomaly_score"] = anomaly_score
    results["is_anomaly"] = is_anomaly

    anchorage_predictions = assign_nearest_anchorage(
        built.vessels,
        anchorage_clusters,
        eps_km,
    )
    results = results.merge(
        anchorage_predictions.drop(columns=["end_lat", "end_lon"]),
        on="MMSI",
        how="left",
    )
    return results.sort_values(
        ["is_anomaly", "anomaly_score"],
        ascending=[False, False],
        ignore_index=True,
    )


def build_route_catalog(
    vessels: pd.DataFrame,
    signature: pd.DataFrame,
    labels: pd.Series,
    route_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습된 모든 항로 라벨에 대한 요약표와 중심선 테이블을 만든다."""

    route_rows: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []

    signature_scaler = route_info["signature_scaler"]
    route_centroids = route_info["route_centroids"]
    points = int(signature.shape[1] / 2)

    labeled = vessels.copy()
    labeled["route_label"] = labels.astype(str).to_numpy()

    for route_label, group in labeled.groupby("route_label", sort=True):
        route_rows.append(
            {
                "route_label": route_label,
                "vessel_count": int(len(group)),
                "mean_start_lat": float(group["start_lat"].mean()),
                "mean_start_lon": float(group["start_lon"].mean()),
                "mean_end_lat": float(group["end_lat"].mean()),
                "mean_end_lon": float(group["end_lon"].mean()),
                "mean_distance_km": float(group["total_distance_km"].mean()),
                "mean_duration_hours": float(group["duration_hours"].mean()),
            }
        )

        centroid_scaled = np.asarray(route_centroids[str(route_label)]).reshape(1, -1)
        centroid = signature_scaler.inverse_transform(centroid_scaled).reshape(points, 2)
        for idx, (lat, lon) in enumerate(centroid):
            center_rows.append(
                {
                    "route_label": route_label,
                    "step": idx,
                    "Latitude": float(lat),
                    "Longitude": float(lon),
                }
            )

    return pd.DataFrame(route_rows), pd.DataFrame(center_rows)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """numpy/pandas 값을 JSON으로 변환 가능한 형태로 바꿔 실행 요약을 저장한다."""

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def train_and_run(args: argparse.Namespace) -> None:
    """항로 모델 번들을 학습하고 예측을 실행한 뒤 모든 출력을 저장한다.

    이 함수는 스크립트의 orchestration 계층이다. 데이터를 읽고 정제하고,
    특징을 만들고, 항로 라벨을 생성하거나 읽으며, 분류기와 이상/정박지
    헬퍼를 학습한다. 이후 joblib 번들을 저장하고 요청 데이터셋을 예측한 뒤,
    프로젝트의 다른 부분에서 사용할 CSV/JSON 산출물을 내보낸다.
    """

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.train_data.resolve()
    predict_path = args.predict_data.resolve() if args.predict_data else train_path
    model_path = args.model_out.resolve()

    print(f"Loading training data: {train_path}")
    train_points = load_and_clean_ais(train_path)
    print(
        f"Clean training points: {len(train_points):,}, "
        f"vessels: {train_points['MMSI'].nunique():,}"
    )

    train_built = build_features(
        train_points,
        args.route_points,
        args.early_fraction,
        args.early_window_hours,
    )
    target_col = resolve_target_column(train_built.vessels, args.target_col)
    route_labels, route_info = train_route_labels(
        train_built.vessels,
        train_built.signature,
        target_col,
        args.route_clusters,
        args.random_state,
    )

    x_train, feature_cols = make_feature_matrix(
        train_built.vessels,
        train_built.early_signature,
    )
    classifier, metrics = train_classifier(x_train, route_labels, args.random_state)
    strict_route_evaluation = evaluate_strict_route_holdout(
        train_built.vessels,
        train_built.signature,
        x_train,
        target_col,
        args.route_clusters,
        args.random_state,
    )
    temporal_route_evaluation = evaluate_temporal_route_holdout(
        train_built.vessels,
        train_built.signature,
        x_train,
        target_col,
        args.route_clusters,
        args.random_state,
    )
    route_labels_by_mmsi = pd.Series(
        route_labels.astype(str).to_numpy(),
        index=train_built.vessels["MMSI"],
    )
    early_window_metrics = evaluate_early_windows(
        train_built.clean_points,
        route_labels_by_mmsi,
        args.route_points,
        args.early_fraction,
        args.early_eval_windows,
        args.random_state,
    )
    anomaly_thresholds = train_anomaly_thresholds(
        train_built.signature,
        route_labels,
        route_info,
        args.anomaly_quantile,
    )

    stop_events, anchorage_clusters = cluster_anchorages(
        detect_stop_events(train_points, args.slow_sog),
        args.anchorage_eps_km,
        args.anchorage_min_samples,
    )

    bundle = {
        "classifier": classifier,
        "feature_cols": feature_cols,
        "route_points": args.route_points,
        "early_fraction": args.early_fraction,
        "early_window_hours": args.early_window_hours,
        "route_info": route_info,
        "anomaly_thresholds": anomaly_thresholds,
        "anchorage_clusters": anchorage_clusters,
        "metadata": {
            "train_data": str(train_path),
            "label_source": route_info["label_source"],
            "target_col": target_col,
            "classifier_feature_scope": "early_track_only",
            "early_window_hours": args.early_window_hours,
            "early_fraction_fallback": args.early_fraction,
            "full_track_features_used_for": [
                "route_labeling",
                "anomaly_distance",
                "reporting_outputs",
            ],
            "slow_sog": args.slow_sog,
            "anchorage_eps_km": args.anchorage_eps_km,
            "anchorage_min_samples": args.anchorage_min_samples,
        },
    }
    joblib.dump(bundle, model_path)

    route_catalog, route_centers = build_route_catalog(
        train_built.vessels,
        train_built.signature,
        route_labels,
        route_info,
    )

    print(f"Loading prediction data: {predict_path}")
    predict_points = load_and_clean_ais(predict_path)
    predict_built = build_features(
        predict_points,
        args.route_points,
        args.early_fraction,
        args.early_window_hours,
    )
    predictions = predict_routes(
        predict_built,
        bundle,
        anchorage_clusters,
        args.anchorage_eps_km,
    )

    anomaly_ships = predictions.loc[predictions["is_anomaly"]].copy()
    anchorage_predictions = predictions[
        [
            "MMSI",
            "predicted_route",
            "predicted_anchorage_id",
            "predicted_anchorage_lat",
            "predicted_anchorage_lon",
            "anchorage_distance_km",
            "anchorage_confidence",
        ]
    ].copy()

    train_built.vessels.assign(route_label=route_labels.to_numpy()).to_csv(
        output_dir / "training_vessel_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    route_catalog.to_csv(output_dir / "route_catalog.csv", index=False, encoding="utf-8-sig")
    route_centers.to_csv(output_dir / "route_centers_long.csv", index=False, encoding="utf-8-sig")
    early_window_metrics.to_csv(
        output_dir / "route_early_window_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(output_dir / "route_predictions.csv", index=False, encoding="utf-8-sig")
    anomaly_ships.to_csv(output_dir / "anomaly_ships.csv", index=False, encoding="utf-8-sig")
    anchorage_predictions.to_csv(
        output_dir / "anchorage_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stop_events.to_csv(output_dir / "anchorage_stop_events.csv", index=False, encoding="utf-8-sig")
    anchorage_clusters.to_csv(
        output_dir / "anchorage_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "train_data": str(train_path),
        "predict_data": str(predict_path),
        "model_path": str(model_path),
        "clean_train_points": int(len(train_points)),
        "train_vessels": int(len(train_built.vessels)),
        "predicted_vessels": int(len(predictions)),
        "route_classes": int(route_labels.nunique()),
        "label_source": route_info["label_source"],
        "classifier_feature_scope": "early_track_only",
        "early_window_hours": args.early_window_hours,
        "early_fraction_fallback": args.early_fraction,
        "full_track_features_used_for": [
            "route_labeling",
            "anomaly_distance",
            "reporting_outputs",
        ],
        "metrics": metrics,
        "strict_route_evaluation": strict_route_evaluation,
        "temporal_route_evaluation": temporal_route_evaluation,
        "early_window_metrics": early_window_metrics.to_dict("records"),
        "anomaly_thresholds": anomaly_thresholds,
        "anomaly_count": int(predictions["is_anomaly"].sum()),
        "anchorage_cluster_count": int(len(anchorage_clusters)),
        "outputs": {
            "route_predictions": str(output_dir / "route_predictions.csv"),
            "anomaly_ships": str(output_dir / "anomaly_ships.csv"),
            "anchorage_predictions": str(output_dir / "anchorage_predictions.csv"),
            "anchorage_clusters": str(output_dir / "anchorage_clusters.csv"),
            "route_catalog": str(output_dir / "route_catalog.csv"),
            "route_centers_long": str(output_dir / "route_centers_long.csv"),
            "route_early_window_metrics": str(output_dir / "route_early_window_metrics.csv"),
        },
    }
    save_json(output_dir / "run_summary.json", summary)

    print("\nDone.")
    print(f"Route classes: {route_labels.nunique():,}")
    if "holdout_accuracy" in metrics:
        print(f"Holdout accuracy: {metrics['holdout_accuracy']:.4f}")
        print(f"Holdout macro F1: {metrics['holdout_f1_macro']:.4f}")
    if strict_route_evaluation.get("available"):
        print(
            "Strict route holdout: "
            f"accuracy={strict_route_evaluation['accuracy']:.4f}, "
            f"macro_f1={strict_route_evaluation['macro_f1']:.4f}"
        )
    if temporal_route_evaluation.get("available"):
        print(
            "Temporal route holdout: "
            f"accuracy={temporal_route_evaluation['accuracy']:.4f}, "
            f"macro_f1={temporal_route_evaluation['macro_f1']:.4f}"
        )
    print(f"Predicted vessels: {len(predictions):,}")
    print(f"Anomaly vessels: {int(predictions['is_anomaly'].sum()):,}")
    print(f"Anchorage clusters: {len(anchorage_clusters):,}")
    print(f"Model saved: {model_path}")
    print(f"Outputs saved: {output_dir}")


def main() -> None:
    """항로 분석 모델 생성을 위한 CLI 진입점."""

    args = parse_args()
    train_and_run(args)


if __name__ == "__main__":
    main()
