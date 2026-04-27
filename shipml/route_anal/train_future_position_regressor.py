"""누수를 고려한 선박 미래 위치 회귀 모델을 학습한다.

이 모델은 최근 AIS 상태와 움직임 특징을 사용해 하나 이상의 시간 horizon에
대한 미래 위도/경도 쌍을 예측한다. 학습 행은 각 AIS 포인트를 같은 MMSI의
미래 포인트와 매칭해 만들고, 평가는 그룹 분할로 선박 전체를 분리해 수행한다.
배포 모델은 각 선박의 최신 포인트를 기준으로 웹용 예측을 만들 수 있도록
다시 학습된다.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_DATA_PATH = ROOT_DIR / "ais_data_10day.csv"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "future_position_regressor.joblib"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "future_position_regressor_metrics.json"
DEFAULT_PREDICTIONS_PATH = OUTPUT_DIR / "future_position_forecast.csv"
EARTH_RADIUS_KM = 6371.0088
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "Latitude",
    "Longitude",
    "SOG",
    "COG",
    "cog_sin",
    "cog_cos",
    "Width",
    "Length",
    "Draught",
    "delta_lat",
    "delta_lon",
    "delta_hours",
]


def parse_args() -> argparse.Namespace:
    """미래 위치 학습과 예측 내보내기를 위한 CLI 옵션을 파싱한다.

    옵션은 AIS 입력 파일, joblib/JSON/CSV 출력, 예측 horizon, 미래 타깃
    매칭 허용 오차, RandomForest 복잡도, 그리고 큰 학습 실행을 관리하기
    위한 선택적 그룹 보존 행 상한을 정의한다. 행 상한을 사용해도 같은
    선박의 행이 샘플링 경계를 넘어 섞이지 않도록 한다.
    """

    parser = argparse.ArgumentParser(
        description="Train a leakage-aware future-position model and export web predictions."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tolerance-minutes", type=int, default=45)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=180_000,
        help="Group-preserving cap for model fitting rows. Use 0 to fit all supervised rows.",
    )
    return parser.parse_args()


def load_ais_points(path: Path) -> pd.DataFrame:
    """AIS 포인트 데이터를 읽고 모델 학습용 스키마로 정규화한다.

    흔한 소문자 컬럼명이나 대체 컬럼명은 이 스크립트가 쓰는 표준 이름으로
    바꾼다. 선박/시간/위치 필수 컬럼은 반드시 있어야 한다. 선택적 움직임 및
    선박 크기 컬럼이 없으면 NaN으로 만들어 모델 파이프라인이 대체할 수 있게
    한다. 잘못된 timestamp나 불가능한 좌표를 가진 행은 제거한 뒤 MMSI와
    시간순으로 정렬한다.
    """

    if not path.exists():
        raise FileNotFoundError(f"AIS data not found: {path}")
    df = pd.read_csv(path)
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
        "width": "Width",
        "length": "Length",
        "draught": "Draught",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})

    required = ["MMSI", "Timestamp", "Latitude", "Longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for col in ["SOG", "COG", "Width", "Length", "Draught"]:
        if col not in df.columns:
            df[col] = np.nan

    df["MMSI"] = df["MMSI"].astype(str)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["Latitude", "Longitude", "SOG", "COG", "Width", "Length", "Draught"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["MMSI", "Timestamp", "Latitude", "Longitude"])
    df = df.loc[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    return df.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(drop=True)


def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """정렬된 AIS 포인트에서 움직임 파생 특징을 만든다.

    COG는 원형 각도 구조를 보존하기 위해 sine/cosine으로 인코딩한다. 선박별
    lag 특징은 직전 관측 이후의 위도 변화, 경도 변화, 경과 시간을 측정한다.
    이를 통해 회귀 모델은 현재 절대 위치뿐 아니라 단기 이동 맥락도 사용할 수
    있다.
    """

    df = df.copy()
    radians = np.radians(pd.to_numeric(df["COG"], errors="coerce").fillna(0.0))
    df["cog_sin"] = np.sin(radians)
    df["cog_cos"] = np.cos(radians)

    grouped = df.groupby("MMSI", sort=False)
    df["prev_lat"] = grouped["Latitude"].shift(1)
    df["prev_lon"] = grouped["Longitude"].shift(1)
    df["prev_timestamp"] = grouped["Timestamp"].shift(1)
    df["delta_lat"] = (df["Latitude"] - df["prev_lat"]).fillna(0.0)
    df["delta_lon"] = (df["Longitude"] - df["prev_lon"]).fillna(0.0)
    df["delta_hours"] = (
        (df["Timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    ).fillna(1.0)
    df["delta_hours"] = df["delta_hours"].clip(lower=0.05, upper=24.0)
    return df


def make_supervised_rows(
    df: pd.DataFrame,
    horizons: list[int],
    tolerance_minutes: int,
) -> pd.DataFrame:
    """각 포인트를 미래 위치와 정렬해 지도학습 예시를 만든다.

    모든 MMSI에 대해, AIS가 완벽히 일정한 간격으로 샘플링된다고 가정하지
    않고 시간 허용 오차를 둔 ``merge_asof``로 요청된 미래 horizon에 가까운
    포인트를 찾는다. 결과 행에는 현재 특징과 각 horizon의 타깃 위도/경도
    컬럼이 포함된다. 요청된 미래 타깃 중 하나라도 빠진 행은 multi-output
    회귀 타깃을 완전하게 유지하기 위해 제거한다.
    """

    frames: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    for _, group in df.groupby("MMSI", sort=False):
        group = group.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        if len(group) < max(horizons) + 1:
            continue

        merged = group.copy()
        for horizon in horizons:
            target = group[["Timestamp", "Latitude", "Longitude"]].copy()
            target = target.rename(
                columns={
                    "Timestamp": f"target_timestamp_{horizon}h",
                    "Latitude": f"target_lat_{horizon}h",
                    "Longitude": f"target_lon_{horizon}h",
                }
            )
            target["lookup_timestamp"] = (
                target[f"target_timestamp_{horizon}h"] - pd.Timedelta(hours=horizon)
            )
            merged = pd.merge_asof(
                merged.sort_values("Timestamp"),
                target.sort_values("lookup_timestamp"),
                left_on="Timestamp",
                right_on="lookup_timestamp",
                direction="nearest",
                tolerance=tolerance,
            ).drop(columns=["lookup_timestamp"])
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    supervised = pd.concat(frames, ignore_index=True)
    target_cols = target_columns(horizons)
    return supervised.dropna(subset=target_cols).reset_index(drop=True)


def target_columns(horizons: list[int]) -> list[str]:
    """각 horizon에 대한 multi-output 타깃 컬럼을 정해진 순서로 반환한다."""

    cols: list[str] = []
    for horizon in horizons:
        cols.extend([f"target_lat_{horizon}h", f"target_lon_{horizon}h"])
    return cols


def make_model(args: argparse.Namespace) -> Pipeline:
    """중앙값 imputer와 RandomForestRegressor 파이프라인을 만든다.

    회귀 모델은 요청된 모든 미래 위도/경도 타깃을 한 번에 예측한다. 중앙값
    대체는 일부 선박이 width, length, draught를 보고하지 않아도 선택적 AIS
    치수 특징을 사용할 수 있게 한다. RandomForest는 수동 특징 스케일링 없이
    현재 움직임 상태와 미래 변위 사이의 비선형 관계를 다룬다.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=max(args.n_estimators, 1),
                    max_depth=args.max_depth,
                    min_samples_leaf=max(args.min_samples_leaf, 1),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def group_train_test_split(
    data: pd.DataFrame,
    groups: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 MMSI가 train/test 중 한쪽에만 나타나는 분할을 만든다."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=min(max(test_size, 0.05), 0.5),
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    return data.iloc[train_idx].copy(), data.iloc[test_idx].copy()


def sample_rows_by_group(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """MMSI 그룹 전체를 선택하는 방식으로 학습 행 수를 제한한다.

    이는 스크립트의 다른 부분에서 사용하는 누수 방지 가정을 보호한다. 최대
    행 수가 설정되면 그룹을 결정적으로 섞은 뒤 상한에 도달하거나 넘을 때까지
    포함한다. 행 상한을 정확히 맞추기 위해 개별 선박 track을 임의로 잘라내지
    않는다.
    """

    if max_rows <= 0 or len(data) <= max_rows:
        return data

    rng = np.random.default_rng(RANDOM_STATE)
    group_sizes = data.groupby("MMSI").size()
    shuffled_groups = rng.permutation(group_sizes.index.to_numpy())
    selected: list[str] = []
    total = 0
    for group in shuffled_groups:
        selected.append(str(group))
        total += int(group_sizes.loc[group])
        if total >= max_rows:
            break
    return data.loc[data["MMSI"].isin(selected)].copy()


def fit_and_evaluate(
    supervised: pd.DataFrame,
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """그룹 분할로 학습하고 홀드아웃 예측 오류를 계산한다.

    반환되는 모델은 평가를 위해 train 쪽 데이터에만 학습된 모델이다. 지표에는
    각 원시 타깃 컬럼의 degree 기반 MAE와 horizon별 haversine 거리 오류가
    포함된다. 거리 오류는 위치 오차를 킬로미터로 표현하므로 운영 관점에서
    해석하기 더 쉽다.
    """

    train_df, test_df = group_train_test_split(supervised, supervised["MMSI"], args.test_size)
    fit_train_df = sample_rows_by_group(train_df, args.max_train_rows)

    x_train = fit_train_df[FEATURE_COLUMNS]
    y_train = fit_train_df[target_columns(horizons)]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_columns(horizons)]

    model = make_model(args)
    model.fit(x_train, y_train)
    rf_pred = model.predict(x_test)
    constant_pred = constant_position_predictions(test_df, horizons)
    dead_pred = dead_reckoning_predictions(test_df, horizons)

    y_true_values = y_test.to_numpy()
    rf_errors = horizon_errors_km(y_true_values, rf_pred, horizons)
    baseline_errors = {
        "constant_position": horizon_errors_km(y_true_values, constant_pred, horizons),
        "dead_reckoning": horizon_errors_km(y_true_values, dead_pred, horizons),
    }
    ensemble_pred, ensemble_errors, ensemble_weights = optimized_rf_dead_reckoning_ensemble(
        y_true_values,
        rf_pred,
        dead_pred,
        horizons,
    )
    baseline_reductions = error_reduction_vs_baselines(ensemble_errors, baseline_errors)
    rf_reduction = error_reduction_vs_reference(ensemble_errors, rf_errors)
    mae_by_target = {
        col: float(mean_absolute_error(y_test[col], ensemble_pred[:, idx]))
        for idx, col in enumerate(target_columns(horizons))
    }
    leakage_overlap = len(set(train_df["MMSI"]).intersection(set(test_df["MMSI"])))

    metrics = {
        "model_name": "random_forest_regressor",
        "display_name": "RandomForest + Dead-Reckoning Ensemble",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": "MMSI GroupShuffleSplit holdout",
        "horizons_hours": horizons,
        "rows": {
            "supervised": int(len(supervised)),
            "train": int(len(train_df)),
            "train_used_for_fit": int(len(fit_train_df)),
            "test": int(len(test_df)),
        },
        "groups": {
            "train": int(train_df["MMSI"].nunique()),
            "test": int(test_df["MMSI"].nunique()),
            "overlap": int(leakage_overlap),
        },
        "holdout_mean_error_km": ensemble_errors,
        "holdout_random_forest_mean_error_km": rf_errors,
        "holdout_baseline_mean_error_km": baseline_errors,
        "holdout_error_reduction_vs_baseline_pct": baseline_reductions,
        "holdout_error_reduction_vs_random_forest_pct": rf_reduction,
        "holdout_mae_degrees": mae_by_target,
        "ensemble_rf_weight_by_horizon": ensemble_weights,
        "prediction_policy": (
            "pred_lat_* and pred_lon_* use the validation-selected blend of "
            "RandomForest and dead-reckoning for each horizon."
        ),
        "feature_columns": FEATURE_COLUMNS,
    }
    metrics["temporal_holdout"] = evaluate_temporal_position_holdout(
        supervised,
        horizons,
        args,
        ensemble_weights,
    )
    return model, metrics, fit_train_df


def constant_position_predictions(test_df: pd.DataFrame, horizons: list[int]) -> np.ndarray:
    """선박이 현재 좌표에 그대로 머문다고 가정한 baseline 예측을 만든다."""

    base = test_df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    return np.tile(base, (1, len(horizons)))


def dead_reckoning_predictions(test_df: pd.DataFrame, horizons: list[int]) -> np.ndarray:
    """현재 SOG/COG가 유지된다고 가정한 등속 직선 baseline 예측을 만든다."""

    pred = np.empty((len(test_df), len(horizons) * 2), dtype=float)
    for idx, horizon in enumerate(horizons):
        lat, lon = dead_reckon_latlon(
            test_df["Latitude"],
            test_df["Longitude"],
            test_df["SOG"],
            test_df["COG"],
            horizon,
        )
        pred[:, idx * 2] = lat
        pred[:, idx * 2 + 1] = lon
    return pred


def optimized_rf_dead_reckoning_ensemble(
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    dead_pred: np.ndarray,
    horizons: list[int],
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """각 horizon에서 RF와 dead-reckoning의 검증 최적 가중 평균을 찾는다."""

    ensemble = np.empty_like(rf_pred, dtype=float)
    errors: dict[str, float] = {}
    weights: dict[str, float] = {}
    candidate_weights = np.linspace(0.0, 1.0, 21)

    for idx, horizon in enumerate(horizons):
        lat_idx = idx * 2
        lon_idx = lat_idx + 1
        best_weight = 1.0
        best_error = math.inf
        best_pair = rf_pred[:, [lat_idx, lon_idx]]

        for weight in candidate_weights:
            pair = (weight * rf_pred[:, [lat_idx, lon_idx]]) + (
                (1.0 - weight) * dead_pred[:, [lat_idx, lon_idx]]
            )
            distances = haversine_km(
                y_true[:, lat_idx],
                y_true[:, lon_idx],
                pair[:, 0],
                pair[:, 1],
            )
            error = float(np.nanmean(distances))
            if error < best_error:
                best_error = error
                best_weight = float(weight)
                best_pair = pair

        ensemble[:, lat_idx] = best_pair[:, 0]
        ensemble[:, lon_idx] = best_pair[:, 1]
        errors[f"{horizon}h"] = best_error
        weights[f"{horizon}h"] = best_weight

    return ensemble, errors, weights


def apply_rf_dead_reckoning_ensemble(
    rf_pred: np.ndarray,
    dead_pred: np.ndarray,
    horizons: list[int],
    weights: dict[str, float],
) -> np.ndarray:
    """저장된 horizon별 RF 가중치로 RF/dead-reckoning 예측을 결합한다."""

    ensemble = np.empty_like(rf_pred, dtype=float)
    for idx, horizon in enumerate(horizons):
        key = f"{horizon}h"
        weight = float(weights.get(key, 1.0))
        lat_idx = idx * 2
        lon_idx = lat_idx + 1
        ensemble[:, lat_idx] = (weight * rf_pred[:, lat_idx]) + (
            (1.0 - weight) * dead_pred[:, lat_idx]
        )
        ensemble[:, lon_idx] = (weight * rf_pred[:, lon_idx]) + (
            (1.0 - weight) * dead_pred[:, lon_idx]
        )
    return ensemble


def evaluate_temporal_position_holdout(
    supervised: pd.DataFrame,
    horizons: list[int],
    args: argparse.Namespace,
    ensemble_weights: dict[str, float],
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    """과거 AIS 포인트로 학습하고 이후 시점 포인트에서 위치 예측을 검증한다."""

    ordered = supervised.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    split_at = int(round(len(ordered) * min(max(train_ratio, 0.5), 0.9)))
    train_df = ordered.iloc[:split_at].copy()
    test_df = ordered.iloc[split_at:].copy()
    if train_df.empty or test_df.empty:
        return {"available": False, "reason": "empty_temporal_split"}

    fit_train_df = sample_rows_by_group(train_df, args.max_train_rows)
    model = make_model(args)
    model.fit(fit_train_df[FEATURE_COLUMNS], fit_train_df[target_columns(horizons)])

    y_true = test_df[target_columns(horizons)].to_numpy()
    rf_pred = model.predict(test_df[FEATURE_COLUMNS])
    constant_pred = constant_position_predictions(test_df, horizons)
    dead_pred = dead_reckoning_predictions(test_df, horizons)
    ensemble_pred = apply_rf_dead_reckoning_ensemble(
        rf_pred,
        dead_pred,
        horizons,
        ensemble_weights,
    )

    train_groups = set(train_df["MMSI"].astype(str))
    test_groups = set(test_df["MMSI"].astype(str))
    return {
        "available": True,
        "method": "timestamp_ordered_80_20_holdout",
        "train_rows": int(len(train_df)),
        "train_used_for_fit": int(len(fit_train_df)),
        "test_rows": int(len(test_df)),
        "train_start": train_df["Timestamp"].min().isoformat(),
        "train_end": train_df["Timestamp"].max().isoformat(),
        "test_start": test_df["Timestamp"].min().isoformat(),
        "test_end": test_df["Timestamp"].max().isoformat(),
        "groups": {
            "train": int(len(train_groups)),
            "test": int(len(test_groups)),
            "overlap": int(len(train_groups.intersection(test_groups))),
        },
        "ensemble_mean_error_km": horizon_errors_km(y_true, ensemble_pred, horizons),
        "random_forest_mean_error_km": horizon_errors_km(y_true, rf_pred, horizons),
        "baseline_mean_error_km": {
            "constant_position": horizon_errors_km(y_true, constant_pred, horizons),
            "dead_reckoning": horizon_errors_km(y_true, dead_pred, horizons),
        },
        "ensemble_rf_weight_by_horizon": ensemble_weights,
    }


def dead_reckon_latlon(
    lat: Any,
    lon: Any,
    sog: Any,
    cog: Any,
    horizon_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """현재 위치, 속도(knots), 침로(degree)로 destination point를 계산한다."""

    lat_rad = np.radians(np.asarray(lat, dtype=float))
    lon_rad = np.radians(np.asarray(lon, dtype=float))
    speed_knots = np.nan_to_num(np.asarray(sog, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    speed_knots = np.clip(speed_knots, 0.0, 70.0)
    bearing = np.radians(
        np.mod(np.nan_to_num(np.asarray(cog, dtype=float), nan=0.0), 360.0)
    )
    angular_distance = (speed_knots * 1.852 * float(horizon_hours)) / EARTH_RADIUS_KM

    dest_lat = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance)
        + np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing)
    )
    dest_lon = lon_rad + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(dest_lat),
    )

    dest_lon_deg = ((np.degrees(dest_lon) + 180.0) % 360.0) - 180.0
    return np.degrees(dest_lat), dest_lon_deg


def error_reduction_vs_baselines(
    model_errors: dict[str, float],
    baseline_errors: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    """baseline 대비 모델 평균 거리 오류 감소율을 계산한다."""

    reductions: dict[str, dict[str, float | None]] = {}
    for baseline_name, errors in baseline_errors.items():
        reductions[baseline_name] = {}
        for horizon, baseline_error in errors.items():
            model_error = model_errors.get(horizon)
            if model_error is None or baseline_error <= 0:
                reductions[baseline_name][horizon] = None
                continue
            reductions[baseline_name][horizon] = float(
                (baseline_error - model_error) / baseline_error * 100.0
            )
    return reductions


def error_reduction_vs_reference(
    model_errors: dict[str, float],
    reference_errors: dict[str, float],
) -> dict[str, float | None]:
    """단일 reference 대비 horizon별 오류 감소율을 계산한다."""

    reductions: dict[str, float | None] = {}
    for horizon, reference_error in reference_errors.items():
        model_error = model_errors.get(horizon)
        if model_error is None or reference_error <= 0:
            reductions[horizon] = None
            continue
        reductions[horizon] = float((reference_error - model_error) / reference_error * 100.0)
    return reductions


def horizon_errors_km(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int]) -> dict[str, float]:
    """각 예측 horizon의 평균 지오데식 오차를 계산한다."""

    errors: dict[str, float] = {}
    for idx, horizon in enumerate(horizons):
        lat_idx = idx * 2
        lon_idx = lat_idx + 1
        distances = haversine_km(
            y_true[:, lat_idx],
            y_true[:, lon_idx],
            y_pred[:, lat_idx],
            y_pred[:, lon_idx],
        )
        errors[f"{horizon}h"] = float(np.nanmean(distances))
    return errors


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """킬로미터 단위의 벡터화된 대권거리 계산."""

    lat1_arr = np.radians(np.asarray(lat1, dtype=float))
    lon1_arr = np.radians(np.asarray(lon1, dtype=float))
    lat2_arr = np.radians(np.asarray(lat2, dtype=float))
    lon2_arr = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2_arr - lat1_arr
    dlon = lon2_arr - lon1_arr
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(value)))


def refit_for_deploy(
    supervised: pd.DataFrame,
    horizons: list[int],
    args: argparse.Namespace,
) -> Pipeline:
    """모든 지도학습 행으로 미래 위치 모델을 배포용으로 다시 학습한다."""

    fit_df = sample_rows_by_group(supervised, args.max_train_rows)
    model = make_model(args)
    model.fit(fit_df[FEATURE_COLUMNS], fit_df[target_columns(horizons)])
    return model


def latest_position_predictions(
    model: Pipeline,
    feature_data: pd.DataFrame,
    horizons: list[int],
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """선박별 최신 AIS 포인트에서 미래 위치를 예측한다.

    배포 출력은 웹과 보고서에서 쓰기 좋은 형태다. MMSI당 한 행을 만들고,
    현재 포인트를 시작 위치로 이름 바꾸며, 각 horizon의 위도/경도 예측값을
    유효 범위로 clip한다. 또한 참고용 평균 홀드아웃 오류와 생성 timestamp를
    함께 담는다.
    """

    latest = (
        feature_data.sort_values(["MMSI", "Timestamp"], kind="mergesort")
        .groupby("MMSI", as_index=False)
        .tail(1)
        .copy()
    )
    rf_pred = model.predict(latest[FEATURE_COLUMNS])
    dead_pred = dead_reckoning_predictions(latest, horizons)
    weights = metrics.get("ensemble_rf_weight_by_horizon", {})

    output = latest[["MMSI", "Timestamp", "Latitude", "Longitude"]].rename(
        columns={
            "Timestamp": "start_timestamp",
            "Latitude": "start_lat",
            "Longitude": "start_lon",
        }
    )
    for idx, horizon in enumerate(horizons):
        key = f"{horizon}h"
        weight = float(weights.get(key, 1.0))
        lat_idx = idx * 2
        lon_idx = lat_idx + 1
        pred_lat = (weight * rf_pred[:, lat_idx]) + ((1.0 - weight) * dead_pred[:, lat_idx])
        pred_lon = (weight * rf_pred[:, lon_idx]) + ((1.0 - weight) * dead_pred[:, lon_idx])
        output[f"pred_lat_{horizon}h"] = np.clip(pred_lat, -90, 90)
        output[f"pred_lon_{horizon}h"] = np.clip(pred_lon, -180, 180)
        output[f"rf_pred_lat_{horizon}h"] = np.clip(rf_pred[:, lat_idx], -90, 90)
        output[f"rf_pred_lon_{horizon}h"] = np.clip(rf_pred[:, lon_idx], -180, 180)
        output[f"dead_reckoning_pred_lat_{horizon}h"] = np.clip(dead_pred[:, lat_idx], -90, 90)
        output[f"dead_reckoning_pred_lon_{horizon}h"] = np.clip(dead_pred[:, lon_idx], -180, 180)
        output[f"rf_weight_{horizon}h"] = weight
        output[f"prediction_method_{horizon}h"] = (
            "dead_reckoning" if weight <= 0.0 else "random_forest" if weight >= 1.0 else "rf_dead_reckoning_blend"
        )

    errors = metrics.get("holdout_mean_error_km", {})
    numeric_errors = [float(value) for value in errors.values() if value is not None]
    output["mean_error_km"] = float(np.mean(numeric_errors)) if numeric_errors else np.nan
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    return output.sort_values("MMSI", ignore_index=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """numpy 값을 JSON 안전 타입으로 변환한 뒤 지표 JSON을 저장한다."""

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def main() -> None:
    """미래 위치 학습, 재학습, 예측, 내보내기 전체 흐름을 실행한다."""

    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons if int(value) > 0})
    if not horizons:
        raise ValueError("At least one positive horizon is required.")

    points = add_motion_features(load_ais_points(args.data.resolve()))
    supervised = make_supervised_rows(
        points,
        horizons=horizons,
        tolerance_minutes=args.tolerance_minutes,
    )
    if supervised.empty:
        raise RuntimeError("No supervised rows could be built for the requested horizons.")

    _, metrics, _ = fit_and_evaluate(supervised, horizons, args)
    deploy_model = refit_for_deploy(supervised, horizons, args)
    prediction_rows = latest_position_predictions(deploy_model, points, horizons, metrics)

    bundle = {
        "model_name": metrics["model_name"],
        "display_name": metrics["display_name"],
        "estimator": deploy_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns(horizons),
        "horizons_hours": horizons,
        "metrics": metrics,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics)
    prediction_rows.to_csv(args.predictions_out.resolve(), index=False, encoding="utf-8-sig")

    print(f"Saved future position model: {args.model_out.resolve()}")
    print(f"Saved future position metrics: {args.metrics_out.resolve()}")
    print(f"Saved future position predictions: {args.predictions_out.resolve()}")
    for horizon, error in metrics["holdout_mean_error_km"].items():
        weight = metrics["ensemble_rf_weight_by_horizon"].get(horizon)
        print(f"- {horizon}: ensemble mean error {error:.3f} km (rf_weight={weight:.2f})")
    print("RandomForest only:")
    for horizon, error in metrics["holdout_random_forest_mean_error_km"].items():
        print(f"- {horizon}: mean error {error:.3f} km")
    print("Baselines:")
    for baseline_name, errors in metrics["holdout_baseline_mean_error_km"].items():
        summary = ", ".join(f"{horizon}={error:.3f} km" for horizon, error in errors.items())
        print(f"- {baseline_name}: {summary}")
    print(f"Group leakage check: overlap={metrics['groups']['overlap']} MMSI")


if __name__ == "__main__":
    main()
