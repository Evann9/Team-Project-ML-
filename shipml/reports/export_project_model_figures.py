from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import koreanmatplotlib  # type: ignore  # noqa: F401
except ImportError:
    try:
        import koreanize_matplotlib  # type: ignore  # noqa: F401
    except ImportError:
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SHIPML_DIR = Path(__file__).resolve().parents[1]
TYPE_OUTPUTS = SHIPML_DIR / "type_anal" / "outputs"
ROUTE_OUTPUTS = SHIPML_DIR / "route_anal" / "outputs"
DEFAULT_OUTPUT_DIR = SHIPML_DIR / "reports" / "figures"

DEFAULT_TYPE_METRICS = TYPE_OUTPUTS / "ship_type_classifier_group_split_metrics.json"
DEFAULT_CLASS_METRICS = TYPE_OUTPUTS / "ship_type_classifier_class_metrics.csv"
DEFAULT_FEATURE_IMPORTANCE = TYPE_OUTPUTS / "ship_type_classifier_feature_importance.csv"
DEFAULT_CONFUSION_PAIRS = TYPE_OUTPUTS / "ship_type_classifier_confusion_pairs.csv"
DEFAULT_ROUTE_SUMMARY = ROUTE_OUTPUTS / "run_summary.json"
DEFAULT_ROUTE_PREDICTIONS = ROUTE_OUTPUTS / "route_predictions_with_types.csv"
DEFAULT_FUTURE_METRICS = ROUTE_OUTPUTS / "future_position_regressor_metrics.json"
DEFAULT_FUTURE_FORECAST = ROUTE_OUTPUTS / "future_position_forecast.csv"

SHIPTYPE_KO = {
    "Cargo": "Cargo",
    "Dredging": "Dredging",
    "Fishing": "Fishing",
    "HSC": "HSC",
    "Law enforcement": "Law enforcement",
    "Military": "Military",
    "Passenger": "Passenger",
    "Pilot": "Pilot",
    "Pleasure": "Pleasure",
    "SAR": "SAR",
    "Sailing": "Sailing",
    "Tanker": "Tanker",
    "Towing": "Towing",
    "Tug": "Tug",
}

FEATURE_KO = {
    "length": "length",
    "draught": "draught",
    "width": "width",
    "sog": "sog",
    "cog_sin": "cog_sin",
    "cog_cos": "cog_cos",
    "heading_sin": "heading_sin",
    "heading_cos": "heading_cos",
    "cog": "cog",
    "heading": "heading",
    "navigationalstatus": "navigationalstatus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export presentation-ready PNG figures for the ShipML project."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sample-vessels", type=int, default=80)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_model_score_summary(
    type_metrics: dict[str, Any],
    route_summary: dict[str, Any],
    future_metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    ship_best = type_metrics.get("best_metrics", {})
    route_metrics = route_summary.get("metrics", {})
    future_errors = future_metrics.get("holdout_mean_error_km", {})
    future_baselines = future_metrics.get("holdout_baseline_mean_error_km", {})

    labels = ["선종 분류\n정확도", "선종 분류\n매크로 F1", "항로 분류\n정확도", "항로 분류\n매크로 F1"]
    values = [
        ship_best.get("test_accuracy"),
        ship_best.get("macro_f1"),
        route_metrics.get("holdout_accuracy"),
        route_metrics.get("holdout_f1_macro"),
    ]
    values = [float(value) if value is not None else np.nan for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = ["#0f766e", "#0f766e", "#2563eb", "#2563eb"]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("분류 모델 성능")
    axes[0].set_ylabel("점수")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        if np.isfinite(value):
            axes[0].text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    future_keys = list(future_errors.keys())
    future_labels = [label_horizon(key) for key in future_keys]
    future_series: list[tuple[str, dict[str, Any], str]] = [
        ("RF+DR 앙상블", future_errors, "#be185d")
    ]
    rf_errors = future_metrics.get("holdout_random_forest_mean_error_km", {})
    if rf_errors:
        future_series.append(("RandomForest", rf_errors, "#2563eb"))
    baseline_labels = {
        "constant_position": ("현재 위치 유지", "#64748b"),
        "dead_reckoning": ("등속 직선", "#f59e0b"),
    }
    for key, (label, color) in baseline_labels.items():
        errors = future_baselines.get(key, {})
        if errors:
            future_series.append((label, errors, color))

    x = np.arange(len(future_keys))
    width = min(0.24, 0.78 / max(len(future_series), 1))
    for series_idx, (series_label, errors, color) in enumerate(future_series):
        offset = (series_idx - (len(future_series) - 1) / 2.0) * width
        values = [float(errors.get(key, np.nan)) for key in future_keys]
        axes[1].bar(x + offset, values, width=width, color=color, label=series_label)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                axes[1].text(
                    x[idx] + offset,
                    value + 0.12,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=8,
                )
    axes[1].set_xticks(x, future_labels)
    axes[1].set_title("미래 좌표 예측 평균 오차")
    axes[1].set_ylabel("평균 오차 (km)")
    axes[1].grid(axis="y", alpha=0.25)
    if len(future_series) > 1:
        axes[1].legend(fontsize=8)

    fig.suptitle("ShipML 주요 모델 결과 요약", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output_dir / "01_model_result_overview.png")


def save_ship_type_class_f1(class_metrics: pd.DataFrame, output_dir: Path) -> Path | None:
    if class_metrics.empty:
        return None
    df = class_metrics.sort_values("f1_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.34)))
    ax.barh(df["shiptype"].map(label_shiptype), df["f1_score"], color="#0f766e")
    ax.set_xlim(0, 1.05)
    ax.set_title("선종별 F1 점수")
    ax.set_xlabel("F1 점수")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(df["f1_score"]):
        ax.text(float(value) + 0.015, idx, f"{float(value):.2f}", va="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir / "02_ship_type_class_f1_scores.png")


def save_ship_type_feature_importance(
    feature_importance: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> Path | None:
    if feature_importance.empty:
        return None
    df = feature_importance.head(max(top_n, 1)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.34)))
    ax.barh(df["feature"].map(label_feature), df["importance"], color="#155e75")
    ax.set_title("선종 분류 주요 변수 중요도")
    ax.set_xlabel("중요도")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir / "03_ship_type_top_features.png")


def save_ship_type_confusions(
    confusion_pairs: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> Path | None:
    if confusion_pairs.empty:
        return None
    df = confusion_pairs.head(max(top_n, 1)).copy()
    df["pair"] = df["actual"].map(label_shiptype) + " -> " + df["predicted"].map(label_shiptype)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.36)))
    ax.barh(df["pair"], df["count"], color="#be185d")
    ax.set_title("주요 선종 혼동 관계")
    ax.set_xlabel("건수")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir / "04_ship_type_top_confusion_pairs.png")


def save_route_distribution(routes: pd.DataFrame, output_dir: Path) -> Path | None:
    if routes.empty:
        return None
    df = routes.copy()
    df["is_anomaly"] = df["is_anomaly"].map(parse_bool)
    counts = (
        df.groupby(["predicted_route", "is_anomaly"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "정상", True: "이상 항로"})
    )
    for col in ["정상", "이상 항로"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["전체"] = counts["정상"] + counts["이상 항로"]
    counts = counts.sort_values("전체", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(counts))
    ax.bar(x, counts["정상"], color="#2563eb", label="정상")
    ax.bar(x, counts["이상 항로"], bottom=counts["정상"], color="#c2410c", label="이상 항로")
    ax.set_title("예측 항로별 선박 수와 이상 항로")
    ax.set_ylabel("선박 수")
    ax.set_xticks(x, [label_route(label) for label in counts.index], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir / "05_route_distribution_anomalies.png")


def save_route_shiptype_heatmap(routes: pd.DataFrame, output_dir: Path) -> Path | None:
    if routes.empty or "predicted_shiptype" not in routes.columns:
        return None
    pivot = pd.crosstab(routes["predicted_route"], routes["predicted_shiptype"])
    top_shiptypes = pivot.sum(axis=0).sort_values(ascending=False).head(8).index
    pivot = pivot[top_shiptypes].sort_index()
    pivot = pivot.rename(columns=SHIPTYPE_KO)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_title("예측 항로 패턴별 선종 분포")
    ax.set_xlabel("예측 선종")
    ax.set_ylabel("예측 항로")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), [label_route(label) for label in pivot.index])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="선박 수")
    fig.tight_layout()
    return save_figure(fig, output_dir / "06_route_ship_type_heatmap.png")


def save_future_forecast_map(
    future_forecast: pd.DataFrame,
    output_dir: Path,
    sample_vessels: int,
) -> Path | None:
    if future_forecast.empty:
        return None
    df = future_forecast.head(max(sample_vessels, 1)).copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["start_lon"], df["start_lat"], s=12, color="#172026", alpha=0.5, label="현재 위치")
    horizons = [(1, "#0f766e"), (2, "#2563eb"), (3, "#be185d")]
    for horizon, color in horizons:
        lat_col = f"pred_lat_{horizon}h"
        lon_col = f"pred_lon_{horizon}h"
        if lat_col not in df.columns or lon_col not in df.columns:
            continue
        ax.scatter(df[lon_col], df[lat_col], s=10, color=color, alpha=0.5, label=f"{horizon}시간 후")
        for row in df.itertuples(index=False):
            start_lon = getattr(row, "start_lon")
            start_lat = getattr(row, "start_lat")
            end_lon = getattr(row, lon_col)
            end_lat = getattr(row, lat_col)
            ax.plot([start_lon, end_lon], [start_lat, end_lat], color=color, alpha=0.12, linewidth=0.8)
    ax.set_title("미래 좌표 예측 샘플")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.legend(loc="best", markerscale=1.5)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return save_figure(fig, output_dir / "07_future_position_forecast_sample.png")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def label_shiptype(value: Any) -> str:
    text = str(value)
    return SHIPTYPE_KO.get(text, text)


def label_feature(value: Any) -> str:
    text = str(value)
    if "=" in text:
        source, option = text.split("=", 1)
        return f"{FEATURE_KO.get(source, source)}={option}"
    return FEATURE_KO.get(text, text)


def label_route(value: Any) -> str:
    text = str(value)
    if text.startswith("route_"):
        return f"항로 {text.removeprefix('route_')}"
    return text


def label_horizon(value: Any) -> str:
    text = str(value)
    if text.endswith("h") and text[:-1].isdigit():
        return f"{text[:-1]}시간"
    return text


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    type_metrics = read_json(DEFAULT_TYPE_METRICS)
    route_summary = read_json(DEFAULT_ROUTE_SUMMARY)
    future_metrics = read_json(DEFAULT_FUTURE_METRICS)
    class_metrics = read_csv(DEFAULT_CLASS_METRICS)
    feature_importance = read_csv(DEFAULT_FEATURE_IMPORTANCE)
    confusion_pairs = read_csv(DEFAULT_CONFUSION_PAIRS)
    routes = read_csv(DEFAULT_ROUTE_PREDICTIONS)
    future_forecast = read_csv(DEFAULT_FUTURE_FORECAST)

    saved: list[Path | None] = [
        save_model_score_summary(type_metrics, route_summary, future_metrics, output_dir),
        save_ship_type_class_f1(class_metrics, output_dir),
        save_ship_type_feature_importance(feature_importance, output_dir, args.top_n),
        save_ship_type_confusions(confusion_pairs, output_dir, args.top_n),
        save_route_distribution(routes, output_dir),
        save_route_shiptype_heatmap(routes, output_dir),
        save_future_forecast_map(future_forecast, output_dir, args.sample_vessels),
    ]

    for path in saved:
        if path is not None:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
