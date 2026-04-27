"""MMSI 기준 그룹 홀드아웃 평가로 선박 종류 분류기를 학습한다.

이 스크립트는 기존 행 단위 분할 학습기의 누수 방지 버전이다. 평가 시
선박 식별자를 그룹 컬럼으로 사용하여 같은 MMSI가 train과 test 양쪽에
동시에 나타나지 않게 한다. 요청된 분류기 계열을 그룹 분할에서 비교한 뒤,
선택된 모델을 전체 행에 다시 학습시키고 배포 가능한 joblib 번들, 간결한
지표, CSV 진단 파일을 저장한다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ship_type_model import (  # type: ignore  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        refit_full,
        save_json,
        split_columns,
    )
else:
    from .ship_type_model import (  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        refit_full,
        save_json,
        split_columns,
    )


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "ais_ship_type_with_mmsi.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split_evaluation.json"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split.joblib"
)
DEFAULT_MODEL_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split_metrics.json"
)
DEFAULT_CLASS_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_class_metrics.csv"
)
DEFAULT_CONFUSION_PAIRS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_confusion_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    """그룹 분할 모델 평가와 내보내기를 위한 CLI 옵션을 파싱한다.

    인자는 입출력 위치, 누수 방지 그룹으로 쓸 컬럼, 비교할 모델 계열,
    그리고 기존 행 단위 랜덤 분할을 기준선으로 함께 실행할지 여부를
    제어한다. 기준선 옵션은 선박 누수 때문에 성능이 얼마나 낙관적으로
    보였는지 정량화할 때 유용하다.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ship-type classifiers with a leakage-resistant group split. "
            "The grouping column, usually MMSI, is used only for splitting and is "
            "removed from model features."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV containing MMSI and shiptype columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file where evaluation metrics are saved.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved deployable group-split-selected model bundle.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_MODEL_METRICS_PATH,
        help="Compact metrics JSON for the deployable group-split-selected model.",
    )
    parser.add_argument(
        "--class-metrics-out",
        type=Path,
        default=DEFAULT_CLASS_METRICS_PATH,
        help="CSV with per-class precision/recall/F1/support for the best group-split model.",
    )
    parser.add_argument(
        "--confusion-pairs-out",
        type=Path,
        default=DEFAULT_CONFUSION_PAIRS_PATH,
        help="CSV with actual -> predicted confusion pairs for the best group-split model.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="mmsi",
        help="Group column used to prevent the same vessel from appearing in both splits.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Approximate test ratio. StratifiedGroupKFold uses 1/test_size folds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "voting", "xgboost"],
        help=(
            "Models to compare. Supported: logistic_regression random_forest "
            "voting xgboost knn svc."
        ),
    )
    parser.add_argument(
        "--compare-random-split",
        action="store_true",
        help="Also run the old row-level random split to show the leakage gap.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip isotonic calibration for predicted ship-type confidence values.",
    )
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, requested: str) -> str:
    """사용자가 입력한 컬럼명을 대소문자 구분 없이 찾는다.

    CSV 파일은 ``mmsi``, ``MMSI``처럼 대소문자가 다를 수 있다. 이 헬퍼는
    명령행 사용성을 유연하게 유지하면서도, 요청한 그룹 컬럼이 없을 때는
    명확한 오류를 낸다.
    """

    lookup = {col.lower(): col for col in df.columns}
    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f"Group column '{requested}' not found. Available columns: {list(df.columns)}"
        )
    return resolved


def add_trig_features(df: pd.DataFrame) -> pd.DataFrame:
    """원형 AIS 각도 컬럼에 sine/cosine 인코딩을 추가한다.

    COG와 heading은 360도에서 다시 0도로 이어지므로, 원시 숫자로만 다루면
    실제로 가까운 359도와 1도가 멀리 떨어진 값처럼 보인다. 삼각함수 특징은
    이런 원형 구조를 보존한다. 원본 각도 컬럼이 있고 파생 컬럼이 아직 없을
    때만 새로 추가한다.
    """

    df = df.copy()
    for angle_col in ["cog", "heading"]:
        if angle_col not in df.columns:
            continue
        angle = pd.to_numeric(df[angle_col], errors="coerce")
        radians = np.radians(angle.fillna(0.0))
        sin_col = f"{angle_col}_sin"
        cos_col = f"{angle_col}_cos"
        if sin_col not in df.columns:
            df[sin_col] = np.sin(radians)
        if cos_col not in df.columns:
            df[cos_col] = np.cos(radians)
    return df


def group_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    """각 MMSI가 train 또는 test 한쪽에만 속하도록 행을 분할한다.

    선호하는 splitter는 StratifiedGroupKFold다. 선박 그룹 경계를 지키면서도
    선박 종류 클래스 분포를 최대한 보존하기 때문이다. 데이터가 너무 작거나
    이 전략을 쓰기에 충분히 다양하지 않으면 GroupShuffleSplit으로
    fallback한다. 이 경우 클래스 균형은 약해질 수 있지만 누수 경계는 여전히
    유지된다.
    """

    test_size = min(max(test_size, 0.05), 0.5)
    n_splits = max(2, int(round(1.0 / test_size)))
    group_counts = groups.value_counts()
    usable = len(group_counts) >= n_splits and y.nunique() > 1

    if usable:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        method = f"StratifiedGroupKFold(n_splits={n_splits})"
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        method = "GroupShuffleSplit"

    return (
        x.iloc[train_idx],
        x.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        groups.iloc[train_idx],
        groups.iloc[test_idx],
        method,
    )


def evaluate_specs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """요청된 모든 모델 spec을 그룹 홀드아웃에서 학습하고 평가한다.

    후보 파이프라인은 공통 ``ship_type_model`` 팩토리에서 생성하므로 기존
    학습기와 전처리가 일관된다. 결과는 정확도를 우선으로, macro F1을
    다음 기준으로 정렬한다. macro F1을 정렬 키에 포함하면 소수 선박 종류를
    더 잘 처리하는 모델이 동률을 깰 수 있다.
    """

    categorical_cols, numeric_cols = split_columns(x_train)
    specs, skipped = model_specs(categorical_cols, numeric_cols, set(model_names))
    results: list[dict[str, Any]] = []

    for spec in specs:
        metrics = fit_spec_with_predictions(spec, x_train, x_test, y_train, y_test)
        results.append(metrics)

    results.sort(key=lambda item: (item["macro_f1"], item["test_accuracy"]), reverse=True)
    return results, skipped


def fit_spec_with_predictions(
    spec: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """후보 하나를 학습하고 지표와 오분류 상세를 반환한다.

    공통 ``fit_spec`` 헬퍼와 같은 흐름이지만, 보고용 confusion-pair 행도
    보관한다. 선택적 라벨 인코딩은 숫자 클래스 ID가 필요한 estimator에만
    적용하며, 모든 report가 원래 선박 종류 라벨을 쓰도록 지표 계산 전에
    예측값을 복원한다.
    """

    label_encoder: LabelEncoder | None = None
    fit_y: pd.Series | np.ndarray = y_train

    if spec.requires_label_encoding:
        label_encoder = LabelEncoder()
        fit_y = label_encoder.fit_transform(y_train)

    spec.estimator.fit(x_train, fit_y)
    pred = spec.estimator.predict(x_test)
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred.astype(int))

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    confusion_pairs = confusion_pair_rows(y_test, pred, spec.name, spec.display_name)
    return {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_pairs": confusion_pairs,
        "top_confusion_pairs": confusion_pairs[:12],
    }


def confusion_pair_rows(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    display_name: str,
) -> list[dict[str, Any]]:
    """혼동 행렬에서 실제값-예측값 오류 쌍을 정렬해 반환한다.

    이 CSV는 모델이 어떤 클래스를 혼동하는지 강조하기 위한 것이므로 정답
    예측은 제외한다. 각 행에는 실제 클래스의 support와 해당 클래스 안에서의
    오류율도 포함한다. 이를 통해 자주 발생하는 작은 실수와 희귀 클래스의
    심각한 실패를 구분할 수 있다.
    """

    labels = sorted(set(y_true.astype(str)).union(set(map(str, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows: list[dict[str, Any]] = []
    for row_idx, actual in enumerate(labels):
        actual_total = int(cm[row_idx, :].sum())
        for col_idx, predicted in enumerate(labels):
            count = int(cm[row_idx, col_idx])
            if actual == predicted or count == 0:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "display_name": display_name,
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "actual_support": actual_total,
                    "actual_error_rate": float(count / max(actual_total, 1)),
                }
            )
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def class_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """sklearn의 클래스별 classification report를 CSV 행으로 평탄화한다.

    이 내보내기는 보고서나 대시보드에서 클래스별로 살펴보기 위한 것이므로,
    accuracy, macro avg, weighted avg 같은 집계 섹션은 제외한다.
    """

    report = metrics.get("classification_report", {})
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "model_name": metrics["model_name"],
                "display_name": metrics["display_name"],
                "shiptype": label,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1_score": values.get("f1-score"),
                "support": values.get("support"),
            }
        )
    return rows


def random_split_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    test_size: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """그룹 분할과 비교하기 위해 기존 행 단위 분할을 실행한다.

    행 단위 분할은 같은 선박의 포인트를 train과 test 양쪽에 배치할 수 있어,
    처음 보는 선박에 대한 실제 성능보다 좋아 보일 수 있다. 이 함수는
    선택 실행이며, 호출자가 명시적으로 비교를 요청할 때만 기준선 지표를
    기록한다.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return evaluate_specs(x_train, x_test, y_train, y_test, model_names)


def leakage_report(
    train_groups: pd.Series,
    test_groups: pd.Series,
) -> dict[str, Any]:
    """선박 그룹이 홀드아웃 경계를 넘어갔는지 요약한다."""

    train_set = set(train_groups.astype(str))
    test_set = set(test_groups.astype(str))
    overlap = train_set.intersection(test_set)
    return {
        "train_groups": int(len(train_set)),
        "test_groups": int(len(test_set)),
        "overlap_groups": int(len(overlap)),
        "overlap_ratio_of_test_groups": float(len(overlap) / max(len(test_set), 1)),
    }


def class_counts(y: pd.Series) -> dict[str, int]:
    """분할 진단에 쓸 안정적이고 JSON 친화적인 클래스 개수를 반환한다."""

    return {str(label): int(count) for label, count in y.value_counts().sort_index().items()}


def train_deploy_bundle(
    x: pd.DataFrame,
    y: pd.Series,
    best_metrics: dict[str, Any],
    all_metrics: list[dict[str, Any]],
    skipped: dict[str, str],
    split_info: dict[str, Any],
    probability_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """그룹 분할에서 선택된 모델을 재학습하고 저장 산출물을 만든다.

    평가는 MMSI 그룹 홀드아웃에서 수행하지만, 모델 계열을 선택한 뒤 배포에는
    사용 가능한 모든 라벨 데이터를 쓰는 편이 좋다. 이 함수는 winning model
    spec을 다시 만들고 전체 특징 행렬에 재학습한 뒤, estimator, 특징 스키마,
    라벨 encoder, 지표, 그리고 MMSI가 모델 특징이 아니라 분할 용도로만
    쓰였음을 설명하는 평가 메타데이터를 함께 저장한다.
    """

    categorical_cols, numeric_cols = split_columns(x)
    fresh_specs, _ = model_specs(categorical_cols, numeric_cols, {best_metrics["model_name"]})
    if not fresh_specs:
        raise RuntimeError(f"Could not recreate model spec: {best_metrics['model_name']}")

    final_estimator, final_label_encoder = refit_full(fresh_specs[0], x, y)
    bundle = {
        "target": TARGET,
        "model_name": best_metrics["model_name"],
        "display_name": best_metrics["display_name"],
        "estimator": final_estimator,
        "label_encoder": final_label_encoder,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(x)),
        "target_classes": sorted(y.unique().tolist()),
        "best_metrics": best_metrics,
        "all_metrics": all_metrics,
        "skipped_models": skipped,
        "evaluation": {
            "method": "mmsi_group_split",
            "split": split_info,
            "note": (
                "The selected model is refit on all rows for deployment after "
                "MMSI group-split evaluation. MMSI is not used as a feature."
            ),
        },
    }
    if probability_calibration:
        bundle["probability_calibration"] = probability_calibration
    return bundle


def build_probability_calibration(
    best_metrics: dict[str, Any],
    x_train: pd.DataFrame,
    x_calib: pd.DataFrame,
    y_train: pd.Series,
    y_calib: pd.Series,
) -> dict[str, Any] | None:
    """그룹 홀드아웃 예측 확률을 이용해 클래스별 isotonic calibrator를 학습한다."""

    categorical_cols, numeric_cols = split_columns(x_train)
    fresh_specs, _ = model_specs(categorical_cols, numeric_cols, {best_metrics["model_name"]})
    if not fresh_specs:
        return None

    spec = fresh_specs[0]
    label_encoder: LabelEncoder | None = None
    fit_y: pd.Series | np.ndarray = y_train
    if spec.requires_label_encoding:
        label_encoder = LabelEncoder()
        fit_y = label_encoder.fit_transform(y_train)

    spec.estimator.fit(x_train, fit_y)
    if not hasattr(spec.estimator, "predict_proba"):
        return None

    raw_proba = np.asarray(spec.estimator.predict_proba(x_calib))
    if label_encoder is not None:
        class_order = [str(value) for value in label_encoder.classes_]
    else:
        class_order = [str(value) for value in getattr(spec.estimator, "classes_", [])]
    if raw_proba.shape[1] != len(class_order):
        return None

    calibrators: list[IsotonicRegression | None] = []
    calibrated = np.zeros_like(raw_proba, dtype=float)
    for idx, label in enumerate(class_order):
        binary_target = (y_calib.astype(str).to_numpy() == label).astype(int)
        if binary_target.min() == binary_target.max():
            calibrators.append(None)
            calibrated[:, idx] = raw_proba[:, idx]
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba[:, idx], binary_target)
        calibrators.append(calibrator)
        calibrated[:, idx] = calibrator.predict(raw_proba[:, idx])

    calibrated = np.clip(calibrated, 0.0, 1.0)
    row_sums = calibrated.sum(axis=1)
    valid = row_sums > 0
    calibrated[valid] = calibrated[valid] / row_sums[valid, None]
    calibrated[~valid] = raw_proba[~valid]

    pred = spec.estimator.predict(x_calib)
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred.astype(int))
    before_confidence = predicted_label_confidence(pred, raw_proba, class_order)
    after_confidence = predicted_label_confidence(pred, calibrated, class_order)

    return {
        "method": "one_vs_rest_isotonic",
        "classes": class_order,
        "calibrators": calibrators,
        "calibration_rows": int(len(x_calib)),
        "calibration_note": (
            "Isotonic calibrators were fit on the MMSI group holdout predictions. "
            "The deployed estimator is still refit on all rows; calibration only "
            "adjusts reported confidence values."
        ),
        "ece_before": expected_calibration_error(y_calib, pred, before_confidence),
        "ece_after": expected_calibration_error(y_calib, pred, after_confidence),
    }


def predicted_label_confidence(
    pred: np.ndarray,
    proba: np.ndarray,
    class_order: list[str],
) -> np.ndarray:
    """예측 라벨에 해당하는 proba 컬럼을 confidence 벡터로 추출한다."""

    lookup = {label: idx for idx, label in enumerate(class_order)}
    confidence = np.full(len(pred), np.nan)
    for idx, label in enumerate(np.asarray(pred, dtype=str)):
        class_idx = lookup.get(label)
        if class_idx is None:
            confidence[idx] = float(np.nanmax(proba[idx]))
        else:
            confidence[idx] = float(proba[idx, class_idx])
    return confidence


def expected_calibration_error(
    y_true: pd.Series,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    """예측 confidence와 실제 정답률 사이의 expected calibration error."""

    y_true_arr = y_true.astype(str).to_numpy()
    y_pred_arr = np.asarray(y_pred, dtype=str)
    conf = np.asarray(confidence, dtype=float)
    valid = np.isfinite(conf)
    if not valid.any():
        return float("nan")

    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]
    conf = conf[valid]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not mask.any():
            continue
        accuracy = np.mean(y_true_arr[mask] == y_pred_arr[mask])
        avg_confidence = float(np.mean(conf[mask]))
        error += float(mask.mean()) * abs(float(accuracy) - avg_confidence)
    return float(error)


def main() -> None:
    """그룹 분할 학습, 보고, 내보내기 전체 워크플로를 실행한다."""

    args = parse_args()
    df = add_trig_features(load_type_data(args.data.resolve()))
    group_col = resolve_column(df, args.group_col)
    df[group_col] = df[group_col].astype(str)

    feature_drop_cols = [TARGET, group_col]
    x = df.drop(columns=feature_drop_cols)
    y = df[TARGET].astype(str)
    groups = df[group_col]

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
        split_method,
    ) = group_train_test_split(x, y, groups, args.test_size)

    group_results, skipped = evaluate_specs(x_train, x_test, y_train, y_test, args.models)

    split_info = {
        "method": split_method,
        "requested_test_size": args.test_size,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "leakage_check": leakage_report(train_groups, test_groups),
    }

    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()),
        "target": TARGET,
        "group_col": group_col,
        "group_col_used_as_feature": False,
        "feature_columns": x.columns.tolist(),
        "split": split_info,
        "group_split_metrics": group_results,
        "skipped_models": skipped,
    }

    if args.compare_random_split:
        random_results, random_skipped = random_split_baseline(x, y, args.models, args.test_size)
        output["row_level_random_split_metrics"] = random_results
        output["row_level_random_split_note"] = (
            "This is the old row-level split. It may be optimistic because points "
            "from the same MMSI can appear in both train and test."
        )
        output["skipped_models"].update(random_skipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.resolve(), output)

    if not group_results:
        raise RuntimeError("No group-split model results were produced.")

    best_group_result = group_results[0]
    class_rows = class_metric_rows(best_group_result)
    confusion_rows = best_group_result.get("confusion_pairs", [])
    pd.DataFrame(class_rows).to_csv(
        args.class_metrics_out.resolve(),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(confusion_rows).to_csv(
        args.confusion_pairs_out.resolve(),
        index=False,
        encoding="utf-8-sig",
    )

    probability_calibration = None
    if not args.no_calibration:
        probability_calibration = build_probability_calibration(
            best_group_result,
            x_train,
            x_test,
            y_train,
            y_test,
        )

    bundle = train_deploy_bundle(
        x=x,
        y=y,
        best_metrics=best_group_result,
        all_metrics=group_results,
        skipped=skipped,
        split_info=split_info,
        probability_calibration=probability_calibration,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics_summary(bundle))

    print(f"Saved group-split metrics: {args.output.resolve()}")
    print(f"Saved deploy model: {args.model_out.resolve()}")
    print(f"Saved deploy model metrics: {args.metrics_out.resolve()}")
    print(f"Saved class metrics: {args.class_metrics_out.resolve()}")
    print(f"Saved confusion pairs: {args.confusion_pairs_out.resolve()}")
    if probability_calibration:
        print(
            "Calibration ECE: "
            f"before={probability_calibration['ece_before']:.4f}, "
            f"after={probability_calibration['ece_after']:.4f}"
        )
    print(f"Rows: train={len(x_train):,}, test={len(x_test):,}")
    print(
        "Group leakage check: "
        f"overlap={output['split']['leakage_check']['overlap_groups']:,} groups"
    )
    print("\nGroup split results:")
    for result in group_results:
        print(
            f"- {result['display_name']}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"weighted_f1={result['weighted_f1']:.4f}"
        )

    if args.compare_random_split:
        print("\nRow-level random split results:")
        for result in output["row_level_random_split_metrics"]:
            print(
                f"- {result['display_name']}: "
                f"accuracy={result['test_accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}, "
                f"weighted_f1={result['weighted_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
