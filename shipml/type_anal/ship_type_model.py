"""AIS 선박 종류 분류기 학습과 추론에 쓰이는 공통 유틸리티.

이 모듈은 선박 종류 모델을 만드는 중심 팩토리 역할을 한다. 특징 CSV를
읽고, 전처리 파이프라인을 만들고, 후보 분류기를 정의한 뒤, 홀드아웃
분할에서 가장 좋은 후보를 고른다. 이후 배포용으로 선택 모델을 전체
데이터에 다시 학습시키고, 다른 스크립트가 재사용할 수 있는 joblib 번들로
저장한다. 이 파일의 기존 행 단위 분할 방식은 호환성을 위해 유지되어
있으며, 그룹 분할 스크립트들은 같은 헬퍼를 가져다 쓰면서 MMSI 기준
평가로 선박 누수를 줄인다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "ais_ship_type_features.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "ship_type_classifier_row_split_legacy.joblib"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "ship_type_classifier_row_split_legacy_metrics.json"
TARGET = "shiptype"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    """하나의 후보 분류기를 설명하는 선언형 설정 객체.

    코드에서는 단순 estimator 객체만 넘기지 않고 ModelSpec을 사용한다.
    이렇게 하면 각 후보가 사용자에게 보여줄 이름, 학습할 sklearn Pipeline,
    그리고 학습 전에 타깃 라벨을 정수 ID로 바꿔야 하는지 여부를 함께
    가진다. XGBoost는 내부적으로 숫자 클래스 ID를 사용하지만, 일반
    sklearn 분류기들은 원래의 문자열 선박 종류 라벨을 바로 사용할 수 있다.
    """

    name: str
    display_name: str
    estimator: Pipeline
    requires_label_encoding: bool = False


def parse_args() -> argparse.Namespace:
    """기존 행 단위 분할 모델 생성기의 CLI 옵션을 파싱한다.

    인자는 입력 CSV, 출력 산출물 위치, 후보 모델 이름, 홀드아웃 비율을
    지정한다. 데이터셋 컬럼이나 타깃 라벨에 의존하는 검증은 실제 데이터를
    읽은 뒤에 수행하므로, 여기서는 단순한 인자 파싱만 담당한다.
    """

    parser = argparse.ArgumentParser(
        description="Train and save the best practical AIS ship-type classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="AIS type-analysis CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved joblib bundle path.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Saved metrics JSON path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "voting", "xgboost"],
        help=(
            "Models to compare. Supported: logistic_regression random_forest "
            "voting xgboost knn svc. SVC/KNN can be slow on this dataset."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout ratio used for model comparison.",
    )
    return parser.parse_args()


def load_type_data(path: Path) -> pd.DataFrame:
    """AIS 선박 종류 특징 테이블을 읽고 정규화한다.

    타깃 컬럼은 반드시 있어야 하며, 앞뒤 공백을 제거한 문자열 라벨로
    유지한다. 타깃과 명시적 범주형 컬럼을 제외한 모든 컬럼은 숫자로
    변환하여, 이후 imputer가 누락값이나 잘못된 값을 일관되게 처리할 수
    있게 한다. 선박 종류가 없는 예시는 지도학습에 사용할 수 없으므로
    제거한다.
    """

    if not path.exists():
        raise FileNotFoundError(f"Ship-type data not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    if TARGET not in df.columns:
        raise ValueError(f"{path.name} is missing target column: {TARGET}")

    for col in df.columns:
        if col != TARGET and col != "navigationalstatus":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df.dropna(subset=[TARGET])


def split_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """특징 컬럼을 범주형 그룹과 숫자형 그룹으로 나눈다.

    sklearn의 ColumnTransformer는 명시적인 컬럼 목록이 필요하다. object 타입
    컬럼은 범주형으로 보고 결측값 대체와 원-핫 인코딩을 거친다. 그 외
    컬럼은 숫자형으로 보고 중앙값 대체를 수행하며, 모델 계열에 따라
    선택적으로 스케일링을 적용한다.
    """

    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    return categorical_cols, numeric_cols


def make_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """모든 분류기 파이프라인이 공유하는 전처리 단계를 만든다.

    AIS 데이터에는 선박 치수나 운항값이 비어 있는 경우가 많으므로 숫자형
    컬럼은 중앙값으로 결측값을 대체한다. 선형, 거리 기반, 커널 모델은
    크기가 다른 특징이 최적화를 지배하지 않도록 스케일링을 요구한다.
    트리 기반 모델은 단조 스케일 변환에 분할 기준이 크게 영향을 받지
    않으므로 불필요한 스케일링을 생략한다.

    범주형 컬럼은 최빈값 대체와 원-핫 인코딩을 사용한다.
    ``handle_unknown="ignore"``를 지정해, 추론 시 학습 때 없던 운항 상태
    범주가 들어와도 예측이 계속 진행되도록 한다.
    """

    if scale_numeric:
        numeric_transformer: Any = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = SimpleImputer(strategy="median")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def model_specs(
    categorical_cols: list[str],
    numeric_cols: list[str],
    requested: set[str],
) -> tuple[list[ModelSpec], dict[str, str]]:
    """호출자가 요청한 후보 모델 파이프라인들을 만든다.

    반환되는 각 ModelSpec은 전처리 단계와 분류기 단계를 포함한 완전한
    sklearn Pipeline을 가진다. 선택 의존성이 없거나 알 수 없는 모델 이름이
    들어오면 전체 실행을 실패시키지 않고 ``skipped``에 이유를 기록한다.
    덕분에 현재 환경에서 사용할 수 있는 후보만 비교할 수 있다.

    후보들은 의도적으로 같은 전처리 헬퍼를 공유한다. 이렇게 해야 결측값
    처리나 범주형 인코딩 차이가 아니라 estimator 자체의 성능 차이를 비교할
    수 있다.
    """

    specs: list[ModelSpec] = []
    skipped: dict[str, str] = {}

    # 선형 기준 모델: 빠르고 해석하기 쉬우며 숫자형 특징 스케일링이 필요하다.
    if "logistic_regression" in requested:
        specs.append(
            ModelSpec(
                name="logistic_regression",
                display_name="LogisticRegression",
                estimator=Pipeline(
                    steps=[
                        ("preprocessor", make_preprocessor(categorical_cols, numeric_cols, True)),
                        (
                            "classifier",
                            LogisticRegression(max_iter=2000, class_weight="balanced"),
                        ),
                    ]
                ),
            )
        )

    # 표 형태 AIS 특징에 강한 실용 기본값이며 숫자형 스케일링이 필요 없다.
    if "random_forest" in requested:
        specs.append(
            ModelSpec(
                name="random_forest",
                display_name="RandomForest",
                estimator=Pipeline(
                    steps=[
                        ("preprocessor", make_preprocessor(categorical_cols, numeric_cols, False)),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=200,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                ),
            )
        )

    # 소프트 보팅 앙상블은 선형 모델과 트리 모델을 섞어 더 안정적인 후보를 만든다.
    if "voting" in requested:
        specs.append(
            ModelSpec(
                name="voting",
                display_name="VotingClassifier",
                estimator=Pipeline(
                    steps=[
                        ("preprocessor", make_preprocessor(categorical_cols, numeric_cols, True)),
                        (
                            "classifier",
                            VotingClassifier(
                                estimators=[
                                    (
                                        "lr",
                                        LogisticRegression(
                                            max_iter=2000,
                                            class_weight="balanced",
                                        ),
                                    ),
                                    (
                                        "rf",
                                        RandomForestClassifier(
                                            n_estimators=200,
                                            random_state=RANDOM_STATE,
                                            n_jobs=-1,
                                            class_weight="balanced",
                                        ),
                                    ),
                                    (
                                        "et",
                                        ExtraTreesClassifier(
                                            n_estimators=200,
                                            random_state=RANDOM_STATE,
                                            n_jobs=-1,
                                            class_weight="balanced",
                                        ),
                                    ),
                                ],
                                voting="soft",
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            )
        )

    # 선택적 그래디언트 부스팅 트리 모델이며 xgboost가 없으면 조용히 건너뛴다.
    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            skipped["xgboost"] = "xgboost is not installed in this Python environment."
        else:
            specs.append(
                ModelSpec(
                    name="xgboost",
                    display_name="XGBoost",
                    estimator=Pipeline(
                        steps=[
                            (
                                "preprocessor",
                                make_preprocessor(categorical_cols, numeric_cols, False),
                            ),
                            (
                                "classifier",
                                XGBClassifier(
                                    n_estimators=300,
                                    max_depth=8,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    objective="multi:softprob",
                                    eval_metric="mlogloss",
                                    tree_method="hist",
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                    requires_label_encoding=True,
                )
            )

    # 거리 기반 기준 모델: KNN은 특징 거리 계산을 사용하므로 스케일링이 필수다.
    if "knn" in requested:
        from sklearn.neighbors import KNeighborsClassifier

        specs.append(
            ModelSpec(
                name="knn",
                display_name="KNeighborsClassifier",
                estimator=Pipeline(
                    steps=[
                        ("preprocessor", make_preprocessor(categorical_cols, numeric_cols, True)),
                        (
                            "classifier",
                            KNeighborsClassifier(
                                n_neighbors=7,
                                weights="distance",
                                metric="minkowski",
                                p=2,
                            ),
                        ),
                    ]
                ),
            )
        )

    # 커널 SVM은 정확할 수 있지만 이 데이터셋에서는 비용이 커서 선택 실행으로 둔다.
    if "svc" in requested:
        from sklearn.svm import SVC

        specs.append(
            ModelSpec(
                name="svc",
                display_name="SVC",
                estimator=Pipeline(
                    steps=[
                        ("preprocessor", make_preprocessor(categorical_cols, numeric_cols, True)),
                        (
                            "classifier",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                class_weight="balanced",
                                probability=True,
                            ),
                        ),
                    ]
                ),
            )
        )

    unknown = requested.difference(
        {"logistic_regression", "random_forest", "voting", "xgboost", "knn", "svc"}
    )
    for name in sorted(unknown):
        skipped[name] = "unknown model name"

    return specs, skipped


def fit_spec(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, LabelEncoder | None, dict[str, Any]]:
    """후보 모델 하나를 학습하고 홀드아웃 분할에서 평가한다.

    이 함수는 estimator별 차이 하나를 숨겨준다. XGBoost는 정수 클래스 ID를
    기대하지만, 대부분의 sklearn 분류기는 문자열 라벨을 그대로 받는다.
    라벨 인코딩을 사용하는 경우에는 지표 계산 전에 예측값을 다시 선박 종류
    문자열로 복원하여 모든 후보를 같은 라벨 공간에서 비교한다.

    반환 지표에는 집계 점수와 전체 classification report가 포함된다. 학습된
    estimator도 튜플에 포함되지만, 배포 경로에서는 선택 모델을 새로 만들고
    전체 행에 다시 학습시키므로 저장 산출물에는 홀드아웃 학습 상태가
    남지 않는다.
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
    metrics = {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "classification_report": report,
    }
    return spec.estimator, label_encoder, metrics


def refit_full(
    spec: ModelSpec,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[Pipeline, LabelEncoder | None]:
    """선택된 후보를 사용 가능한 모든 라벨 행에 다시 학습한다.

    평가는 홀드아웃 분할로 수행하지만, 모델 계열이 선택된 뒤의 배포 모델은
    전체 데이터셋에서 학습하는 편이 좋다. 이 함수는 ``fit_spec``과 같은
    선택적 라벨 인코딩 로직을 반복하고, 학습된 Pipeline과 추론 시 숫자
    예측값을 원래 선박 종류 라벨로 되돌리는 데 필요한 encoder를 함께
    반환한다.
    """

    label_encoder: LabelEncoder | None = None
    fit_y: pd.Series | np.ndarray = y
    if spec.requires_label_encoding:
        label_encoder = LabelEncoder()
        fit_y = label_encoder.fit_transform(y)
    spec.estimator.fit(x, fit_y)
    return spec.estimator, label_encoder


def train_best_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    requested_models: list[str] | None = None,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """후보 분류기를 학습하고, 최적 모델을 고른 뒤 번들로 저장한다.

    기존 워크플로는 stratified 행 단위 분할로 후보 모델을 정확도와 macro F1
    기준으로 비교한다. 가장 좋은 ModelSpec을 선택한 뒤에는 해당 estimator를
    새로 만들어 전체 특징 테이블에 학습시키고, 이후 예측 코드에 필요한
    메타데이터와 함께 저장한다.

    반환 및 저장되는 번들에는 estimator, 선택적 라벨 encoder, 특징 스키마,
    타깃 클래스, 모든 후보 지표, 건너뛴 모델의 이유, 타임스탬프가 포함된다.
    다운스트림 코드는 원시 Pipeline 대신 이 번들을 사용해야 학습 때의 특징
    순서와 라벨 복원 동작을 정확히 재현할 수 있다.
    """

    df = load_type_data(data_path)
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    categorical_cols, numeric_cols = split_columns(x)
    requested = set(requested_models or ["logistic_regression", "random_forest", "voting", "xgboost"])
    specs, skipped = model_specs(categorical_cols, numeric_cols, requested)

    if not specs:
        raise RuntimeError("No ship-type model candidates are available.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: list[dict[str, Any]] = []
    best_spec: ModelSpec | None = None
    best_sort_key: tuple[float, float] | None = None

    for spec in specs:
        _, _, metrics = fit_spec(spec, x_train, x_test, y_train, y_test)
        results.append(metrics)
        sort_key = (metrics["test_accuracy"], metrics["macro_f1"])
        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_spec = spec

    if best_spec is None:
        raise RuntimeError("No ship-type model could be fitted.")

    # 저장 모델이 전체 데이터에 깔끔하게 학습되도록 선택 estimator를 새로 만든다.
    fresh_specs, _ = model_specs(categorical_cols, numeric_cols, {best_spec.name})
    final_estimator, final_label_encoder = refit_full(fresh_specs[0], x, y)
    best_metrics = next(item for item in results if item["model_name"] == best_spec.name)

    bundle = {
        "target": TARGET,
        "model_name": best_spec.name,
        "display_name": best_spec.display_name,
        "estimator": final_estimator,
        "label_encoder": final_label_encoder,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(df)),
        "target_classes": sorted(y.unique().tolist()),
        "best_metrics": best_metrics,
        "all_metrics": sorted(results, key=lambda item: item["test_accuracy"], reverse=True),
        "skipped_models": skipped,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    save_json(metrics_path, metrics_summary(bundle))
    return bundle


def save_json(path: Path, data: dict[str, Any]) -> None:
    """numpy 값을 표준 Python 객체로 바꿔 JSON을 저장한다.

    지표 딕셔너리에는 pandas/sklearn이 만든 numpy scalar나 배열이 들어가는
    경우가 많다. ``json.dumps``는 이를 직접 직렬화할 수 없으므로, 이 헬퍼가
    흔한 numpy 컨테이너를 변환한다. 출력 파일의 한글 등 비 ASCII 라벨은
    읽을 수 있는 형태로 유지한다.
    """

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def metrics_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """전체 모델 번들에서 간결한 지표 JSON을 만든다.

    sklearn classification report 전체는 크기가 클 수 있고 이미 joblib 번들에
    포함되어 있다. 이 요약은 긴 report 블록을 제거하되, 대시보드나 빠른
    확인에 필요한 모델 식별 정보, 특징 스키마, 후보 점수, 건너뛴 모델,
    선택적 평가 메타데이터는 유지한다.
    """

    best_metrics = {
        key: value
        for key, value in bundle["best_metrics"].items()
        if key != "classification_report"
    }
    summary = {
        "model_name": bundle["model_name"],
        "display_name": bundle["display_name"],
        "trained_at": bundle["trained_at"],
        "train_rows": bundle["train_rows"],
        "target_classes": bundle["target_classes"],
        "feature_columns": bundle["feature_columns"],
        "best_metrics": best_metrics,
        "all_metrics": [
            {key: value for key, value in item.items() if key != "classification_report"}
            for item in bundle["all_metrics"]
        ],
        "skipped_models": bundle["skipped_models"],
    }
    if "evaluation" in bundle:
        summary["evaluation"] = bundle["evaluation"]
    if "probability_calibration" in bundle:
        calibration = bundle["probability_calibration"]
        summary["probability_calibration"] = {
            key: value
            for key, value in calibration.items()
            if key != "calibrators"
        }
    return summary


def load_or_train_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    force_train: bool = False,
    requested_models: list[str] | None = None,
) -> dict[str, Any]:
    """기존 모델 번들을 읽거나, 요청 시 새 모델을 학습한다.

    예측이 필요하지만 joblib 산출물이 이미 있는지 신경 쓰고 싶지 않은
    스크립트를 위한 편의 진입점이다. ``force_train=True``를 넘기면 캐시를
    무시하고 현재 CSV와 요청 후보 목록으로 번들을 다시 만든다.
    """

    if model_path.exists() and not force_train:
        return joblib.load(model_path)
    return train_best_model(
        data_path=data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        requested_models=requested_models,
    )


def predict_ship_types(
    bundle: dict[str, Any],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """선박 종류 라벨과 신뢰도처럼 사용할 확률값을 예측한다.

    항로에서 파생된 입력 특징에는 학습 때 존재하던 컬럼이 빠져 있을 수 있다.
    이 함수는 없는 컬럼을 NaN으로 추가하고, 저장된 특징 스키마와 같은 순서로
    DataFrame을 재정렬한 뒤, Pipeline의 imputer가 결측값을 채우게 한다.
    모델이 라벨 encoder를 사용했다면 숫자 예측값을 원래 라벨로 복원한다.
    확률 출력은 가능할 때 최대 클래스 확률이며, ``predict_proba``가 없는
    estimator에서는 NaN을 반환한다.
    """

    x = features.copy()
    for col in bundle["feature_columns"]:
        if col not in x.columns:
            x[col] = np.nan
    x = x[bundle["feature_columns"]]

    estimator: Pipeline = bundle["estimator"]
    pred = estimator.predict(x)
    label_encoder: LabelEncoder | None = bundle.get("label_encoder")
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred.astype(int))

    probabilities = np.full(len(x), np.nan)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        proba_classes = probability_class_order(bundle, estimator)
        proba, proba_classes = apply_probability_calibration(
            bundle,
            np.asarray(proba),
            proba_classes,
        )
        probabilities = predicted_class_confidence(pred, proba, proba_classes)

    return np.asarray(pred, dtype=object), probabilities


def probability_class_order(bundle: dict[str, Any], estimator: Pipeline) -> list[str]:
    """predict_proba 컬럼 순서에 대응하는 클래스 라벨을 찾는다."""

    label_encoder: LabelEncoder | None = bundle.get("label_encoder")
    if label_encoder is not None:
        return [str(value) for value in label_encoder.classes_]
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        return [str(value) for value in classes]
    return [str(value) for value in bundle.get("target_classes", [])]


def apply_probability_calibration(
    bundle: dict[str, Any],
    proba: np.ndarray,
    proba_classes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """저장된 one-vs-rest calibrator로 클래스 확률을 보정하고 정규화한다."""

    calibration = bundle.get("probability_calibration")
    if not calibration or "calibrators" not in calibration:
        return proba, proba_classes

    target_classes = [str(value) for value in calibration.get("classes", proba_classes)]
    raw_lookup = {label: idx for idx, label in enumerate(proba_classes)}
    aligned = np.zeros((len(proba), len(target_classes)), dtype=float)
    for idx, label in enumerate(target_classes):
        raw_idx = raw_lookup.get(label)
        if raw_idx is not None and raw_idx < proba.shape[1]:
            aligned[:, idx] = proba[:, raw_idx]

    calibrated = np.zeros_like(aligned)
    for idx, calibrator in enumerate(calibration["calibrators"]):
        if calibrator is None:
            calibrated[:, idx] = aligned[:, idx]
        else:
            calibrated[:, idx] = calibrator.predict(aligned[:, idx])

    calibrated = np.clip(calibrated, 0.0, 1.0)
    row_sums = calibrated.sum(axis=1)
    valid = row_sums > 0
    calibrated[valid] = calibrated[valid] / row_sums[valid, None]
    calibrated[~valid] = aligned[~valid]
    return calibrated, target_classes


def predicted_class_confidence(
    pred: np.ndarray,
    proba: np.ndarray,
    proba_classes: list[str],
) -> np.ndarray:
    """각 예측 라벨에 대응하는 확률 컬럼 값을 confidence로 반환한다."""

    lookup = {label: idx for idx, label in enumerate(proba_classes)}
    confidence = np.full(len(pred), np.nan)
    for idx, label in enumerate(np.asarray(pred, dtype=str)):
        class_idx = lookup.get(label)
        if class_idx is None or class_idx >= proba.shape[1]:
            confidence[idx] = float(np.nanmax(proba[idx]))
        else:
            confidence[idx] = float(proba[idx, class_idx])
    return confidence


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """한 좌표쌍에서 다른 좌표쌍으로 향하는 초기 방위각을 반환한다.

    항로 요약에는 원본 AIS COG 값이 항상 있지 않으므로, 이 헬퍼는 시작/끝
    좌표에서 heading처럼 사용할 각도를 만든다. 좌표가 없으면 0도를 반환해
    특징 생성이 중단되지 않게 하고, 이후 imputer/model이 중립적인 fallback을
    처리하도록 둔다.
    """

    if any(pd.isna(value) for value in [lat1, lon1, lat2, lon2]):
        return 0.0
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    dlon = math.radians(float(lon2) - float(lon1))
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def route_rows_to_type_features(routes: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """항로 예측 행을 선박 종류 분류기 특징으로 변환한다.

    선박 종류 모델은 AIS 점 단위와 비슷한 특징으로 학습되지만, 항로 예측
    출력은 선박 단위 요약이다. 이 adapter는 항로 요약에서 필요한 컬럼을
    합성한다. ``mean_sog``에서 속도를, 항로 방위각에서 heading/course를,
    항로 메타데이터에서 선박 치수를, 파생 방위각에서 삼각함수 각도 특징을
    만든다. 저장 번들이 기대하지만 항로 데이터에 없는 특징은 NaN으로
    추가해, 학습 때와 같은 imputation 규칙이 결측값을 처리하도록 한다.
    """

    bearings = routes.apply(
        lambda row: bearing_degrees(
            row.get("start_lat"),
            row.get("start_lon"),
            row.get("end_lat"),
            row.get("end_lon"),
        ),
        axis=1,
    )
    bearing_rad = np.radians(pd.to_numeric(bearings, errors="coerce").fillna(0.0))

    base = pd.DataFrame(index=routes.index)
    base["navigationalstatus"] = "Under way using engine"
    base["sog"] = pd.to_numeric(routes.get("mean_sog"), errors="coerce")
    base["cog"] = bearings
    base["heading"] = bearings
    base["width"] = pd.to_numeric(routes.get("width"), errors="coerce")
    base["length"] = pd.to_numeric(routes.get("length"), errors="coerce")
    base["draught"] = pd.to_numeric(routes.get("draught"), errors="coerce")
    base["cog_sin"] = np.sin(bearing_rad)
    base["cog_cos"] = np.cos(bearing_rad)
    base["heading_sin"] = np.sin(bearing_rad)
    base["heading_cos"] = np.cos(bearing_rad)

    for col in feature_columns:
        if col not in base.columns:
            base[col] = np.nan
    return base[feature_columns]


def main() -> None:
    """기존 선박 종류 모델을 학습하고 저장하는 CLI 진입점."""

    args = parse_args()
    bundle = train_best_model(
        data_path=args.data.resolve(),
        model_path=args.model_out.resolve(),
        metrics_path=args.metrics_out.resolve(),
        requested_models=args.models,
        test_size=args.test_size,
    )
    best = bundle["best_metrics"]
    print(f"Best ship-type model: {bundle['display_name']}")
    print(f"Test accuracy: {best['test_accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Model saved: {args.model_out.resolve()}")
    print(f"Metrics saved: {args.metrics_out.resolve()}")


if __name__ == "__main__":
    main()
