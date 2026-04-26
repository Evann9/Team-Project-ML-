"""MMSI 그룹 분할 선박 종류 분류기의 하이퍼파라미터 튜닝.

이 스크립트는 같은 선박 그룹이 검증 fold와 외부 홀드아웃에 동시에 나타나지
않도록 하면서 RandomForest 후보를 튜닝한다. 흐름은 의도적으로 nested
구조다. 먼저 외부 MMSI 그룹 홀드아웃을 만들고, 학습 쪽 데이터에서만
StratifiedGroupKFold로 하이퍼파라미터를 찾는다. 그 다음 튜닝된 모델을
건드리지 않은 홀드아웃에서 평가하고, 마지막으로 선택 설정을 전체 행에
다시 학습시켜 배포한다.
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
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from train_ship_type_classifier_group_split import (  # type: ignore  # noqa: E402
        DEFAULT_DATA_PATH,
        add_trig_features,
        class_counts,
        group_train_test_split,
        leakage_report,
        resolve_column,
    )
    from ship_type_model import (  # type: ignore  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        save_json,
        split_columns,
    )
else:
    from .train_ship_type_classifier_group_split import (  # noqa: E402
        DEFAULT_DATA_PATH,
        add_trig_features,
        class_counts,
        group_train_test_split,
        leakage_report,
        resolve_column,
    )
    from .ship_type_model import (  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        save_json,
        split_columns,
    )


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuning_results.json"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuned_group_split.joblib"
)
DEFAULT_MODEL_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuned_group_split_metrics.json"
)


def parse_args() -> argparse.Namespace:
    """RandomForest 그룹 인식 튜닝을 위한 CLI 옵션을 파싱한다.

    표준 입출력 경로 외에도, 외부 홀드아웃 크기, 내부 교차검증 fold 수,
    랜덤 탐색 예산, 더 빠른 탐색을 위한 선택적 그룹 보존 다운샘플링, 그리고
    탐색 wrapper와 RandomForest estimator 자체의 병렬 처리 설정을 각각
    제어한다.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Tune the MMSI group-split ship-type RandomForest model without "
            "letting the same vessel appear in both train and validation folds."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV containing mmsi and shiptype columns.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="mmsi",
        help="Group column used for leakage-safe splitting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Detailed tuning/evaluation metrics JSON.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved tuned model bundle.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_MODEL_METRICS_PATH,
        help="Compact metrics JSON for the tuned model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="External MMSI group holdout ratio.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Internal StratifiedGroupKFold folds for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="Number of random hyperparameter combinations to test.",
    )
    parser.add_argument(
        "--max-search-groups",
        type=int,
        default=0,
        help=(
            "Optional cap on MMSI groups used only during the search. "
            "0 means use all training groups."
        ),
    )
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs inside RandomForest.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "accuracy"],
        help="Metric used to choose the best hyperparameters.",
    )
    return parser.parse_args()


def parameter_space() -> dict[str, list[Any]]:
    """RandomForest 파이프라인의 하이퍼파라미터 탐색 공간을 반환한다.

    튜닝 대상 estimator가 sklearn Pipeline이므로 파라미터 이름에는
    ``classifier__`` 접두사가 붙는다. 값 목록은 트리 개수, 깊이,
    leaf/split 정규화, 특징 서브샘플링, 클래스 가중치, bootstrap 동작을
    포함한다. 이들은 표 형태 AIS 분류 문제에서 성능과 과적합에 영향을 줄
    가능성이 큰 조정값들이다.
    """

    return {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [None, 12, 20, 32, 48],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", 0.5, None],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
        "classifier__bootstrap": [True, False],
    }


def scoring_map() -> dict[str, Any]:
    """사용자 친화적 점수 이름을 sklearn scoring 식별자로 매핑한다."""

    return {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
    }


def prepare_data(
    data_path: Path,
    group_col_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """특징을 읽고 각도 인코딩을 추가한 뒤 타깃/그룹 컬럼을 분리한다.

    숫자 MMSI 포맷 차이로 우연한 불일치가 생기지 않도록 그룹 컬럼은
    문자열로 변환한다. 분류기에 직접 누수되는 것을 막기 위해 그룹 컬럼과
    타깃은 특징 행렬에서 제거한다.
    """

    df = add_trig_features(load_type_data(data_path))
    group_col = resolve_column(df, group_col_name)
    df[group_col] = df[group_col].astype(str)

    x = df.drop(columns=[TARGET, group_col])
    y = df[TARGET].astype(str)
    groups = df[group_col]
    return x, y, groups, group_col


def sample_search_groups(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    max_groups: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """선박 그룹을 쪼개지 않고 내부 탐색 데이터만 선택적으로 줄인다.

    전체 AIS 테이블에서 하이퍼파라미터 탐색을 하면 비용이 클 수 있다. 그룹
    상한이 주어지면 이 함수는 개별 행이 아니라 MMSI 그룹 전체를 샘플링하므로
    탐색 부분집합에서도 누수 방지 가정이 유지된다. 각 그룹의 최빈 클래스를
    기준으로, 예시 수가 충분할 때는 선택 그룹을 stratify하려고 시도한다.
    """

    if max_groups <= 0 or train_groups.nunique() <= max_groups:
        return x_train, y_train, train_groups

    group_labels = (
        pd.DataFrame({"group": train_groups.astype(str), "target": y_train.astype(str)})
        .groupby("group", sort=False)["target"]
        .agg(lambda values: values.mode().iloc[0])
    )
    stratify = group_labels if group_labels.value_counts().min() >= 2 else None
    selected_groups, _ = train_test_split(
        group_labels.index.to_numpy(),
        train_size=max_groups,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    selected = set(map(str, selected_groups))
    mask = train_groups.astype(str).isin(selected)
    return x_train.loc[mask], y_train.loc[mask], train_groups.loc[mask]


def make_random_forest_pipeline(
    x_train: pd.DataFrame,
    estimator_n_jobs: int,
) -> Any:
    """공유 RandomForest Pipeline을 만들고 estimator 병렬성을 설정한다.

    파이프라인은 ``model_specs``에서 가져오므로 튜닝은 그룹 분할 학습기와
    정확히 같은 전처리 및 기준 estimator 정의를 사용한다. 큰 실행에서 CPU
    코어를 과도하게 잡아먹지 않도록 탐색 수준 병렬성과 트리 생성 병렬성을
    따로 설정하기 위해 여기서 ``estimator_n_jobs``를 주입한다.
    """

    categorical_cols, numeric_cols = split_columns(x_train)
    specs, skipped = model_specs(categorical_cols, numeric_cols, {"random_forest"})
    if skipped:
        raise RuntimeError(f"RandomForest spec could not be built: {skipped}")
    if not specs:
        raise RuntimeError("RandomForest spec was not created.")
    estimator = specs[0].estimator
    estimator.set_params(classifier__n_jobs=estimator_n_jobs)
    return estimator


def top_confusion_pairs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """가장 자주 발생한 실제값-예측값 오분류 쌍을 추출한다."""

    labels = sorted(set(y_true.astype(str)).union(set(map(str, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, Any]] = []
    for row_idx, actual in enumerate(labels):
        for col_idx, predicted in enumerate(labels):
            count = int(cm[row_idx, col_idx])
            if actual != predicted and count > 0:
                pairs.append(
                    {"actual": actual, "predicted": predicted, "count": count}
                )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:top_n]


def evaluate_holdout(
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    """튜닝된 파라미터를 외부 train 행에 학습하고 외부 홀드아웃을 평가한다.

    탐색에 사용한 estimator를 건드리지 않은 템플릿으로 남기기 위해 학습 전에
    clone한다. 지표는 MMSI가 겹치지 않는 홀드아웃에서 계산하며, 전체
    classification report와 빠른 오류 확인을 위한 주요 혼동 쌍을 포함한다.
    """

    model = clone(estimator)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "model_name": "random_forest_group_tuned",
        "display_name": "RandomForest Tuned (MMSI Group Split)",
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
        "top_confusion_pairs": top_confusion_pairs(y_test, pred),
    }
    return model, metrics


def compact_cv_results(search: RandomizedSearchCV, top_n: int = 10) -> list[dict[str, Any]]:
    """긴 RandomizedSearchCV 결과를 작은 순위표로 변환한다.

    전체 ``cv_results_`` 객체에는 시간 정보와 split별 컬럼이 많이 들어 있다.
    이 헬퍼는 상위 설정의 순위, 평균/표준편차 검증 점수, 파라미터만 남긴다.
    덕분에 튜닝 JSON을 읽기 좋게 유지하면서도 선택 파라미터를 감사하는 데
    필요한 정보는 보존할 수 있다.
    """

    results = pd.DataFrame(search.cv_results_)
    score_col = f"mean_test_{search.refit}" if isinstance(search.refit, str) else "mean_test_score"
    if score_col not in results.columns:
        score_col = "mean_test_score"

    cols = [
        "rank_test_" + search.refit if isinstance(search.refit, str) else "rank_test_score",
        score_col,
        "std_test_" + search.refit if isinstance(search.refit, str) else "std_test_score",
        "params",
    ]
    cols = [col for col in cols if col in results.columns]
    top = results.sort_values(score_col, ascending=False).head(top_n)
    return top[cols].to_dict("records")


def train_full_bundle(
    best_estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    holdout_metrics: dict[str, Any],
    search: RandomizedSearchCV,
    split_info: dict[str, Any],
) -> dict[str, Any]:
    """튜닝된 RandomForest를 전체 행에 다시 학습하고 모델 번들을 구성한다.

    탐색과 홀드아웃 평가는 최종 하이퍼파라미터를 결정하지만, 저장 산출물은
    모든 라벨 예시에서 학습해야 한다. 번들은 다른 선박 종류 학습기와 같은
    구조를 따르며, 홀드아웃 지표와 교차검증 탐색 메타데이터를 모두 기록해
    배포 estimator가 어떻게 선택되었는지 확인할 수 있게 한다.
    """

    final_estimator = clone(best_estimator)
    final_estimator.fit(x, y)
    categorical_cols, numeric_cols = split_columns(x)
    best_metrics = {
        **holdout_metrics,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_refit_metric": str(search.refit),
    }
    return {
        "target": TARGET,
        "model_name": "random_forest_group_tuned",
        "display_name": "RandomForest Tuned (MMSI Group Split)",
        "estimator": final_estimator,
        "label_encoder": None,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(x)),
        "target_classes": sorted(y.unique().tolist()),
        "best_metrics": best_metrics,
        "all_metrics": [best_metrics],
        "skipped_models": {},
        "evaluation": {
            "method": "nested_mmsi_group_tuning",
            "split": split_info,
            "note": (
                "Hyperparameters were selected with StratifiedGroupKFold on "
                "the external training split, then evaluated on a separate "
                "MMSI group holdout. The saved estimator is refit on all rows."
            ),
        },
    }


def main() -> None:
    """nested 그룹 인식 튜닝을 실행하고 지표와 모델을 저장한다."""

    args = parse_args()
    x, y, groups, group_col = prepare_data(args.data.resolve(), args.group_col)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
        split_method,
    ) = group_train_test_split(x, y, groups, args.test_size)

    x_search, y_search, search_groups = sample_search_groups(
        x_train,
        y_train,
        train_groups,
        args.max_search_groups,
    )

    estimator = make_random_forest_pipeline(x_search, args.estimator_n_jobs)
    cv = StratifiedGroupKFold(
        n_splits=max(args.cv_folds, 2),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scorers = scoring_map()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space(),
        n_iter=max(args.n_iter, 1),
        scoring=scorers,
        refit=args.scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=args.search_n_jobs,
        verbose=2,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_search, y_search, groups=search_groups)

    tuned_estimator = clone(estimator).set_params(**search.best_params_)
    tuned_estimator.set_params(classifier__n_jobs=args.estimator_n_jobs)
    _, holdout_metrics = evaluate_holdout(
        tuned_estimator,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    split_info = {
        "outer_split_method": split_method,
        "outer_requested_test_size": args.test_size,
        "outer_train_rows": int(len(x_train)),
        "outer_test_rows": int(len(x_test)),
        "outer_train_class_counts": class_counts(y_train),
        "outer_test_class_counts": class_counts(y_test),
        "outer_leakage_check": leakage_report(train_groups, test_groups),
        "inner_cv_method": f"StratifiedGroupKFold(n_splits={max(args.cv_folds, 2)})",
        "inner_search_rows": int(len(x_search)),
        "inner_search_groups": int(search_groups.nunique()),
        "group_col": group_col,
        "group_col_used_as_feature": False,
    }

    bundle = train_full_bundle(
        best_estimator=tuned_estimator,
        x=x,
        y=y,
        holdout_metrics=holdout_metrics,
        search=search,
        split_info=split_info,
    )

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()),
        "target": TARGET,
        "group_col": group_col,
        "feature_columns": x.columns.tolist(),
        "scoring": args.scoring,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_results_top": compact_cv_results(search),
        "holdout_metrics": holdout_metrics,
        "split": split_info,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.resolve(), output)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics_summary(bundle))

    print(f"Saved tuning metrics: {args.output.resolve()}")
    print(f"Saved tuned model: {args.model_out.resolve()}")
    print(f"Saved tuned model metrics: {args.metrics_out.resolve()}")
    print(f"Best CV {args.scoring}: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in search.best_params_.items():
        print(f"- {key}: {value}")
    print(
        "Holdout: "
        f"accuracy={holdout_metrics['test_accuracy']:.4f}, "
        f"macro_f1={holdout_metrics['macro_f1']:.4f}, "
        f"weighted_f1={holdout_metrics['weighted_f1']:.4f}"
    )
    print(
        "Group leakage check: "
        f"overlap={split_info['outer_leakage_check']['overlap_groups']:,} groups"
    )


if __name__ == "__main__":
    main()
