from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ship_type_model import (  # type: ignore  # noqa: E402
        RANDOM_STATE,
        TARGET,
        fit_spec,
        load_type_data,
        model_specs,
        save_json,
        split_columns,
    )
else:
    from .ship_type_model import (  # noqa: E402
        RANDOM_STATE,
        TARGET,
        fit_spec,
        load_type_data,
        model_specs,
        save_json,
        split_columns,
    )


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "ais_mmsi_incl.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_group_split_metrics.json"
)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, requested: str) -> str:
    lookup = {col.lower(): col for col in df.columns}
    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f"Group column '{requested}' not found. Available columns: {list(df.columns)}"
        )
    return resolved


def add_trig_features(df: pd.DataFrame) -> pd.DataFrame:
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
    categorical_cols, numeric_cols = split_columns(x_train)
    specs, skipped = model_specs(categorical_cols, numeric_cols, set(model_names))
    results: list[dict[str, Any]] = []

    for spec in specs:
        _, _, metrics = fit_spec(spec, x_train, x_test, y_train, y_test)
        metrics["confusion_matrix"] = confusion_matrix_summary(y_test, metrics)
        results.append(metrics)

    results.sort(key=lambda item: (item["test_accuracy"], item["macro_f1"]), reverse=True)
    return results, skipped


def confusion_matrix_summary(y_test: pd.Series, metrics: dict[str, Any]) -> dict[str, Any]:
    labels = sorted(y_test.unique().tolist())
    # Recomputeing predictions is avoided here; the full classification report is already saved.
    return {"labels": labels, "note": "See classification_report for per-class metrics."}


def random_split_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    test_size: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
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
    return {str(label): int(count) for label, count in y.value_counts().sort_index().items()}


def main() -> None:
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

    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()),
        "target": TARGET,
        "group_col": group_col,
        "group_col_used_as_feature": False,
        "feature_columns": x.columns.tolist(),
        "split": {
            "method": split_method,
            "requested_test_size": args.test_size,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_class_counts": class_counts(y_train),
            "test_class_counts": class_counts(y_test),
            "leakage_check": leakage_report(train_groups, test_groups),
        },
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

    print(f"Saved group-split metrics: {args.output.resolve()}")
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
