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


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "ais_data_with_trig.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "best_ship_type_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "best_ship_type_model_metrics.json"
TARGET = "shiptype"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    estimator: Pipeline
    requires_label_encoding: bool = False


def parse_args() -> argparse.Namespace:
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
    categorical_cols = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    return categorical_cols, numeric_cols


def make_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
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
    specs: list[ModelSpec] = []
    skipped: dict[str, str] = {}

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

    # Recreate the chosen estimator so the saved model is cleanly fitted on all data.
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
    return {
        "model_name": bundle["model_name"],
        "display_name": bundle["display_name"],
        "trained_at": bundle["trained_at"],
        "train_rows": bundle["train_rows"],
        "target_classes": bundle["target_classes"],
        "feature_columns": bundle["feature_columns"],
        "best_metrics": {
            key: value
            for key, value in bundle["best_metrics"].items()
            if key != "classification_report"
        },
        "all_metrics": [
            {key: value for key, value in item.items() if key != "classification_report"}
            for item in bundle["all_metrics"]
        ],
        "skipped_models": bundle["skipped_models"],
    }


def load_or_train_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    force_train: bool = False,
    requested_models: list[str] | None = None,
) -> dict[str, Any]:
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
        probabilities = np.asarray(proba).max(axis=1)

    return np.asarray(pred, dtype=object), probabilities


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
