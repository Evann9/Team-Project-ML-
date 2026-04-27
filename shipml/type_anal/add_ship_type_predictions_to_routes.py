from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ship_type_model import (  # type: ignore  # noqa: E402
        DEFAULT_DATA_PATH,
        DEFAULT_METRICS_PATH,
        DEFAULT_MODEL_PATH,
        load_or_train_model,
        metrics_summary,
        predict_ship_types,
        route_rows_to_type_features,
        save_json,
    )
else:
    from .ship_type_model import (  # noqa: E402
        DEFAULT_DATA_PATH,
        DEFAULT_METRICS_PATH,
        DEFAULT_MODEL_PATH,
        load_or_train_model,
        metrics_summary,
        predict_ship_types,
        route_rows_to_type_features,
        save_json,
    )


SHIPML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_PREDICTIONS = SHIPML_DIR / "route_anal" / "outputs" / "route_predictions.csv"
DEFAULT_OUTPUT = SHIPML_DIR / "route_anal" / "outputs" / "route_predictions_with_types.csv"
DEFAULT_SUMMARY = SHIPML_DIR / "route_anal" / "outputs" / "route_type_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach predicted ship types to route prediction rows."
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=DEFAULT_ROUTE_PREDICTIONS,
        help="route_predictions.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for route predictions enriched with ship types.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="JSON summary path.",
    )
    parser.add_argument(
        "--type-data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="AIS ship-type training CSV.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved ship-type model bundle.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Saved ship-type model metrics JSON.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain model even if a saved model already exists.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "voting", "xgboost"],
        help="Candidate ship-type models used when training is needed.",
    )
    return parser.parse_args()


def build_summary(routes: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    type_counts = (
        routes["predicted_shiptype"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename_axis("shiptype")
        .reset_index(name="vessel_count")
        .to_dict("records")
    )
    route_type_counts = (
        routes.groupby(["predicted_shiptype", "predicted_route"], dropna=False)
        .size()
        .rename("vessel_count")
        .reset_index()
        .sort_values(["predicted_shiptype", "vessel_count"], ascending=[True, False])
        .to_dict("records")
    )
    calibration = bundle.get("probability_calibration", {})
    probability_note = (
        "predicted_shiptype_probability is calibrated with "
        f"{calibration.get('method', 'a saved calibration model')}."
        if calibration
        else (
            "predicted_shiptype_probability is an uncalibrated model confidence "
            "score unless the saved classifier was calibrated separately."
        )
    )
    return {
        "rows": int(len(routes)),
        "model": metrics_summary(bundle),
        "probability_note": probability_note,
        "shiptype_counts": type_counts,
        "route_type_counts": route_type_counts,
    }


def enrich_routes(
    routes_path: Path = DEFAULT_ROUTE_PREDICTIONS,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    type_data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    force_train: bool = False,
    requested_models: list[str] | None = None,
) -> pd.DataFrame:
    if not routes_path.exists():
        raise FileNotFoundError(f"Route prediction CSV not found: {routes_path}")

    bundle = load_or_train_model(
        model_path=model_path,
        data_path=type_data_path,
        metrics_path=metrics_path,
        force_train=force_train,
        requested_models=requested_models,
    )

    routes = pd.read_csv(routes_path)
    features = route_rows_to_type_features(routes, bundle["feature_columns"])
    predicted, probability = predict_ship_types(bundle, features)

    routes["predicted_shiptype"] = predicted
    routes["predicted_shiptype_probability"] = probability
    routes["shiptype_model"] = bundle["model_name"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = build_summary(routes, bundle)
    save_json(summary_path, summary)
    return routes


def main() -> None:
    args = parse_args()
    routes = enrich_routes(
        routes_path=args.routes.resolve(),
        output_path=args.output.resolve(),
        summary_path=args.summary_out.resolve(),
        type_data_path=args.type_data.resolve(),
        model_path=args.model.resolve(),
        metrics_path=args.metrics.resolve(),
        force_train=args.force_train,
        requested_models=args.models,
    )

    counts = routes["predicted_shiptype"].fillna("Unknown").astype(str).value_counts()
    print(f"Enriched routes saved: {args.output.resolve()}")
    print(f"Rows: {len(routes):,}")
    print("Top ship types:")
    for shiptype, count in counts.head(10).items():
        print(f"- {shiptype}: {int(count):,}")

    summary_path = args.summary_out.resolve()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = summary["model"]
        print(
            "Type model: "
            f"{model['display_name']} "
            f"(accuracy={model['best_metrics']['test_accuracy']:.4f})"
        )


if __name__ == "__main__":
    main()
