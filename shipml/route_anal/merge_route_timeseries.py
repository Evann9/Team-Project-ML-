from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_NAME = "route_timeseries_merged.csv"
REQUIRED_COLUMNS = {"MMSI", "Timestamp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge route analysis CSV files and sort by vessel/time."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that contains the CSV files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to route_timeseries_merged.csv in the input directory.",
    )
    return parser.parse_args()


def resolve_output_path(input_dir: Path, output: Path | None) -> Path:
    if output is None:
        return input_dir / DEFAULT_OUTPUT_NAME
    return output if output.is_absolute() else input_dir / output


def collect_csv_files(input_dir: Path, output_path: Path) -> list[Path]:
    files = []
    output_path = output_path.resolve()

    for path in sorted(input_dir.glob("*.csv")):
        if path.resolve() == output_path:
            continue
        files.append(path)

    return files


def load_csv(path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required columns: {missing_text}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce").astype("Int64")

    invalid_mask = df["Timestamp"].isna() | df["MMSI"].isna()
    dropped_rows = int(invalid_mask.sum())

    if dropped_rows:
        df = df.loc[~invalid_mask].copy()

    return df, dropped_rows


def merge_route_csvs(input_dir: Path, output_path: Path) -> Path:
    csv_files = collect_csv_files(input_dir, output_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in {input_dir}")

    frames = []
    dropped_rows = 0

    for csv_file in csv_files:
        df, dropped = load_csv(csv_file)
        frames.append(df)
        dropped_rows += dropped
        print(f"Loaded {csv_file.name}: {len(df):,} rows")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(drop=True)
    merged["Timestamp"] = merged["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nMerged files: {len(csv_files)}")
    print(f"Total rows: {len(merged):,}")
    print(f"Unique vessels (MMSI): {merged['MMSI'].nunique():,}")
    print(f"Dropped invalid rows: {dropped_rows:,}")
    print(f"Saved merged CSV: {output_path}")

    return output_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = resolve_output_path(input_dir, args.output).resolve()
    merge_route_csvs(input_dir, output_path)


if __name__ == "__main__":
    main()
