from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ais_data.csv"
Towing_RAW_PATH = BASE_DIR / "ais_Towing_only.csv"
Towing_CLEAN_PATH = BASE_DIR / "ais_Towing_cleaned.csv"

# 0이면 안 되는 선박 제원만 정제 대상으로 둡니다.
# heading/cog는 0도 유효한 방향 값일 수 있어 제외했습니다.
ZERO_AS_MISSING_COLS = ["width", "length", "draught"]
NUMERIC_ANALYSIS_COLS = ["sog", "cog", "heading", "width", "length", "draught"]
ANGLE_RANGE_COLS = {"cog": (0, 360), "heading": (0, 360)}
NONNEGATIVE_COLS = ["sog"]


def print_divider() -> None:
    print("-" * 40)


df = pd.read_csv(DATA_PATH)

# 저장 과정에서 생긴 인덱스 컬럼이 있으면 제거합니다.
unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

if "shiptype" not in df.columns:
    raise ValueError("'shiptype' 컬럼이 없어 Towing 데이터를 분류할 수 없습니다.")

initial_count = len(df)
df_Towing = df[df["shiptype"].astype(str).str.strip() == "Towing"].copy()
Towing_count = len(df_Towing)

if Towing_count == 0:
    raise ValueError("'Towing' 값이 있는 데이터가 없습니다.")

df_Towing.to_csv(Towing_RAW_PATH, index=False)

print_divider()
print("Towing 선박 데이터 분석")
print_divider()
print(f"전체 데이터 개수: {initial_count}행")
print(f"Towing 데이터 개수: {Towing_count}행")
print(f"Towing 비율: {(Towing_count / initial_count) * 100:.2f}%")
print(f"Towing 선박 고유 MMSI 수: {df_Towing['mmsi'].nunique()}개")

missing_ratio = (df_Towing.isna().mean() * 100).round(2)
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

print_divider()
print("[Towing 데이터 결측치 비율(%)]")
if missing_ratio.empty:
    print("결측치가 없습니다.")
else:
    print(missing_ratio.to_string())

df_cleaned = df_Towing.copy()
for col in ZERO_AS_MISSING_COLS:
    if col in df_cleaned.columns:
        df_cleaned.loc[df_cleaned[col] <= 0, col] = pd.NA

required_cols = [col for col in ZERO_AS_MISSING_COLS if col in df_cleaned.columns]
removal_summary = {}

before_dropna = len(df_cleaned)
df_cleaned = df_cleaned.dropna(subset=required_cols)
removal_summary["width/length/draught 결측 또는 0 이하"] = before_dropna - len(df_cleaned)

for col, (lower, upper) in ANGLE_RANGE_COLS.items():
    if col in df_cleaned.columns:
        before_filter = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned[col].isna() | df_cleaned[col].between(lower, upper)]
        removal_summary[f"{col} 범위 이탈"] = before_filter - len(df_cleaned)

for col in NONNEGATIVE_COLS:
    if col in df_cleaned.columns:
        before_filter = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned[col].isna() | (df_cleaned[col] >= 0)]
        removal_summary[f"{col} 음수값"] = before_filter - len(df_cleaned)

df_cleaned.to_csv(Towing_CLEAN_PATH, index=False)

cleaned_count = len(df_cleaned)
dropped_count = Towing_count - cleaned_count

print_divider()
print("[정제 결과]")
print(f"정제 전 데이터 개수: {Towing_count}행")
print(f"정제 후 데이터 개수: {cleaned_count}행")
print(f"삭제된 데이터 개수: {dropped_count}행")
print(f"남은 데이터 비율: {(cleaned_count / Towing_count) * 100:.2f}%")
print("[제거 기준별 삭제 건수]")
for reason, count in removal_summary.items():
    print(f"{reason}: {count}행")

if "navigationalstatus" in df_cleaned.columns:
    print_divider()
    print("[항해 상태 분포]")
    print(df_cleaned["navigationalstatus"].value_counts().to_string())

analysis_cols = [col for col in NUMERIC_ANALYSIS_COLS if col in df_cleaned.columns]
if analysis_cols:
    print_divider()
    print("[수치형 변수 기초 통계]")
    print(df_cleaned[analysis_cols].describe().round(2).to_string())

print_divider()
print(f"분류된 Towing 원본 데이터 저장: {Towing_RAW_PATH.name}")
print(f"정제된 Towing 데이터 저장: {Towing_CLEAN_PATH.name}")
print_divider()
