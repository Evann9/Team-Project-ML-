import pandas as pd
import numpy as np


# =========================
# 메모리 최적화 함수
# =========================
def reduce_memory_usage(df, exclude_cols=None):
    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


# =========================
# AIS 전처리 함수
# =========================
def preprocess_ais(df, resample_rule="1h"):
    df = df.copy()

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 컬럼명 통일
    rename_map = {
        "# Timestamp": "Timestamp",
        "timestamp": "Timestamp",
        "TimeStamp": "Timestamp",

        "mmsi": "MMSI",

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
        "draft": "Draught",

        "Ship type": "shiptype",
        "ship_type": "shiptype",
        "Ship Type": "shiptype",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 필요한 컬럼만 유지
    keep_cols = [
        "Timestamp",
        "MMSI",
        "Latitude",
        "Longitude",
        "SOG",
        "COG",
        "Width",
        "Length",
        "Draught",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols]

    # 필수 컬럼 확인
    required_cols = ["Timestamp", "MMSI", "Latitude", "Longitude"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    # Timestamp 변환
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # MMSI는 일단 숫자로 검증
    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")

    # 수치형 컬럼 변환
    num_cols = [
        "Latitude",
        "Longitude",
        "SOG",
        "COG",
        "Width",
        "Length",
        "Draught",
    ]

    for col in [c for c in num_cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 이상치 처리
    # =========================

    # MMSI는 9자리 기준
    df.loc[~df["MMSI"].between(100000000, 999999998), "MMSI"] = np.nan

    # 위도 / 경도
    df.loc[~df["Latitude"].between(-90, 90), "Latitude"] = np.nan
    df.loc[~df["Longitude"].between(-180, 180), "Longitude"] = np.nan

    # SOG: 속도
    if "SOG" in df.columns:
        df.loc[(df["SOG"] < 0) | (df["SOG"] > 70), "SOG"] = np.nan

    # COG: 진행 방향
    if "COG" in df.columns:
        df.loc[~df["COG"].between(0, 360), "COG"] = np.nan

    # Width, Length, Draught 음수 제거
    for col in ["Width", "Length", "Draught"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # =========================
    # 필수 결측치 제거
    # =========================
    essential_cols = [
        col for col in ["Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG"]
        if col in df.columns
    ]

    df = df.dropna(subset=essential_cols)

    # MMSI는 식별자라서 문자열로 고정
    df["MMSI"] = df["MMSI"].astype("int64").astype("string")

    # =========================
    # Width, Length, Draught 결측치 대체
    # =========================
    for col in ["Width", "Length", "Draught"]:
        if col in df.columns:
            group_mean = df.groupby("MMSI")[col].transform("mean")
            df[col] = df[col].fillna(group_mean)
            df[col] = df[col].fillna(df[col].mean())

    # =========================
    # 중복 제거
    # =========================
    df = df.drop_duplicates(
        subset=["Timestamp", "MMSI", "Latitude", "Longitude"]
    )

    # =========================
    # 선박별 시간순 정렬
    # =========================
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    # =========================
    # 1시간 단위 리샘플링
    # =========================
    if resample_rule is not None:
        df = df.set_index("Timestamp")

        df = (
            df.groupby("MMSI")
            .resample(resample_rule)
            .mean(numeric_only=True)
            .dropna(subset=["Latitude", "Longitude"])
            .reset_index()
        )

        # 리샘플링 후 MMSI 다시 문자열 고정
        df["MMSI"] = df["MMSI"].astype("string")

    # =========================
    # 메모리 최적화
    # MMSI는 절대 float32로 줄이면 안 됨
    # =========================
    df = reduce_memory_usage(df, exclude_cols=["MMSI"])

    return df


# =========================
# 실행 코드
# =========================

input_path = "aisdk-2026-01-10.csv"
output_path = "26-01-10.csv"

df = pd.read_csv(input_path)

# 원본 컬럼명 정리
df.columns = df.columns.str.strip()

# Timestamp 컬럼명 사전 통일
if "# Timestamp" in df.columns:
    df = df.rename(columns={"# Timestamp": "Timestamp"})
elif "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "Timestamp"})

# 전처리 전 MMSI 확인용
before_mmsi = set(
    pd.to_numeric(df["MMSI"], errors="coerce")
    .dropna()
    .astype("int64")
    .astype(str)
) if "MMSI" in df.columns else set()

print("원본 데이터 크기:", df.shape)
print("원본 MMSI 개수:", len(before_mmsi))
print(df["MMSI"].astype(str).str.len().value_counts())

# 전처리 실행
clean_df = preprocess_ais(df, resample_rule="1h")

# 전처리 후 MMSI 확인
after_mmsi = set(clean_df["MMSI"].astype(str))
new_mmsi = after_mmsi - before_mmsi

print("전처리 데이터 크기:", clean_df.shape)
print("전처리 후 MMSI 개수:", len(after_mmsi))
print("전처리 후 새로 생긴 MMSI 개수:", len(new_mmsi))
print("새로 생긴 MMSI 예시:", list(new_mmsi)[:10])

# 저장
clean_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("파일 저장 완료")
print("전처리 파일:", output_path)
print(clean_df.head())