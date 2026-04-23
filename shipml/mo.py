# ais_data_ultra_cleaned.csv의 cog와 heading의 각도 sin/ cos 로 변경하는 코드 작성
import pandas as pd
import numpy as np

# 1. 데이터 로드
df = pd.read_csv("ais_data_ultra_cleaned.csv")

# 2. Heading 결측치를 COG 값으로 대체
df['heading'] = df['heading'].fillna(df['cog'])

# 2. 각도(Degree)를 라디안(Radian)으로 변환 후 Sin/Cos 계산 함수
def transform_angles(df, column):
    # 0~360도 범위를 라디안으로 변환
    radians = np.deg2rad(df[column])
    df[f'{column}_sin'] = np.sin(radians)
    df[f'{column}_cos'] = np.cos(radians)
    return df

# 3. COG 및 Heading 변환 적용
df = transform_angles(df, 'cog')
df = transform_angles(df, 'heading')

# 4. 결과 확인 및 저장 (필요 시)
print(df[['cog', 'cog_sin', 'cog_cos', 'heading', 'heading_sin', 'heading_cos']].head())
df.to_csv("ais_data_with_trig.csv", index=False)
