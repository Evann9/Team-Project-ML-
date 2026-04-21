import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Unnamed: 0 : 그냥 저장 과정에서 붙은 인덱스 컬럼
# mmsi : 선박 고유 식별번호
# navigationalstatus : 항해 상태
# sog(Speed Over Ground) : 속력
# cog(Course Over Ground) : 진행 방향
# heading : 선수 방향
# shiptype : 선박 종류
# width : 선박 폭
# length : 선박 길이
# draught : 흘수 : 선박이 물 위에 떠 있을 때 수면에서 선체 최하부(용골)까지의 수직 거리

# 항해 상태 분포 해석
# Under way using engine: 307,581 
# Unknown value: 17,259
# Constrained by her draught: 12,287
# Engaged in fishing(어업 활동): 10,798
# Moored: 4,121

# 선박 종류 분포 해석
# Cargo: 190,027
# Tanker: 78,535
# Fishing: 25,481
# Passenger: 17,825
# Tug: 10,112
# Military: 7,239

# 수치형 변수 해석
# 속도 sog
# 평균: 12.12
# 중앙값: 11.3
# 75%: 13.3
# 95%: 18.5
# 최대: 214.0

# 대부분 선박은 보통 중저속 항해를 하고 있다.
# 평균과 중앙값이 비슷해서 전체적으로는 큰 왜곡이 없지만, 최대값 214는 너무 크다.
# 이건 현실적으로 이상치이거나 입력 오류일 가능성이 높다.

# 방향 cog, heading
# cog 중앙값: 168.7
# heading 중앙값: 170
# 최대 heading: 507

# 방향값은 원형 데이터라 평균 자체는 큰 의미가 약하다.
# 중요한 건 범위가 0~360 부근이어야 하는데, heading 최대가 507이라는 점이다.
# 이건 분명히 비정상값이 섞여 있다는 뜻이다.

# 폭 width
# 중앙값: 17
# 평균: 19.95

# 길이 length
# 중앙값: 115
# 평균: 124.97
# 길이와 폭 분포를 보면 전반적으로 중대형 선박 비중이 높다고 해석할 수 있다.
# Cargo, Tanker 비중이 높은 것과 일치한다.

# 흘수 draught
# 중앙값: 6.1
# 평균: 6.57
# 흘수도 중간 정도 이상 값이 많아서,
# 이 데이터가 단순 소형 레저선 위주가 아니라 상선과 실무 선박 중심이라는 점을 다시 확인할 수 있다.


# 선박 종류별 특성 해석
# 중앙값 기준으로 보면 꽤 차이가 난다.
# Passenger: 길이 178, 속도 16.3
# Tanker: 길이 174, 속도 12.0
# Cargo: 길이 119, 속도 11.2
# HSC: 속도 34.5로 매우 빠름
# Military: 길이 43, 속도 11.7
# Fishing: 길이 17, 속도 7.4
# Tug: 길이 32, 속도 8.6
# 해석:
# Passenger는 길고 비교적 빠르다
# Tanker는 크고 흘수도 깊다
# Cargo는 가장 전형적인 상선 패턴이다
# Fishing은 소형이고 느리다
# Tug는 짧고 느리다
# Military는 생각보다 길이 중앙값이 작다

# 마지막 점은 중요하다.
# Military라고 해서 전부 대형 군함은 아니라는 뜻이다.
# 소형 군용 선박이나 지원선까지 섞였을 가능성이 크다.

# 결측치 해석
# sog: 458
# cog: 3,169
# heading: 20,614
# width: 3,711
# length: 3,743
# draught: 25,543
# 특히 heading과 draught 결측이 많다.
# 그래서 이 데이터는 전처리 없이 바로 모델에 넣기 어렵다.
# 해석상으로는:
# 방향 정보는 일부 선박에서 불완전할 수 있음
# 흘수는 모든 선박이 항상 정확히 입력하지 않는다는 뜻
# 결측 자체가 선박 유형과 관련이 있을 수도 있음
# 즉, 단순 삭제보다 결측 패턴 자체도 분석 가치가 있다.



data = pd.read_csv("ais_data.csv", index_col=0)
print(data.head(3), data.shape) # (358351, 10)
print(data.dtypes)
# mmsi                    int64
# navigationalstatus     object
# sog                   float64
# cog                   float64
# heading               float64
# shiptype               object
# width                 float64
# length                float64
# draught               float64
# dtype: object

