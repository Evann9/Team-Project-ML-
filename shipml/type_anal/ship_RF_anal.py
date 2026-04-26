# RandomForest 모델
"""단독 RandomForest 선박 종류 분류기 실험 스크립트.

이 스크립트는 RandomForest 학습 흐름을 직접 읽기 좋게 펼쳐둔 버전이다.
가공된 AIS 특징 CSV를 읽고, 행을 train/test로 나누고, 숫자형/범주형 컬럼의
전처리를 구성한 뒤, 분류기를 학습하고 집계 지표와 혼동 기반 평가 진단을
출력한다. 빠른 모델 계열 실험에 유용하며, 배포 가능한 번들 저장 로직은
공통 학습 스크립트에 있다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "ais_ship_type_features.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    """혼동 행렬의 off-diagonal 항목 중 가장 자주 발생한 항목을 출력한다.

    혼동 행렬에는 정답과 오답 예측이 모두 들어 있다. 이 헬퍼는 대각선을
    제외하고 남은 오류를 개수 기준으로 정렬한 뒤, 실제값-예측값 클래스 혼동
    중 가장 큰 항목을 출력한다. 전체 행렬을 훑지 않아도 모델의 약점을 볼 수
    있게 하기 위함이다.
    """

    errors = []
    for actual in cm_df.index:
        for predicted in cm_df.columns:
            if actual != predicted and cm_df.loc[actual, predicted] > 0:
                errors.append((actual, predicted, int(cm_df.loc[actual, predicted])))

    errors.sort(key=lambda x: x[2], reverse=True)

    print(f"\ntop {top_n} confusion pairs:")
    if not errors:
        print("no misclassifications")
        return

    for actual, predicted, count in errors[:top_n]:
        print(f"{actual} -> {predicted}: {count}")

# 특징 테이블을 읽고 지도학습 타깃을 입력 컬럼에서 분리한다.
# 이 파일은 현재 작업 디렉터리에 type-analysis 실험에 쓰는 생성된
# ``ais_ship_type_features.csv`` 데이터셋이 있다고 가정한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 학습 전에 스키마와 라벨 분포를 출력해, 클래스 불균형이나 예상 밖의 범주형
# 컬럼을 콘솔에서 바로 확인할 수 있게 한다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# stratified 분할은 양쪽 파티션의 선박 종류 클래스 비율을 대략 유지하므로,
# 빠른 홀드아웃 지표를 더 쉽게 해석할 수 있게 한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# 트리 모델은 숫자형 스케일링이 필요 없지만, 결측값 대체와 범주형 상태값의
# 원-핫 indicator 컬럼 확장은 여전히 필요하다.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# Pipeline은 학습 데이터에서 배운 전처리가 RandomForest 학습과 예측 전에
# 동일하게 적용되도록 보장한다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]
)

# 전처리와 분류기를 포함한 전체 파이프라인을 train 행에만 학습한다.
model.fit(X_train, y_train)

# train과 holdout 양쪽 예측을 만들어, train 정확도와 test 정확도의 차이로
# 과적합 여부를 확인한다.
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
report = classification_report(y_test, test_pred, output_dict=True)

print("\ntrain accuracy:", train_acc)
print("test accuracy:", test_acc)
print("macro f1:", report["macro avg"]["f1-score"])
print("weighted f1:", report["weighted avg"]["f1-score"])

print("\nclassification report:")
print(classification_report(y_test, test_pred))

# 라벨이 붙은 혼동 행렬을 만든 뒤 가장 큰 오류들을 요약한다.
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
