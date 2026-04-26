# KNeighborsClassifier 모델
"""단독 KNeighbors 선박 종류 분류기 실험 스크립트.

이 스크립트는 AIS 종류 특징에 거리 기반 KNN 기준선을 실행한다. KNN은 특징
거리로 행을 비교하므로, 학습 전에 숫자형 스케일링과 범주형 원-핫 확장을
사용한다. 다른 모델 계열 실험과 비교할 수 있도록 빠른 홀드아웃 지표와
혼동 요약을 출력한다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ais_ship_type_features.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    """혼동 행렬에서 가장 흔한 잘못된 클래스 쌍 예측을 출력한다."""

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

# 가공된 AIS 특징 CSV를 읽고 predictor에서 선박 종류 타깃을 분리한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 거리 기반 모델을 학습하기 전에 스키마 이상이나 클래스 불균형을 쉽게 찾을 수
# 있도록 간단한 데이터 점검 정보를 출력한다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# 더 공정한 빠른 지표를 위해 홀드아웃 분할에서 클래스 비율을 안정적으로 유지한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# KNN은 거리에 직접 의존하므로 숫자형 컬럼 스케일링이 필수다. 범주형 값은
# 임의의 정수 코드가 아니라 binary 거리 성분으로 작동하도록 원-핫 인코딩한다.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
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

# 거리 가중치는 선박 종류 투표 시 가까운 AIS 특징 이웃이 먼 이웃보다 더 큰
# 영향을 갖도록 한다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
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
)

# 파이프라인을 학습한다. KNN에서 학습은 주로 변환된 학습 예시를 저장하는 과정이다.
model.fit(X_train, y_train)

# train과 holdout 행을 예측해 암기나 낮은 일반화 성능을 확인한다.
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

# 홀드아웃 혼동 행렬에서 가장 큰 오분류 쌍을 요약한다.
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
