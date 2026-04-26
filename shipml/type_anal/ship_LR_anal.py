# LogisticRegression 모델
"""단독 LogisticRegression 선박 종류 분류기 실험 스크립트.

이 스크립트는 선형 모델 기준선을 처음부터 끝까지 보여준다. 가공된 AIS 특징
CSV를 읽고, 데이터를 분할하고, 숫자형 특징을 결측값 대체/스케일링하며,
범주형 특징을 원-핫 인코딩한다. 이후 balanced LogisticRegression 분류기를
학습하고 홀드아웃 진단을 출력한다. 배포 가능한 모델 번들 저장보다는 명확한
비교와 이해에 초점을 둔다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ais_ship_type_features.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    """가장 자주 발생한 잘못된 실제값-예측값 클래스 쌍을 출력한다."""

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

# 특징 테이블을 읽고 모델 입력에서 타깃 라벨을 분리한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 모델 학습 전에 스키마와 타깃 분포를 출력해, 불균형과 예상 밖의 컬럼을
# 확인할 수 있게 한다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# 빠른 모델 계열 비교를 위해 stratified 행 단위 홀드아웃을 사용한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# 로지스틱 회귀는 특징 스케일에 민감하므로 숫자형 컬럼은 중앙값 대체 후
# 표준화한다. 범주형 컬럼은 결측값 대체와 원-핫 인코딩을 적용하며, 예측 시
# 모르는 범주는 무시한다.
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

# balanced 클래스 가중치는 최적화 과정에서 희귀 선박 종류를 보정한다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ]
)

# train/test 변환 차이를 피하기 위해 전처리와 분류기를 함께 학습한다.
model.fit(X_train, y_train)

# train과 holdout 예측을 비교해 과소적합이나 과적합을 확인한다.
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

# 홀드아웃 예측을 라벨이 붙은 혼동 행렬로 변환한 뒤, 가장 큰 off-diagonal
# 오류를 출력한다.
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
