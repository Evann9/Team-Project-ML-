# SVC 모델 - 너무 느려서 안돌아감
"""단독 SVC 선박 종류 분류기 실험 스크립트.

이 파일은 support-vector 분류기 실험을 다른 모델 계열 스크립트와 같은
형태로 유지한다. 특징을 읽고, 스케일링과 원-핫 인코딩을 적용한 뒤, balanced
클래스 가중치를 쓰는 RBF-kernel SVC를 학습하고 홀드아웃 지표를 출력한다.
kernel SVC는 학습 행이 많을 때 확장성이 좋지 않아 이 데이터셋에서는 느릴 수
있다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DATA_PATH = "ais_ship_type_features.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    """라벨이 붙은 혼동 행렬에서 가장 큰 off-diagonal 항목을 출력한다."""

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

# 특징 행을 읽고 지도학습 선박 종류 타깃을 분리한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# SVC 실행 시간이 길 수 있으므로 학습 전에 스키마와 타깃 분포를 출력한다.
# 이렇게 하면 긴 학습을 기다리기 전에 잘못된 입력 파일을 더 빨리 발견할 수 있다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# stratification은 빠른 홀드아웃 추정에서 클래스 균형을 보존한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# Kernel SVC는 특징 공간의 거리를 사용하므로 숫자형 표준화가 필요하다.
# 원-핫 인코딩은 범주형 navigation-status 값을 처리한다.
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

# balanced 클래스 가중치는 희귀 선박 종류 클래스를 무시하는 경향을 줄인다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            ),
        ),
    ]
)

# 전체 파이프라인을 train 행에 학습한다. RBF SVC 학습은 sample 수가 늘수록
# 빠르게 비싸지므로 트리 모델보다 훨씬 느릴 수 있다.
model.fit(X_train, y_train)

# train과 holdout 성능을 모두 평가해 과적합을 드러낸다.
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

# 혼동 행렬을 만들고 가장 자주 발생한 클래스 혼동을 출력한다.
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
