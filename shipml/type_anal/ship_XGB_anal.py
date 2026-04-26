# XGBoost 모델
"""단독 XGBoost 선박 종류 분류기 실험 스크립트.

이 스크립트는 AIS 종류 특징에서 XGBoost 다중 클래스 분류기를 평가한다.
XGBoost는 정수 클래스 ID가 필요하므로 학습 전에 선박 종류 라벨을 인코딩하고,
지표를 출력하기 전에 다시 디코딩한다. 배포용 및 그룹 분할 워크플로는 공통
학습 헬퍼를 통해 같은 모델링 아이디어를 재사용하며, 이 파일은 직접 실험용
버전이다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


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

# 가공된 AIS 특징을 읽고 입력 컬럼과 문자열 라벨을 분리한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# XGBoost를 위해 라벨을 정수로 인코딩하기 전에 스키마와 클래스 분포를 확인한다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# XGBoost의 sklearn API는 이 objective에서 숫자형 다중 클래스 라벨을 기대한다.
# 예측값을 다시 선박 종류 이름으로 변환할 수 있도록 encoder를 보관한다.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 인코딩된 라벨로 stratify하면 원래 문자열 타깃과 같은 클래스 비율을 보존한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# 트리 부스팅은 숫자형 스케일링이 필요 없다. 표 형태 행렬을 만들기 위해
# 누락된 숫자값은 중앙값으로 대체하고 범주형 값은 원-핫 인코딩한다.
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

# 분류기는 다중 클래스 선박 종류 예측용으로 설정되어 있으며, 나중에 평가
# 데이터가 제공되면 학습 중 log-loss를 보고한다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                random_state=42,
            ),
        ),
    ]
)

# 전처리와 XGBoost 분류기를 인코딩된 라벨에 함께 학습한다.
model.fit(X_train, y_train)

# 사람이 읽기 쉬운 지표와 혼동 행을 계산하기 전에 예측값을 디코딩한다.
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred_labels = label_encoder.inverse_transform(train_pred)
test_pred_labels = label_encoder.inverse_transform(test_pred)
y_train_labels = label_encoder.inverse_transform(y_train)
y_test_labels = label_encoder.inverse_transform(y_test)

train_acc = accuracy_score(y_train_labels, train_pred_labels)
test_acc = accuracy_score(y_test_labels, test_pred_labels)
report = classification_report(y_test_labels, test_pred_labels, output_dict=True)

print("\ntrain accuracy:", train_acc)
print("test accuracy:", test_acc)
print("macro f1:", report["macro avg"]["f1-score"])
print("weighted f1:", report["weighted avg"]["f1-score"])

print("\nclassification report:")
print(classification_report(y_test_labels, test_pred_labels))

# 원래 라벨 공간에서 혼동 행렬을 만든다.
labels = sorted(label_encoder.classes_)
cm = confusion_matrix(y_test_labels, test_pred_labels, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
