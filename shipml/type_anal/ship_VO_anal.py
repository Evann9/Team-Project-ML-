# Voting 모델
"""단독 soft-voting 선박 종류 분류기 실험 스크립트.

이 스크립트는 LogisticRegression, RandomForestClassifier, ExtraTreesClassifier를
soft voting으로 섞은 앙상블을 비교한다. 파이프라인은 공유 전처리를 사용하고,
AIS 종류 특징에 앙상블을 학습한 뒤, 홀드아웃 지표와 혼동 요약을 출력한다.
후보를 배포용 학습기로 승격하기 전에 읽기 쉬운 실험으로 확인하는 데 유용하다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ais_ship_type_features.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    """빠른 오류 검토를 위해 가장 큰 off-diagonal 혼동 쌍을 출력한다."""

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

# 가공된 특징 테이블을 읽고 타깃 라벨을 분리한다.
df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 앙상블 성능은 희귀 클래스와 예상 밖의 범주형 특징 변화에 영향을 받을 수
# 있으므로 스키마와 라벨 분포를 출력한다.
print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

# 충분히 표현된 각 선박 종류가 train/test 양쪽에 나타나도록 stratified
# 홀드아웃을 사용한다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

# 앙상블에 선형 learner가 포함되어 있으므로 숫자형 특징을 표준화한다.
# 트리 learner는 스케일링에 강건하므로, 빠른 비교에서는 스케일된 표현을
# 공유해도 괜찮다.
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

# soft voting은 서로 보완적인 learner들의 클래스 확률을 평균낸다. 여기서는
# 선형 모델, bagged forest, extremely randomized forest를 사용한다.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            VotingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                        ),
                    ),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        ),
                    ),
                ],
                voting="soft",
                n_jobs=1,
            ),
        ),
    ]
)

# 전체 앙상블 파이프라인을 train 행에만 학습한다.
model.fit(X_train, y_train)

# train과 holdout 예측을 비교해 앙상블이 암기하는지 확인한다.
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
report = classification_report(y_test, test_pred, output_dict=True)

print("\ntrain accuracy:", train_acc)
print("test accuracy:", test_acc)
print("macro f1:", report["macro avg"]["f1-score"])
print("weighted f1:", report["weighted avg"]["f1-score"])
# train accuracy: 0.9999957447894947
# test accuracy: 0.9641884531590414
# macro f1: 0.9499696517032737
# weighted f1: 0.9637415880404236

print("\nclassification report:")
print(classification_report(y_test, test_pred))
# classification report:
#                  precision    recall  f1-score   support

#           Cargo       0.95      0.99      0.97     34389
#        Dredging       0.98      0.94      0.96       795
#         Fishing       1.00      0.99      1.00      2585
#             HSC       0.99      0.97      0.98       740
# Law enforcement       0.98      0.94      0.96       320
#        Military       0.98      0.99      0.99       865
#       Passenger       0.99      0.98      0.99      1928
#           Pilot       1.00      1.00      1.00       395
#        Pleasure       0.82      0.90      0.86        20
#             SAR       0.94      0.93      0.94       101
#         Sailing       0.82      0.78      0.80        36
#          Tanker       0.98      0.89      0.93     14859
#          Towing       0.96      0.92      0.94       186
#             Tug       0.99      1.00      0.99      1533

#        accuracy                           0.96     58752
#       macro avg       0.96      0.95      0.95     58752
#    weighted avg       0.96      0.96      0.96     58752

# 홀드아웃 예측을 혼동 행렬로 변환하고, 상세 report 뒤에 가장 큰 오분류 쌍을
# 나열한다.
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
# top 10 confusion pairs:
# Tanker -> Cargo: 1622
# Cargo -> Tanker: 269
# Dredging -> Cargo: 39
# Passenger -> Cargo: 30
# Cargo -> Passenger: 14
# Towing -> Tug: 14
# Fishing -> Cargo: 11
# HSC -> Cargo: 11
# Cargo -> Dredging: 8
# Law enforcement -> Military: 6
