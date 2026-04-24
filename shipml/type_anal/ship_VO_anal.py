# Voting 모델

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ais_data_with_trig.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
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


df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)
print(df.head())

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

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

model.fit(X_train, y_train)

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