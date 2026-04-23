import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")

df = pd.read_csv("shiptype/shiptype_cleaned/ais_Military_cleaned.csv")

# 1. 기본 확인
print(df.shape)
print(df.info())
print(df.isna().mean().sort_values(ascending=False) * 100)
print(df.describe())

numeric_cols = ["sog", "cog", "heading", "width", "length", "draught"]
numeric_cols = [col for col in numeric_cols if col in df.columns]

# 2. 히스토그램
df[numeric_cols].hist(bins=30, figsize=(10, 8))
plt.suptitle("Military Numeric Feature Distributions")
plt.tight_layout()
plt.show()

# 3. 박스플롯
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.show()

# 4. 상관관계 히트맵
corr_cols = ["sog", "width", "length", "draught"]
corr_cols = [col for col in corr_cols if col in df.columns]

plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 5. 산점도
sns.pairplot(df[corr_cols].dropna())
plt.show()

# 6. 상태별 분포
if "navigationalstatus" in df.columns:
    plt.figure(figsize=(10, 8))
    sns.countplot(data=df, y="navigationalstatus",
                order=df["navigationalstatus"].value_counts().index)
    plt.title("Navigational Status Distribution")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df, x="navigationalstatus", y="sog")
    plt.xticks(rotation=45)
    plt.title("SOG by Navigational Status")
    plt.show()