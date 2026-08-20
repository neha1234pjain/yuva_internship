"""
Week 1: Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: Titanic passenger data (public, via seaborn-data repo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
df = pd.read_csv("titanic.csv")
print("Raw shape:", df.shape)
print(df.dtypes)
print(df.isna().sum())

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
df_clean = df.copy()

# a) Drop duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Duplicates removed: {before - len(df_clean)}")

# b) 'deck' is ~77% missing -> drop the column (too sparse to impute reliably)
df_clean = df_clean.drop(columns=["deck"])

# c) 'age' missing (~20%) -> impute with median age WITHIN each passenger class,
#    since age correlates with class (first class skewed older)
df_clean["age"] = df_clean.groupby("pclass")["age"].transform(
    lambda x: x.fillna(x.median())
)

# d) 'embarked'/'embark_town' missing (2 rows) -> impute with mode
df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])
df_clean["embark_town"] = df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0])

# e) Correct data types
df_clean["survived"] = df_clean["survived"].astype("category")
df_clean["pclass"] = df_clean["pclass"].astype("category")
df_clean["sex"] = df_clean["sex"].astype("category")
df_clean["alone"] = df_clean["alone"].astype(bool)

print("\nCleaned shape:", df_clean.shape)
print("Remaining missing values:\n", df_clean.isna().sum())

df_clean.to_csv("titanic_clean.csv", index=False)

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
summary_stats = df_clean.describe(include="all")
summary_stats.to_csv("summary_stats.csv")
print(summary_stats)

# --- Visualization 1: Missing values heatmap (on RAW data, to justify cleaning) ---
plt.figure(figsize=(8, 5))
sns.heatmap(df.isna(), cbar=False, cmap="rocket_r")
plt.title("Missing Values Before Cleaning (Titanic Raw Data)")
plt.tight_layout()
plt.savefig("viz1_missing_values.png")
plt.show()
plt.close()

# --- Visualization 2: Age distribution by survival ---
plt.figure(figsize=(8, 5))
sns.histplot(data=df_clean, x="age", hue="survived", multiple="stack", bins=30, palette="viridis")
plt.title("Age Distribution by Survival Outcome")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig("viz2_age_distribution.png")
plt.show()
plt.close()

# --- Visualization 3: Correlation heatmap of numeric features ---
plt.figure(figsize=(7, 6))
numeric_df = df_clean[["age", "fare", "sibsp", "parch"]].copy()
numeric_df["survived"] = df_clean["survived"].astype(int)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig("viz3_correlation.png")
plt.show()
plt.close()

# --- Visualization 4 (bonus): Survival rate by class and sex ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df_clean, x="pclass", y=df_clean["survived"].astype(int), hue="sex", palette="Set2")
plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("viz4_survival_by_class_sex.png")
plt.show()
plt.close()

print("\nAll visualizations saved.")

# ---------------------------------------------------------
# 4. KEY INSIGHTS (printed for documentation)
# ---------------------------------------------------------
insights = []
overall_rate = df_clean["survived"].astype(int).mean()
insights.append(f"Overall survival rate: {overall_rate:.1%}")

female_rate = df_clean[df_clean.sex == "female"]["survived"].astype(int).mean()
male_rate = df_clean[df_clean.sex == "male"]["survived"].astype(int).mean()
insights.append(f"Female survival rate: {female_rate:.1%} vs Male: {male_rate:.1%}")

class1_rate = df_clean[df_clean.pclass == 1]["survived"].astype(int).mean()
class3_rate = df_clean[df_clean.pclass == 3]["survived"].astype(int).mean()
insights.append(f"1st class survival: {class1_rate:.1%} vs 3rd class: {class3_rate:.1%}")

for i in insights:
    print(i)

with open("insights.txt", "w") as f:
    f.write("\n".join(insights))
