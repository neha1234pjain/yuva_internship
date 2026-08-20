"""
Week 5: Comprehensive Data Science Project Reporting and Strategic Recommendations
This script pulls together the Week 1-4 workflow into one place: reloads the
cleaned data, re-runs the key summary numbers from each week, and regenerates
the 4 headline visualizations referenced in the final report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("titanic_clean.csv")
print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ==================================================================
# WEEK 1 RECAP: Cleaning summary + age/survival chart
# ==================================================================
print("\n--- WEEK 1 RECAP ---")
print(f"Overall survival rate: {df['survived'].mean():.1%}")

plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="age", hue="survived", multiple="stack", bins=30, palette="viridis")
plt.title("Week 1: Age Distribution by Survival")
plt.tight_layout()
plt.savefig("final_fig1_age_distribution.png")
plt.show()
plt.close()

# ==================================================================
# WEEK 2 RECAP: Class + sex combined survival rate
# ==================================================================
print("\n--- WEEK 2 RECAP ---")
pivot = df.pivot_table(index="class", columns="sex", values="survived", aggfunc="mean")
pivot = pivot.reindex(["First", "Second", "Third"])
print(pivot)

plt.figure(figsize=(7, 4.5))
pivot.plot(kind="bar", ax=plt.gca(), color=["#e78ac3", "#8da0cb"], width=0.7)
plt.title("Week 2: Survival Rate by Class + Sex")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.legend(title="Sex", labels=["Women", "Men"])
plt.tight_layout()
plt.savefig("final_fig2_class_sex_combined.png")
plt.show()
plt.close()

# ==================================================================
# WEEK 3 RECAP: Hypothesis test results
# ==================================================================
print("\n--- WEEK 3 RECAP ---")
contingency_class = pd.crosstab(df["class"], df["survived"])
chi2_class, p_class, _, _ = stats.chi2_contingency(contingency_class)
print(f"Class vs Survival: chi2={chi2_class:.2f}, p={p_class:.6f}")

contingency_sex = pd.crosstab(df["sex"], df["survived"])
chi2_sex, p_sex, _, _ = stats.chi2_contingency(contingency_sex)
print(f"Sex vs Survival: chi2={chi2_sex:.2f}, p={p_sex:.6f}")

fare_survived = df[df.survived == 1]["fare"]
fare_died = df[df.survived == 0]["fare"]
t_stat, p_fare = stats.ttest_ind(fare_survived, fare_died, equal_var=False)
print(f"Fare (survived vs died): t={t_stat:.2f}, p={p_fare:.6f}")

plt.figure(figsize=(7, 4.5))
sns.countplot(data=df, x="sex", hue="survived", palette=["#de2d26", "#2ca25f"])
plt.title(f"Week 3: Sex vs Survival (chi-square p = {p_sex:.6f})")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("final_fig3_sex_survival_test.png")
plt.show()
plt.close()

# ==================================================================
# WEEK 4 RECAP: Re-train models, compare ROC curves
# ==================================================================
print("\n--- WEEK 4 RECAP ---")
features = df[["pclass", "sex", "age", "fare", "sibsp", "parch"]].copy()
features["sex"] = features["sex"].map({"male": 0, "female": 1})
target = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
acc_log = accuracy_score(y_test, log_reg.predict(X_test_scaled))

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_prob_tree = tree.predict_proba(X_test)[:, 1]
acc_tree = accuracy_score(y_test, tree.predict(X_test))

print(f"Logistic Regression accuracy: {acc_log:.1%}")
print(f"Decision Tree accuracy: {acc_tree:.1%}")

plt.figure(figsize=(6, 5))
for y_prob, name, color in [(y_prob_log, "Logistic Regression", "#2ca25f"),
                              (y_prob_tree, "Decision Tree", "#de2d26")]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Week 4: Model ROC Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("final_fig4_roc_comparison.png")
plt.show()
plt.close()

# ==================================================================
# FINAL SUMMARY TABLE (printed for the report)
# ==================================================================
print("\n--- FINAL PROJECT SUMMARY ---")
summary = pd.DataFrame({
    "Week": [1, 2, 3, 4],
    "Focus": ["Data cleaning & EDA", "Storytelling visualization",
              "Hypothesis testing", "ML model"],
    "Key Result": [
        f"{df.shape[0]} clean records; survival rate {df['survived'].mean():.1%}",
        f"1st class women {pivot.loc['First','female']:.0%} vs 3rd class men {pivot.loc['Third','male']:.0%}",
        f"Class p={p_class:.4f}, Sex p={p_sex:.6f}, Fare p={p_fare:.6f}",
        f"LogReg {acc_log:.1%} acc / Tree {acc_tree:.1%} acc"
    ]
})
print(summary.to_string(index=False))
summary.to_csv("final_project_summary.csv", index=False)

print("\nAll Week 5 recap visualizations and summary saved.")
