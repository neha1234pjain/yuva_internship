"""
Week 4: Machine Learning Model Development and Evaluation
Building on Weeks 1-3: predicting Titanic survival using the features
(class, sex, age, fare) that Week 3 confirmed are statistically significant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("titanic_clean.csv")

# ------------------------------------------------------------------
# 1. FEATURE PREPARATION
# ------------------------------------------------------------------
# Using the features Week 3 confirmed as statistically significant:
# pclass, sex, fare, age. Also adding sibsp/parch (family size) as
# reasonable additional predictors.
features = df[["pclass", "sex", "age", "fare", "sibsp", "parch"]].copy()
features["sex"] = features["sex"].map({"male": 0, "female": 1})
target = df["survived"]

print("Feature matrix shape:", features.shape)
print(features.head())

# ------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Scale features (helps logistic regression converge and treats features fairly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. MODEL 1: LOGISTIC REGRESSION
# ------------------------------------------------------------------
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_log))

# ------------------------------------------------------------------
# 4. MODEL 2: DECISION TREE (for comparison)
# ------------------------------------------------------------------
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)  # trees don't need scaling
y_pred_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:, 1]

print("\n=== Decision Tree (max_depth=4) ===")
print(classification_report(y_test, y_pred_tree))

# ------------------------------------------------------------------
# 5. METRICS SUMMARY
# ------------------------------------------------------------------
def get_metrics(y_true, y_pred, name):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

metrics_summary = pd.DataFrame([
    get_metrics(y_test, y_pred_log, "Logistic Regression"),
    get_metrics(y_test, y_pred_tree, "Decision Tree")
])
print("\n=== Metrics Summary ===")
print(metrics_summary)
metrics_summary.to_csv("model_metrics.csv", index=False)

# ------------------------------------------------------------------
# 6. VISUALIZATION 1: Confusion matrices side by side
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, name in zip(axes, [y_pred_log, y_pred_tree], ["Logistic Regression", "Decision Tree"]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Did not survive", "Survived"],
                yticklabels=["Did not survive", "Survived"])
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("model1_confusion_matrices.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# 7. VISUALIZATION 2: ROC curves
# ------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for y_prob, name, color in [(y_prob_log, "Logistic Regression", "#2ca25f"),
                              (y_prob_tree, "Decision Tree", "#de2d26")]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model2_roc_curves.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# 8. VISUALIZATION 3 (bonus): Feature importance (tree) / coefficients (logreg)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

coefs = pd.Series(log_reg.coef_[0], index=features.columns).sort_values()
axes[0].barh(coefs.index, coefs.values, color=["#de2d26" if v < 0 else "#2ca25f" for v in coefs.values])
axes[0].set_title("Logistic Regression Coefficients\n(scaled features)")
axes[0].set_xlabel("Coefficient (impact on survival log-odds)")

importances = pd.Series(tree.feature_importances_, index=features.columns).sort_values()
axes[1].barh(importances.index, importances.values, color="#3182bd")
axes[1].set_title("Decision Tree Feature Importance")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("model3_feature_importance.png")
plt.show()
plt.close()

print("\nAll models trained, evaluated, and visualizations saved.")
