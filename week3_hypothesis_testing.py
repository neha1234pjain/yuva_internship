"""
Week 3: Statistical Analysis and Hypothesis Testing in Python
Building on the cleaned Titanic dataset from Weeks 1 & 2.

Hypotheses tested:
H1: Passenger class and survival are NOT independent (chi-square test)
H2: Sex and survival are NOT independent (chi-square test)
H3: Survivors paid a significantly different average fare than non-survivors (t-test)
H4 (bonus): Average fare differs significantly across the 3 passenger classes (ANOVA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("titanic_clean.csv")

alpha = 0.05
results_log = []

# ------------------------------------------------------------------
# H1: Chi-square test — Passenger Class vs Survival
# ------------------------------------------------------------------
contingency_class = pd.crosstab(df["class"], df["survived"])
chi2_class, p_class, dof_class, expected_class = stats.chi2_contingency(contingency_class)

print("=== H1: Class vs Survival (Chi-square) ===")
print(contingency_class)
print(f"Chi2 = {chi2_class:.3f}, dof = {dof_class}, p-value = {p_class:.6f}")
results_log.append(("H1 Class vs Survival", chi2_class, p_class, p_class < alpha))

# ------------------------------------------------------------------
# H2: Chi-square test — Sex vs Survival
# ------------------------------------------------------------------
contingency_sex = pd.crosstab(df["sex"], df["survived"])
chi2_sex, p_sex, dof_sex, expected_sex = stats.chi2_contingency(contingency_sex)

print("\n=== H2: Sex vs Survival (Chi-square) ===")
print(contingency_sex)
print(f"Chi2 = {chi2_sex:.3f}, dof = {dof_sex}, p-value = {p_sex:.6f}")
results_log.append(("H2 Sex vs Survival", chi2_sex, p_sex, p_sex < alpha))

# ------------------------------------------------------------------
# H3: Independent t-test — Fare paid: survivors vs non-survivors
# ------------------------------------------------------------------
fare_survived = df[df.survived == 1]["fare"]
fare_died = df[df.survived == 0]["fare"]

t_stat, p_fare = stats.ttest_ind(fare_survived, fare_died, equal_var=False)  # Welch's t-test
print("\n=== H3: Fare — Survivors vs Non-survivors (t-test) ===")
print(f"Mean fare (survived) = {fare_survived.mean():.2f}, Mean fare (died) = {fare_died.mean():.2f}")
print(f"t = {t_stat:.3f}, p-value = {p_fare:.6f}")
results_log.append(("H3 Fare: survived vs died", t_stat, p_fare, p_fare < alpha))

# ------------------------------------------------------------------
# H4 (bonus): One-way ANOVA — Fare across 3 classes
# ------------------------------------------------------------------
fare_first = df[df["class"] == "First"]["fare"]
fare_second = df[df["class"] == "Second"]["fare"]
fare_third = df[df["class"] == "Third"]["fare"]

f_stat, p_anova = stats.f_oneway(fare_first, fare_second, fare_third)
print("\n=== H4: Fare across Passenger Classes (ANOVA) ===")
print(f"F = {f_stat:.3f}, p-value = {p_anova:.6f}")
results_log.append(("H4 Fare across classes", f_stat, p_anova, p_anova < alpha))

# ------------------------------------------------------------------
# VISUALIZATIONS supporting the tests
# ------------------------------------------------------------------

# Viz 1: Survival counts by class (supports H1)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="class", hue="survived", order=["First", "Second", "Third"], palette=["#de2d26", "#2ca25f"])
plt.title(f"Survival Counts by Class (Chi-square p = {p_class:.4f})")
plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("test1_class_survival_counts.png")
plt.show()
plt.close()

# Viz 2: Survival counts by sex (supports H2)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="sex", hue="survived", palette=["#de2d26", "#2ca25f"])
plt.title(f"Survival Counts by Sex (Chi-square p = {p_sex:.6f})")
plt.xlabel("Sex")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("test2_sex_survival_counts.png")
plt.show()
plt.close()

# Viz 3: Fare distribution by survival (supports H3)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="survived", y="fare", palette=["#de2d26", "#2ca25f"])
plt.title(f"Fare Paid by Survival Outcome (t-test p = {p_fare:.6f})")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare Paid (£)")
plt.ylim(0, 300)
plt.tight_layout()
plt.savefig("test3_fare_by_survival_boxplot.png")
plt.show()
plt.close()

# Viz 4: Fare distribution by class (supports H4)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="class", y="fare", order=["First", "Second", "Third"], palette="Set2")
plt.title(f"Fare Paid by Passenger Class (ANOVA p = {p_anova:.2e})")
plt.xlabel("Passenger Class")
plt.ylabel("Fare Paid (£)")
plt.ylim(0, 300)
plt.tight_layout()
plt.savefig("test4_fare_by_class_boxplot.png")
plt.show()
plt.close()

print("\nAll hypothesis tests complete and visualizations saved.")

# Save results summary to CSV for reference
results_df = pd.DataFrame(results_log, columns=["Test", "Statistic", "p_value", "Significant_at_0.05"])
results_df.to_csv("hypothesis_test_results.csv", index=False)
print(results_df)
