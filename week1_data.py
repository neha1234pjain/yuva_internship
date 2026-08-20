# ============================================
# WEEK 1 - THREE EDA VISUALIZATIONS
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to store the three figures
os.makedirs("Week1_Visualizations", exist_ok=True)

# Set a professional Seaborn theme
sns.set_theme(style="whitegrid")


# --------------------------------------------
# FIGURE 1: Passenger Survival Count
# --------------------------------------------

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    x="survived",
    data=df
)

plt.title("Passenger Survival Count", fontsize=16, fontweight="bold")
plt.xlabel("Survival Status", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)

# Add labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fontsize=11)

plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.tight_layout()

plt.savefig(
    "Week1_Visualizations/Figure_1_Survival_Count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# --------------------------------------------
# FIGURE 2: Age Distribution
# --------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=20,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers",
          fontsize=16, fontweight="bold")
plt.xlabel("Age", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)

plt.tight_layout()

plt.savefig(
    "Week1_Visualizations/Figure_2_Age_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# --------------------------------------------
# FIGURE 3: Correlation Heatmap
# --------------------------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Variables",
          fontsize=16, fontweight="bold")

plt.tight_layout()

plt.savefig(
    "Week1_Visualizations/Figure_3_Correlation_Heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print("==========================================")
print("All 3 visualizations created successfully!")
print("==========================================")
print("Figure 1: Survival Count")
print("Figure 2: Age Distribution")
print("Figure 3: Correlation Heatmap")
print()
print("Files are saved inside:")
print("Week1_Visualizations/")