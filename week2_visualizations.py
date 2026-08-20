"""
Week 2: Advanced Data Visualization and Storytelling with Python
Building on the cleaned Titanic dataset from Week 1.
Goal: tell a clear visual story for a non-technical audience.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

df = pd.read_csv("titanic_clean.csv")
df["survived_label"] = df["survived"].map({0: "Did not survive", 1: "Survived"})

palette_survival = {"Survived": "#2ca25f", "Did not survive": "#de2d26"}

# ------------------------------------------------------------------
# STORY BEAT 1: The headline number — who made it, who didn't
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
counts = df["survived_label"].value_counts()
bars = plt.bar(counts.index, counts.values,
                color=[palette_survival[c] for c in counts.index])
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h}\n({h/len(df):.0%})",
              ha="center", fontsize=11, fontweight="bold")
plt.title("Only 4 in 10 Passengers Survived the Titanic", fontsize=14, fontweight="bold")
plt.ylabel("Number of Passengers")
plt.ylim(0, max(counts.values) * 1.25)
plt.tight_layout()
plt.savefig("story1_headline.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# STORY BEAT 2: Class mattered — a lot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
class_survival = df.groupby("class")["survived"].mean().reindex(["First", "Second", "Third"])
bars = plt.bar(class_survival.index, class_survival.values,
                color=["#31a354", "#addd8e", "#de2d26"])
for bar, val in zip(bars, class_survival.values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.0%}",
              ha="center", fontsize=12, fontweight="bold")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Your Ticket Class Was Almost Your Fate", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.ylim(0, 0.8)
plt.tight_layout()
plt.savefig("story2_class.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# STORY BEAT 3: Sex was the strongest single factor
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sex_survival = df.groupby("sex")["survived"].mean().reindex(["female", "male"])
bars = plt.bar(["Women", "Men"], sex_survival.values, color=["#e78ac3", "#8da0cb"])
for bar, val in zip(bars, sex_survival.values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.0%}",
              ha="center", fontsize=12, fontweight="bold")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("'Women and Children First' Shows Clearly in the Data", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.ylim(0, 0.9)
plt.tight_layout()
plt.savefig("story3_sex.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# STORY BEAT 4: Class AND sex together — the combined picture
# ------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
pivot = df.pivot_table(index="class", columns="sex", values="survived", aggfunc="mean")
pivot = pivot.reindex(["First", "Second", "Third"])
pivot.plot(kind="bar", ax=plt.gca(), color=["#e78ac3", "#8da0cb"], width=0.7)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("The Combined Effect: Class + Sex Determined Survival Odds", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.xticks(rotation=0)
plt.legend(title="Sex", labels=["Women", "Men"])
plt.tight_layout()
plt.savefig("story4_class_sex_combined.png")
plt.show()
plt.close()

# ------------------------------------------------------------------
# STORY BEAT 5: Age + fare + survival in one view (bubble/scatter)
# ------------------------------------------------------------------
plt.figure(figsize=(9, 6))
scatter = plt.scatter(df["age"], df["fare"],
                       c=df["survived"].map({0: "#de2d26", 1: "#2ca25f"}),
                       alpha=0.55, s=40, edgecolor="white", linewidth=0.3)
plt.title("Where Age and Fare Meet: Higher Fares Cluster with Survival", fontsize=13, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Fare Paid (£)")
plt.ylim(0, 300)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Survived', markerfacecolor='#2ca25f', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Did not survive', markerfacecolor='#de2d26', markersize=10)
]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("story5_age_fare_scatter.png")
plt.show()
plt.close()

print("All 5 storytelling visualizations generated.")

# Quick stats for the narrative text
print("Class survival:\n", class_survival)
print("Sex survival:\n", sex_survival)
print("Combined pivot:\n", pivot)
